from __future__ import annotations

import csv
import json
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from rag.chunking import ChunkingConfig, DocumentChunk, split_text_sliding_window

# Backward-compatible alias used across the codebase.
Document = DocumentChunk


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_pdf_pages(path: Path) -> List[Tuple[int, str]]:
    """Best-effort PDF reader; returns (page_idx, text) tuples."""
    try:
        import PyPDF2  # type: ignore
    except Exception:
        return []
    pages: List[Tuple[int, str]] = []
    try:
        with path.open("rb") as stream:
            reader = PyPDF2.PdfReader(stream)
            for page_idx, page in enumerate(reader.pages):
                extracted = page.extract_text() or ""
                pages.append((page_idx, extracted))
    except Exception:
        return []
    return pages


def _read_docx_file(path: Path) -> str:
    """Reads DOCX with optional dependency."""
    try:
        import docx  # type: ignore
    except Exception:
        return ""
    try:
        document = docx.Document(str(path))
        return "\n".join([para.text for para in document.paragraphs])
    except Exception:
        return ""


def _read_price_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames:
            return [dict(row) for row in reader]
        handle.seek(0)
        reader_generic = csv.reader(handle)
        return [{"row": " ".join(row)} for row in reader_generic]


def _read_price_excel(path: Path) -> List[Dict[str, Any]]:
    """Reads Excel via pandas if available; returns list of row dicts."""
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return []
    try:
        df = pd.read_excel(path)
    except Exception:
        return []
    return df.fillna("").to_dict(orient="records")


def _build_doc_id(path: Path, suffix: str | None = None) -> str:
    base = path.stem
    return f"{base}:{suffix}" if suffix is not None else base


def _base_metadata(
    path: Path, source_type: str, extra: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    stat = path.stat() if path.exists() else None
    meta: Dict[str, Any] = {
        "source": path.name,
        "source_path": str(path.resolve()),
        "source_type": source_type,
        "mime_type": mimetypes.guess_type(path.name)[0] or source_type,
        "lang": None,
        "ingested_at": datetime.utcnow().isoformat(),
        "created_at": datetime.utcfromtimestamp(stat.st_ctime).isoformat() if stat else None,
        "updated_at": datetime.utcfromtimestamp(stat.st_mtime).isoformat() if stat else None,
    }
    if extra:
        meta.update(extra)
    return meta


def _chunk_text(
    text: str,
    *,
    doc_id: str,
    page: Optional[int],
    chunking: ChunkingConfig,
    base_metadata: Dict[str, Any],
) -> List[Document]:
    return split_text_sliding_window(
        text,
        doc_id=doc_id,
        page=page,
        base_metadata=base_metadata,
        chunk_size_tokens=chunking.chunk_size_tokens,
        chunk_overlap_tokens=chunking.chunk_overlap_tokens,
        tokenizer=chunking.tokenizer,
    )


def _documents_from_rows(
    rows: Iterable[Dict[str, Any]],
    path: Path,
    source_type: str,
    chunking: ChunkingConfig,
    extra_metadata: Dict[str, Any] | None = None,
) -> List[Document]:
    docs: List[Document] = []
    for idx, row in enumerate(rows):
        text = " ".join(f"{k}: {v}" for k, v in row.items() if v not in (None, ""))
        doc_id = _build_doc_id(path, f"row_{idx}")
        metadata = _base_metadata(path, source_type, extra_metadata)
        metadata.update({"row_index": idx})
        docs.extend(_chunk_text(text, doc_id=doc_id, page=None, chunking=chunking, base_metadata=metadata))
    return docs


def load_price_documents(
    path: str | Path,
    chunking: ChunkingConfig | None = None,
    base_metadata: Dict[str, Any] | None = None,
) -> List[Document]:
    """Loads price lists from CSV/Excel and returns chunked documents."""
    cfg = chunking or ChunkingConfig()
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    rows: List[Dict[str, Any]]
    if suffix in {".csv"}:
        rows = _read_price_csv(file_path)
    elif suffix in {".xlsx", ".xls"}:
        rows = _read_price_excel(file_path)
    else:
        return []
    return _documents_from_rows(rows, file_path, "price_list", cfg, base_metadata)


def load_contract_documents(
    path: str | Path,
    chunking: ChunkingConfig | None = None,
    base_metadata: Dict[str, Any] | None = None,
) -> List[Document]:
    """Loads contract/regulation docs (txt/pdf/docx) and chunks them."""
    cfg = chunking or ChunkingConfig()
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        text = _read_text_file(file_path)
        pages = [(None, text)] if text else []
    elif suffix == ".pdf":
        pages = _read_pdf_pages(file_path)
    elif suffix in {".docx", ".doc"}:
        text = _read_docx_file(file_path)
        pages = [(None, text)] if text else []
    else:
        pages = []

    if not pages:
        return []

    docs: List[Document] = []
    for page_idx, text in pages:
        metadata = _base_metadata(file_path, "contract", base_metadata)
        doc_id = _build_doc_id(file_path, str(page_idx) if page_idx is not None else None)
        docs.extend(
            _chunk_text(
                text,
                doc_id=doc_id,
                page=page_idx if page_idx is not None else None,
                chunking=cfg,
                base_metadata=metadata,
            )
        )
    return docs


def load_messages_documents(
    messages: Sequence[Dict[str, Any]],
    source_name: str = "chatlog",
    chunking: ChunkingConfig | None = None,
    base_metadata: Dict[str, Any] | None = None,
) -> List[Document]:
    """
    Converts chat/email logs into documents.

    Expected message dict fields: `id`, `text`, `author`, `client_id`, `timestamp`.
    """
    cfg = chunking or ChunkingConfig()
    docs: List[Document] = []
    for idx, msg in enumerate(messages):
        text = msg.get("text") or ""
        if not text:
            continue
        meta = {
            "source_type": "message",
            "source": source_name,
            "author": msg.get("author"),
            "client_id": msg.get("client_id"),
            "timestamp": msg.get("timestamp"),
        }
        if base_metadata:
            meta.update(base_metadata)
        doc_id = f"{source_name}:{msg.get('id', idx)}"
        docs.extend(_chunk_text(text, doc_id=doc_id, page=None, chunking=cfg, base_metadata=meta))
    return docs


def ingest_from_dir(
    directory: str | Path,
    chunking: ChunkingConfig | None = None,
    default_lang: str | None = "ru",
) -> List[Document]:
    """
    Ingests mixed corpus from a directory.

    Supports .txt/.pdf/.docx files and CSV/Excel price lists.
    """
    cfg = chunking or ChunkingConfig()
    root = Path(directory)
    docs: List[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        base_meta = {"lang": default_lang}
        if suffix in {".txt", ".pdf", ".docx", ".doc"}:
            docs.extend(load_contract_documents(path, cfg, base_meta))
        elif suffix in {".csv", ".xlsx", ".xls"}:
            docs.extend(load_price_documents(path, cfg, base_meta))
        elif suffix in {".json"} and path.name.startswith("messages"):
            try:
                messages = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(messages, list):
                    docs.extend(load_messages_documents(messages, path.stem, cfg, base_meta))
            except Exception:
                continue
    return docs
