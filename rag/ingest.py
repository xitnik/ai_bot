from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class DocumentChunk:
    doc_id: str
    text: str
    source: str


def chunk_text(text: str, max_tokens: int = 120) -> List[str]:
    tokens = text.split()
    chunks: List[str] = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        if chunk_tokens:
            chunks.append(" ".join(chunk_tokens))
    return chunks


def load_local_corpus(root: Path) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    for path in sorted(root.glob("*.txt")):
        content = path.read_text(encoding="utf-8")
        for idx, chunk in enumerate(chunk_text(content)):
            chunks.append(DocumentChunk(doc_id=f"{path.stem}:{idx}", text=chunk, source=str(path)))
    return chunks


def ingest_from_dir(directory: str | Path) -> List[DocumentChunk]:
    return load_local_corpus(Path(directory))
