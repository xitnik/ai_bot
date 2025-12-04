from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

DEFAULT_CHUNK_SIZE_TOKENS = 768
DEFAULT_CHUNK_OVERLAP_TOKENS = 128


def _approximate_tokenize_with_offsets(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Best-effort tokenizer with offsets.

    - Splits on whitespace.
    - Approximates token count by 4 characters per token when words are long.
    - TODO: replace with model-aware tokenizer once available in runtime.
    """
    tokens: List[str] = []
    offsets: List[Tuple[int, int]] = []
    for match in re.finditer(r"\S+", text):
        word = match.group()
        start, end = match.start(), match.end()
        token_count = max(1, math.ceil(len(word) / 4))
        for idx in range(token_count):
            # Keep offsets to map back to original text bounds.
            tokens.append(word if token_count == 1 else f"{word}#{idx}")
            offsets.append((start, end))
    return tokens, offsets


def _approximate_offsets_from_tokens(text: str, tokens: Sequence[str]) -> List[Tuple[int, int]]:
    """
    Fallback offsets builder when a tokenizer without offsets is provided.
    Searches sequentially to avoid wrong matches for repeated tokens.
    """
    offsets: List[Tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        idx = text.find(token, cursor)
        if idx == -1:
            # If we cannot find a token, approximate using previous cursor.
            start = cursor
            end = min(len(text), start + max(1, len(token)))
        else:
            start = idx
            end = start + len(token)
            cursor = end
        offsets.append((start, end))
    return offsets


@dataclass
class DocumentChunk:
    """
    Canonical representation of a chunked document.

    id: unique chunk id
    doc_id: id of source document (file/logical document)
    text: chunk text
    page: optional page number for paged formats
    start_char / end_char: byte offsets into original text
    metadata: arbitrary metadata, preserved across ingestion/retrieval
    embedding: optional vector attached post-embedding
    """

    id: str
    text: str
    doc_id: Optional[str] = None
    page: Optional[int] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", ""))

    def __post_init__(self) -> None:
        if self.doc_id is None:
            self.doc_id = self.id


Tokenizer = Callable[[str], List[str]]


@dataclass
class ChunkingConfig:
    """Configuration controlling sliding-window chunking."""

    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS
    tokenizer: Optional[Tokenizer] = None


def _tokenize_with_offsets(
    text: str, tokenizer: Optional[Tokenizer]
) -> Tuple[List[str], List[Tuple[int, int]]]:
    if tokenizer is None:
        return _approximate_tokenize_with_offsets(text)
    tokens = tokenizer(text)
    if not tokens:
        return [], []
    offsets = _approximate_offsets_from_tokens(text, tokens)
    return tokens, offsets


def _chunk_bounds(
    offsets: Sequence[Tuple[int, int]], start: int, end: int
) -> Tuple[Optional[int], Optional[int]]:
    if not offsets or start >= len(offsets) or start >= end:
        return None, None
    safe_end = min(len(offsets), end) - 1
    return offsets[start][0], offsets[safe_end][1]


def split_text_sliding_window(
    text: str,
    *,
    doc_id: str,
    page: Optional[int] = None,
    base_metadata: Optional[Dict[str, Any]] = None,
    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    tokenizer: Optional[Tokenizer] = None,
) -> List[DocumentChunk]:
    """
    Splits text into overlapping token-based chunks.

    Args:
        text: full document text.
        doc_id: stable id of the source document (e.g., file path).
        page: optional page index for paged docs.
        base_metadata: metadata propagated to every chunk.
        chunk_size_tokens: window size.
        chunk_overlap_tokens: overlap between consecutive windows.
        tokenizer: optional tokenization function (falls back to approximate).
    """
    tokens, offsets = _tokenize_with_offsets(text, tokenizer)
    if not tokens:
        return []
    step = max(1, chunk_size_tokens - chunk_overlap_tokens)
    chunks: List[DocumentChunk] = []
    for start in range(0, len(tokens), step):
        end = min(len(tokens), start + chunk_size_tokens)
        if start >= end:
            break
        start_char, end_char = _chunk_bounds(offsets, start, end)
        chunk_text = text[start_char:end_char] if start_char is not None else text
        metadata = dict(base_metadata or {})
        metadata.update(
            {
                "doc_id": doc_id,
                "page": page,
                "start_char": start_char,
                "end_char": end_char,
                "chunk_index": len(chunks),
                "chunk_size_tokens": chunk_size_tokens,
                "chunk_overlap_tokens": chunk_overlap_tokens,
            }
        )
        chunk = DocumentChunk(
            id=f"{doc_id}::chunk_{len(chunks)}",
            doc_id=doc_id,
            text=chunk_text.strip(),
            page=page,
            start_char=start_char,
            end_char=end_char,
            metadata=metadata,
        )
        chunks.append(chunk)
    return chunks
