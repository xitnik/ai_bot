from __future__ import annotations

from rag.chunking import ChunkingConfig, split_text_sliding_window


def test_sliding_window_overlap_and_bounds() -> None:
    text = "one two three four five six seven eight nine ten"
    chunks = split_text_sliding_window(
        text,
        doc_id="doc",
        chunk_size_tokens=4,
        chunk_overlap_tokens=2,
    )

    assert len(chunks) >= 3
    # Ensure overlap: every consecutive chunk starts before previous ends.
    for first, second in zip(chunks, chunks[1:]):
        assert first.end_char is None or second.start_char is None or first.end_char > second.start_char
    # Chunk ids should be deterministic.
    assert chunks[0].id == "doc::chunk_0"


def test_chunk_metadata_propagates_base_fields() -> None:
    cfg = ChunkingConfig(chunk_size_tokens=3, chunk_overlap_tokens=1)
    base_meta = {"source": "fixture.txt", "lang": "ru"}
    chunks = split_text_sliding_window(
        "alpha beta gamma delta",
        doc_id="doc-123",
        page=7,
        base_metadata=base_meta,
        chunk_size_tokens=cfg.chunk_size_tokens,
        chunk_overlap_tokens=cfg.chunk_overlap_tokens,
        tokenizer=cfg.tokenizer,
    )

    assert chunks
    meta = chunks[0].metadata
    assert meta["doc_id"] == "doc-123"
    assert meta["page"] == 7
    assert meta["source"] == "fixture.txt"
    assert meta["lang"] == "ru"
