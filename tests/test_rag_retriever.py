from __future__ import annotations

from pathlib import Path

from rag.ingest import ingest_from_dir
from rag.retriever import RagEngine


def test_rag_retriever_prefers_matching_chunk(tmp_path):
    # Копируем фикстуры в temp, чтобы не зависеть от пути.
    source = Path("fixtures/rag_docs")
    for file in source.glob("*.txt"):
        data = file.read_text()
        (tmp_path / file.name).write_text(data)

    chunks = ingest_from_dir(tmp_path)
    engine = RagEngine(chunks)
    results = engine.retrieve("need procurement best offer", k=2)
    assert results
    top_chunk, score = results[0]
    assert "procurement" in top_chunk.text.lower()
    assert score > 0
