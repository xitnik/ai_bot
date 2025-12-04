from __future__ import annotations

import types

import pytest

from rag.ingest import Document
from rag.pipeline import RagPipeline, SessionContext
from rag.retriever import HybridRetriever, RetrieverConfig
from vector_index import InMemoryVectorStore


class FakeEmbedder:
    """Deterministic embedder for tests."""

    async def get_text_embedding(self, text: str) -> list[float]:
        return [1.0 if "policy" in text or "oak" in text else 0.0]


@pytest.mark.asyncio
async def test_hybrid_retriever_prefers_matching_doc() -> None:
    docs = [
        Document(id="match", text="pricing policy and discount rules", metadata={"lang": "en"}, embedding=[1.0]),
        Document(id="other", text="unrelated text", metadata={"lang": "en"}, embedding=[0.0]),
    ]
    store = InMemoryVectorStore()
    await store.add_documents(docs)
    retriever = HybridRetriever(
        store,
        FakeEmbedder(),
        documents=docs,
        config=RetrieverConfig(enable_reranker=False, lexical_top_k=2, dense_top_k=2, hybrid_top_k=2),
    )

    results = await retriever.retrieve("discount policy", filters={"lang": "en"})

    assert results
    assert results[0].document.id == "match"


@pytest.mark.asyncio
async def test_rag_pipeline_answer_uses_chat(monkeypatch) -> None:
    docs = [
        Document(id="d1", text="pricing policy doc", metadata={"source": "fixture", "lang": "en"}, embedding=[1.0]),
    ]
    store = InMemoryVectorStore()
    await store.add_documents(docs)
    embedder = FakeEmbedder()
    retriever = HybridRetriever(
        store,
        embedder,
        documents=docs,
        config=RetrieverConfig(enable_reranker=False, lexical_top_k=1, dense_top_k=1, hybrid_top_k=1),
    )

    # Fake chat completion object.
    completion = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="final answer"))]
    )
    monkeypatch.setattr("rag.pipeline.chat", lambda *args, **kwargs: completion)

    pipeline = RagPipeline(retriever=retriever, embedder=embedder, vector_index=store, documents=docs)
    result = await pipeline.answer("pricing question", session=SessionContext(lang="en"))

    assert result.answer == "final answer"
    assert result.used_documents and result.used_documents[0].document.id == "d1"
    assert "d1" in result.debug_info.get("retrieved_ids", [])
