from __future__ import annotations

import pytest

from rag.ingest import Document
from rag.retrieval import DenseRetriever, RetrievalConfig
from rag.vector_store import InMemoryStoreAdapter


class StaticEmbedder:
    async def get_text_embedding(self, text: str) -> list[float]:
        return [1.0, 0.0]


@pytest.mark.asyncio
async def test_dense_retriever_returns_top_hit_without_reranker() -> None:
    store = InMemoryStoreAdapter()
    doc = Document(id="doc1", text="relevant", metadata={"lang": "ru"}, embedding=[1.0, 0.0])
    await store.upsert([doc])
    retriever = DenseRetriever(
        store=store,
        embedder=StaticEmbedder(),
        reranker=None,
        config=RetrievalConfig(reranker_enabled=False, knn_top_k=3, final_top_k=1),
    )

    results = await retriever.retrieve("query")

    assert results
    assert results[0].document.id == "doc1"
