from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from embeddings_client import EmbeddingsClient
from metrics import REGISTRY
from rag.reranker import Reranker, build_reranker
from rag.vector_store import VectorStore
from vector_index import ScoredDocument

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    knn_top_k: int = 50
    final_top_k: int = 8
    reranker_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"


class DenseRetriever:
    """
    Two-stage dense retriever: KNN over vector store followed by optional cross-encoder rerank.
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingsClient,
        reranker: Optional[Reranker],
        config: Optional[RetrievalConfig] = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._config = config or RetrievalConfig()
        self._reranker = reranker

    @classmethod
    async def build(
        cls, store: VectorStore, embedder: EmbeddingsClient, config: Optional[RetrievalConfig] = None
    ) -> "DenseRetriever":
        cfg = config or RetrievalConfig()
        reranker = None
        if cfg.reranker_enabled:
            reranker = await build_reranker(cfg.reranker_model)
        return cls(store=store, embedder=embedder, reranker=reranker, config=cfg)

    async def retrieve(
        self, query: str, filters: Optional[Dict[str, Any]] = None
    ) -> List[ScoredDocument]:
        filters = filters or {}
        embed_start = time.perf_counter()
        try:
            query_embedding = await self._embedder.get_text_embedding(query)
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning("Failed to embed query, returning empty retrieval", extra={"error": str(exc)})
            return []
        REGISTRY.histogram("rag_query_embed_latency_ms").observe(
            (time.perf_counter() - embed_start) * 1000
        )

        knn_start = time.perf_counter()
        knn_results = await self._store.query(
            query_embedding, top_k=self._config.knn_top_k, filters=filters
        )
        REGISTRY.histogram("rag_knn_latency_ms").observe(
            (time.perf_counter() - knn_start) * 1000
        )
        REGISTRY.histogram("rag_knn_results_count").observe(len(knn_results))
        if not knn_results:
            return []

        ordered = knn_results
        if self._config.reranker_enabled and self._reranker is not None:
            rerank_start = time.perf_counter()
            ordered = await self._reranker.rerank(
                query, knn_results, top_k=self._config.final_top_k
            )
            REGISTRY.histogram("rag_rerank_latency_ms").observe(
                (time.perf_counter() - rerank_start) * 1000
            )
        else:
            ordered = ordered[: self._config.final_top_k]

        before_ids = [item.chunk.id for item in knn_results[:5]]
        after_ids = [item.chunk.id for item in ordered[:5]]
        logger.debug("Retrieval order", extra={"before": before_ids, "after": after_ids})
        return [ScoredDocument(document=item.chunk, score=item.score) for item in ordered]
