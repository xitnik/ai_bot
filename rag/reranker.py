from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import List, Protocol

from metrics import REGISTRY
from rag.vector_store import ScoredChunk

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    async def rerank(self, query: str, candidates: List[ScoredChunk], top_k: int) -> List[ScoredChunk]: ...


@dataclass
class CrossEncoderReranker:
    """Cross-encoder reranker built on sentence-transformers."""

    model_name: str
    _model: object | None = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("sentence-transformers is required for reranking") from exc
        self._model = CrossEncoder(self.model_name)
        return self._model

    async def rerank(self, query: str, candidates: List[ScoredChunk], top_k: int) -> List[ScoredChunk]:
        if not candidates:
            return candidates
        try:
            model = self._load_model()
        except Exception as exc:
            logger.warning("Reranker unavailable, skipping", extra={"error": str(exc)})
            return candidates[:top_k]

        start = time.perf_counter()
        pairs = [[query, cand.chunk.text] for cand in candidates]
        try:
            scores = await asyncio.to_thread(model.predict, pairs)
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning("Reranker prediction failed, returning original order", extra={"error": str(exc)})
            return candidates[:top_k]
        REGISTRY.histogram("rag_rerank_latency_ms").observe((time.perf_counter() - start) * 1000)
        reranked: List[ScoredChunk] = []
        for cand, score in zip(candidates, scores):
            reranked.append(ScoredChunk(chunk=cand.chunk, score=float(score)))
        reranked.sort(key=lambda item: item.score, reverse=True)
        debug_before = [c.chunk.id for c in candidates[:5]]
        debug_after = [c.chunk.id for c in reranked[:5]]
        logger.debug("Rerank order changed", extra={"before": debug_before, "after": debug_after})
        return reranked[:top_k]


async def build_reranker(model_name: str) -> Reranker | None:
    """Factory handling supported reranker models."""
    name = model_name.lower()
    if "qwen" in name:
        logger.warning("Qwen reranker not implemented; skipping rerank")
        return None
    return CrossEncoderReranker(model_name=model_name)
