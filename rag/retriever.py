from __future__ import annotations

import asyncio
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from embeddings_client import EmbeddingsClient
from rag.ingest import Document, DocumentChunk
from vector_index import ScoredDocument, VectorIndex


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in text.split() if t.strip()]


def _hash_embedding(text: str, dim: int = 64) -> List[float]:
    vector = [0.0] * dim
    for idx, byte in enumerate(text.encode("utf-8")):
        vector[idx % dim] += (byte % 31) / 255.0
    norm = sum(v * v for v in vector) ** 0.5 or 1.0
    return [v / norm for v in vector]


@dataclass
class RetrieverConfig:
    """Configurable knobs for hybrid retrieval."""

    lexical_top_k: int = 8
    dense_top_k: int = 12
    hybrid_top_k: int = 8
    rerank_top_k: int = 5
    lexical_weight: float = 0.35
    dense_weight: float = 0.65
    min_score: float = 0.0
    enable_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"


class HybridRetriever:
    """
    Hybrid retriever: lexical BM25 + dense search + optional cross-encoder reranking.
    """

    def __init__(
        self,
        vector_index: VectorIndex,
        embedder: EmbeddingsClient,
        documents: Iterable[Document],
        config: Optional[RetrieverConfig] = None,
    ) -> None:
        self._index = vector_index
        self._embedder = embedder
        self._config = config or RetrieverConfig()
        self._documents = list(documents)
        self._lexical_df: Dict[str, int] = defaultdict(int)
        self._avg_doc_len: float = 0.0
        self._reranker = None
        self._cache: Dict[str, List[ScoredDocument]] = {}
        self._cache_size = 128
        self._build_lexical_index()

    def _build_lexical_index(self) -> None:
        total_len = 0
        for doc in self._documents:
            tokens = _tokenize(doc.text)
            total_len += len(tokens)
            for token in set(tokens):
                self._lexical_df[token] += 1
        self._avg_doc_len = total_len / len(self._documents) if self._documents else 0.0

    def _idf(self, term: str) -> float:
        return math.log((1 + len(self._documents)) / (1 + self._lexical_df.get(term, 0))) + 1.0

    def _bm25(self, query_tokens: List[str], doc_tokens: List[str], k1: float = 1.5, b: float = 0.75) -> float:
        if not doc_tokens:
            return 0.0
        doc_len = len(doc_tokens)
        counts = Counter(doc_tokens)
        score = 0.0
        for term in query_tokens:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            idf = self._idf(term)
            denom = tf + k1 * (1 - b + b * doc_len / (self._avg_doc_len or 1))
            score += idf * (tf * (k1 + 1)) / denom
        return score

    def _lexical_search(self, query: str, filters: Optional[Dict[str, Any]]) -> List[ScoredDocument]:
        tokens = _tokenize(query)
        results: List[ScoredDocument] = []
        for doc in self._documents:
            if filters and not self._metadata_matches(doc.metadata, filters):
                continue
            doc_tokens = _tokenize(doc.text)
            score = self._bm25(tokens, doc_tokens)
            if score > 0:
                results.append(ScoredDocument(document=doc, score=score))
        results.sort(key=lambda item: item.score, reverse=True)
        return results[: self._config.lexical_top_k]

    async def _dense_search(self, query: str, filters: Optional[Dict[str, Any]]) -> List[ScoredDocument]:
        try:
            embedding = await self._embedder.get_text_embedding(query)
        except Exception:
            embedding = _hash_embedding(query)
        return self._index.search(embedding, filters=filters or {}, top_k=self._config.dense_top_k)

    def _metadata_matches(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        from vector_index import InMemoryVectorStore  # avoid cycle in type hints

        helper = getattr(InMemoryVectorStore(), "_passes_filters")
        return helper(metadata, filters)  # type: ignore[arg-type]

    async def retrieve(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[ScoredDocument]:
        cache_key = self._cache_key(query, filters or {})
        if cache_key in self._cache:
            return list(self._cache[cache_key])
        lexical = self._lexical_search(query, filters)
        dense = await self._dense_search(query, filters)

        combined: Dict[str, ScoredDocument] = {}
        for item in lexical:
            combined[item.document.id] = ScoredDocument(
                document=item.document, score=item.score * self._config.lexical_weight
            )
        for item in dense:
            prev = combined.get(item.document.id)
            base_score = prev.score if prev else 0.0
            combined[item.document.id] = ScoredDocument(
                document=item.document, score=base_score + item.score * self._config.dense_weight
            )

        scored = sorted(combined.values(), key=lambda it: it.score, reverse=True)
        top = scored[: self._config.hybrid_top_k]
        if self._config.enable_reranker:
            reranked = await self._maybe_rerank(query, top)
            result = reranked
        else:
            result = top
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = list(result)
        return result

    async def _maybe_rerank(self, query: str, candidates: List[ScoredDocument]) -> List[ScoredDocument]:
        if not candidates:
            return candidates
        reranker = await self._get_reranker()
        if reranker is None:
            return candidates

        top_candidates = candidates[: self._config.rerank_top_k]
        pairs = [[query, cand.document.text] for cand in top_candidates]
        try:
            scores = await asyncio.to_thread(reranker.predict, pairs)
        except Exception:
            return candidates
        reranked: List[ScoredDocument] = []
        for cand, score in zip(top_candidates, scores):
            reranked.append(ScoredDocument(document=cand.document, score=float(score)))
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked + candidates[len(top_candidates) :]

    async def _get_reranker(self) -> Any:
        if self._reranker is not None:
            return self._reranker
        if not self._config.enable_reranker:
            return None
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception:
            self._reranker = None
            return None
        self._reranker = CrossEncoder(self._config.reranker_model)
        return self._reranker

    def _cache_key(self, query: str, filters: Dict[str, Any]) -> str:
        items = sorted(filters.items())
        return f"{query}|{items}"


class RagEngine:
    """
    Legacy lexical-only retriever kept for backwards compatibility and fast tests.
    """

    def __init__(self, chunks: Iterable[DocumentChunk]) -> None:
        self.chunks = list(chunks)
        self._df: Dict[str, int] = defaultdict(int)
        self._build_index()

    def _build_index(self) -> None:
        for chunk in self.chunks:
            seen = set()
            for token in _tokenize(chunk.text):
                if token not in seen:
                    self._df[token] += 1
                    seen.add(token)

    def _idf(self, token: str) -> float:
        return math.log((1 + len(self.chunks)) / (1 + self._df.get(token, 0))) + 1.0

    def score(self, query: str, chunk: DocumentChunk) -> float:
        q_tokens = _tokenize(query)
        c_tokens = _tokenize(chunk.text)
        q_counts = Counter(q_tokens)
        c_counts = Counter(c_tokens)
        score = 0.0
        for token, q_freq in q_counts.items():
            score += q_freq * c_counts.get(token, 0) * self._idf(token)
        return score

    def retrieve(self, query: str, k: int = 3):
        scored = [(chunk, self.score(query, chunk)) for chunk in self.chunks]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [pair for pair in scored[:k] if pair[1] > 0]
