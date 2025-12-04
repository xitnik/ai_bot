from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence

from alternatives_models import Hit
from rag.ingest import Document


@dataclass(frozen=True)
class ScoredDocument:
    """Retrieval result with attached score."""

    document: Document
    score: float


class VectorIndex(Protocol):
    """High-level protocol for document vector indexes."""

    def add_documents(self, documents: List[Document]) -> None:
        ...

    def search(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> List[ScoredDocument]:
        ...


class VectorStore(VectorIndex, Protocol):
    """
    Legacy protocol kept for compatibility with alternatives_agent.
    Exposes product-centric upsert/search alongside document search.
    """

    def upsert_product(self, product_id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        ...

    def knn_search(self, vector: List[float], k: int, filters: Dict[str, Any]) -> List[Hit]:
        ...


class InMemoryVectorStore:
    """In-memory implementation for development/tests."""

    def __init__(self) -> None:
        self._vectors: Dict[str, List[float]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._documents: Dict[str, Document] = {}

    # --- New document-centric API ---
    def add_documents(self, documents: List[Document]) -> None:
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(f"Document {doc.id} has no embedding")
            self._documents[doc.id] = doc
            self._vectors[doc.id] = list(doc.embedding)
            self._metadata[doc.id] = dict(doc.metadata)

    def search(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> List[ScoredDocument]:
        matched_ids = self._filter_ids(filters or {})
        scored: List[ScoredDocument] = []
        for doc_id in matched_ids:
            vector = self._vectors.get(doc_id)
            if vector is None:
                continue
            score = self._cosine_similarity(query_embedding, vector)
            scored.append(ScoredDocument(document=self._documents[doc_id], score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    # --- Legacy product-centric wrappers ---
    def upsert_product(self, product_id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        """
        Stores product vector; kept for compatibility with AlternativesAgent.
        Text is synthesized from metadata to avoid empty payloads.
        """
        text = " ".join(f"{k}:{v}" for k, v in metadata.items())
        doc = Document(id=product_id, text=text, metadata=metadata, embedding=vector)
        self.add_documents([doc])

    def knn_search(self, vector: List[float], k: int, filters: Dict[str, Any]) -> List[Hit]:
        results = self.search(vector, filters=filters, top_k=k)
        return [
            Hit(product_id=item.document.id, score=item.score, metadata=item.document.metadata)
            for item in results
        ]

    # --- Internal helpers ---
    def _filter_ids(self, filters: Dict[str, Any]) -> List[str]:
        passed: List[str] = []
        for doc_id, meta in self._metadata.items():
            if self._passes_filters(meta, filters):
                passed.append(doc_id)
        return passed

    def _passes_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        for key, expected in filters.items():
            if expected is None:
                continue
            if key == "dimensions":
                if not self._dimensions_match(metadata.get("dimensions"), expected):
                    return False
                continue
            actual = metadata.get(key)
            if isinstance(expected, dict):
                if not self._range_match(actual, expected):
                    return False
            elif isinstance(expected, (list, tuple, set)):
                if actual not in expected:
                    return False
            else:
                if actual != expected:
                    return False
        return True

    def _dimensions_match(self, candidate_dims: Any, filters: Dict[str, Any]) -> bool:
        if not filters:
            return True
        if not isinstance(candidate_dims, dict):
            return False
        for key in ("length", "width", "thickness"):
            bounds = filters.get(key)
            if bounds is None:
                continue
            if not isinstance(bounds, dict):
                return False
            value = candidate_dims.get(key)
            if value is None:
                return False
            if not self._range_match(value, bounds):
                return False
        return True

    def _range_match(self, actual: Any, expected: Dict[str, Any]) -> bool:
        """Supports numeric/date ranges expressed as {'min': ..., 'max': ...}."""
        if actual is None:
            return False
        min_v = expected.get("min")
        max_v = expected.get("max")
        try:
            value = float(actual)
        except (TypeError, ValueError):
            # Attempt to parse dates.
            try:
                from datetime import datetime

                value_dt = datetime.fromisoformat(str(actual))
                value = value_dt.timestamp()
            except Exception:
                return False
        if min_v is not None and value < float(min_v):
            return False
        if max_v is not None and value > float(max_v):
            return False
        return True

    def _cosine_similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)


class QdrantVectorIndex:
    """
    Thin wrapper around qdrant-client. Optional dependency; raise on missing import.
    """

    def __init__(
        self,
        collection: str,
        vector_size: int,
        client: Any = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        distance: str = "Cosine",
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client import models as qmodels
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("qdrant-client is required for QdrantVectorIndex") from exc

        self.collection = collection
        self._models = qmodels
        self._client = client or QdrantClient(url=url, api_key=api_key)
        # Ensure collection exists.
        try:
            self._client.get_collection(collection)
        except Exception:
            self._client.recreate_collection(
                collection_name=collection,
                vectors_config=qmodels.VectorParams(size=vector_size, distance=getattr(qmodels.Distance, distance)),
            )

    def add_documents(self, documents: List[Document]) -> None:
        payloads = []
        vectors = []
        ids = []
        for doc in documents:
            if doc.embedding is None:
                continue
            ids.append(doc.id)
            vectors.append(doc.embedding)
            payloads.append(doc.metadata)
        if not ids:
            return
        self._client.upsert(
            collection_name=self.collection,
            points=self._models.Batch(ids=ids, vectors=vectors, payloads=payloads),
        )

    def search(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> List[ScoredDocument]:
        qfilter = None
        if filters:
            try:
                qfilter = self._models.Filter(
                    must=[self._models.FieldCondition(key=k, match=self._models.MatchValue(value=v)) for k, v in filters.items()]
                )
            except Exception:
                qfilter = None
        hits = self._client.search(
            collection_name=self.collection,
            query_vector=query_embedding,
            query_filter=qfilter,
            limit=top_k,
        )
        results: List[ScoredDocument] = []
        for hit in hits:
            doc_meta = dict(getattr(hit, "payload", {}) or {})
            doc = Document(id=str(hit.id), text=doc_meta.get("text", ""), metadata=doc_meta, embedding=None)
            results.append(ScoredDocument(document=doc, score=getattr(hit, "score", 0.0) or 0.0))
        return results
