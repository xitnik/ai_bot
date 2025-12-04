from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence

from sqlalchemy import select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import db
from alternatives_models import Hit
from rag.ingest import Document


@dataclass(frozen=True)
class ScoredDocument:
    """Retrieval result with attached score."""

    document: Document
    score: float


def _range_match(actual: Any, expected: Dict[str, Any]) -> bool:
    """Supports numeric/date ranges expressed as {'min': ..., 'max': ...}."""
    if actual is None:
        return False
    min_v = expected.get("min")
    max_v = expected.get("max")
    try:
        value = float(actual)
    except (TypeError, ValueError):
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


def _dimensions_match(candidate_dims: Any, filters: Dict[str, Any]) -> bool:
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
        if not _range_match(value, bounds):
            return False
    return True


def _passes_filters(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    for key, expected in filters.items():
        if expected is None:
            continue
        if key == "dimensions":
            if not _dimensions_match(metadata.get("dimensions"), expected):
                return False
            continue
        actual = metadata.get(key)
        if isinstance(expected, dict):
            if not _range_match(actual, expected):
                return False
        elif isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        else:
            if actual != expected:
                return False
    return True


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorIndex(Protocol):
    """High-level protocol for document vector indexes."""

    async def add_documents(self, documents: List[Document]) -> None:
        ...

    async def search(
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

    async def upsert_product(
        self, product_id: str, vector: List[float], metadata: Dict[str, Any]
    ) -> None:
        ...

    async def knn_search(
        self, vector: List[float], k: int, filters: Dict[str, Any]
    ) -> List[Hit]:
        ...


class InMemoryVectorStore:
    """In-memory implementation for development/tests."""

    def __init__(self) -> None:
        self._vectors: Dict[str, List[float]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._documents: Dict[str, Document] = {}

    # --- New document-centric API ---
    async def add_documents(self, documents: List[Document]) -> None:
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(f"Document {doc.id} has no embedding")
            self._documents[doc.id] = doc
            self._vectors[doc.id] = list(doc.embedding)
            self._metadata[doc.id] = dict(doc.metadata)

    async def search(
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
            score = _cosine_similarity(query_embedding, vector)
            scored.append(ScoredDocument(document=self._documents[doc_id], score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    # --- Legacy product-centric wrappers ---
    async def upsert_product(
        self, product_id: str, vector: List[float], metadata: Dict[str, Any]
    ) -> None:
        """
        Stores product vector; kept for compatibility with AlternativesAgent.
        Text is synthesized from metadata to avoid empty payloads.
        """
        text = " ".join(f"{k}:{v}" for k, v in metadata.items())
        doc = Document(id=product_id, text=text, metadata=metadata, embedding=vector)
        await self.add_documents([doc])

    async def knn_search(self, vector: List[float], k: int, filters: Dict[str, Any]) -> List[Hit]:
        results = await self.search(vector, filters=filters, top_k=k)
        return [
            Hit(product_id=item.document.id, score=item.score, metadata=item.document.metadata)
            for item in results
        ]

    # --- Internal helpers ---
    def _filter_ids(self, filters: Dict[str, Any]) -> List[str]:
        passed: List[str] = []
        for doc_id, meta in self._metadata.items():
            if _passes_filters(meta, filters):
                passed.append(doc_id)
        return passed


class MySQLVectorStore(VectorStore):
    """Персистентный VectorStore на MySQL с вычислением косинусного сходства в приложении."""

    def __init__(
        self,
        session_factory: Optional[async_sessionmaker[AsyncSession]] = None,
        max_candidates: int = 500,
    ) -> None:
        self._session_factory = session_factory or db.AsyncSessionLocal
        self._max_candidates = max_candidates

    async def add_documents(self, documents: List[Document]) -> None:
        async with self._session_factory() as session:
            for doc in documents:
                if doc.embedding is None:
                    raise ValueError(f"Document {doc.id} has no embedding")
                payload = {
                    "doc_id": doc.id,
                    "text": doc.text,
                    "source": doc.metadata.get("source"),
                    "source_type": doc.metadata.get("source_type"),
                    "client_id": doc.metadata.get("client_id"),
                    "product_id": doc.metadata.get("product_id"),
                    "lang": doc.metadata.get("lang"),
                    "metadata": dict(doc.metadata),
                    "embedding": list(doc.embedding),
                    "embedding_model": doc.metadata.get("embedding_model"),
                }
                stmt = mysql_insert(db.DocumentModel).values(**payload)
                stmt = stmt.on_duplicate_key_update(
                    text=stmt.inserted.text,
                    source=stmt.inserted.source,
                    source_type=stmt.inserted.source_type,
                    client_id=stmt.inserted.client_id,
                    product_id=stmt.inserted.product_id,
                    lang=stmt.inserted.lang,
                    metadata=stmt.inserted.metadata,
                    embedding=stmt.inserted.embedding,
                    embedding_model=stmt.inserted.embedding_model,
                    ingested_at=db.func.now(),
                )
                await session.execute(stmt)
            await session.commit()

    async def search(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> List[ScoredDocument]:
        filters = filters or {}
        async with self._session_factory() as session:
            stmt = select(db.DocumentModel).where(db.DocumentModel.embedding.is_not(None))
            if filters.get("client_id"):
                stmt = stmt.where(db.DocumentModel.client_id == str(filters["client_id"]))
            if filters.get("product_id"):
                stmt = stmt.where(db.DocumentModel.product_id == str(filters["product_id"]))
            if filters.get("lang"):
                stmt = stmt.where(db.DocumentModel.lang == str(filters["lang"]))
            if filters.get("source_type"):
                stmt = stmt.where(db.DocumentModel.source_type == str(filters["source_type"]))
            stmt = stmt.limit(self._max_candidates)
            result = await session.execute(stmt)
            rows = result.scalars().all()

        scored: List[ScoredDocument] = []
        for row in rows:
            meta = dict(row.meta or {})
            meta.setdefault("source", row.source)
            meta.setdefault("source_type", row.source_type)
            if row.client_id:
                meta.setdefault("client_id", row.client_id)
            if row.product_id:
                meta.setdefault("product_id", row.product_id)
            if row.lang:
                meta.setdefault("lang", row.lang)
            if not _passes_filters(meta, filters):
                continue
            embedding = row.embedding or []
            score = _cosine_similarity(query_embedding, embedding)
            document = Document(id=row.doc_id, text=row.text, metadata=meta, embedding=embedding)
            scored.append(ScoredDocument(document=document, score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    async def upsert_product(
        self, product_id: str, vector: List[float], metadata: Dict[str, Any]
    ) -> None:
        async with self._session_factory() as session:
            payload = {
                "product_id": product_id,
                "vector": list(vector),
                "metadata": dict(metadata),
                "species": metadata.get("species"),
                "grade": metadata.get("grade"),
                "price": metadata.get("price"),
                "in_stock": metadata.get("in_stock", True),
                "dimensions": metadata.get("dimensions"),
            }
            stmt = mysql_insert(db.ProductVectorModel).values(**payload)
            stmt = stmt.on_duplicate_key_update(
                vector=stmt.inserted.vector,
                metadata=stmt.inserted.metadata,
                species=stmt.inserted.species,
                grade=stmt.inserted.grade,
                price=stmt.inserted.price,
                in_stock=stmt.inserted.in_stock,
                dimensions=stmt.inserted.dimensions,
                updated_at=db.func.now(),
            )
            await session.execute(stmt)
            await session.commit()

    async def knn_search(self, vector: List[float], k: int, filters: Dict[str, Any]) -> List[Hit]:
        async with self._session_factory() as session:
            stmt = select(db.ProductVectorModel)
            if filters.get("species"):
                stmt = stmt.where(db.ProductVectorModel.species == filters["species"])
            if filters.get("grade"):
                stmt = stmt.where(db.ProductVectorModel.grade == filters["grade"])
            if filters.get("in_stock") is not None:
                stmt = stmt.where(db.ProductVectorModel.in_stock == bool(filters["in_stock"]))
            stmt = stmt.limit(self._max_candidates)
            result = await session.execute(stmt)
            rows = result.scalars().all()

        hits: List[Hit] = []
        for row in rows:
            meta = dict(row.meta or {})
            meta.setdefault("species", row.species)
            meta.setdefault("grade", row.grade)
            meta.setdefault("price", row.price)
            meta.setdefault("dimensions", row.dimensions)
            meta.setdefault("in_stock", row.in_stock)
            if not _passes_filters(meta, filters):
                continue
            score = _cosine_similarity(vector, row.vector or [])
            hits.append(Hit(product_id=row.product_id, score=score, metadata=meta))
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:k]


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
                vectors_config=qmodels.VectorParams(
                    size=vector_size, distance=getattr(qmodels.Distance, distance)
                ),
            )

    async def add_documents(self, documents: List[Document]) -> None:
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
        await asyncio.to_thread(
            self._client.upsert,
            collection_name=self.collection,
            points=self._models.Batch(ids=ids, vectors=vectors, payloads=payloads),
        )

    async def search(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> List[ScoredDocument]:
        qfilter = None
        if filters:
            try:
                qfilter = self._models.Filter(
                    must=[
                        self._models.FieldCondition(
                            key=k, match=self._models.MatchValue(value=v)
                        )
                        for k, v in filters.items()
                    ]
                )
            except Exception:
                qfilter = None
        hits = await asyncio.to_thread(
            self._client.search,
            collection_name=self.collection,
            query_vector=query_embedding,
            query_filter=qfilter,
            limit=top_k,
        )
        results: List[ScoredDocument] = []
        for hit in hits:
            doc_meta = dict(getattr(hit, "payload", {}) or {})
            doc = Document(
                id=str(hit.id),
                text=doc_meta.get("text", ""),
                metadata=doc_meta,
                embedding=None,
            )
            results.append(
                ScoredDocument(document=doc, score=getattr(hit, "score", 0.0) or 0.0)
            )
        return results
