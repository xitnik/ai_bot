from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from config import get_settings
from rag.chunking import DocumentChunk
from vector_index import InMemoryVectorStore, MySQLVectorStore, ScoredDocument

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoredChunk:
    chunk: DocumentChunk
    score: float


class VectorStore(Protocol):
    async def upsert(self, chunks: List[DocumentChunk]) -> None: ...

    async def query(
        self, query_embedding: List[float], top_k: int, filters: Optional[Dict[str, Any]] = None
    ) -> List[ScoredChunk]: ...


class VectorStoreIndexAdapter:
    """
    Adapts new VectorStore protocol to legacy VectorIndex expected by HybridRetriever.
    """

    def __init__(self, store: VectorStore) -> None:
        self._store = store

    async def add_documents(self, documents: List[DocumentChunk]) -> None:
        await self._store.upsert(documents)

    async def search(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> List[ScoredDocument]:
        results = await self._store.query(query_embedding, top_k=top_k, filters=filters)
        return [ScoredDocument(document=item.chunk, score=item.score) for item in results]


class LegacyIndexStore(VectorStore):
    """
    Wraps existing VectorIndex implementations to new VectorStore protocol.
    Useful for reusing MySQLVectorStore until dedicated backend is added.
    """

    def __init__(self, index: MySQLVectorStore) -> None:
        self._index = index

    async def upsert(self, chunks: List[DocumentChunk]) -> None:
        await self._index.add_documents(chunks)

    async def query(
        self, query_embedding: List[float], top_k: int, filters: Optional[Dict[str, Any]] = None
    ) -> List[ScoredChunk]:
        results = await self._index.search(query_embedding, filters=filters or {}, top_k=top_k)
        return [ScoredChunk(chunk=item.document, score=item.score) for item in results]


class InMemoryStoreAdapter(VectorStore):
    """Adapter over existing in-memory store to conform to ScoredChunk protocol."""

    def __init__(self) -> None:
        self._store = InMemoryVectorStore()

    async def upsert(self, chunks: List[DocumentChunk]) -> None:
        await self._store.add_documents(chunks)

    async def query(
        self, query_embedding: List[float], top_k: int, filters: Optional[Dict[str, Any]] = None
    ) -> List[ScoredChunk]:
        results = await self._store.search(query_embedding, filters=filters or {}, top_k=top_k)
        return [ScoredChunk(chunk=item.document, score=item.score) for item in results]


class ChromaVectorStore(VectorStore):
    """Chroma-backed vector store with local persistence."""

    def __init__(self, persist_path: str, collection_name: str = "rag_chunks") -> None:
        try:
            import chromadb  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("chromadb is required for ChromaVectorStore") from exc
        self._client = chromadb.PersistentClient(path=persist_path)
        self._collection = self._client.get_or_create_collection(collection_name)
        self._persist_path = persist_path
        self._collection_name = collection_name
        logger.info(
            "ChromaVectorStore initialized",
            extra={"persist_path": persist_path, "collection": collection_name},
        )

    async def upsert(self, chunks: List[DocumentChunk]) -> None:
        ids: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []
        documents: List[str] = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.id} missing embedding")
            ids.append(chunk.id)
            embeddings.append(list(chunk.embedding))
            metadatas.append(dict(chunk.metadata))
            documents.append(chunk.text)
        if not ids:
            return
        await asyncio.to_thread(
            self._collection.upsert,
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

    async def query(
        self, query_embedding: List[float], top_k: int, filters: Optional[Dict[str, Any]] = None
    ) -> List[ScoredChunk]:
        result = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters or {},
        )
        ids = (result.get("ids") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        scored: List[ScoredChunk] = []
        for idx, chunk_id in enumerate(ids):
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            text = documents[idx] if idx < len(documents) else metadata.get("text", "")
            distance = distances[idx] if idx < len(distances) else None
            score = 1 - float(distance) if distance is not None else 0.0
            chunk = DocumentChunk(
                id=str(chunk_id),
                doc_id=metadata.get("doc_id") or str(chunk_id),
                text=text,
                metadata=metadata or {},
            )
            scored.append(ScoredChunk(chunk=chunk, score=score))
        return scored


class QdrantVectorStore(VectorStore):
    """Stub for future Qdrant backend."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - stub
        raise NotImplementedError(
            "QdrantVectorStore is not implemented yet. Configure RAG_VECTOR_BACKEND=chroma or memory."
        )


class PgVectorStore(VectorStore):
    """Stub for future pgvector backend."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - stub
        raise NotImplementedError(
            "PgVectorStore is not implemented yet. Configure RAG_VECTOR_BACKEND=chroma or memory."
        )


def build_vector_store() -> VectorStore:
    """Factory selecting backend from config."""
    settings = get_settings().rag.vector
    backend = (settings.backend or "chroma").lower()
    if backend == "mysql":
        logger.info("Using legacy MySQLVectorStore backend")
        return LegacyIndexStore(MySQLVectorStore())
    if backend == "chroma":
        try:
            return ChromaVectorStore(settings.persist_path, settings.collection_name)
        except Exception as exc:
            logger.warning(
                "Falling back to in-memory store because Chroma init failed",
                extra={"error": str(exc)},
            )
            return InMemoryStoreAdapter()
    if backend == "qdrant":
        logger.warning("Qdrant backend not implemented, using in-memory store")
        return InMemoryStoreAdapter()
    if backend == "pgvector":
        logger.warning("PgVector backend not implemented, using in-memory store")
        return InMemoryStoreAdapter()
    logger.warning("Unknown vector backend %s, using in-memory", backend)
    return InMemoryStoreAdapter()
