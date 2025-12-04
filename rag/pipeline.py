from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import db
from config import get_settings
from embeddings_client import EmbeddingsClient
from llm_client import chat
from metrics import REGISTRY
from ner import extract_entities
from rag.corrective import CorrectiveRag, RetrievalEvaluator
from rag.ingest import Document, ingest_from_dir
from rag.knowledge_graph import KG
from rag.retriever import HybridRetriever, RetrieverConfig
from rag.self_rag import SelfRagOrchestrator
from vector_index import MySQLVectorStore, ScoredDocument, VectorIndex


@dataclass
class SessionContext:
    """Lightweight context passed from gateway/agents into RAG."""

    user_id: Optional[str] = None
    client_id: Optional[str] = None
    product_id: Optional[str] = None
    lang: Optional[str] = None
    entities: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RagResult:
    answer: str
    used_documents: List[ScoredDocument]
    debug_info: Dict[str, Any] = field(default_factory=dict)


class RagPipeline:
    """High-level RAG orchestration: retrieval + LLM generation."""

    def __init__(
        self,
        retriever: HybridRetriever,
        embedder: EmbeddingsClient,
        vector_index: VectorIndex,
        documents: Sequence[Document],
    ) -> None:
        self._retriever = retriever
        self._embedder = embedder
        self._vector_index = vector_index
        self._documents = list(documents)
        self._doc_map: Dict[str, Document] = {doc.id: doc for doc in documents}

    @classmethod
    async def from_default_corpus(
        cls,
        data_dir: str = "fixtures/rag_docs",
        retriever_config: Optional[RetrieverConfig] = None,
    ) -> "RagPipeline":
        await db.init_db()
        embedder = EmbeddingsClient()
        docs = ingest_from_dir(data_dir)
        documents_with_vectors = await _embed_documents(embedder, docs)
        index = MySQLVectorStore()
        await index.add_documents(documents_with_vectors)
        if get_settings().rag.enable_knowledge_graph:
            await KG.bulk_add_documents(documents_with_vectors)
        retriever = HybridRetriever(index, embedder, documents=documents_with_vectors, config=retriever_config)
        return cls(retriever=retriever, embedder=embedder, vector_index=index, documents=documents_with_vectors)

    async def answer(self, query: str, session: Optional[SessionContext] = None, mode: Optional[str] = None) -> RagResult:
        session = session or SessionContext()
        rag_settings = get_settings().rag
        effective_mode = mode or rag_settings.default_mode

        if self._is_simple_query(query, rag_settings.simple_query_max_tokens):
            answer = await self._answer_without_rag(query)
            return RagResult(answer=answer, used_documents=[], debug_info={"mode": "skipped"})

        if effective_mode == "self-rag":
            orchestrator = SelfRagOrchestrator(self._retriever, max_iterations=rag_settings.max_selfrag_iterations)
            try:
                answer, docs, debug = await orchestrator.run(query)
                debug["mode"] = "self-rag"
                return RagResult(answer=answer, used_documents=docs, debug_info=debug)
            except Exception:
                effective_mode = "basic"  # fallback

        filters = self._build_filters(session)
        retrieved = await self._retrieve_with_filters(query, filters, session)
        entities = session.entities if session and session.entities is not None else extract_entities(query)

        if effective_mode == "crag":
            evaluator = RetrievalEvaluator(min_score=rag_settings.retriever_min_score)
            crag = CorrectiveRag(self._retriever, evaluator, max_retries=rag_settings.max_crag_retries)

            def _gen_answer(docs: List[ScoredDocument]) -> str:
                return _llm_answer(query, docs, entities)

            answer, docs, debug = await crag.run(query, _gen_answer, filters)
            debug["mode"] = "crag"
            return RagResult(answer=answer, used_documents=docs, debug_info=debug)

        # basic
        answer = _llm_answer(query, retrieved, entities)
        debug = {
            "filters": filters,
            "entities": entities,
            "retrieved_ids": [item.document.id for item in retrieved],
            "mode": "basic",
        }
        return RagResult(answer=answer, used_documents=retrieved, debug_info=debug)

    async def retrieve(self, query: str, session: Optional[SessionContext] = None) -> List[ScoredDocument]:
        filters = self._build_filters(session)
        return await self._retrieve_with_filters(query, filters)

    async def _retrieve_with_filters(
        self, query: str, filters: Dict[str, Any], session: Optional[SessionContext] = None
    ) -> List[ScoredDocument]:
        retrieval_start = time.perf_counter()
        retrieved = await self._retriever.retrieve(query, filters=filters)
        retrieved = await self._augment_with_kg(retrieved, session)
        REGISTRY.histogram("rag_retrieval_latency_ms").observe((time.perf_counter() - retrieval_start) * 1000)
        REGISTRY.histogram("rag_retrieved_docs_count").observe(len(retrieved))
        return retrieved

    def _build_filters(self, session: Optional[SessionContext]) -> Dict[str, Any]:
        filters: Dict[str, Any] = {}
        if not session:
            return filters
        if session.client_id:
            filters["client_id"] = session.client_id
        if session.product_id:
            filters["product_id"] = session.product_id
        if session.lang:
            filters["lang"] = session.lang
        for entity in session.entities or []:
            etype = entity.get("type")
            value = entity.get("value")
            if etype == "wood_species":
                filters.setdefault("species", value)
            if etype == "sku":
                filters.setdefault("product_id", value)
        filters.update(session.metadata or {})
        return filters

    async def _augment_with_kg(self, retrieved: List[ScoredDocument], session: Optional[SessionContext]) -> List[ScoredDocument]:
        settings = get_settings().rag
        if not settings.enable_knowledge_graph or not session:
            return retrieved
        related_ids = await KG.get_related_documents(session.client_id, session.product_id)
        existing = {item.document.id for item in retrieved}
        augmented = list(retrieved)
        for doc_id in related_ids:
            if doc_id in existing:
                continue
            doc = self._doc_map.get(doc_id)
            if not doc:
                continue
            augmented.append(ScoredDocument(document=doc, score=0.0))
        return augmented

    def _is_simple_query(self, query: str, max_tokens: int) -> bool:
        tokens = query.split()
        if len(tokens) > max_tokens:
            return False
        lowered = query.lower()
        important = ["price", "pric", "цен", "contract", "договор", "policy", "регламент"]
        if any(key in lowered for key in important):
            return False
        return "?" not in query

    async def _answer_without_rag(self, query: str) -> str:
        try:
            completion = await asyncio.to_thread(
                chat,
                "gpt5",
                [{"role": "system", "content": "Short helpful assistant."}, {"role": "user", "content": query}],
                temperature=0.3,
            )
            return _extract_text(completion)
        except Exception:
            return "Для этого запроса не требуется поиск по базе."


async def _embed_documents(embedder: EmbeddingsClient, documents: Sequence[Document]) -> List[Document]:
    embedded: List[Document] = []
    for doc in documents:
        try:
            vector = await embedder.get_text_embedding(doc.text)
            doc.embedding = vector
            doc.metadata.setdefault("embedding_model", embedder.default_model)
            embedded.append(doc)
        except Exception:
            # Deterministic fallback to allow offline dev; avoid dropping documents.
            doc.embedding = _hashed_embedding(doc.text)
            doc.metadata.setdefault("embedding_model", "hashed-fallback")
            embedded.append(doc)
    return embedded


def _hashed_embedding(text: str, dim: int = 64) -> List[float]:
    vector = [0.0] * dim
    encoded = text.encode("utf-8")
    for i, byte in enumerate(encoded):
        vector[i % dim] += (byte % 31) / 255.0
    norm = sum(v * v for v in vector) ** 0.5 or 1.0
    return [v / norm for v in vector]


def _format_context(retrieved: Sequence[ScoredDocument]) -> str:
    lines: List[str] = []
    for item in retrieved:
        meta = item.document.metadata or {}
        source = meta.get("source") or meta.get("source_type") or "doc"
        lines.append(f"[doc:{item.document.id} | {source}] {item.document.text}")
    return "\n".join(lines)


def _extract_text(completion: Any) -> str:
    if not completion or not getattr(completion, "choices", None):
        return ""
    first = completion.choices[0]
    message = getattr(first, "message", None)
    if message and getattr(message, "content", None):
        return message.content
    return getattr(first, "text", "") or ""


def _llm_answer(query: str, retrieved: Sequence[ScoredDocument], entities: Sequence[Dict[str, Any]]) -> str:
    context_blocks = _format_context(retrieved)
    system_prompt = (
        "You are a retrieval-augmented assistant for sales/procurement.\n"
        "Use ONLY provided context snippets. Quote sources with [doc:id] markers.\n"
        "If context is insufficient, say so explicitly."
    )
    user_prompt = (
        f"Question: {query}\n\n"
        f"Entities: {entities}\n\n"
        f"Context snippets:\n{context_blocks}\n\n"
        "Answer in Russian, concise, include sources."
    )
    try:
        completion = chat(
            "gpt5",
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0,
        )
        return _extract_text(completion)
    except Exception:
        return "Не удалось получить ответ от LLM."


_PIPELINE: Optional[RagPipeline] = None
_LOCK = asyncio.Lock()


async def _get_pipeline() -> RagPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        async with _LOCK:
            if _PIPELINE is None:
                _PIPELINE = await RagPipeline.from_default_corpus()
    return _PIPELINE


async def rag_answer(query: str, session: Optional[SessionContext] = None) -> RagResult:
    pipeline = await _get_pipeline()
    return await pipeline.answer(query, session=session or SessionContext())


async def rag_retrieve(query: str, session: Optional[SessionContext] = None) -> List[ScoredDocument]:
    pipeline = await _get_pipeline()
    return await pipeline.retrieve(query, session=session or SessionContext())
