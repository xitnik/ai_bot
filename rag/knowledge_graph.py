from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

from sqlalchemy import select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import db
from rag.ingest import Document


class AsyncKnowledgeGraph:
    """MySQL-подложка для хранения сущностей/связей с документами."""

    def __init__(self, session_factory: Optional[async_sessionmaker[AsyncSession]] = None) -> None:
        self._session_factory = session_factory or db.AsyncSessionLocal

    async def _upsert_entity(
        self, session: db.AsyncSession, entity_id: str, entity_type: str, attributes: Optional[Dict[str, str]]
    ) -> None:
        stmt = mysql_insert(db.KGEntityModel).values(
            entity_id=entity_id,
            entity_type=entity_type,
            attributes=attributes or {},
        )
        stmt = stmt.on_duplicate_key_update(
            entity_type=stmt.inserted.entity_type,
            attributes=stmt.inserted.attributes,
            updated_at=db.func.now(),
        )
        await session.execute(stmt)

    async def _add_edge(self, session: db.AsyncSession, src: str, dst: str, relation: str) -> None:
        stmt = mysql_insert(db.KGEdgeModel).values(src_id=src, dst_id=dst, relation=relation)
        # deduplicate edges by source/destination/relation
        stmt = stmt.on_duplicate_key_update(relation=stmt.inserted.relation)
        await session.execute(stmt)

    async def add_document(self, document: Document) -> None:
        async with self._session_factory() as session:
            await self._write_document(session, document)
            await session.commit()

    async def bulk_add_documents(self, documents: Iterable[Document]) -> None:
        async with self._session_factory() as session:
            for doc in documents:
                await self._write_document(session, doc)
            await session.commit()

    async def _write_document(self, session: db.AsyncSession, document: Document) -> None:
        doc_id = document.id
        await self._upsert_entity(
            session,
            doc_id,
            "Document",
            {"source_type": str(document.metadata.get("source_type", ""))},
        )
        client_id = document.metadata.get("client_id")
        product_id = document.metadata.get("product_id")
        if client_id:
            cid = str(client_id)
            await self._upsert_entity(session, cid, "Client", {})
            await self._add_edge(session, cid, doc_id, "HAS_DOCUMENT")
        if product_id:
            pid = str(product_id)
            await self._upsert_entity(session, pid, "Product", {})
            await self._add_edge(session, pid, doc_id, "DESCRIBED_IN")

    async def get_related_documents(
        self, client_id: Optional[str] = None, product_id: Optional[str] = None
    ) -> List[str]:
        related: Set[str] = set()
        async with self._session_factory() as session:
            if client_id:
                related.update(await self._neighbors(session, str(client_id)))
            if product_id:
                related.update(await self._neighbors(session, str(product_id)))
        return list(related)

    async def _neighbors(self, session: db.AsyncSession, entity_id: str) -> Set[str]:
        stmt = select(db.KGEdgeModel.dst_id).where(
            db.KGEdgeModel.src_id == entity_id,
            db.KGEdgeModel.relation.in_(["HAS_DOCUMENT", "DESCRIBED_IN"]),
        )
        result = await session.execute(stmt)
        return {row[0] for row in result.all()}


# Singleton for lightweight usage.
KG = AsyncKnowledgeGraph()
