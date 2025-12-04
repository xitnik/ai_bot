from __future__ import annotations

import asyncio
from typing import Dict, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import db
from models import SessionDTO


class SessionStore:
    """Контракт для хранения сессий (можно заменить на Redis)."""

    async def get(self, user_id: str, channel: str) -> Optional[SessionDTO]:
        raise NotImplementedError

    async def save(self, session: SessionDTO) -> None:
        raise NotImplementedError

    async def delete(self, session_id: str) -> None:
        raise NotImplementedError


class InMemorySessionStore(SessionStore):
    """Небольшой in-memory кеш; потокобезопасность через lock."""

    def __init__(self) -> None:
        self._data: Dict[Tuple[str, str], SessionDTO] = {}
        self._by_id: Dict[str, Tuple[str, str]] = {}
        self._lock = asyncio.Lock()

    async def get(self, user_id: str, channel: str) -> Optional[SessionDTO]:
        key = (user_id, channel)
        async with self._lock:
            return self._data.get(key)

    async def save(self, session: SessionDTO) -> None:
        key = (session.user_id, session.channel)
        async with self._lock:
            self._data[key] = session
            self._by_id[session.session_id] = key

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            key = self._by_id.pop(session_id, None)
            if key:
                self._data.pop(key, None)


class MySQLSessionStore(SessionStore):
    """Хранилище сессий в MySQL с простыми upsert-операциями."""

    def __init__(self, session_factory: Optional[async_sessionmaker[AsyncSession]] = None) -> None:
        self._session_factory = session_factory or db.AsyncSessionLocal

    async def get(self, user_id: str, channel: str) -> Optional[SessionDTO]:
        async with self._session_factory() as session:
            stmt = (
                select(db.SessionModel)
                .where(db.SessionModel.user_id == user_id, db.SessionModel.channel == channel)
                .order_by(db.SessionModel.last_event_at.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            model = result.scalars().first()
            if not model:
                return None
            return SessionDTO.model_validate(model)

    async def save(self, session_obj: SessionDTO) -> None:
        async with self._session_factory() as session:
            stmt = mysql_insert(db.SessionModel).values(
                session_id=session_obj.session_id,
                user_id=session_obj.user_id,
                channel=session_obj.channel,
                state=session_obj.state,
            )
            stmt = stmt.on_duplicate_key_update(
                state=stmt.inserted.state, last_event_at=db.func.now()
            )
            await session.execute(stmt)
            await session.commit()

    async def delete(self, session_id: str) -> None:
        async with self._session_factory() as session:
            existing = await session.get(db.SessionModel, session_id)
            if existing:
                await session.delete(existing)
                await session.commit()
