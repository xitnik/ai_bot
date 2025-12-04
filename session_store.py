from __future__ import annotations

import asyncio
from typing import Dict, Optional, Tuple

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
