from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, BigInteger, DateTime, Index, Integer, String, func, select, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# JSONB в Postgres и JSON для совместимости с SQLite в тестах
JSONType = JSONB().with_variant(JSON, "sqlite")


class Base(DeclarativeBase):
    pass


def _default_db_url() -> str:
    return os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./bot.db")


def _build_engine(url: Optional[str] = None):
    return create_async_engine(url or _default_db_url(), echo=False, future=True)


# Глобальный движок; переинициализируется в configure_engine для тестов
engine = _build_engine()
AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine, expire_on_commit=False
)


class SessionModel(Base):
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    channel: Mapped[str] = mapped_column(String(64), index=True)
    state: Mapped[str] = mapped_column(String(64), default="idle")
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_event_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (Index("ix_sessions_user_channel", "user_id", "channel"),)


class EventModel(Base):
    __tablename__ = "events"

    # Integer проще автоинкрементируется в SQLite при тестовом прогоне.
    event_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    trace_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    session_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    user_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    payload: Mapped[dict] = mapped_column(JSONType, nullable=False)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error: Mapped[Optional[dict]] = mapped_column(JSONType, nullable=True)

    __table_args__ = (
        Index("ix_events_trace", "trace_id"),
        Index("ix_events_session", "session_id"),
        Index("ix_events_type", "event_type"),
    )


async def init_db() -> None:
    """Создание схемы для тестового/локального запуска."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def configure_engine(url: str) -> None:
    """Переинициализирует движок для тестов или другого окружения."""
    global engine, AsyncSessionLocal
    engine = _build_engine(url)
    AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db_session() -> AsyncSession:
    """Зависимость FastAPI для выдачи сессии."""
    async with AsyncSessionLocal() as session:
        yield session


async def fetch_latest_session(
    session: AsyncSession, user_id: str, channel: str
) -> Optional[SessionModel]:
    stmt = (
        select(SessionModel)
        .where(SessionModel.user_id == user_id, SessionModel.channel == channel)
        .order_by(SessionModel.last_event_at.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalars().first()


async def touch_session_state(
    session: AsyncSession, session_id: str, next_state: str
) -> None:
    """Обновляем состояние и временную метку активности."""
    await session.execute(
        update(SessionModel)
            .where(SessionModel.session_id == session_id)
            .values(state=next_state, last_event_at=func.now())
    )
    await session.commit()
