from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    select,
    update,
)
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from config import get_settings


class Base(DeclarativeBase):
    pass


def _default_db_url() -> str:
    settings = get_settings().database
    return settings.async_url()


def _build_engine(url: Optional[str] = None) -> AsyncEngine:
    settings = get_settings().database
    target_url = url or _default_db_url()
    kwargs: dict[str, Any] = {
        "echo": settings.mysql_echo,
        "pool_pre_ping": True,
        "future": True,
    }
    if target_url.startswith("sqlite"):
        # SQLite драйвер не поддерживает pool_size/pool_pre_ping.
        kwargs.pop("pool_pre_ping", None)
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["pool_size"] = settings.mysql_pool_size
    return create_async_engine(target_url, **kwargs)


# Глобальный движок; переинициализируется в configure_engine при тестах/смене окружения.
engine: AsyncEngine = _build_engine()
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

    event_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    trace_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    session_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    user_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_events_trace", "trace_id"),
        Index("ix_events_session", "session_id"),
        Index("ix_events_type", "event_type"),
    )


class SalesSessionModel(Base):
    __tablename__ = "sales_sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    state: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class DocumentModel(Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    text: Mapped[str] = mapped_column(Text(), nullable=False)
    source: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    source_type: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)
    client_id: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)
    product_id: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)
    lang: Mapped[Optional[str]] = mapped_column(String(8), index=True, nullable=True)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, nullable=False, default=dict)
    embedding: Mapped[Optional[list[float]]] = mapped_column(JSON, nullable=True)
    embedding_model: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_documents_client", "client_id"),
        Index("ix_documents_product", "product_id"),
        Index("ix_documents_source_type", "source_type"),
        Index("ix_documents_lang", "lang"),
    )


class ProductVectorModel(Base):
    __tablename__ = "product_vectors"

    product_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    vector: Mapped[list[float]] = mapped_column(JSON, nullable=False)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    species: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)
    grade: Mapped[Optional[str]] = mapped_column(String(32), index=True, nullable=True)
    price: Mapped[Optional[float]] = mapped_column(Numeric(12, 2), nullable=True)
    in_stock: Mapped[bool] = mapped_column(Boolean, default=True)
    dimensions: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("ix_product_vectors_species", "species"),
        Index("ix_product_vectors_grade", "grade"),
        Index("ix_product_vectors_in_stock", "in_stock"),
    )


class KGEntityModel(Base):
    __tablename__ = "kg_entities"

    entity_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    entity_type: Mapped[str] = mapped_column(String(32), index=True)
    attributes: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class KGEdgeModel(Base):
    __tablename__ = "kg_edges"

    edge_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    src_id: Mapped[str] = mapped_column(String(128), ForeignKey("kg_entities.entity_id"), nullable=False)
    dst_id: Mapped[str] = mapped_column(String(128), ForeignKey("kg_entities.entity_id"), nullable=False)
    relation: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_kg_edges_src_rel", "src_id", "relation"),
        UniqueConstraint("src_id", "dst_id", "relation", name="uq_kg_edge"),
    )


async def init_db(engine_override: Optional[AsyncEngine] = None) -> None:
    """Создание схемы для локального запуска/миграций."""
    db_engine = engine_override or engine
    async with db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_engine() -> None:
    await engine.dispose()


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


async def upsert_sales_session_state(session_id: str, state: dict[str, Any]) -> None:
    async with AsyncSessionLocal() as session:
        dialect = session.bind.dialect.name if session.bind else ""
        if dialect == "sqlite":
            stmt = sqlite_insert(SalesSessionModel).values(session_id=session_id, state=state)
            stmt = stmt.on_conflict_do_update(
                index_elements=[SalesSessionModel.session_id],
                set_={"state": stmt.excluded.state, "updated_at": func.now()},
            )
        else:
            stmt = mysql_insert(SalesSessionModel).values(session_id=session_id, state=state)
            stmt = stmt.on_duplicate_key_update(state=stmt.inserted.state, updated_at=func.now())
        await session.execute(stmt)
        await session.commit()


async def load_sales_session_state(session_id: str) -> dict[str, Any]:
    async with AsyncSessionLocal() as session:
        record = await session.get(SalesSessionModel, session_id)
        if not record:
            return {}
        return dict(record.state or {})
