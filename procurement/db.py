from __future__ import annotations

import os
from typing import AsyncGenerator, Optional

import sqlalchemy as sa
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, relationship

from config import get_settings

try:  # SQLAlchemy <2.0 совместимость
    from sqlalchemy.ext.asyncio import async_sessionmaker
except ImportError:  # pragma: no cover
    from sqlalchemy.orm import sessionmaker as async_sessionmaker

# Базовая декларативная модель SQLAlchemy.


Base = declarative_base()

json_variant = sa.JSON()


class RFQSpecRecord(Base):
    __tablename__ = "rfq_specs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    spec = Column(json_variant, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    rfqs = relationship("RFQ", back_populates="spec_record")


class RFQ(Base):
    __tablename__ = "rfq_requests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    spec_id = Column(Integer, ForeignKey("rfq_specs.id"), nullable=False)
    vendor_id = Column(Integer, nullable=False)
    status = Column(String(16), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    spec_record = relationship("RFQSpecRecord", back_populates="rfqs")
    offers = relationship("Offer", back_populates="rfq")


class Offer(Base):
    __tablename__ = "rfq_offers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    rfq_id = Column(Integer, ForeignKey("rfq_requests.id"), nullable=False)
    price_per_unit = Column(Float, nullable=True)
    min_batch = Column(Float, nullable=True)
    lead_time_days = Column(Integer, nullable=True)
    terms_text = Column(Text, nullable=False, default="")
    vendor_score = Column(Float, nullable=True)
    raw_text = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    rfq = relationship("RFQ", back_populates="offers")


def _database_url() -> str:
    """Получение URL MySQL для procurement-модуля."""

    settings = get_settings().database
    env_override = os.getenv("PROCUREMENT_DATABASE_URL")
    return env_override or settings.procurement_async_url()


def create_engine(url: Optional[str] = None) -> AsyncEngine:
    """Создает новый AsyncEngine для заданного URL."""

    target_url = url or _database_url()
    kwargs = {"echo": False, "future": True}
    if target_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["pool_pre_ping"] = True
    return create_async_engine(target_url, **kwargs)


engine: AsyncEngine = create_engine()
AsyncSessionMaker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def init_db(async_engine: Optional[AsyncEngine] = None) -> None:
    """Инициализация схемы БД."""

    engine_to_use = async_engine or engine
    async with engine_to_use.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Зависимость FastAPI для выдачи сессии."""

    async with AsyncSessionMaker() as session:
        yield session


__all__ = [
    "Base",
    "RFQSpecRecord",
    "RFQ",
    "Offer",
    "engine",
    "AsyncSessionMaker",
    "init_db",
    "get_session",
    "create_engine",
]
