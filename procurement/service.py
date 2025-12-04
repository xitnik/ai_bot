from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag.pipeline import SessionContext as RagSessionContext
from rag.pipeline import rag_retrieve

from . import schemas
from .db import RFQ, Offer, RFQSpecRecord

logger = logging.getLogger(__name__)


def compose_rfq(spec: schemas.RFQSpec, vendor: schemas.Vendor) -> dict:
    """
    Формирует структурированный payload для отправки по email или API.
    """

    payload = {
        "recipient": vendor.address,
        "subject": f"RFQ: {spec.species} {spec.grade}",
        "greeting": f"Здравствуйте, {vendor.name}!",
        "details": {
            "species": spec.species,
            "grade": spec.grade,
            "volume": spec.volume,
            "delivery_terms": spec.delivery_terms,
            "deadline": spec.deadline.isoformat(),
        },
        "instructions": (
            "Просьба ответить ценой за единицу, минимальной партией, сроком поставки"
            " и условиями оплаты."
        ),
    }
    references = _retrieve_rag_references(spec)
    if references:
        payload["references"] = references
    return payload


async def save_rfq(
    spec: schemas.RFQSpec, vendor: schemas.Vendor, payload: dict, session: AsyncSession
) -> int:
    """
    Создает RFQ spec (если еще нет) и заявку к конкретному вендору.
    """

    spec_dict = spec.model_dump(mode="json")
    logger.debug("RFQ payload preview for vendor %s: %s", vendor.name, payload)
    result = await session.execute(select(RFQSpecRecord).where(RFQSpecRecord.spec == spec_dict))
    spec_record = result.scalar_one_or_none()
    if spec_record is None:
        spec_record = RFQSpecRecord(spec=spec_dict)
        session.add(spec_record)
        await session.flush()

    rfq = RFQ(spec_id=spec_record.id, vendor_id=vendor.id, status="SENT")
    session.add(rfq)
    await session.commit()
    await session.refresh(rfq)
    logger.info("RFQ %s saved for vendor %s", rfq.id, vendor.name)
    return rfq.id


async def update_status(rfq_id: int, status: schemas.RFQStatus, session: AsyncSession) -> None:
    """Обновляет статус RFQ, бросает ошибку если запись не найдена."""

    result = await session.execute(select(RFQ).where(RFQ.id == rfq_id))
    rfq = result.scalar_one_or_none()
    if rfq is None:
        raise LookupError(f"RFQ {rfq_id} not found")
    rfq.status = status
    await session.commit()


async def store_offer(
    rfq_id: int,
    parsed: schemas.OfferCore,
    raw_text: str,
    session: AsyncSession,
    vendor_score: Optional[float] = None,
) -> Offer:
    """Сохраняет оффер, обеспечивая идемпотентность по raw_text."""

    existing = await session.execute(
        select(Offer).where(Offer.rfq_id == rfq_id, Offer.raw_text == raw_text)
    )
    found = existing.scalar_one_or_none()
    if found:
        return found

    rfq = await session.get(RFQ, rfq_id)
    if rfq is None:
        raise LookupError(f"RFQ {rfq_id} not found")

    offer = Offer(
        rfq_id=rfq_id,
        price_per_unit=parsed.price_per_unit,
        min_batch=parsed.min_batch,
        lead_time_days=parsed.lead_time_days,
        terms_text=parsed.terms_text,
        vendor_score=vendor_score,
        raw_text=raw_text,
    )
    session.add(offer)
    rfq.status = "ANSWERED"
    await session.commit()
    await session.refresh(offer)
    return offer


async def get_offers_for_spec(spec_id: int, session: AsyncSession) -> list[Offer]:
    """Возвращает все офферы для заданной спецификации."""

    result = await session.execute(
        select(Offer).join(RFQ, RFQ.id == Offer.rfq_id).where(RFQ.spec_id == spec_id)
    )
    return list(result.scalars().all())


def _retrieve_rag_references(spec: schemas.RFQSpec) -> list[dict[str, Any]]:
    """
    Sync helper to pull a couple of RAG snippets for the RFQ.
    Falls back silently if retrieval fails.
    """
    try:
        query = f"Условия поставки {spec.species} {spec.grade} {spec.delivery_terms}"

        async def _run() -> list[dict[str, Any]]:
            ctx = RagSessionContext(lang="ru")
            docs = await rag_retrieve(query, session=ctx)
            return [
                {
                    "id": d.document.id,
                    "text": d.document.text[:300],
                    "source": d.document.metadata.get("source"),
                }
                for d in docs
            ]

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            return []
        return asyncio.run(_run())
    except Exception:
        return []


async def fetch_rfq_with_spec(rfq_id: int, session: AsyncSession) -> tuple[RFQ, dict[str, Any]]:
    """Получает RFQ и его исходную спецификацию как dict."""

    rfq = await session.get(RFQ, rfq_id)
    if rfq is None:
        raise LookupError(f"RFQ {rfq_id} not found")
    spec_record = await session.get(RFQSpecRecord, rfq.spec_id)
    spec_dict: dict[str, Any] = spec_record.spec if spec_record else {}
    return rfq, spec_dict


__all__ = [
    "compose_rfq",
    "save_rfq",
    "update_status",
    "store_offer",
    "get_offers_for_spec",
    "fetch_rfq_with_spec",
]
