from __future__ import annotations

import logging
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from . import schemas
from .db import Offer, get_session
from .parser import parse_vendor_reply
from .selector import score_offers, select_best_offer
from .sender import MemoryRFQSender
from .service import compose_rfq, fetch_rfq_with_spec, get_offers_for_spec, save_rfq, store_offer

router = APIRouter()
logger = logging.getLogger(__name__)
sender = MemoryRFQSender()
SessionDep = Annotated[AsyncSession, Depends(get_session)]


def _offer_to_out(offer: Offer, score: Optional[float] = None) -> schemas.OfferWithScore:
    return schemas.OfferWithScore(
        id=offer.id,
        rfq_id=offer.rfq_id,
        price_per_unit=offer.price_per_unit,
        min_batch=offer.min_batch,
        lead_time_days=offer.lead_time_days,
        terms_text=offer.terms_text,
        vendor_score=offer.vendor_score,
        raw_text=offer.raw_text,
        created_at=offer.created_at,
        score=score if score is not None else 0.0,
    )


@router.post("/agents/procurement/rfq")
async def create_rfq(
    payload: schemas.RFQCreateRequest, session: SessionDep
) -> list[int]:
    """Создает RFQ для списка поставщиков."""

    rfq_ids: list[int] = []
    for vendor in payload.vendors:
        structured = compose_rfq(payload.spec, vendor)
        try:
            rfq_id = await save_rfq(payload.spec, vendor, structured, session)
        except Exception as exc:
            logger.exception("Failed to save RFQ for vendor %s", vendor.name)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
            ) from exc
        rfq_ids.append(rfq_id)
        try:
            await sender.send_rfq(vendor, structured)
        except Exception:
            logger.exception("Stub sender failed for RFQ %s", rfq_id)
    return rfq_ids


@router.post("/agents/procurement/parse_reply")
async def parse_reply(
    payload: schemas.ParseReplyRequest, session: SessionDep
) -> schemas.OfferOut:
    """Парсит письмо поставщика и сохраняет оффер."""

    try:
        rfq, _spec_dict = await fetch_rfq_with_spec(payload.rfq_id, session)
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    try:
        parsed = await parse_vendor_reply(payload.raw_text)
    except Exception as exc:
        logger.exception("Parse failed for RFQ %s", payload.rfq_id)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    try:
        offer = await store_offer(rfq.id, parsed, payload.raw_text, session)
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to store offer for RFQ %s", payload.rfq_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc

    return schemas.OfferOut(
        id=offer.id,
        rfq_id=offer.rfq_id,
        price_per_unit=offer.price_per_unit,
        min_batch=offer.min_batch,
        lead_time_days=offer.lead_time_days,
        terms_text=offer.terms_text,
        vendor_score=offer.vendor_score,
        raw_text=offer.raw_text,
        created_at=offer.created_at,
    )


@router.get("/agents/procurement/best_offer/{spec_id}")
async def best_offer(
    spec_id: int, session: SessionDep
) -> schemas.BestOfferResponse:
    """Агрегирует офферы по спецификации и выбирает лучший."""

    offers = await get_offers_for_spec(spec_id, session)
    if not offers:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No offers for spec")

    weights = {"price_per_unit": 0.5, "lead_time_days": 0.3, "vendor_score": 0.2}
    best = select_best_offer(list(offers), weights)
    scored = score_offers(offers, weights)
    comparison = [_offer_to_out(off, sc) for off, sc in scored]
    comparison.sort(key=lambda item: item.score, reverse=True)
    best_scored = next((item for item in comparison if item.id == best.id), comparison[0])
    return schemas.BestOfferResponse(best_offer=best_scored, comparison=comparison)


__all__ = ["router"]
