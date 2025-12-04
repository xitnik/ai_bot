from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# Базовые контракты для взаимодействия с агентом.


class RFQSpec(BaseModel):
    species: str
    grade: str
    volume: float
    delivery_terms: str
    deadline: datetime


class Vendor(BaseModel):
    id: int
    name: str
    channel: Literal["email", "api"]
    address: str


class OfferCore(BaseModel):
    price_per_unit: Optional[float] = Field(default=None)
    min_batch: Optional[float] = Field(default=None)
    lead_time_days: Optional[int] = Field(default=None)
    terms_text: str


class RFQCreateRequest(BaseModel):
    spec: RFQSpec
    vendors: list[Vendor]


class ParseReplyRequest(BaseModel):
    rfq_id: int
    raw_text: str


class OfferOut(BaseModel):
    id: int
    rfq_id: int
    price_per_unit: Optional[float]
    min_batch: Optional[float]
    lead_time_days: Optional[int]
    terms_text: str
    vendor_score: Optional[float]
    raw_text: str
    created_at: datetime


class OfferWithScore(OfferOut):
    score: float


class BestOfferResponse(BaseModel):
    best_offer: OfferWithScore
    comparison: list[OfferWithScore]


RFQStatus = Literal["SENT", "PENDING", "ANSWERED", "FAILED"]


def offer_core_json_schema() -> dict[str, Any]:
    """Возвращает JSON схему OfferCore для передачи в LLM."""

    # Прямое построение схемы, чтобы не зависеть от внутренностей Pydantic.
    return {
        "title": "OfferCore",
        "type": "object",
        "properties": {
            "price_per_unit": {"type": "number"},
            "min_batch": {"type": "number"},
            "lead_time_days": {"type": "integer"},
            "terms_text": {"type": "string"},
        },
        "required": ["terms_text"],
        "additionalProperties": False,
    }
