from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


AlternativeType = Literal["direct", "functional", "price_low", "price_high"]


class Dimensions(BaseModel):
    """Габариты изделия в миллиметрах."""

    length: float
    width: float
    thickness: float


class AlternativesRequest(BaseModel):
    """Запрос на подбор альтернатив."""

    query_text: str
    hard_filters: Dict[str, Any] = Field(default_factory=dict)
    k: int = 5
    price_band: Optional[Literal["cheaper", "premium"]] = None

    model_config = ConfigDict(extra="ignore")


class AlternativeItem(BaseModel):
    """Элемент ответа с типом рекомендации."""

    product_id: str
    similarity: float
    type: AlternativeType
    reason: str


class AlternativesResult(BaseModel):
    """Структура ответа агенту."""

    alternatives: List[AlternativeItem] = Field(default_factory=list)


class Hit(BaseModel):
    """Результат поиска по векторному индексу."""

    product_id: str
    score: float
    metadata: Dict[str, Any]

    model_config = ConfigDict(frozen=True)

