from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class SalesRequest(BaseModel):
    """Запрос на запуск продажного агента."""

    session_id: str
    user_message: str
    context: Dict[str, Any] = Field(default_factory=dict)


class SalesResponse(BaseModel):
    """Финальный ответ, отдаваемый пользователю."""

    reply: str
    suggested_actions: List[str]
    next_state: str
    used_tools: List[str]


class PlannerOutput(BaseModel):
    """Структурированный черновик после планирования."""

    model_config = ConfigDict(extra="allow")

    reply_draft: str
    actions: List[str]
    next_state: str


class PricingToolInput(BaseModel):
    """Аргументы для вызова ценообразования."""

    model_config = ConfigDict(extra="forbid")

    order_spec: Dict[str, Any] = Field(default_factory=dict)


class PricingResult(BaseModel):
    """Минимальная модель ответа ценообразования."""

    model_config = ConfigDict(extra="allow")

    currency: Optional[str] = None
    amount: Optional[float] = None
    items: Optional[List[Dict[str, Any]]] = None


class StockToolInput(BaseModel):
    """Аргументы для проверки стока."""

    model_config = ConfigDict(extra="forbid")

    sku: str


class StockInfo(BaseModel):
    """Ответ по наличию товара."""

    model_config = ConfigDict(extra="allow")

    sku: str
    available: bool
    quantity: Optional[int] = None
    location: Optional[str] = None


class AlternativesToolInput(BaseModel):
    """Аргументы для поиска альтернатив."""

    model_config = ConfigDict(extra="forbid")

    query_text: str
    hard_filters: Dict[str, Any] = Field(default_factory=dict)
    k: int = 5
    price_band: Optional[Literal["cheaper", "premium"]] = None
