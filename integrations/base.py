from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol


@dataclass
class LeadPayload:
    user_id: str
    name: str
    phone: str | None = None
    email: str | None = None
    comment: str | None = None
    intent: str | None = None
    idempotency_key: str | None = None


class ChannelClient(Protocol):
    async def send_message(self, chat_id: str, text: str) -> None:  # pragma: no cover - интерфейс
        ...


class CRMClient(Protocol):
    async def create_lead(self, payload: LeadPayload) -> Dict[str, Any]:  # pragma: no cover - интерфейс
        ...


class StockClient(Protocol):
    async def get_stock(self, sku: str) -> Dict[str, Any]:  # pragma: no cover - интерфейс
        ...


class MarketplaceClient(Protocol):
    async def publish_message(self, payload: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover - интерфейс
        ...
