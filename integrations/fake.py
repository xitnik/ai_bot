from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from .base import ChannelClient, CRMClient, LeadPayload, MarketplaceClient, StockClient


class InMemoryTelegram(ChannelClient):
    def __init__(self) -> None:
        self.messages: List[tuple[str, str]] = []

    async def send_message(self, chat_id: str, text: str) -> None:
        await asyncio.sleep(0)
        self.messages.append((chat_id, text))


class InMemoryMAX(ChannelClient):
    def __init__(self) -> None:
        self.messages: List[Dict[str, str]] = []

    async def send_message(self, chat_id: str, text: str) -> None:
        await asyncio.sleep(0)
        self.messages.append({"chat_id": chat_id, "text": text})


class InMemoryAvito(MarketplaceClient):
    def __init__(self) -> None:
        self.published: List[Dict[str, Any]] = []

    async def publish_message(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0)
        self.published.append(payload)
        return {"status": "ok", "id": len(self.published)}


class InMemoryBitrix(CRMClient):
    def __init__(self) -> None:
        self.leads: Dict[str, LeadPayload] = {}

    async def create_lead(self, payload: LeadPayload) -> Dict[str, Any]:
        await asyncio.sleep(0)
        key = payload.idempotency_key or f"{payload.user_id}-{payload.intent}"
        if key in self.leads:
            return {"status": "duplicate", "lead_id": self.leads[key].idempotency_key}
        self.leads[key] = payload
        return {"status": "created", "lead_id": key}


class InMemoryOneC(StockClient):
    def __init__(self) -> None:
        self.stock: Dict[str, Dict[str, Any]] = {}

    async def get_stock(self, sku: str) -> Dict[str, Any]:
        await asyncio.sleep(0)
        return self.stock.get(sku, {"sku": sku, "available": False, "quantity": 0})
