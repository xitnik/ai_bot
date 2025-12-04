from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import httpx

from config import get_settings
from .base import ChannelClient, CRMClient, LeadPayload, MarketplaceClient, StockClient
from .fake import InMemoryAvito, InMemoryBitrix, InMemoryMAX, InMemoryOneC, InMemoryTelegram


def _http_client(timeout: float = 5.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout)


class TelegramClient(ChannelClient):
    def __init__(self, token: Optional[str] = None, client: Optional[httpx.AsyncClient] = None) -> None:
        self.token = token or get_settings().integrations.telegram_bot_token
        self._client = client or _http_client()

    async def send_message(self, chat_id: str, text: str) -> None:
        # Поддерживает как реальный Telegram Bot API, так и заглушку через env.
        if self.token.startswith("fake-"):
            fake = InMemoryTelegram()
            await fake.send_message(chat_id, text)
            return
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        await self._client.post(url, json={"chat_id": chat_id, "text": text})


class MAXClient(ChannelClient):
    def __init__(self, api_token: Optional[str] = None, base_url: Optional[str] = None) -> None:
        settings = get_settings()
        self.api_token = api_token or settings.integrations.max_api_token
        self.base_url = base_url or "https://api.max.example.com"
        self._client = _http_client()

    async def send_message(self, chat_id: str, text: str) -> None:
        if self.api_token.startswith("fake-"):
            fake = InMemoryMAX()
            await fake.send_message(chat_id, text)
            return
        headers = {"Authorization": f"Bearer {self.api_token}"}
        await self._client.post(f"{self.base_url}/messages", json={"chat_id": chat_id, "text": text}, headers=headers)


class AvitoClient(MarketplaceClient):
    def __init__(self, api_token: Optional[str] = None, base_url: Optional[str] = None) -> None:
        settings = get_settings()
        self.api_token = api_token or settings.integrations.avito_api_token
        self.base_url = base_url or "https://api.avito.example.com"
        self._client = _http_client()

    async def publish_message(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.api_token.startswith("fake-"):
            fake = InMemoryAvito()
            return await fake.publish_message(payload)
        headers = {"Authorization": f"Bearer {self.api_token}"}
        response = await self._client.post(f"{self.base_url}/messages", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()


class BitrixClient(CRMClient):
    def __init__(self, webhook_url: Optional[str] = None, client: Optional[httpx.AsyncClient] = None) -> None:
        self.webhook_url = webhook_url or get_settings().integrations.bitrix24_webhook_url
        self._client = client or _http_client()
        self._fake = InMemoryBitrix()

    async def create_lead(self, payload: LeadPayload) -> Dict[str, Any]:
        if "example" in self.webhook_url or self.webhook_url.startswith("fake-"):
            return await self._fake.create_lead(payload)
        data = {
            "fields": {
                "TITLE": payload.intent or "Lead",
                "NAME": payload.name,
                "PHONE": [{"VALUE": payload.phone}] if payload.phone else [],
                "EMAIL": [{"VALUE": payload.email}] if payload.email else [],
                "COMMENTS": payload.comment,
                "UF_CRM_TRACE_ID": payload.idempotency_key,
            },
            "params": {"REGISTER_SONET_EVENT": "Y"},
        }
        response = await self._client.post(self.webhook_url, json=data)
        response.raise_for_status()
        body = response.json()
        return {"status": "created", "lead_id": body.get("result")}


class OneCClient(StockClient):
    def __init__(self, base_url: Optional[str] = None, client: Optional[httpx.AsyncClient] = None) -> None:
        self.base_url = base_url or get_settings().integrations.onec_base_url
        self._client = client or _http_client()
        self._fake = InMemoryOneC()

    async def get_stock(self, sku: str) -> Dict[str, Any]:
        if "localhost" in self.base_url:
            return await self._fake.get_stock(sku)
        response = await self._client.get(f"{self.base_url}/stock", params={"sku": sku})
        response.raise_for_status()
        return response.json()


def build_default_clients(use_fake: bool = False) -> dict[str, Any]:
    """Фабрика для Supervisor: возвращает канальные и CRM клиенты."""
    if use_fake:
        return {
            "telegram": InMemoryTelegram(),
            "max": InMemoryMAX(),
            "avito": InMemoryAvito(),
            "bitrix": InMemoryBitrix(),
            "onec": InMemoryOneC(),
        }
    return {
        "telegram": TelegramClient(),
        "max": MAXClient(),
        "avito": AvitoClient(),
        "bitrix": BitrixClient(),
        "onec": OneCClient(),
    }
