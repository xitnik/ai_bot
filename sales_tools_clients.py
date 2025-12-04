from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx

from alternatives_models import AlternativesRequest, AlternativesResult
from sales_models import PricingResult, StockInfo

DEFAULT_BASE_URL = os.getenv("SALES_AGENTS_BASE_URL", "http://localhost:8000")


class SalesToolsClient:
    """HTTP-клиент для внутренних агентов и складского API."""

    def __init__(
        self,
        http_client: Optional[httpx.AsyncClient] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            base_url=base_url or DEFAULT_BASE_URL, timeout=10.0
        )

    async def aclose(self) -> None:
        """Закрывает HTTP клиент, если он создан здесь."""
        if self._owns_client:
            await self._client.aclose()

    async def call_pricing_agent(self, order_spec: Dict[str, Any]) -> PricingResult:
        """Делает POST на агент ценообразования."""
        response = await self._client.post("/agents/pricing/run", json={"order_spec": order_spec})
        response.raise_for_status()
        return PricingResult.model_validate(response.json())

    async def check_stock(self, sku: str) -> StockInfo:
        """Делает GET на складской API."""
        response = await self._client.get("/1c_api/stock", params={"sku": sku})
        response.raise_for_status()
        return StockInfo.model_validate(response.json())

    async def find_alternatives(self, query: Dict[str, Any]) -> AlternativesResult:
        """Делает POST на агент подбора альтернатив."""
        payload = AlternativesRequest(
            query_text=query.get("query_text") or query.get("text") or "",
            hard_filters=query.get("hard_filters") or {},
            k=int(query.get("k") or 5),
            price_band=query.get("price_band"),
        ).model_dump(mode="json")
        response = await self._client.post("/agents/alternatives/run", json=payload)
        response.raise_for_status()
        return AlternativesResult.model_validate(response.json())
