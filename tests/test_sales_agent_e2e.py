from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import httpx
from fastapi.testclient import TestClient

from sales_agent import SalesAgentService, create_app, get_sales_agent_service
from sales_tools_clients import SalesToolsClient


class DummyToolCallFunction:
    """Упрощенные структуры для эмуляции OpenAI ответов в интеграционном тесте."""

    def __init__(self, name: str, arguments: Dict[str, Any]) -> None:
        self.name = name
        self.arguments = json.dumps(arguments)


class DummyToolCall:
    def __init__(self, call_id: str, name: str, arguments: Dict[str, Any]) -> None:
        self.id = call_id
        self.type = "function"
        self.function = DummyToolCallFunction(name, arguments)


class DummyMessage:
    def __init__(
        self, content: Optional[str], tool_calls: Optional[List[DummyToolCall]] = None
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class DummyChoice:
    def __init__(self, message: DummyMessage) -> None:
        self.message = message


class DummyCompletion:
    def __init__(self, message: DummyMessage, model: str = "gpt5") -> None:
        self.choices = [DummyChoice(message)]
        self.model = model
        self.usage = {"prompt_tokens": 5, "completion_tokens": 5}


def test_sales_agent_e2e() -> None:
    http_calls: Dict[str, int] = {"pricing": 0, "stock": 0, "alternatives": 0}

    # Ответы подменяются через MockTransport, чтобы агент дергал HTTP слой.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/agents/pricing/run":
            http_calls["pricing"] += 1
            payload = json.loads(request.content.decode())
            qty = payload["order_spec"].get("qty", 1)
            return httpx.Response(200, json={"currency": "USD", "amount": 25 * qty})
        if request.url.path == "/1c_api/stock":
            http_calls["stock"] += 1
            sku = request.url.params.get("sku")
            return httpx.Response(200, json={"sku": sku, "available": True, "quantity": 3})
        if request.url.path == "/agents/alternatives/run":
            http_calls["alternatives"] += 1
            return httpx.Response(
                200,
                json={
                    "alternatives": [
                        {
                            "product_id": "alt-1",
                            "similarity": 0.8,
                            "type": "functional",
                            "reason": "stub",
                        }
                    ]
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    async_client = httpx.AsyncClient(transport=transport, base_url="http://testserver")
    tools_client = SalesToolsClient(http_client=async_client, base_url="http://testserver")

    call_counter = {"value": 0}

    def llm_stub(model: str, messages: List[Dict[str, Any]], **_: Any) -> DummyCompletion:
        idx = call_counter["value"]
        call_counter["value"] += 1
        if idx == 0:
            tool_calls = [
                DummyToolCall(
                    "call-1",
                    "call_pricing_agent",
                    {"order_spec": {"sku": "sku-9", "qty": 2}},
                ),
                DummyToolCall("call-2", "check_stock", {"sku": "sku-9"}),
                DummyToolCall(
                    "call-3",
                    "find_alternatives",
                    {"query_text": "sku-9", "hard_filters": {"sku": "sku-9"}, "k": 3},
                ),
            ]
            return DummyCompletion(DummyMessage(content=None, tool_calls=tool_calls))
        if idx == 1:
            payload = {
                "reply_draft": "Price ready, stock confirmed.",
                "actions": ["go_to_checkout", "show_alternatives"],
                "next_state": "ready_for_checkout",
            }
            return DummyCompletion(DummyMessage(content=json.dumps(payload)))
        return DummyCompletion(DummyMessage(content="Styled reply!"), model="yagpt-lora-sales")

    service = SalesAgentService(
        tools_client=tools_client,
        planner_chat=llm_stub,
        styler_chat=llm_stub,
    )

    app = create_app()
    app.dependency_overrides[get_sales_agent_service] = lambda: service

    with TestClient(app) as client:
        payload = {
            "session_id": "sess-e2e",
            "user_message": "Need sku-9",
            "context": {"locale": "ru"},
        }
        response = client.post("/agents/sales/run", json=payload)

    asyncio.run(async_client.aclose())

    assert response.status_code == 200
    body = response.json()
    assert body["reply"]
    assert body["next_state"] == "ready_for_checkout"
    assert "go_to_checkout" in body["suggested_actions"]
    assert set(body["used_tools"]) >= {"call_pricing_agent", "check_stock", "find_alternatives"}
    assert http_calls["pricing"] == 1
    assert http_calls["stock"] == 1
    assert http_calls["alternatives"] == 1
