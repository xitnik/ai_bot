from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import httpx
import pytest

from sales_agent import SalesAgentService
from sales_models import SalesRequest
from sales_tools_clients import SalesToolsClient


class DummyToolCallFunction:
    """Упрощенный аналог FunctionCall из OpenAI ответа."""

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
        self.usage = {"prompt_tokens": 10, "completion_tokens": 20}


@pytest.mark.asyncio
async def test_planner_invokes_pricing_and_stock_tools() -> None:
    pricing_requests: List[Dict[str, Any]] = []
    stock_requests: List[Dict[str, Any]] = []

    # Мокаем HTTP-агентов через MockTransport.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/agents/pricing/run":
            pricing_requests.append(json.loads(request.content.decode()))
            return httpx.Response(200, json={"currency": "USD", "amount": 42})
        if request.url.path == "/1c_api/stock":
            stock_requests.append(dict(request.url.params))
            return httpx.Response(
                200, json={"sku": request.url.params.get("sku"), "available": True, "quantity": 7}
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    async_client = httpx.AsyncClient(transport=transport, base_url="http://testserver")
    tools_client = SalesToolsClient(http_client=async_client, base_url="http://testserver")

    call_counter = {"value": 0}

    def planner_stub(model: str, messages: List[Dict[str, Any]], **_: Any) -> DummyCompletion:
        idx = call_counter["value"]
        call_counter["value"] += 1
        if idx == 0:
            tool_calls = [
                DummyToolCall(
                    "call-1",
                    "call_pricing_agent",
                    {"order_spec": {"sku": "sku-1", "qty": 1}},
                ),
                DummyToolCall("call-2", "check_stock", {"sku": "sku-1"}),
            ]
            return DummyCompletion(DummyMessage(content=None, tool_calls=tool_calls))
        if idx == 1:
            payload = {
                "reply_draft": "Price and stock ready",
                "actions": ["go_to_checkout"],
                "next_state": "quoted",
            }
            return DummyCompletion(DummyMessage(content=json.dumps(payload)))
        return DummyCompletion(DummyMessage(content="Styled response"), model="yagpt-lora-sales")

    service = SalesAgentService(
        tools_client=tools_client,
        planner_chat=planner_stub,
        styler_chat=planner_stub,
    )
    request = SalesRequest(session_id="sess-1", user_message="I need sku-1", context={})

    response = await service.run(request)
    await async_client.aclose()

    assert pricing_requests == [{"order_spec": {"sku": "sku-1", "qty": 1}}]
    assert stock_requests == [{"sku": "sku-1"}]
    assert set(response.used_tools) == {"call_pricing_agent", "check_stock"}
    assert response.reply == "Styled response"


@pytest.mark.asyncio
async def test_styler_fallback_to_draft_on_error() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={}))
    async_client = httpx.AsyncClient(transport=transport, base_url="http://testserver")
    tools_client = SalesToolsClient(http_client=async_client, base_url="http://testserver")

    call_counter = {"value": 0}

    def planner_stub(model: str, messages: List[Dict[str, Any]], **_: Any) -> DummyCompletion:
        idx = call_counter["value"]
        call_counter["value"] += 1
        if idx == 0:
            return DummyCompletion(DummyMessage(content=None, tool_calls=[]))
        payload = {
            "reply_draft": "Draft reply",
            "actions": ["ask_more"],
            "next_state": "awaiting_info",
        }
        return DummyCompletion(DummyMessage(content=json.dumps(payload)))

    def styler_stub(*_: Any, **__: Any) -> DummyCompletion:
        raise RuntimeError("styler_fail")

    service = SalesAgentService(
        tools_client=tools_client,
        planner_chat=planner_stub,
        styler_chat=styler_stub,
    )
    request = SalesRequest(session_id="sess-2", user_message="hello", context={})

    response = await service.run(request)
    await async_client.aclose()

    assert response.reply == "Draft reply"
    assert response.suggested_actions == ["ask_more"]
    assert response.used_tools == []
