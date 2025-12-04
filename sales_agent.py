from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import ValidationError
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import db
from alternatives_models import AlternativesResult
from config import get_settings
from llm_client import chat
from logging_utils import log_event
from metrics import REGISTRY
from sales_models import (
    AlternativesToolInput,
    PlannerOutput,
    PricingResult,
    PricingToolInput,
    SalesRequest,
    SalesResponse,
    StockInfo,
    StockToolInput,
)
from sales_tools_clients import SalesToolsClient
from rag.pipeline import SessionContext as RagSessionContext
from rag.pipeline import rag_retrieve

PLANNER_SYSTEM_PROMPT = (
    "You are a sales planner. Decide which tools to call and what to ask next. "
    "Never guess facts; always call tools for prices, stock, and alternatives."
)
STYLER_SYSTEM_PROMPT = "Rewrite the reply in our brand tone. Do not change any facts or numbers."

ChatCallable = Callable[..., Any]


class SalesSessionRepository:
    """Слой доступа к состоянию продажных сессий в MySQL."""

    def __init__(self, session_factory: Optional[async_sessionmaker[AsyncSession]] = None) -> None:
        self._session_factory = session_factory or db.AsyncSessionLocal

    async def load(self, session_id: str) -> Dict[str, Any]:
        async with self._session_factory() as session:
            record = await session.get(db.SalesSessionModel, session_id)
            if not record:
                return {}
            return dict(record.state or {})

    async def save(self, session_id: str, state: Dict[str, Any]) -> None:
        async with self._session_factory() as session:
            dialect = session.bind.dialect.name if session.bind else ""
            if dialect == "sqlite":
                stmt = sqlite_insert(db.SalesSessionModel).values(
                    session_id=session_id, state=state
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=[db.SalesSessionModel.session_id],
                    set_={"state": stmt.excluded.state, "updated_at": db.func.now()},
                )
            else:
                stmt = mysql_insert(db.SalesSessionModel).values(session_id=session_id, state=state)
                stmt = stmt.on_duplicate_key_update(state=stmt.inserted.state, updated_at=db.func.now())
            await session.execute(stmt)
            await session.commit()


def _styler_model_for_role(role: str) -> str:
    settings = get_settings()
    adapter_id = settings.llm.lora_adapter_id
    if adapter_id:
        return f"yagpt-lora-{role}"
    return "yagpt-lora-sales"


class SalesAgentService:
    """Оркестратор вызовов планировщика, тулов и стилизации."""

    def __init__(
        self,
        tools_client: SalesToolsClient,
        planner_chat: Optional[ChatCallable] = None,
        styler_chat: Optional[ChatCallable] = None,
        session_repo: Optional[SalesSessionRepository] = None,
    ) -> None:
        self.tools_client = tools_client
        self.planner_chat = planner_chat or chat
        self.styler_chat = styler_chat or self.planner_chat
        self.session_repo = session_repo or SalesSessionRepository()

    async def run(self, request: SalesRequest) -> SalesResponse:
        """Главный пайплайн агента."""
        trace_id = str(uuid.uuid4())
        session_snapshot = await self.session_repo.load(request.session_id)
        used_tools: List[str] = []

        planner_output = await self._run_planner(
            request=request,
            session_snapshot=session_snapshot,
            trace_id=trace_id,
            used_tools=used_tools,
        )
        styled_reply = await self._run_styler(
            planner_output.reply_draft, trace_id, request.session_id
        )
        await self.session_repo.save(request.session_id, {"next_state": planner_output.next_state})
        REGISTRY.counter("sales_agent_calls_total").inc()

        return SalesResponse(
            reply=styled_reply,
            suggested_actions=planner_output.actions,
            next_state=planner_output.next_state,
            used_tools=used_tools,
        )

    async def _run_planner(
        self,
        request: SalesRequest,
        session_snapshot: Dict[str, Any],
        trace_id: str,
        used_tools: List[str],
    ) -> PlannerOutput:
        # Собираем стартовые сообщения для GPT-5.
        rag_snippets = await self._fetch_rag_knowledge(request)
        context_block = json.dumps(
            {"context": request.context, "session_state": session_snapshot, "rag": rag_snippets},
            ensure_ascii=False,
        )
        user_content = f"User message: {request.user_message}\nContext: {context_block}"
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        tool_definitions = self._tool_schemas()
        start_time = time.perf_counter()
        first_completion = await asyncio.to_thread(
            self.planner_chat,
            "gpt5",
            messages,
            temperature=0,
            tools=tool_definitions,
            tool_choice="auto",
        )
        planner_latency_ms = (time.perf_counter() - start_time) * 1000
        REGISTRY.histogram("planner_latency_avg").observe(planner_latency_ms)
        log_event(
            "planner_llm_call",
            {
                "trace_id": trace_id,
                "session_id": request.session_id,
                "stage": "initial",
                "model": getattr(first_completion, "model", "gpt5"),
                "latency_ms": planner_latency_ms,
                "usage": getattr(first_completion, "usage", None),
            },
        )

        first_choice = first_completion.choices[0]
        assistant_message = getattr(first_choice, "message", None)
        tool_calls = getattr(assistant_message, "tool_calls", None) or []

        assistant_message_param: Dict[str, Any] = {
            "role": "assistant",
            "content": getattr(assistant_message, "content", None),
        }
        if tool_calls:
            assistant_message_param["tool_calls"] = [
                {
                    "id": getattr(tool_call, "id", ""),
                    "type": getattr(tool_call, "type", "function"),
                    "function": {
                        "name": getattr(tool_call.function, "name", ""),
                        "arguments": getattr(tool_call.function, "arguments", "{}"),
                    },
                }
                for tool_call in tool_calls
            ]

        tool_messages: List[Dict[str, Any]] = []
        for tool_call in tool_calls:
            tool_result = await self._execute_tool_call(
                tool_call=tool_call,
                trace_id=trace_id,
                session_id=request.session_id,
                used_tools=used_tools,
            )
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": getattr(tool_call, "id", ""),
                    "name": getattr(tool_call.function, "name", ""),
                    "content": json.dumps(tool_result, ensure_ascii=False),
                }
            )

        second_messages: List[Dict[str, Any]] = messages + [assistant_message_param] + tool_messages
        planner_schema = PlannerOutput.model_json_schema()
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "planner_output",
                "schema": planner_schema,
                "strict": True,
            },
        }

        second_start = time.perf_counter()
        second_completion = await asyncio.to_thread(
            self.planner_chat,
            "gpt5",
            second_messages,
            temperature=0,
            response_format=response_format,
        )
        second_latency_ms = (time.perf_counter() - second_start) * 1000
        REGISTRY.histogram("planner_latency_avg").observe(second_latency_ms)
        log_event(
            "planner_llm_call",
            {
                "trace_id": trace_id,
                "session_id": request.session_id,
                "stage": "final",
                "model": getattr(second_completion, "model", "gpt5"),
                "latency_ms": second_latency_ms,
                "usage": getattr(second_completion, "usage", None),
            },
        )

        try:
            content = second_completion.choices[0].message.content or "{}"
            planner_output = PlannerOutput.model_validate_json(content)
        except (ValidationError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=500, detail="planner_output_invalid") from exc
        return planner_output

    async def _fetch_rag_knowledge(self, request: SalesRequest) -> List[Dict[str, Any]]:
        """Retrieves top context snippets for the planner."""
        try:
            session = RagSessionContext(
                client_id=request.context.get("client_id"),
                product_id=request.context.get("product_id"),
                lang=request.context.get("lang"),
                metadata=request.context.get("metadata", {}),
            )
            retrieved = await rag_retrieve(request.user_message, session=session)
            return [
                {
                    "id": item.document.id,
                    "text": item.document.text,
                    "source": item.document.metadata.get("source") or item.document.metadata.get("source_type"),
                }
                for item in retrieved
            ]
        except Exception:
            return []

    async def _execute_tool_call(
        self,
        tool_call: Any,
        trace_id: str,
        session_id: str,
        used_tools: List[str],
    ) -> Dict[str, Any]:
        """Диспетчеризирует вызовы тулов по имени."""
        tool_name = getattr(tool_call.function, "name", "")
        arguments_str = getattr(tool_call.function, "arguments", "{}") or "{}"
        try:
            arguments = json.loads(arguments_str)
        except json.JSONDecodeError:
            arguments = {}

        handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {
            "call_pricing_agent": self._validated_pricing_call,
            "check_stock": self._validated_stock_call,
            "find_alternatives": self._validated_alternatives_call,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            log_event(
                "tool_call",
                {
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "tool": tool_name,
                    "status": "unknown_tool",
                },
            )
            raise HTTPException(status_code=400, detail=f"Unknown tool {tool_name}")

        start = time.perf_counter()
        try:
            result = await handler(arguments)
            status = "ok"
            used_tools.append(tool_name)
        except Exception as exc:  # pragma: no cover - защитное логирование
            status = "error"
            result = {"error": str(exc)}
            log_event(
                "tool_call",
                {
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "tool": tool_name,
                    "status": status,
                    "latency_ms": (time.perf_counter() - start) * 1000,
                    "args": arguments,
                },
            )
            raise

        log_event(
            "tool_call",
            {
                "trace_id": trace_id,
                "session_id": session_id,
                "tool": tool_name,
                "status": status,
                "latency_ms": (time.perf_counter() - start) * 1000,
                "args": arguments,
            },
        )
        return result

    async def _validated_pricing_call(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        validated = PricingToolInput.model_validate(arguments)
        result: PricingResult = await self.tools_client.call_pricing_agent(validated.order_spec)
        return result.model_dump()

    async def _validated_stock_call(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        validated = StockToolInput.model_validate(arguments)
        result: StockInfo = await self.tools_client.check_stock(validated.sku)
        return result.model_dump()

    async def _validated_alternatives_call(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        validated = AlternativesToolInput.model_validate(arguments)
        result: AlternativesResult = await self.tools_client.find_alternatives(
            validated.model_dump()
        )
        return result.model_dump()

    def _tool_schemas(self) -> List[Dict[str, Any]]:
        # Возвращаем JSON Schema для подсказки GPT-5.
        return [
            {
                "type": "function",
                "function": {
                    "name": "call_pricing_agent",
                    "description": "Получить цену и скидки для заказа.",
                    "parameters": PricingToolInput.model_json_schema(),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "check_stock",
                    "description": "Проверить наличие конкретного SKU на складе.",
                    "parameters": StockToolInput.model_json_schema(),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "find_alternatives",
                    "description": "Найти альтернативные товары.",
                    "parameters": AlternativesToolInput.model_json_schema(),
                },
            },
        ]

    async def _run_styler(self, reply_draft: str, trace_id: str, session_id: str) -> str:
        messages = [
            {"role": "system", "content": STYLER_SYSTEM_PROMPT},
            {"role": "user", "content": reply_draft},
        ]
        styler_model = _styler_model_for_role("sales")
        start = time.perf_counter()
        try:
            completion = await asyncio.to_thread(
                self.styler_chat,
                styler_model,
                messages,
                temperature=0.3,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            REGISTRY.histogram("styler_latency_ms").observe(latency_ms)
            styled_reply = completion.choices[0].message.content
            if not styled_reply:
                raise ValueError("empty_styler_reply")
            log_event(
                "styler_call",
                {
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "model": getattr(completion, "model", styler_model),
                    "latency_ms": latency_ms,
                    "status": "ok",
                },
            )
            return styled_reply
        except Exception as exc:
            log_event(
                "styler_call",
                {
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "model": styler_model,
                    "latency_ms": (time.perf_counter() - start) * 1000,
                    "status": "error",
                    "error": str(exc),
                },
            )
            return reply_draft

def _default_base_url() -> str:
    return get_settings().integrations.sales_agents_base_url


async def get_sales_agent_service(request: Request) -> SalesAgentService:
    """Стандартный провайдер сервиса через состояние FastAPI."""
    return request.app.state.sales_agent_service


def create_app() -> FastAPI:
    app = FastAPI()

    @app.on_event("startup")
    async def startup_event() -> None:
        from observability import init_logging, get_service_name

        init_logging(f"{get_service_name()}-sales")
        await db.init_db()
        base_url = _default_base_url()
        http_client = httpx.AsyncClient(base_url=base_url, timeout=10.0)
        tools_client = SalesToolsClient(http_client=http_client)
        app.state.http_client = http_client
        app.state.sales_agent_service = SalesAgentService(tools_client=tools_client)

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        http_client: Optional[httpx.AsyncClient] = getattr(app.state, "http_client", None)
        if http_client:
            await http_client.aclose()
        await db.dispose_engine()

    @app.post("/agents/sales/run", response_model=SalesResponse)
    async def run_sales_endpoint(
        sales_request: SalesRequest,
        service: SalesAgentService = Depends(get_sales_agent_service),
    ) -> SalesResponse:
        return await service.run(sales_request)

    return app


app = create_app()
