from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Optional

import httpx

from agents.base import AgentInput
from agents.circuit_breaker import CircuitBreakerRegistry
from agents.supervisor import Supervisor
from events_logger import log_event
from models import Message, SessionDTO
from ner import extract_entities
from metrics import REGISTRY
from otel import get_tracer
from rag.ingest import ingest_from_dir
from rag.retriever import RagEngine
from rag.pipeline import SessionContext as RagSessionContext
from rag.pipeline import rag_retrieve

_RAG_ENGINE: Optional[RagEngine] = None
_SUPERVISOR = Supervisor()
_CIRCUIT = CircuitBreakerRegistry()
_STATE_TRANSITIONS: Dict[str, str] = {
    "idle": "sales_followup",
    "sales": "sales_followup",
    "pricing": "pricing_quote",
    "alternatives": "alternatives_suggest",
    "procurement": "procurement_flow",
    "calculator": "calculator_flow",
}


def _get_rag_engine() -> Optional[RagEngine]:
    global _RAG_ENGINE
    if _RAG_ENGINE is None:
        try:
            chunks = ingest_from_dir("fixtures/rag_docs")
            _RAG_ENGINE = RagEngine(chunks)
        except Exception:
            _RAG_ENGINE = None
    return _RAG_ENGINE

async def enrich_message(message: Message) -> dict:
    """Обогащалка: нормализуем текст, выделяем сущности и RAG-контекст."""
    normalized = message.text.lower().strip()
    entities = extract_entities(message.text)

    snippets: List[str] = []
    try:
        rag_session = RagSessionContext(
            user_id=message.user_id,
            client_id=None,
            product_id=None,
            lang="ru",
            entities=entities,
        )
        retrieved = await rag_retrieve(normalized, session=rag_session)
        snippets = [item.document.text for item in retrieved]
    except Exception:
        rag = _get_rag_engine()
        if rag:
            try:
                results = rag.retrieve(normalized, k=2)
                snippets = [doc.text for doc, _ in results]
            except Exception:
                snippets = []
    return {"normalized_text": normalized, "entities": entities, "knowledge": snippets}


def _next_state(current: str, intent: str) -> str:
    """Простая FSM: если уже в состоянии, удерживаем его, иначе идем по переходу."""
    if current and current not in ("idle", ""):
        return current
    return _STATE_TRANSITIONS.get(intent, "sales_followup")


async def route_message(session: SessionDTO, message: Message, context: dict) -> dict:
    """FSM + Supervisor: выбираем интент, агентов, A/B вариант и стейт."""
    agent_input = AgentInput(
        message=message.text,
        user_id=message.user_id,
        session_id=session.session_id,
        intent=session.state,
        context=context,
    )
    decision = _SUPERVISOR.route(agent_input)

    payloads = {
        agent: {"message": message.model_dump(), "context": context, "state": session.state}
        for agent in decision.agents
    }
    return {
        "intent": decision.intent,
        "agents": decision.agents,
        "payloads": payloads,
        "next_state": _next_state(session.state, decision.intent),
        "variant": decision.variant,
    }


async def call_agent(agent_name: str, payload: dict, trace_id: str, session_id: str) -> dict:
    """Обертка над вызовом агента с логированием, трейсами и circuit breaker."""
    tracer = get_tracer()
    error_payload: Optional[Dict[str, Any]] = None
    start = time.perf_counter()
    result: Dict[str, Any] = {}

    with tracer.start_as_current_span(
        f"AgentRun:{agent_name}",
        attributes={
            "agent.name": agent_name,
            "session.id": session_id,
            "trace.id": trace_id,
        },
    ):
        async def _call() -> Dict[str, Any]:
            base_url = os.getenv("AGENTS_BASE_URL")
            if base_url:
                async with httpx.AsyncClient(timeout=5) as client:
                    response = await client.post(
                        f"{base_url}/agents/{agent_name}/run",
                        json=payload,
                    )
                    response.raise_for_status()
                    return response.json()
            # Стаб агентского ответа для локальной разработки.
            await asyncio.sleep(0)
            return {"agent": agent_name, "status": "ok", "data": {"echo": payload}}

        result = await _CIRCUIT.run(agent_name, _call)
        if result.get("status") == "error" and "error" in result:
            error_payload = result.get("error")

        latency_ms = int((time.perf_counter() - start) * 1000)
        await log_event(
            trace_id,
            session_id,
            "AgentRun",
            {"agent": agent_name, "payload": payload, "result": result},
            latency_ms=latency_ms,
            error=error_payload,
        )
        return result


async def call_integration(name: str, payload: dict, trace_id: str, session_id: str) -> dict:
    """Обертка для интеграций (1С/CRM) по аналогии с агентами."""
    tracer = get_tracer()
    start = time.perf_counter()
    error_payload: Optional[Dict[str, Any]] = None
    result: Dict[str, Any] = {}

    with tracer.start_as_current_span(
        f"Integration:{name}",
        attributes={"integration.name": name, "trace.id": trace_id, "session.id": session_id},
    ):
        try:
            base_url = os.getenv("INTEGRATIONS_BASE_URL")
            if base_url:
                async with httpx.AsyncClient(timeout=5) as client:
                    response = await client.post(
                        f"{base_url}/integrations/{name}/call", json=payload
                    )
                    response.raise_for_status()
                    result = response.json()
            else:
                await asyncio.sleep(0)
                result = {"integration": name, "status": "ok", "data": {"echo": payload}}
        except Exception as exc:  # noqa: BLE001
            error_payload = {"message": str(exc), "type": exc.__class__.__name__}
            result = {"integration": name, "status": "error", "error": error_payload}

        latency_ms = int((time.perf_counter() - start) * 1000)
        await log_event(
            trace_id,
            session_id,
            "Integration",
            {"integration": name, "payload": payload, "result": result},
            latency_ms=latency_ms,
            error=error_payload,
        )
        return result


async def gather_agent_calls(decision: dict, trace_id: str, session_id: str) -> dict:
    """Запускаем агентов параллельно; ошибки не пробрасываем наружу."""
    tasks = {
        agent: asyncio.create_task(
            call_agent(agent, decision["payloads"].get(agent, {}), trace_id, session_id)
        )
        for agent in decision["agents"]
    }
    results: Dict[str, Any] = {}
    for agent, task in tasks.items():
        try:
            results[agent] = await task
        except Exception as exc:  # noqa: BLE001
            # Логируем как деградацию; сам call_agent уже написал событие.
            results[agent] = {"agent": agent, "status": "error", "error": str(exc)}
    return results


def _validate_agent_result(agent: str, result: dict) -> tuple[dict, Optional[str]]:
    """
    Мини-контракт: ожидаем статус, data и опциональный error.
    Возвращаем (data, error_text) для удобства.
    """
    status = result.get("status")
    data = result.get("data") if isinstance(result, dict) else None
    error = None
    if status != "ok":
        error = str(result.get("error") or "unknown_error")
    if data is None:
        data = {}
    return data, error


async def _register_fallback_lead(
    message: Message, decision: dict, agent_results: dict, context: dict, trace_id: str
) -> None:
    """
    При деградации собираем минимальные контакты и создаем лид в CRM.
    Заглушка: использует call_integration(bitrix24) с user_id/текстом; доп. поля можно расширить.
    """
    contacts = {"phone": None, "email": None}
    for ent in context.get("entities", []):
        if ent.get("type") == "phone":
            contacts["phone"] = ent.get("value")
        if ent.get("type") == "email":
            contacts["email"] = ent.get("value")
    await call_integration(
        "bitrix24",
        {
            "user_id": message.user_id,
            "comment": f"[fallback] {message.text}",
            "intent": decision.get("intent"),
            "idempotency_key": decision.get("next_state"),
            **contacts,
        },
        trace_id,
        decision.get("next_state") or message.user_id,
    )


async def assemble_reply(
    message: Message, decision: dict, agent_results: dict, context: dict
) -> tuple[str, bool]:
    """Собираем финальный ответ; bool показывает наличие деградации."""
    fallback_needed = any(res.get("status") == "error" for res in agent_results.values())
    validated: Dict[str, dict] = {}
    errors: List[str] = []
    for agent, res in agent_results.items():
        data, err = _validate_agent_result(agent, res)
        validated[agent] = data
        if err:
            errors.append(f"{agent}:{err}")
    if fallback_needed:
        REGISTRY.counter("fallback_total").inc()
        await _register_fallback_lead(message, decision, agent_results, context, context.get("trace_id", ""))
        msg = "Мы обработали запрос, но часть сервисов недоступна. Скоро вернемся с деталями."
        if errors:
            msg += f" ({'; '.join(errors)})"
        return (msg, True)

    # Конкатенируем основные ответы агентов в один текстовый реплай.
    snippets = []
    for agent, data in validated.items():
        summary = data.get("summary") or data.get("echo") or "готово"
        snippets.append(f"{agent}: {summary}")
    if not snippets:
        snippets.append("sales: обновим вас по запросу")

    reply = f"Интент: {decision['intent']}. " + " | ".join(snippets)
    if context.get("entities"):
        reply += f" (контекст: {context['entities']})"
    return reply, False
