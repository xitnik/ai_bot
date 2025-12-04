from __future__ import annotations

import uuid
from typing import Annotated, Any, Dict

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import db
from config import get_settings
from events_logger import log_event
from metrics import REGISTRY
from models import Message, SessionDTO
from observability import get_service_name, init_logging
from orchestrator import (
    assemble_reply,
    call_integration,
    enrich_message,
    gather_agent_calls,
    route_message,
)
from otel import get_tracer
from session_store import MySQLSessionStore
from rag.api import router as rag_router

app = FastAPI(title="Conversation Gateway")
session_store = MySQLSessionStore()
DbSession = Annotated[AsyncSession, Depends(db.get_db_session)]


@app.on_event("startup")
async def startup() -> None:
    init_logging(get_service_name())
    await db.init_db()


@app.on_event("shutdown")
async def shutdown() -> None:
    await db.dispose_engine()


app.include_router(rag_router)


async def normalize_message(payload: Dict[str, Any]) -> Message:
    """Унифицируем вход вебчата в DTO Message."""
    try:
        channel_payload = payload.get("channel_payload") or {}
        return Message(
            user_id=str(payload.get("user_id") or channel_payload.get("from_id") or ""),
            channel=payload.get("channel", channel_payload.get("channel", "webchat")),
            text=str(payload.get("text") or channel_payload.get("text") or ""),
            attachments=payload.get("attachments") or channel_payload.get("attachments"),
        )
    except KeyError as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Missing field: {exc.args[0]}") from exc


async def ensure_session(db_session: AsyncSession, message: Message) -> SessionDTO:
    """Достаем активную сессию или создаем новую."""
    existing = await session_store.get(message.user_id, message.channel)
    if existing:
        await db.touch_session_state(db_session, existing.session_id, existing.state)
        return existing

    session_model = db.SessionModel(
        session_id=str(uuid.uuid4()),
        user_id=message.user_id,
        channel=message.channel,
        state="idle",
    )
    db_session.add(session_model)
    await db_session.commit()
    await db_session.refresh(session_model)
    session_dto = SessionDTO.model_validate(session_model)
    await session_store.save(session_dto)
    return session_dto


async def refresh_session_cache(
    db_session: AsyncSession, session_id: str, next_state: str
) -> None:
    """Обновляем состояние в БД и кешируем актуальный слепок."""
    await db.touch_session_state(db_session, session_id, next_state)
    result = await db_session.execute(
        select(db.SessionModel).where(db.SessionModel.session_id == session_id)
    )
    refreshed = result.scalars().first()
    if refreshed:
        session_obj = SessionDTO.model_validate(refreshed)
        await session_store.save(session_obj)


@app.post("/channels/webchat/message")
async def handle_webchat_message(
    payload: Dict[str, Any], db_session: DbSession
) -> JSONResponse:
    message = await normalize_message(payload)
    session_dto = await ensure_session(db_session, message)
    trace_id = str(uuid.uuid4())
    tracer = get_tracer()

    REGISTRY.counter("requests_total").inc()

    # Открываем корневой спан для цепочки событий.
    try:
        with tracer.start_as_current_span(
            "conversation",
            attributes={
                "trace.id": trace_id,
                "session.id": session_dto.session_id,
                "channel": message.channel,
            },
        ):
            await log_event(
                trace_id,
                session_dto.session_id,
                "Edge.In",
                {"raw": payload, "channel": message.channel, "user_id": message.user_id},
            )

            context = await enrich_message(message)
            await log_event(
                trace_id,
                session_dto.session_id,
                "Enrichment",
                {"context": context, "user_id": message.user_id},
            )

            decision = await route_message(session_dto, message, context)
            await log_event(
                trace_id,
                session_dto.session_id,
                "Decision",
                {
                    "intent": decision["intent"],
                    "prev_state": session_dto.state,
                    "next_state": decision["next_state"],
                    "agents": decision["agents"],
                    "user_id": message.user_id,
                },
            )

            agent_results = await gather_agent_calls(decision, trace_id, session_dto.session_id)
            if decision["intent"] in ("sales", "procurement"):
                await call_integration(
                    "bitrix24",
                    {
                        "user_id": message.user_id,
                        "comment": message.text,
                        "intent": decision["intent"],
                        "idempotency_key": session_dto.session_id,
                    },
                    trace_id,
                    session_dto.session_id,
                )
            reply, degraded = await assemble_reply(
                message, decision, agent_results, context, trace_id
            )

            if degraded:
                await log_event(
                    trace_id,
                    session_dto.session_id,
                    "Fallback",
                    {"agent_results": agent_results, "user_id": message.user_id},
                )

            await refresh_session_cache(db_session, session_dto.session_id, decision["next_state"])
            await log_event(
                trace_id,
                session_dto.session_id,
                "Edge.Out",
                {
                    "reply": reply,
                    "agents": list(agent_results.keys()),
                    "user_id": message.user_id,
                },
            )
    except Exception as exc:  # noqa: BLE001
        # Не роняем процесс, но логируем деградацию.
        await log_event(
            trace_id,
            session_dto.session_id,
            "Fallback",
            {"error": str(exc), "stage": "pipeline", "user_id": message.user_id},
        )
        await log_event(
            trace_id,
            session_dto.session_id,
            "Edge.Out",
            {"reply": "Временная недоступность сервиса", "user_id": message.user_id},
        )
        reply = "Временная недоступность сервиса"
    else:
        pass

    return JSONResponse({"reply": reply, "session_id": session_dto.session_id})


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    """Экспорт метрик для Prometheus-сбора."""
    if not get_settings().observability.metrics_enabled:
        raise HTTPException(status_code=503, detail="metrics_disabled")
    return PlainTextResponse(REGISTRY.render_prometheus(), media_type="text/plain")
