from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import insert

import db


async def log_event(
    trace_id: str,
    session_id: str,
    event_type: str,
    payload: dict,
    latency_ms: Optional[int] = None,
    error: Optional[dict[str, Any]] = None,
) -> None:
    """Асинхронно пишет событие в таблицу events."""
    user_id = payload.get("user_id") if isinstance(payload, dict) else None
    async with db.AsyncSessionLocal() as session:
        stmt = insert(db.EventModel).values(
            trace_id=trace_id,
            session_id=session_id,
            user_id=user_id,
            event_type=event_type,
            payload=payload,
            latency_ms=latency_ms,
            error=error,
        )
        await session.execute(stmt)
        await session.commit()
