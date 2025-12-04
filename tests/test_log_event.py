from __future__ import annotations

from datetime import datetime, timezone
import os

import pytest
from sqlalchemy import select

import db
from events_logger import log_event
from models import Message, SessionDTO
from orchestrator import route_message


@pytest.mark.asyncio
async def test_log_event_persists_payload(tmp_path) -> None:
    db_url = os.getenv("MYSQL_TEST_URL")
    if not db_url:
        pytest.skip("MYSQL_TEST_URL is not configured")
    db.configure_engine(db_url)
    await db.init_db()

    payload = {"user_id": "user-1", "text": "x" * 2048}
    await log_event("trace-1", "session-1", "Edge.In", payload)

    async with db.AsyncSessionLocal() as session:
        result = await session.execute(select(db.EventModel))
        events = result.scalars().all()

    assert len(events) == 1
    assert events[0].trace_id == "trace-1"
    assert events[0].payload["text"].startswith("x")


@pytest.mark.asyncio
async def test_route_message_state_progression() -> None:
    session = SessionDTO(
        session_id="s1",
        user_id="u1",
        channel="webchat",
        state="idle",
        started_at=datetime.now(timezone.utc),
        last_event_at=datetime.now(timezone.utc),
    )

    pricing_msg = Message(user_id="u1", channel="webchat", text="need price", attachments=None)
    decision_pricing = await route_message(session, pricing_msg, {"normalized_text": "need price"})
    assert decision_pricing["intent"] == "pricing"
    assert decision_pricing["next_state"] == "pricing_quote"

    alt_msg = Message(
        user_id="u1", channel="webchat", text="show alternative", attachments=None
    )
    decision_alt = await route_message(session, alt_msg, {"normalized_text": "show alternative"})
    assert decision_alt["intent"] == "alternatives"
    assert decision_alt["next_state"] == "alternatives_suggest"
