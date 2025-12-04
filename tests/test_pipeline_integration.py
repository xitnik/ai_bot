from __future__ import annotations

import importlib
import os
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select

import db
import otel


def _is_subsequence(pattern, sequence) -> bool:
    it = iter(sequence)
    return all(any(item == elem for elem in it) for item in pattern)


@pytest.mark.asyncio
async def test_webchat_pipeline(tmp_path: Path) -> None:
    db_url = os.getenv("MYSQL_TEST_URL")
    if not db_url:
        pytest.skip("MYSQL_TEST_URL is not configured")

    db.configure_engine(db_url)
    import main

    importlib.reload(otel)
    importlib.reload(main)
    otel.reset_traces()

    await db.init_db()

    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/channels/webchat/message",
            json={"user_id": "user-xyz", "text": "need price details"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["reply"]
    session_id = body["session_id"]

    async with db.AsyncSessionLocal() as session:
        res_session = await session.execute(
            select(db.SessionModel).where(db.SessionModel.session_id == session_id)
        )
        session_row = res_session.scalars().first()

        res_events = await session.execute(
            select(db.EventModel.event_type).order_by(db.EventModel.ts, db.EventModel.event_id)
        )
        event_types = [row[0] for row in res_events.all()]

    assert session_row is not None
    assert _is_subsequence(
        ["Edge.In", "Enrichment", "Decision", "AgentRun", "Edge.Out"], event_types
    )

    spans = otel.get_exported_spans()
    assert any(span.name == "conversation" for span in spans)
