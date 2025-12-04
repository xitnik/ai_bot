from __future__ import annotations

import asyncio

from integrations.base import LeadPayload
from integrations.fake import (
    InMemoryAvito,
    InMemoryBitrix,
    InMemoryMAX,
    InMemoryOneC,
    InMemoryTelegram,
)


def test_fake_crm_idempotency():
    crm = InMemoryBitrix()
    payload = LeadPayload(user_id="u1", name="Test", idempotency_key="lead-1")
    first = asyncio.run(crm.create_lead(payload))
    dup = asyncio.run(crm.create_lead(payload))
    assert first["status"] == "created"
    assert dup["status"] == "duplicate"


def test_fake_channels_store_messages():
    tg = InMemoryTelegram()
    max_client = InMemoryMAX()
    asyncio.run(tg.send_message("chat", "hi"))
    asyncio.run(max_client.send_message("chat", "hi"))
    assert tg.messages == [("chat", "hi")]
    assert max_client.messages[0]["text"] == "hi"


def test_fake_avito_publish_and_onec_stock():
    avito = InMemoryAvito()
    onec = InMemoryOneC()
    res = asyncio.run(avito.publish_message({"text": "hello"}))
    assert res["status"] == "ok"
    stock = asyncio.run(onec.get_stock("sku-1"))
    assert stock["sku"] == "sku-1"
