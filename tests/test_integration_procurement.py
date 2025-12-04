from datetime import datetime, timezone

import pytest
from sqlalchemy import select

import procurement.api as api
from procurement.db import RFQSpecRecord
from procurement.schemas import OfferCore, RFQSpec, Vendor


@pytest.mark.asyncio
async def test_happy_path_best_offer_selection(client, monkeypatch):
    async_client, session_maker = client

    offer_map = {
        "premium": OfferCore(
            price_per_unit=120.0, min_batch=10, lead_time_days=12, terms_text="Premium terms"
        ),
        "cheap": OfferCore(
            price_per_unit=80.0, min_batch=5, lead_time_days=10, terms_text="Cheap offer"
        ),
        "fast": OfferCore(price_per_unit=90.0, min_batch=3, lead_time_days=5, terms_text="Fast"),
    }

    async def fake_parse(raw_text: str):
        return offer_map[raw_text]

    monkeypatch.setattr(api, "parse_vendor_reply", fake_parse)

    spec = RFQSpec(
        species="Spruce",
        grade="B",
        volume=25.0,
        delivery_terms="CIF",
        deadline=datetime(2025, 2, 2, tzinfo=timezone.utc),
    )
    vendors = [
        Vendor(id=1, name="Vendor A", channel="email", address="a@example.com"),
        Vendor(id=2, name="Vendor B", channel="api", address="https://b.example.com"),
        Vendor(id=3, name="Vendor C", channel="email", address="c@example.com"),
    ]

    create_resp = await async_client.post(
        "/agents/procurement/rfq",
        json={"spec": spec.model_dump(mode="json"), "vendors": [v.model_dump() for v in vendors]},
    )
    assert create_resp.status_code == 200
    rfq_ids = create_resp.json()
    assert len(rfq_ids) == 3

    # Парсим ответы трех вендоров.
    for rfq_id, key in zip(rfq_ids, ["premium", "cheap", "fast"], strict=False):
        resp = await async_client.post(
            "/agents/procurement/parse_reply", json={"rfq_id": rfq_id, "raw_text": key}
        )
        assert resp.status_code == 200

    async with session_maker() as session:
        res = await session.execute(select(RFQSpecRecord.id))
        spec_id = res.scalars().first()

    assert spec_id is not None
    best_resp = await async_client.get(f"/agents/procurement/best_offer/{spec_id}")
    assert best_resp.status_code == 200
    data = best_resp.json()

    assert data["best_offer"]["raw_text"] == "fast"
    assert len(data["comparison"]) == 3
