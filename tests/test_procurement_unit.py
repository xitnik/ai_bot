from datetime import datetime, timezone

from procurement.db import Offer
from procurement.schemas import RFQSpec, Vendor
from procurement.selector import score_offers, select_best_offer
from procurement.service import compose_rfq


def _make_offer(idx: int, price: float, lead: int, vendor_score: float) -> Offer:
    offer = Offer(
        rfq_id=1,
        price_per_unit=price,
        min_batch=1.0,
        lead_time_days=lead,
        terms_text="terms",
        vendor_score=vendor_score,
        raw_text="reply",
    )
    offer.id = idx  # type: ignore[assignment]
    return offer


def test_select_best_offer_prefers_cheaper_and_faster():
    weights = {"price_per_unit": 0.5, "lead_time_days": 0.3, "vendor_score": 0.2}
    good = _make_offer(1, price=80.0, lead=7, vendor_score=0.9)
    bad = _make_offer(2, price=120.0, lead=15, vendor_score=0.5)

    best = select_best_offer([bad, good], weights)
    assert best is good

    scored = {offer.id: score for offer, score in score_offers([bad, good], weights)}
    assert scored[good.id] > scored[bad.id]


def test_compose_rfq_includes_details():
    spec = RFQSpec(
        species="Pine",
        grade="A",
        volume=10.5,
        delivery_terms="FOB",
        deadline=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    vendor = Vendor(id=1, name="Forest Inc", channel="email", address="vendor@example.com")

    payload = compose_rfq(spec, vendor)

    assert payload["recipient"] == vendor.address
    assert vendor.name in payload["greeting"]
    assert payload["details"]["volume"] == spec.volume
    assert payload["details"]["deadline"] == spec.deadline.isoformat()
