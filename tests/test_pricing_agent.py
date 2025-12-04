from __future__ import annotations

from fastapi.testclient import TestClient

from agents.pricing_agent import app


def test_pricing_agent_calculates_multistep_quote() -> None:
    client = TestClient(app)
    payload = {
        "order_spec": {
            "items": [
                {
                    "species": "oak",
                    "dimensions": {"length_mm": 3000, "width_mm": 150, "thickness_mm": 50},
                    "quantity": 10,
                    "price_per_board_foot": 5.0,
                    "processing": ["sanding"],
                    "rush_level": "fast",
                }
            ],
            "bulk_discounts": [{"threshold_bf": 90, "percent": 5}],
        }
    }
    resp = client.post("/agents/pricing/run", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["currency"] == "USD"
    assert data["total"] > 900  # объем * коэффициенты
    assert data["items"][0]["volume_board_feet"] > 9
    assert data["items"][0]["species_multiplier"] > 1
    assert data["items"][0]["processing_multiplier"] > 1
    assert "bulk discount" in " ".join(data["discounts"]) or data["items"][0]["discount_applied"] >= 0
