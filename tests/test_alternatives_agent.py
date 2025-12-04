from __future__ import annotations

import asyncio

import pytest

# Для прогонов локально убедитесь, что установлены fastapi и её зависимости.
from fastapi.testclient import TestClient

from alternatives_agent import AlternativesAgent, classify_alternative, create_app
from alternatives_models import AlternativesRequest
from vector_index import InMemoryVectorStore


class FakeEmbeddingsClient:
    """Детерминированный эмбеддер для тестов."""

    def __init__(self, vector: list[float]) -> None:
        self.vector = vector

    async def embed_text(self, text: str) -> list[float]:
        return list(self.vector)


def _base_metadata() -> dict:
    return {
        "species": "oak",
        "grade": "A",
        "dimensions": {"length": 100.0, "width": 20.0, "thickness": 5.0},
        "price": 10.0,
        "in_stock": True,
    }


def test_classify_direct_match() -> None:
    base = _base_metadata()
    candidate = {
        "species": "oak",
        "grade": "A",
        "dimensions": {"length": 101.0, "width": 20.0, "thickness": 5.0},
        "price": 10.2,
    }

    alt_type, reason = classify_alternative(candidate, base, price_band=None)

    assert alt_type == "direct"
    assert "price within 10%" in reason


def test_classify_price_bands() -> None:
    base = _base_metadata()
    cheaper = {**base, "price": 8.5}
    premium = {**base, "price": 12.0}

    low_type, low_reason = classify_alternative(cheaper, base, price_band="cheaper")
    high_type, high_reason = classify_alternative(premium, base, price_band="premium")

    assert low_type == "price_low"
    assert "cheaper" in low_reason
    assert high_type == "price_high"
    assert "higher price" in high_reason


@pytest.mark.asyncio
async def test_agent_pipeline_returns_ranked_hits() -> None:
    store = InMemoryVectorStore()
    base = _base_metadata()

    await store.upsert_product(
        "direct",
        [1.0, 0.0, 0.0],
        {
            "species": "oak",
            "grade": "A",
            "dimensions": {"length": 100.0, "width": 20.0, "thickness": 5.0},
            "price": 10.2,
            "in_stock": True,
        },
    )
    await store.upsert_product(
        "cheaper",
        [0.98, 0.02, 0.0],
        {
            "species": "oak",
            "grade": "A",
            "dimensions": {"length": 110.0, "width": 20.0, "thickness": 5.0},
            "price": 8.5,
            "in_stock": True,
        },
    )
    await store.upsert_product(
        "filtered_out",
        [1.0, 0.0, 0.0],
        {
            "species": "pine",
            "grade": "B",
            "dimensions": {"length": 100.0, "width": 20.0, "thickness": 5.0},
            "price": 9.0,
            "in_stock": True,
        },
    )

    agent = AlternativesAgent(store, FakeEmbeddingsClient([1.0, 0.0, 0.0]))
    request = AlternativesRequest(
        query_text="oak plank",
        hard_filters=base,
        k=5,
        price_band="cheaper",
    )

    result = await agent.run(request)

    assert [item.product_id for item in result.alternatives] == ["direct", "cheaper"]
    assert result.alternatives[0].type == "direct"
    assert result.alternatives[1].type == "price_low"


@pytest.mark.asyncio
async def test_agent_returns_empty_when_no_hits() -> None:
    store = InMemoryVectorStore()
    agent = AlternativesAgent(store, FakeEmbeddingsClient([0.0, 1.0, 0.0]))
    request = AlternativesRequest(
        query_text="missing",
        hard_filters=_base_metadata(),
        k=3,
        price_band=None,
    )

    result = await agent.run(request)

    assert result.alternatives == []


def test_fastapi_endpoint() -> None:
    store = InMemoryVectorStore()
    base = _base_metadata()
    asyncio.get_event_loop().run_until_complete(
        store.upsert_product(
            "candidate",
            [1.0, 0.0, 0.0],
            {
                "species": "oak",
                "grade": "A",
                "dimensions": {"length": 101.0, "width": 20.0, "thickness": 5.0},
                "price": 10.1,
                "in_stock": True,
            },
        )
    )
    app = create_app(store=store, embedder=FakeEmbeddingsClient([1.0, 0.0, 0.0]))
    client = TestClient(app)

    response = client.post(
        "/agents/alternatives/run",
        json={
            "query_text": "oak plank",
            "hard_filters": base,
            "k": 3,
            "price_band": None,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["alternatives"][0]["product_id"] == "candidate"
    assert payload["alternatives"][0]["type"] == "direct"
