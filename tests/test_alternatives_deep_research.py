from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from agents.alternatives_deep_research_agent import (
    AlternativesDeepResearchAgent,
    WebSearchClient,
    create_app,
)
from alternatives_deep_models import AlternativesDeepResearchRequest
from alternatives_models import AlternativeItem, AlternativesResult
from alternatives_agent import AlternativesAgent
from alternatives_deep_models import DeepResearchSource


class FakeAlternativesAgent(AlternativesAgent):
    """Фейковый агент альтернатив с детерминированным ответом."""

    def __init__(self) -> None:
        pass

    async def run(self, request: Any) -> AlternativesResult:  # type: ignore[override]
        return AlternativesResult(
            alternatives=[
                AlternativeItem(
                    product_id="oak_1",
                    similarity=0.9,
                    type="direct",
                    reason="test reason",
                )
            ]
        )


class FakeSearchClient(WebSearchClient):
    """Фейковый поиск возвращает статические источники."""

    async def search_forums(self, query: str, k: int = 5, trace_id: str | None = None, session_id: str | None = None) -> List[DeepResearchSource]:  # type: ignore[override]
        return [
            DeepResearchSource(
                url="https://forum.example.com/thread1",
                source_type="forum",
                title="Forum thread",
                snippet=f"Discussion about {query}",
                sentiment="mixed",
            )
        ]

    async def search_reviews(self, query: str, k: int = 5, trace_id: str | None = None, session_id: str | None = None) -> List[DeepResearchSource]:  # type: ignore[override]
        return [
            DeepResearchSource(
                url="https://reviews.example.com/item1",
                source_type="review",
                title="Review",
                snippet=f"Review about {query}",
                sentiment="positive",
            )
        ]

    async def search_articles(self, query: str, k: int = 5, trace_id: str | None = None, session_id: str | None = None) -> List[DeepResearchSource]:  # type: ignore[override]
        return []


class DummyCompletion:
    """Минимальный аналог ChatCompletion для тестов."""

    def __init__(self, content: str) -> None:
        self.choices = [type("Choice", (), {"message": type("Msg", (), {"content": content})()})()]
        self.model = "test-model"


def _fake_chat_response(payload: Dict[str, Any]) -> DummyCompletion:
    return DummyCompletion(json.dumps(payload, ensure_ascii=False))


def _sample_llm_payload() -> Dict[str, Any]:
    return {
        "task_understanding": {
            "base_item_name": "oak plank",
            "domain": "material",
            "key_requirements": ["in_stock"],
            "hard_constraints": ["oak", "A grade"],
            "nice_to_have": ["eco"],
            "notes": "sample",
        },
        "research_plan": {"steps": ["forums", "reviews"]},
        "alternatives": [
            {
                "name": "Candidate A",
                "type": "direct",
                "short_description": "Desc",
                "when_to_choose": "When need budget",
                "pros": ["cheap"],
                "cons": ["lower quality"],
                "price_info_currency": "USD",
                "price_info_range": "10-12",
                "price_relative_to_base": "cheaper",
                "fit_score_0_to_100": 80,
                "risk_notes": ["check stock"],
                "sources": [
                    {
                        "url": "https://reviews.example.com/item1",
                        "source_type": "review",
                        "title": "Review",
                        "snippet": "good choice",
                        "sentiment": "positive",
                    }
                ],
            }
        ],
        "comparison_summary": {
            "key_dimensions": ["price", "quality"],
            "best_for_strict_constraints": "Candidate A",
            "best_overall": "Candidate A",
            "tradeoffs": ["price vs quality"],
        },
        "final_recommendations": {"overall": "Use Candidate A"},
    }


@pytest.mark.asyncio
async def test_deep_research_agent_returns_valid_result() -> None:
    agent = AlternativesDeepResearchAgent(
        alternatives_agent=FakeAlternativesAgent(),
        search_client=FakeSearchClient(),
        chat_callable=lambda model, messages, **kwargs: _fake_chat_response(_sample_llm_payload()),
        search_k=2,
        max_sources=3,
    )

    request = AlternativesDeepResearchRequest(
        query_text="oak plank",
        base_item_name="oak plank",
        constraints=["in_stock"],
        nice_to_have=["eco"],
    )

    result = await agent.run(request)

    assert result.alternatives[0].name == "Candidate A"
    assert result.comparison_summary.best_overall == "Candidate A"
    assert result.task_understanding.base_item_name == "oak plank"


@pytest.mark.asyncio
async def test_deep_research_agent_fallback_on_invalid_json() -> None:
    agent = AlternativesDeepResearchAgent(
        alternatives_agent=FakeAlternativesAgent(),
        search_client=FakeSearchClient(),
        chat_callable=lambda model, messages, **kwargs: DummyCompletion("non-json"),
    )

    request = AlternativesDeepResearchRequest(
        query_text="oak plank",
        constraints=["in_stock"],
    )

    result = await agent.run(request)

    assert result.alternatives == []
    assert result.research_plan["reason"] == "parse_failed"


def test_deep_research_fastapi_endpoint() -> None:
    payload = _sample_llm_payload()

    def _fake_chat(model: str, messages: List[Dict[str, Any]], **kwargs: Any) -> DummyCompletion:
        return _fake_chat_response(payload)

    app = create_app(
        alternatives_agent=FakeAlternativesAgent(),
        search_client=FakeSearchClient(),
        chat_callable=_fake_chat,
    )
    client = TestClient(app)

    response = client.post(
        "/agents/alternatives/deep_research",
        json={
            "query_text": "oak plank",
            "constraints": ["in_stock"],
            "nice_to_have": ["eco"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["data"]["summary"]
    assert body["data"]["research"]["alternatives"][0]["name"] == "Candidate A"
