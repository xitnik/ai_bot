from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

import db
from alternatives_agent import AlternativesAgent, _ingest_stub_catalog
from alternatives_deep_models import (
    AlternativesDeepResearchPayload,
    AlternativesDeepResearchRequest,
    AlternativesDeepResearchResponse,
    AlternativesDeepResearchResult,
    DeepResearchSource,
)
from alternatives_models import AlternativesRequest, AlternativesResult
from config import get_settings
from embeddings_client import EmbeddingsClient
from events_logger import log_event
from llm_client import chat
from vector_index import MySQLVectorStore, VectorStore


async def _safe_log_event(
    trace_id: str,
    session_id: Optional[str],
    event_type: str,
    payload: Dict[str, Any],
    error: Optional[Dict[str, Any]] = None,
) -> None:
    """Логируем события без падения пайплайна при ошибках БД."""
    try:
        await log_event(
            trace_id=trace_id,
            session_id=session_id or "deep-research",
            event_type=event_type,
            payload=payload,
            error=error,
        )
    except Exception:
        return


class WebSearchClient:
    """HTTP-клиент для внешнего поиска форумов/обзоров."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: float = 8.0,
    ) -> None:
        settings = get_settings()
        deep_settings = getattr(settings, "deep_research", None)
        self._base_url = base_url or (deep_settings.search_base_url if deep_settings else "")
        self._api_key = api_key or (deep_settings.search_api_key if deep_settings else "")
        self._timeout = timeout_seconds

    async def search_forums(
        self,
        query: str,
        k: int = 5,
        trace_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[DeepResearchSource]:
        return await self._perform("/search/forums", query, k, "forum", trace_id, session_id)

    async def search_reviews(
        self,
        query: str,
        k: int = 5,
        trace_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[DeepResearchSource]:
        return await self._perform("/search/reviews", query, k, "review", trace_id, session_id)

    async def search_articles(
        self,
        query: str,
        k: int = 5,
        trace_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[DeepResearchSource]:
        return await self._perform("/search/articles", query, k, "article", trace_id, session_id)

    async def _perform(
        self,
        endpoint: str,
        query: str,
        k: int,
        source_type: str,
        trace_id: Optional[str],
        session_id: Optional[str],
    ) -> List[DeepResearchSource]:
        if not self._base_url:
            return []
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        params = {"q": query, "k": k}
        try:
            async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
                response = await client.get(endpoint, params=params, headers=headers)
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:  # noqa: BLE001
            await _safe_log_event(
                trace_id or "deep-research",
                session_id,
                "deep_research.search_error",
                {"endpoint": endpoint, "query": query},
                error={"message": str(exc)},
            )
            return []

        items = payload.get("results") if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            items = []

        results: List[DeepResearchSource] = []
        allowed_types = {"vendor", "review", "forum", "article", "marketplace", "other"}
        for item in items:
            if not isinstance(item, dict):
                continue
            url = item.get("url") or item.get("link") or ""
            snippet = item.get("snippet") or item.get("description") or ""
            if not url or not snippet:
                continue
            normalized_type = source_type if source_type in allowed_types else "other"
            results.append(
                DeepResearchSource(
                    url=url,
                    title=item.get("title") or item.get("name"),
                    snippet=str(snippet)[:500],
                    source_type=normalized_type,  # type: ignore[arg-type]
                    sentiment="neutral",
                )
            )
        return results[:k]


class AlternativesDeepResearchAgent:
    """Агент глубокого SGR-ресерча поверх AlternativesAgent."""

    def __init__(
        self,
        alternatives_agent: AlternativesAgent,
        search_client: WebSearchClient,
        chat_callable=chat,
        search_k: int = 5,
        max_sources: int = 12,
    ) -> None:
        self._alternatives_agent = alternatives_agent
        self._search_client = search_client
        self._chat = chat_callable
        self._search_k = search_k
        self._max_sources = max_sources

    async def run(self, request: AlternativesDeepResearchRequest) -> AlternativesDeepResearchResult:
        trace_id = request.trace_id or str(uuid.uuid4())
        base_item_name = request.base_item_name or request.query_text

        alt_request = AlternativesRequest(
            query_text=request.query_text,
            hard_filters=request.hard_filters,
            k=request.k,
            price_band=request.price_band,
        )
        try:
            base_alternatives = await self._alternatives_agent.run(alt_request)
        except Exception as exc:  # noqa: BLE001
            await _safe_log_event(
                trace_id,
                request.session_id,
                "deep_research.alternatives_error",
                {"query": request.query_text},
                error={"message": str(exc)},
            )
            base_alternatives = AlternativesResult(alternatives=[])

        search_results = await self._collect_search_results(request, base_item_name, trace_id)
        llm_result = await self._run_sgr_llm(request, base_alternatives, search_results, trace_id)
        return llm_result

    async def _collect_search_results(
        self, request: AlternativesDeepResearchRequest, base_item_name: str, trace_id: str
    ) -> List[DeepResearchSource]:
        queries = self._build_queries(request, base_item_name)
        all_results: List[DeepResearchSource] = []
        for query in queries:
            forums = await self._search_client.search_forums(
                query, k=self._search_k, trace_id=trace_id, session_id=request.session_id
            )
            reviews = await self._search_client.search_reviews(
                query, k=self._search_k, trace_id=trace_id, session_id=request.session_id
            )
            articles = await self._search_client.search_articles(
                query,
                k=max(2, self._search_k // 2),
                trace_id=trace_id,
                session_id=request.session_id,
            )
            all_results.extend(forums + reviews + articles)

        unique: Dict[str, DeepResearchSource] = {}
        for item in all_results:
            unique.setdefault(item.url, item)
        return list(unique.values())[: self._max_sources]

    def _build_queries(
        self, request: AlternativesDeepResearchRequest, base_item_name: str
    ) -> List[str]:
        queries = [
            f"{base_item_name} отзывы {request.region or ''}".strip(),
            f"{base_item_name} forum проблемы опыт использования {request.use_case or ''}".strip(),
            f"{base_item_name} сравнение аналоги vs {base_item_name}".strip(),
        ]
        if request.delivery_deadline:
            queries.append(f"{base_item_name} поставка сроки {request.delivery_deadline}")
        if request.certifications:
            certs = " ".join(request.certifications)
            queries.append(f"{base_item_name} сертификация {certs}")
        if request.budget_limit:
            queries.append(f"{base_item_name} budget under {request.budget_limit}")
        return [q for q in queries if q]

    async def _run_sgr_llm(
        self,
        request: AlternativesDeepResearchRequest,
        base_alternatives: AlternativesResult,
        sources: List[DeepResearchSource],
        trace_id: str,
    ) -> AlternativesDeepResearchResult:
        settings = get_settings()
        deep = getattr(settings, "deep_research", None)
        model = getattr(deep, "model", "gpt5")
        temperature = getattr(deep, "temperature", 0.2)
        max_tokens = getattr(deep, "max_output_tokens", 1200)

        system_prompt = (
            "Ты deep-research агент. Следуй Schema-Guided Reasoning:\n"
            "1) Clarify: Понять задачу, базовый товар, сценарий, жесткие ограничения.\n"
            "2) Plan: План исследования и метрики сравнения.\n"
            "3) Search: Форумы/отзывы/обзоры, выделяй короткие цитаты/snippet'ы.\n"
            "4) Synthesize & Compare: сопоставь альтернативы по цене/качеству/рискам/срокам.\n"
            "5) Recommend: сценарные рекомендации (бюджет, надежность, сроки поставки).\n"
            "Формат вывода: только один JSON-объект строго по схеме AlternativesDeepResearchResult "
            "c ключами task_understanding, research_plan, alternatives, comparison_summary, "
            "final_recommendations. В альтернативе указывай fit_score_0_to_100, "
            "price_relative_to_base из ['unknown','cheaper','similar','more_expensive'] и "
            "обязательно список sources {url,title,snippet,source_type,sentiment}. "
            "Никакого текста до или после JSON."
        )

        catalog_alternatives = [
            {
                "product_id": alt.product_id,
                "type": alt.type,
                "reason": alt.reason,
                "similarity": alt.similarity,
            }
            for alt in base_alternatives.alternatives
        ]
        user_payload = {
            "query_text": request.query_text,
            "base_item_name": request.base_item_name or request.query_text,
            "domain": request.domain,
            "use_case": request.use_case,
            "hard_filters": request.hard_filters,
            "constraints": request.constraints,
            "nice_to_have": request.nice_to_have,
            "region": request.region,
            "delivery_deadline": request.delivery_deadline,
            "certifications": request.certifications,
            "budget_limit": request.budget_limit,
            "catalog_alternatives": catalog_alternatives,
            "search_evidence": [src.model_dump() for src in sources],
        }
        user_prompt = (
            "Сформируй SGR-отчет по альтернативам. Используй входные данные ниже. "
            "Обязательно соблюдай схему и не добавляй текст вне JSON.\n"
            f"Входные данные: {json.dumps(user_payload, ensure_ascii=False)}"
        )

        try:
            completion = await asyncio.to_thread(
                self._chat,
                model,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as exc:  # noqa: BLE001
            await _safe_log_event(
                trace_id,
                request.session_id,
                "deep_research.llm_error",
                {"query": request.query_text},
                error={"message": str(exc)},
            )
            return self._fallback_result(request, reason="llm_call_failed")

        content = ""
        try:
            choice = completion.choices[0]
            content = getattr(choice.message, "content", "") or getattr(choice, "text", "") or ""
        except Exception:  # noqa: BLE001
            content = ""

        parsed = self._parse_llm_json(content)
        if parsed:
            return parsed

        await _safe_log_event(
            trace_id,
            request.session_id,
            "deep_research.parse_error",
            {"raw": content[:500]},
        )
        return self._fallback_result(request, reason="parse_failed")

    def _parse_llm_json(self, content: str) -> Optional[AlternativesDeepResearchResult]:
        if not content:
            return None
        candidate_blocks = [content]
        if "{" in content and "}" in content:
            candidate_blocks.append(content[content.find("{") : content.rfind("}") + 1])
        for block in candidate_blocks:
            try:
                payload = json.loads(block)
            except json.JSONDecodeError:
                continue
            try:
                return AlternativesDeepResearchResult.model_validate(payload)
            except ValidationError:
                continue
        return None

    def _fallback_result(
        self, request: AlternativesDeepResearchRequest, reason: str
    ) -> AlternativesDeepResearchResult:
        from alternatives_deep_models import (  # локальный импорт для избежания циклов
            DeepResearchComparisonSummary,
            DeepResearchTaskUnderstanding,
        )

        base_item_name = request.base_item_name or request.query_text
        constraints = request.constraints or [
            f"filters: {json.dumps(request.hard_filters, ensure_ascii=False)}"
        ]
        return AlternativesDeepResearchResult(
            task_understanding=DeepResearchTaskUnderstanding(
                base_item_name=base_item_name,
                domain=request.domain,
                key_requirements=constraints,
                hard_constraints=constraints,
                nice_to_have=request.nice_to_have,
                notes="Fallback: не удалось получить структурированный ответ.",
            ),
            research_plan={"status": "failed", "reason": reason},
            alternatives=[],
            comparison_summary=DeepResearchComparisonSummary(
                key_dimensions=["цена", "качество", "доступность"],
                best_for_strict_constraints="нет данных",
                best_overall="нет данных",
                tradeoffs=["deep research не выполнен из-за ошибки обработки."],
            ),
            final_recommendations={
                "status": "failed",
                "reason": reason,
                "message": (
                    "Не удалось выполнить deep research, используйте каталожные альтернативы."
                ),
            },
        )


def build_summary(result: AlternativesDeepResearchResult) -> str:
    """Короткое summary для orchestrator."""
    best = result.comparison_summary.best_overall
    strict = result.comparison_summary.best_for_strict_constraints
    recommendation = (
        result.final_recommendations.get("overall")
        or result.final_recommendations.get("summary")
    )
    summary_parts = []
    if best and best != "нет данных":
        summary_parts.append(f"Лучший вариант: {best}.")
    if strict and strict != "нет данных":
        summary_parts.append(f"При строгих ограничениях: {strict}.")
    if recommendation:
        summary_parts.append(str(recommendation))
    if not summary_parts:
        summary_parts.append("Deep research выполнен, см. детали.")
    return " ".join(summary_parts)


def create_app(
    *,
    alternatives_agent: Optional[AlternativesAgent] = None,
    search_client: Optional[WebSearchClient] = None,
    embedder: Optional[EmbeddingsClient] = None,
    vector_store: Optional[VectorStore] = None,
    chat_callable=chat,
) -> FastAPI:
    settings = get_settings()
    deep = getattr(settings, "deep_research", None)
    vector_store = vector_store or MySQLVectorStore()
    embedder = embedder or EmbeddingsClient()
    seed_catalog = alternatives_agent is None
    alternatives_agent = alternatives_agent or AlternativesAgent(vector_store, embedder)

    search_client = search_client or WebSearchClient(
        timeout_seconds=getattr(deep, "search_timeout_seconds", 8.0)
    )
    search_k = getattr(deep, "search_top_k", 5)
    max_sources = getattr(deep, "search_max_results", 12)
    agent = AlternativesDeepResearchAgent(
        alternatives_agent=alternatives_agent,
        search_client=search_client,
        chat_callable=chat_callable,
        search_k=search_k,
        max_sources=max_sources,
    )

    app = FastAPI()

    @app.on_event("startup")
    async def startup_event() -> None:
        await db.init_db()
        if seed_catalog:
            await _ingest_stub_catalog(vector_store, embedder)

    @app.post(
        "/agents/alternatives/deep_research",
        response_model=AlternativesDeepResearchResponse,
    )
    async def run_deep_research(
        request: AlternativesDeepResearchRequest,
    ) -> AlternativesDeepResearchResponse:
        try:
            result = await agent.run(request)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            fallback = agent._fallback_result(request, reason="unhandled_exception")  # pylint: disable=protected-access
            payload = AlternativesDeepResearchPayload(summary="Ошибка агента.", research=fallback)
            return AlternativesDeepResearchResponse(
                status="error",
                data=payload,
                error={"message": str(exc)},
            )
        summary = build_summary(result)
        return AlternativesDeepResearchResponse(
            status="ok",
            data=AlternativesDeepResearchPayload(summary=summary, research=result),
        )

    @app.post(
        "/agents/alternatives_deep_research/run",
        response_model=AlternativesDeepResearchResponse,
    )
    async def run_deep_research_alias(
        request: AlternativesDeepResearchRequest,
    ) -> AlternativesDeepResearchResponse:
        return await run_deep_research(request)

    return app


app = create_app()
