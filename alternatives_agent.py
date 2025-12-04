from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI

from alternatives_models import (
    AlternativeItem,
    AlternativeType,
    AlternativesRequest,
    AlternativesResult,
)
from embeddings_client import EmbeddingsClient
from vector_index import InMemoryVectorStore, VectorStore

DIMENSION_TOLERANCE_RATIO = 0.05
PRICE_TOLERANCE_RATIO = 0.10


def _dimensions_close(
    base_dims: Optional[Dict[str, Any]],
    candidate_dims: Optional[Dict[str, Any]],
    tolerance_ratio: float = DIMENSION_TOLERANCE_RATIO,
) -> bool:
    """Сравниваем размеры с заданным относительным допуском."""
    if not base_dims or not candidate_dims:
        return False
    for key in ("length", "width", "thickness"):
        base_value = base_dims.get(key)
        cand_value = candidate_dims.get(key)
        if base_value is None or cand_value is None:
            return False
        if base_value == 0:
            return False
        delta = abs(cand_value - base_value) / base_value
        if delta > tolerance_ratio:
            return False
    return True


def _price_delta_percent(candidate_price: Optional[float], base_price: Optional[float]) -> Optional[float]:
    """Возвращает процентное отклонение цены кандидата от базовой."""
    if candidate_price is None or base_price is None or base_price == 0:
        return None
    return (candidate_price - base_price) / base_price * 100.0


def _percent_text(value: float) -> str:
    """Форматируем проценты без дробной части для детерминизма."""
    return f"{int(round(abs(value)))}%"


def classify_alternative(
    hit_metadata: Dict[str, Any],
    base_metadata: Dict[str, Any],
    price_band: Optional[str],
) -> Tuple[AlternativeType, str]:
    """Определяем тип рекомендации и детерминированную причину."""
    base_species = base_metadata.get("species")
    base_grade = base_metadata.get("grade")
    base_dimensions = base_metadata.get("dimensions")
    base_price = base_metadata.get("price")

    candidate_species = hit_metadata.get("species")
    candidate_grade = hit_metadata.get("grade")
    candidate_dimensions = hit_metadata.get("dimensions")
    candidate_price = hit_metadata.get("price")

    species_match = bool(base_species) and candidate_species == base_species
    grade_match = bool(base_grade) and candidate_grade == base_grade
    dimensions_match = _dimensions_close(base_dimensions, candidate_dimensions)
    price_delta = _price_delta_percent(candidate_price, base_price)

    price_close = price_delta is not None and abs(price_delta) <= PRICE_TOLERANCE_RATIO * 100
    cheaper = price_delta is not None and price_delta <= -PRICE_TOLERANCE_RATIO * 100
    pricier = price_delta is not None and price_delta >= PRICE_TOLERANCE_RATIO * 100

    if species_match and grade_match and dimensions_match and price_close:
        reason = "same species and similar dimensions, price within 10%."
        return "direct", reason

    if price_band == "cheaper" and cheaper:
        suffix = "same grade." if grade_match else "compatible specs."
        cheaper_text = _percent_text(price_delta)
        reason = f"about {cheaper_text} cheaper, {suffix}"
        return "price_low", reason

    if price_band == "premium" and pricier:
        suffix = "similar size." if dimensions_match else "upgraded option."
        premium_text = _percent_text(price_delta)
        reason = f"about {premium_text} higher price, {suffix}"
        return "price_high", reason

    if species_match and grade_match:
        reason = "same species and grade, size differs."
        return "functional", reason

    if species_match or grade_match:
        reason = "partially matching specs, check sizing."
        return "functional", reason

    reason = "similar use-case based on attributes."
    return "functional", reason


class AlternativesAgent:
    """Оркестратор выбора альтернатив поверх векторного поиска."""

    def __init__(self, store: VectorStore, embedder: EmbeddingsClient) -> None:
        self._store = store
        self._embedder = embedder

    async def run(self, request: AlternativesRequest) -> AlternativesResult:
        # Эмбеддинг текста запроса.
        query_vector = await self._embedder.embed_text(request.query_text)
        # Векторный поиск с жесткими фильтрами.
        hits = self._store.knn_search(query_vector, request.k, request.hard_filters)

        alternatives: List[AlternativeItem] = []
        for hit in hits:
            alt_type, reason = classify_alternative(hit.metadata, request.hard_filters, request.price_band)
            alternatives.append(
                AlternativeItem(
                    product_id=hit.product_id,
                    similarity=hit.score,
                    type=alt_type,
                    reason=reason,
                )
            )
        return AlternativesResult(alternatives=alternatives)


def create_app(
    store: Optional[VectorStore] = None,
    embedder: Optional[EmbeddingsClient] = None,
) -> FastAPI:
    # Отдельная фабрика, чтобы проще подменять зависимости в тестах.
    vector_store = store or InMemoryVectorStore()
    embeddings_client = embedder or EmbeddingsClient()
    agent = AlternativesAgent(vector_store, embeddings_client)

    app = FastAPI()

    @app.post("/agents/alternatives/run", response_model=AlternativesResult)
    async def run_alternatives(request: AlternativesRequest) -> AlternativesResult:
        return await agent.run(request)

    return app


# Экземпляр по умолчанию для запуска сервиса.
app = create_app()
