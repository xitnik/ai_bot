from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI

import db
from alternatives_models import (
    AlternativeItem,
    AlternativeType,
    AlternativesRequest,
    AlternativesResult,
)
from embeddings_client import EmbeddingsClient
from vector_index import MySQLVectorStore, VectorStore
from rag.pipeline import SessionContext as RagSessionContext
from rag.pipeline import rag_retrieve

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
        filters = dict(request.hard_filters)
        dims = filters.get("dimensions")
        dims_range = None
        if isinstance(dims, dict) and all(k in dims for k in ("length", "width", "thickness")):
            dims_range = {
                "length": {"min": float(dims["length"]) * (1 - DIMENSION_TOLERANCE_RATIO),
                           "max": float(dims["length"]) * (1 + DIMENSION_TOLERANCE_RATIO)},
                "width": {"min": float(dims["width"]) * (1 - DIMENSION_TOLERANCE_RATIO),
                          "max": float(dims["width"]) * (1 + DIMENSION_TOLERANCE_RATIO)},
                "thickness": {
                    "min": float(dims["thickness"]) * (1 - DIMENSION_TOLERANCE_RATIO),
                    "max": float(dims["thickness"]) * (1 + DIMENSION_TOLERANCE_RATIO),
                },
            }
            filters["dimensions"] = dims_range
        # Цена используется при классификации, но не как жесткий фильтр, чтобы не потерять ближние варианты.
        filters.pop("price", None)

        # Векторный поиск с жесткими фильтрами.
        hits = await self._store.knn_search(query_vector, request.k, filters)
        # Если подбор урезан по габаритам и результатов мало — добавляем ослабленный поиск.
        if dims_range and len(hits) < request.k:
            relaxed_filters = dict(filters)
            relaxed_filters.pop("dimensions", None)
            relaxed_hits = await self._store.knn_search(query_vector, request.k, relaxed_filters)
            # Мержим результаты, сохраняя сортировку по score.
            merged = {hit.product_id: hit for hit in hits}
            for hit in relaxed_hits:
                merged.setdefault(hit.product_id, hit)
            hits = sorted(merged.values(), key=lambda h: h.score, reverse=True)[: request.k]

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
        context_snippets = await self._maybe_retrieve_docs(request)
        return AlternativesResult(alternatives=alternatives, context_snippets=context_snippets)

    async def _maybe_retrieve_docs(self, request: AlternativesRequest) -> List[str]:
        if not request.use_rag_docs:
            return []
        try:
            session = RagSessionContext(
                product_id=request.hard_filters.get("sku") if isinstance(request.hard_filters, dict) else None,
                lang="ru",
            )
            docs = await rag_retrieve(request.query_text, session=session)
            return [doc.document.text for doc in docs]
        except Exception:
            return []


async def _ingest_stub_catalog(store: VectorStore, embedder: EmbeddingsClient) -> None:
    """
    Доп. загрузка каталога из fixtures/catalog.json.
    Если файла нет или embedder недоступен (нет ключей) — мягко пропускаем.
    """
    import json
    from pathlib import Path
    path = Path("fixtures/catalog.json")
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    items = payload if isinstance(payload, list) else payload.get("items", [])
    for item in items:
        try:
            meta = {
                "species": item.get("species"),
                "grade": item.get("grade"),
                "dimensions": item.get("dimensions"),
                "price": item.get("price"),
                "in_stock": item.get("in_stock", True),
            }
            desc = f"{meta.get('species')} {item.get('name','')}"
            vec = await embedder.embed_text(desc)
            await store.upsert_product(item.get("id", item.get("sku", "")), vec, meta)
        except Exception:
            # Если нет env или embedder падает, пропускаем элемент.
            continue


def create_app(
    store: Optional[VectorStore] = None,
    embedder: Optional[EmbeddingsClient] = None,
) -> FastAPI:
    # Отдельная фабрика, чтобы проще подменять зависимости в тестах.
    vector_store = store or MySQLVectorStore()
    embeddings_client = embedder or EmbeddingsClient()
    agent = AlternativesAgent(vector_store, embeddings_client)

    app = FastAPI()

    @app.on_event("startup")
    async def startup_event() -> None:
        await db.init_db()
        if store is None:
            await _ingest_stub_catalog(vector_store, embeddings_client)

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        await db.dispose_engine()

    @app.post("/agents/alternatives/run", response_model=AlternativesResult)
    async def run_alternatives(request: AlternativesRequest) -> AlternativesResult:
        return await agent.run(request)

    return app


# Экземпляр по умолчанию для запуска сервиса.
app = create_app()
