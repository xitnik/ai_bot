from __future__ import annotations

import math
from typing import Any, Dict, List, Protocol

from alternatives_models import Hit


class VectorStore(Protocol):
    """Протокол хранилища векторов для подстановки реальных бекендов."""

    def upsert_product(self, product_id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        ...

    def knn_search(self, vector: List[float], k: int, filters: Dict[str, Any]) -> List[Hit]:
        ...


class InMemoryVectorStore:
    """Простое in-memory хранилище для тестов и разработки."""

    def __init__(self) -> None:
        self._vectors: Dict[str, List[float]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def upsert_product(self, product_id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        # Храним метаданные и вектор без дополнительной сериализации.
        self._vectors[product_id] = list(vector)
        self._metadata[product_id] = dict(metadata)

    def knn_search(self, vector: List[float], k: int, filters: Dict[str, Any]) -> List[Hit]:
        # Перед отбором по сходству сначала фильтруем по атрибутам.
        passed: List[str] = []
        for product_id, meta in self._metadata.items():
            if self._passes_filters(meta, filters):
                passed.append(product_id)

        results: List[Hit] = []
        for product_id in passed:
            stored_vector = self._vectors.get(product_id)
            if stored_vector is None:
                continue
            score = self._cosine_similarity(vector, stored_vector)
            results.append(Hit(product_id=product_id, score=score, metadata=self._metadata[product_id]))

        # Сортируем по убыванию сходства и обрезаем до k результатов.
        results.sort(key=lambda hit: hit.score, reverse=True)
        return results[:k]

    def _passes_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Применяем примитивные фильтры по атрибутам продукта."""
        species = filters.get("species")
        if species and metadata.get("species") != species:
            return False

        grade = filters.get("grade")
        if grade and metadata.get("grade") != grade:
            return False

        if "in_stock" in filters:
            required_stock = bool(filters["in_stock"])
            if bool(metadata.get("in_stock")) != required_stock:
                return False

        if not self._dimensions_match(metadata.get("dimensions"), filters.get("dimensions", {})):
            return False

        return True

    def _dimensions_match(self, candidate_dims: Any, filters: Dict[str, Any]) -> bool:
        """Проверяем диапазоны L/W/T; отсутствие фильтров не ограничивает выборку."""
        if not filters:
            return True
        if not isinstance(candidate_dims, dict):
            return False

        for key in ("length", "width", "thickness"):
            bounds = filters.get(key)
            if not bounds:
                continue
            value = candidate_dims.get(key)
            if value is None:
                return False
            if isinstance(bounds, dict):
                min_v = bounds.get("min")
                max_v = bounds.get("max")
                if min_v is not None and value < float(min_v):
                    return False
                if max_v is not None and value > float(max_v):
                    return False
        return True

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Косинусная близость со страховкой от деления на ноль."""
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)
