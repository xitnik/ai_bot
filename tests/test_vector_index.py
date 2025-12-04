from __future__ import annotations

from vector_index import InMemoryVectorStore


def _sample_dims(length: float) -> dict:
    return {"length": length, "width": 20.0, "thickness": 5.0}


def test_filters_by_species_grade_and_stock() -> None:
    store = InMemoryVectorStore()
    store.upsert_product("oak", [1.0, 0.0], {"species": "oak", "grade": "A", "in_stock": True})
    store.upsert_product("pine", [0.0, 1.0], {"species": "pine", "grade": "A", "in_stock": True})

    hits = store.knn_search(
        [1.0, 0.0],
        k=5,
        filters={"species": "oak", "grade": "A", "in_stock": True},
    )

    assert [hit.product_id for hit in hits] == ["oak"]


def test_filters_by_dimensions_range() -> None:
    store = InMemoryVectorStore()
    store.upsert_product(
        "inside",
        [1.0, 0.0],
        {"species": "oak", "dimensions": _sample_dims(100.0)},
    )
    store.upsert_product(
        "outside",
        [0.0, 1.0],
        {"species": "oak", "dimensions": _sample_dims(150.0)},
    )

    hits = store.knn_search(
        [1.0, 0.0],
        k=5,
        filters={"species": "oak", "dimensions": {"length": {"min": 90, "max": 120}}},
    )

    assert [hit.product_id for hit in hits] == ["inside"]


def test_knn_sorts_by_similarity() -> None:
    store = InMemoryVectorStore()
    store.upsert_product("first", [1.0, 0.0], {"species": "oak"})
    store.upsert_product("second", [0.8, 0.2], {"species": "oak"})
    store.upsert_product("third", [0.0, 1.0], {"species": "oak"})

    hits = store.knn_search([1.0, 0.0], k=2, filters={"species": "oak"})

    assert [hit.product_id for hit in hits] == ["first", "second"]
    assert hits[0].score > hits[1].score
