from __future__ import annotations

import pytest

from rag.ingest import Document
from vector_index import InMemoryVectorStore


def _sample_dims(length: float) -> dict:
    return {"length": length, "width": 20.0, "thickness": 5.0}


@pytest.mark.asyncio
async def test_filters_by_species_grade_and_stock() -> None:
    store = InMemoryVectorStore()
    await store.upsert_product("oak", [1.0, 0.0], {"species": "oak", "grade": "A", "in_stock": True})
    await store.upsert_product("pine", [0.0, 1.0], {"species": "pine", "grade": "A", "in_stock": True})

    hits = await store.knn_search(
        [1.0, 0.0],
        k=5,
        filters={"species": "oak", "grade": "A", "in_stock": True},
    )

    assert [hit.product_id for hit in hits] == ["oak"]


@pytest.mark.asyncio
async def test_filters_by_dimensions_range() -> None:
    store = InMemoryVectorStore()
    await store.upsert_product(
        "inside",
        [1.0, 0.0],
        {"species": "oak", "dimensions": _sample_dims(100.0)},
    )
    await store.upsert_product(
        "outside",
        [0.0, 1.0],
        {"species": "oak", "dimensions": _sample_dims(150.0)},
    )

    hits = await store.knn_search(
        [1.0, 0.0],
        k=5,
        filters={"species": "oak", "dimensions": {"length": {"min": 90, "max": 120}}},
    )

    assert [hit.product_id for hit in hits] == ["inside"]


@pytest.mark.asyncio
async def test_knn_sorts_by_similarity() -> None:
    store = InMemoryVectorStore()
    await store.upsert_product("first", [1.0, 0.0], {"species": "oak"})
    await store.upsert_product("second", [0.8, 0.2], {"species": "oak"})
    await store.upsert_product("third", [0.0, 1.0], {"species": "oak"})

    hits = await store.knn_search([1.0, 0.0], k=2, filters={"species": "oak"})

    assert [hit.product_id for hit in hits] == ["first", "second"]
    assert hits[0].score > hits[1].score


@pytest.mark.asyncio
async def test_document_search_respects_metadata_filters() -> None:
    store = InMemoryVectorStore()
    doc_ru = Document(id="ru", text="привет", metadata={"lang": "ru"}, embedding=[1.0, 0.0])
    doc_en = Document(id="en", text="hello", metadata={"lang": "en"}, embedding=[0.0, 1.0])
    await store.add_documents([doc_ru, doc_en])

    results = await store.search([1.0, 0.0], filters={"lang": "ru"}, top_k=5)

    assert [res.document.id for res in results] == ["ru"]
