from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set

from rag.ingest import Document


@dataclass
class KGEntity:
    id: str
    type: str
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class KGEdge:
    src: str
    dst: str
    relation: str


class KnowledgeGraph:
    """
    Minimal in-memory knowledge graph to capture relations between clients, products, documents and queries.
    Intended as a placeholder for a future graph DB backend.
    """

    def __init__(self) -> None:
        self._entities: Dict[str, KGEntity] = {}
        self._edges: List[KGEdge] = []
        self._doc_index: Dict[str, Set[str]] = {}

    def upsert_entity(self, entity_id: str, entity_type: str, attributes: Optional[Dict[str, str]] = None) -> None:
        attrs = attributes or {}
        existing = self._entities.get(entity_id)
        if existing:
            existing.attributes.update(attrs)
            return
        self._entities[entity_id] = KGEntity(id=entity_id, type=entity_type, attributes=dict(attrs))

    def add_edge(self, src: str, dst: str, relation: str) -> None:
        self._edges.append(KGEdge(src=src, dst=dst, relation=relation))

    def add_document(self, document: Document) -> None:
        doc_id = document.id
        self.upsert_entity(doc_id, "Document", {"source_type": str(document.metadata.get("source_type", ""))})
        client_id = document.metadata.get("client_id")
        product_id = document.metadata.get("product_id")
        if client_id:
            self.upsert_entity(str(client_id), "Client")
            self.add_edge(str(client_id), doc_id, "HAS_DOCUMENT")
        if product_id:
            self.upsert_entity(str(product_id), "Product")
            self.add_edge(str(product_id), doc_id, "DESCRIBED_IN")
        self._doc_index.setdefault(doc_id, set())

    def bulk_add_documents(self, documents: Iterable[Document]) -> None:
        for doc in documents:
            self.add_document(doc)

    def get_related_documents(
        self, client_id: Optional[str] = None, product_id: Optional[str] = None
    ) -> List[str]:
        related: Set[str] = set()
        if client_id:
            related.update(self._neighbors(str(client_id)))
        if product_id:
            related.update(self._neighbors(str(product_id)))
        return list(related)

    def _neighbors(self, entity_id: str) -> Set[str]:
        targets: Set[str] = set()
        for edge in self._edges:
            if edge.src == entity_id and edge.relation in {"HAS_DOCUMENT", "DESCRIBED_IN"}:
                targets.add(edge.dst)
        return targets


# Singleton for lightweight usage.
KG = KnowledgeGraph()
