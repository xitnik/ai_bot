from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rag.pipeline import SessionContext, rag_answer


@dataclass
class RagToolResult:
    answer: str
    sources: List[Dict[str, Any]]
    debug: Dict[str, Any]


class RagTool:
    """Agent-friendly wrapper around the RAG pipeline."""

    async def run(
        self, query: str, *, filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None
    ) -> RagToolResult:
        session = SessionContext(metadata=filters or {})
        result = await rag_answer(query, session=session)
        limited_docs = result.used_documents[: top_k or len(result.used_documents)]
        sources: List[Dict[str, Any]] = []
        for doc in limited_docs:
            meta = doc.document.metadata or {}
            sources.append(
                {
                    "doc_id": meta.get("doc_id") or doc.document.id,
                    "score": doc.score,
                    "title": meta.get("title") or meta.get("source") or meta.get("source_type"),
                }
            )
        return RagToolResult(answer=result.answer, sources=sources, debug=result.debug_info)

