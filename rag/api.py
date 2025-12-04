from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from rag.pipeline import SessionContext, rag_answer

router = APIRouter(prefix="/rag", tags=["rag"])


class RagQuery(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    top_k: Optional[int] = None


class RagSource(BaseModel):
    doc_id: str
    score: float
    title: Optional[str] = None


class RagResponse(BaseModel):
    answer: str
    sources: List[RagSource]
    debug: Dict[str, Any]


@router.post("/query", response_model=RagResponse)
async def rag_query(payload: RagQuery) -> RagResponse:
    session = SessionContext(metadata=payload.filters or {})
    result = await rag_answer(payload.query, session=session)
    limited_docs = result.used_documents[: payload.top_k or len(result.used_documents)]
    sources = [
        RagSource(
            doc_id=doc.document.metadata.get("doc_id") or doc.document.id,
            score=doc.score,
            title=(
                doc.document.metadata.get("title")
                or doc.document.metadata.get("source")
                or doc.document.metadata.get("source_type")
            ),
        )
        for doc in limited_docs
    ]
    return RagResponse(answer=result.answer, sources=sources, debug=result.debug_info)

