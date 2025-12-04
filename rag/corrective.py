from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from llm_client import chat
from metrics import REGISTRY
from rag.retriever import HybridRetriever
from vector_index import ScoredDocument


@dataclass
class EvaluationResult:
    score: float
    action: str
    reason: str


class RetrievalEvaluator:
    """
    Simple evaluator of retrieval quality/answer faithfulness.
    Uses heuristics on scores and number of docs; optionally LLM-judge.
    """

    def __init__(self, min_score: float = 0.2) -> None:
        self._min_score = min_score

    def evaluate(self, query: str, docs: Sequence[ScoredDocument], answer: str | None = None) -> EvaluationResult:
        if not docs:
            return EvaluationResult(score=0.0, action="retry", reason="no_docs")
        top_score = docs[0].score
        if top_score < self._min_score:
            return EvaluationResult(score=top_score, action="retry", reason="low_score")
        if len(docs) < 2:
            return EvaluationResult(score=top_score, action="retry", reason="not_enough_docs")
        if answer:
            try:
                judge_score = self._llm_judge(query, docs, answer)
                if judge_score < self._min_score:
                    return EvaluationResult(score=judge_score, action="retry", reason="judge_low")
            except Exception:
                pass
        return EvaluationResult(score=top_score, action="accept", reason="ok")

    def _llm_judge(self, query: str, docs: Sequence[ScoredDocument], answer: str) -> float:
        """Optional LLM-based judging; returns score 0..1."""
        context = "\n".join(f"[doc:{d.document.id}] {d.document.text}" for d in docs[:3])
        prompt = (
            "Оцени, соответствует ли ответ приведенному контексту. Верни только число 0..1.\n"
            f"Вопрос: {query}\nОтвет: {answer}\nКонтекст:\n{context}"
        )
        completion = chat("gpt5", [{"role": "user", "content": prompt}], temperature=0)
        text = getattr(completion.choices[0].message, "content", "0")
        try:
            return float(text.strip().split()[0])
        except Exception:
            return 0.0


class CorrectiveRag:
    """
    Corrective RAG (CRAG) runner: initial retrieval+answer, evaluate, retry with rewritten query if needed.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        evaluator: RetrievalEvaluator,
        max_retries: int = 1,
    ) -> None:
        self._retriever = retriever
        self._evaluator = evaluator
        self._max_retries = max_retries

    async def run(self, query: str, generate_answer, filters: Dict[str, Any]) -> tuple[str, List[ScoredDocument], Dict[str, Any]]:
        attempts: List[Dict[str, Any]] = []
        retries = 0
        docs = await self._retriever.retrieve(query, filters=filters)
        answer = generate_answer(docs)
        eval_result = self._evaluator.evaluate(query, docs, answer)
        attempts.append({"query": query, "score": eval_result.score, "reason": eval_result.reason})
        while eval_result.action == "retry" and retries < self._max_retries:
            retries += 1
            REGISTRY.counter("rag_crag_retries_count").inc()
            rewritten = await self._rewrite_query(query)
            docs = await self._retriever.retrieve(rewritten, filters=filters)
            answer = generate_answer(docs)
            eval_result = self._evaluator.evaluate(rewritten, docs, answer)
            attempts.append({"query": rewritten, "score": eval_result.score, "reason": eval_result.reason})
        if eval_result.action != "accept":
            REGISTRY.counter("rag_crag_fallback_rate").inc()
            answer = "Недостаточно данных, чтобы ответить точно. Нужны дополнительные документы."
        debug = {"attempts": attempts, "final_score": eval_result.score}
        return answer, docs, debug

    async def _rewrite_query(self, query: str) -> str:
        prompt = (
            "Переформулируй запрос для поиска в базе документов, сохраняя смысл. Кратко, без лишних слов."
        )
        try:
            completion = await asyncio.to_thread(
                chat,
                "gpt5",
                [{"role": "system", "content": prompt}, {"role": "user", "content": query}],
                temperature=0.2,
                max_tokens=64,
            )
            text = getattr(completion.choices[0].message, "content", "").strip()
            return text or query
        except Exception:
            return query
