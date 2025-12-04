from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from llm_client import chat
from metrics import REGISTRY
from rag.retriever import HybridRetriever
from vector_index import ScoredDocument

RETRIEVE_PATTERN = re.compile(r"<RETRIEVE\\s+query=\"([^\"]+)\"\\s*>", re.IGNORECASE)


@dataclass
class SelfRagPlan:
    queries: List[str] = field(default_factory=list)
    raw_plan: str = ""


def parse_retrieve_queries(plan_text: str) -> List[str]:
    """Parses retrieval markers from LLM plan."""
    return [match.group(1).strip() for match in RETRIEVE_PATTERN.finditer(plan_text)]


class SelfRagOrchestrator:
    """
    Lightweight Self-RAG executor:
    1) Ask LLM for plan with <RETRIEVE query="..."> markers.
    2) Run retrieval for each marker, build augmented context.
    3) Ask LLM for final answer + CRITIQUE block.
    """

    def __init__(self, retriever: HybridRetriever, max_iterations: int = 2) -> None:
        self._retriever = retriever
        self._max_iterations = max_iterations

    def _planning_prompt(self, query: str) -> List[Dict[str, str]]:
        content = (
            "You are a self-reflective assistant. Produce a short plan and insert retrieval tags.\n"
            "Use markers <RETRIEVE query=\"...\"></RETRIEVE> where external context is needed.\n"
            "Example: 'Check discount policy <RETRIEVE query=\"pricing discounts 2024\"> then answer.'\n"
            "Keep it concise."
        )
        return [
            {"role": "system", "content": content},
            {"role": "user", "content": f"Task: {query}"},
        ]

    def _answer_prompt(self, query: str, snippets: Sequence[ScoredDocument], plan: str) -> List[Dict[str, str]]:
        context_lines = [
            f"[doc:{item.document.id}] {item.document.text}" for item in snippets
        ]
        context_block = "\n".join(context_lines)
        critique_note = (
            "After the answer, add a CRITIQUE section with self-assessed factuality and which docs were used."
        )
        return [
            {"role": "system", "content": "You are a retrieval-augmented assistant. Only use provided context."},
            {
                "role": "user",
                "content": f"Plan: {plan}\nQuestion: {query}\nContext:\n{context_block}\n\n{critique_note}",
            },
        ]

    def build_plan(self, query: str) -> SelfRagPlan:
        completion = chat("gpt5", self._planning_prompt(query), temperature=0.2, max_tokens=512)
        text = _extract_text(completion)
        return SelfRagPlan(queries=parse_retrieve_queries(text), raw_plan=text)

    async def run(self, query: str) -> tuple[str, List[ScoredDocument], Dict[str, Any]]:
        plan = self.build_plan(query)
        retrieved: List[ScoredDocument] = []
        iterations = 0
        for q in plan.queries[: self._max_iterations]:
            results = await self._retriever.retrieve(q)
            retrieved.extend(results)
            iterations += 1
        REGISTRY.counter("rag_selfrag_iterations").inc(iterations)
        completion = chat("gpt5", self._answer_prompt(query, retrieved, plan.raw_plan), temperature=0)
        answer = _extract_text(completion)
        debug = {"plan": plan.raw_plan, "queries": plan.queries, "iterations": iterations}
        return answer, retrieved, debug


def _extract_text(completion: Any) -> str:
    if not completion or not getattr(completion, "choices", None):
        return ""
    first = completion.choices[0]
    message = getattr(first, "message", None)
    if message and getattr(message, "content", None):
        return message.content
    return getattr(first, "text", "") or ""
