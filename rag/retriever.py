from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

from .ingest import DocumentChunk


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in text.split() if t.strip()]


class RagEngine:
    """Простой BoW retriever без внешних зависимостей."""

    def __init__(self, chunks: Iterable[DocumentChunk]) -> None:
        self.chunks = list(chunks)
        self.df: Dict[str, int] = defaultdict(int)
        self._build_index()

    def _build_index(self) -> None:
        for chunk in self.chunks:
            seen = set()
            for token in _tokenize(chunk.text):
                if token not in seen:
                    self.df[token] += 1
                    seen.add(token)

    def _idf(self, token: str) -> float:
        # Добавляем сглаживание, чтобы не делить на ноль.
        return math.log((1 + len(self.chunks)) / (1 + self.df.get(token, 0))) + 1.0

    def score(self, query: str, chunk: DocumentChunk) -> float:
        q_tokens = _tokenize(query)
        c_tokens = _tokenize(chunk.text)
        q_counts = Counter(q_tokens)
        c_counts = Counter(c_tokens)
        score = 0.0
        for token, q_freq in q_counts.items():
            score += q_freq * c_counts.get(token, 0) * self._idf(token)
        return score

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        scored = [(chunk, self.score(query, chunk)) for chunk in self.chunks]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [pair for pair in scored[:k] if pair[1] > 0]
