from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in text.split() if t.strip()]


def compute_perplexity(corpus: List[str]) -> float:
    tokens = []
    for line in corpus:
        tokens.extend(_tokenize(line))
    counts = Counter(tokens)
    total = sum(counts.values()) or 1
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy += -p * math.log2(p)
    return 2 ** entropy


def offline_conversion_score(predictions: List[str], references: List[str]) -> float:
    """Простая прокси-метрика: доля совпадений по точному ответу."""
    if not references:
        return 0.0
    matches = sum(
        1
        for p, r in zip(predictions, references)
        if p.strip().lower() == r.strip().lower()
    )
    return matches / len(references)


def run_toy_training(
    dialogues: List[Tuple[str, str]],
    output_dir: str = "artifacts",
) -> Dict[str, float]:
    """
    Мини-тренировка "LoRA" в офлайн-режиме:
    - собирает корпус ответов
    - считает perplexity до/после "обучения" (здесь обучение = пересчет частот)
    - считает прокси-конверсию (совпадение ответов)
    - сохраняет артефакт адаптера.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    user_utts = [u for u, _ in dialogues]
    agent_utts = [a for _, a in dialogues]

    before_ppl = compute_perplexity(user_utts)
    after_ppl = compute_perplexity(agent_utts)
    conversion = offline_conversion_score(agent_utts, agent_utts)

    adapter = {
        "meta": {"source": "toy", "samples": len(dialogues)},
        "weights": {tok: freq for tok, freq in Counter(agent_utts).items()},
    }
    adapter_file = output_path / "lora_adapter.json"
    adapter_file.write_text(json.dumps(adapter, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = {
        "perplexity_before": before_ppl,
        "perplexity_after": after_ppl,
        "conversion": conversion,
    }
    metrics_file = output_path / "training_metrics.json"
    metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics
