from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

from agents.base import AgentInput


def _stable_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest(), 16)


@dataclass
class RoutingDecision:
    intent: str
    agents: List[str]
    variant: str
    next_state: str


class Supervisor:
    """Простой supervisor с A/B и ручными правилами интентов."""

    def __init__(self, variants: Tuple[str, str] = ("control", "beta")) -> None:
        self.variants = variants

    def select_variant(self, user_id: str) -> str:
        h = _stable_hash(user_id)
        return self.variants[h % len(self.variants)]

    def route(self, payload: AgentInput) -> RoutingDecision:
        text = payload.message.lower()
        variant = self.select_variant(payload.user_id)

        intent = "sales"
        agents = ["sales"]
        next_state = "sales_followup"

        if "price" in text or "cost" in text:
            intent, agents, next_state = "pricing", ["pricing"], "pricing_quote"
        elif "alternative" in text or "another" in text:
            intent, agents, next_state = "alternatives", ["alternatives"], "alternatives_suggest"
        elif "buy" in text or "order" in text or "procure" in text:
            intent, agents, next_state = "procurement", ["procurement"], "procurement_flow"
        elif "calc" in text or "sum" in text or " + " in text:
            intent, agents, next_state = "calculator", ["calculator"], "calculator_flow"

        # Вариант beta может добавлять калькулятор как вспомогательный агент.
        if variant == "beta" and "price" in text:
            agents = list(dict.fromkeys(agents + ["calculator"]))

        return RoutingDecision(intent=intent, agents=agents, variant=variant, next_state=next_state)
