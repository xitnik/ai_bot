from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol


@dataclass
class AgentInput:
    message: str
    user_id: str
    session_id: str
    intent: str
    context: Dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None


@dataclass
class AgentResult:
    status: str
    data: Dict[str, Any] = field(default_factory=dict)
    error: Dict[str, Any] | None = None


class Agent(Protocol):
    name: str

    async def run(self, payload: AgentInput) -> AgentResult:  # pragma: no cover - интерфейс
        ...
