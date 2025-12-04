from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict


@dataclass
class CircuitState:
    failures: int = 0
    opened_at: float | None = None


class CircuitBreakerRegistry:
    def __init__(self, failure_threshold: int = 3, cooldown_sec: int = 30) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_sec = cooldown_sec
        self._state: Dict[str, CircuitState] = {}
        self._lock = asyncio.Lock()

    async def run(self, key: str, func: Callable[[], Awaitable[Any]], timeout: float = 5.0) -> Any:
        state = await self._get_state(key)
        now = time.time()
        if state.opened_at and now - state.opened_at < self.cooldown_sec:
            return {"status": "open_circuit", "error": "circuit_open"}

        try:
            result = await asyncio.wait_for(func(), timeout=timeout)
            await self._reset(key)
            return result
        except Exception as exc:  # pragma: no cover - защитная логика
            await self._increment_failure(key)
            return {"status": "error", "error": {"message": str(exc)}}

    async def _get_state(self, key: str) -> CircuitState:
        async with self._lock:
            if key not in self._state:
                self._state[key] = CircuitState()
            return self._state[key]

    async def _increment_failure(self, key: str) -> None:
        async with self._lock:
            state = self._state.setdefault(key, CircuitState())
            state.failures += 1
            if state.failures >= self.failure_threshold:
                state.opened_at = time.time()

    async def _reset(self, key: str) -> None:
        async with self._lock:
            self._state[key] = CircuitState()
