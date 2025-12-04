from __future__ import annotations

import time
from typing import Dict


class Counter:
    def __init__(self) -> None:
        self.value = 0

    def inc(self, by: int = 1) -> None:
        self.value += by


class Histogram:
    def __init__(self) -> None:
        self.values: list[float] = []

    def observe(self, value: float) -> None:
        self.values.append(value)

    def avg(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)


class MetricsRegistry:
    def __init__(self) -> None:
        self.counters: Dict[str, Counter] = {}
        self.histograms: Dict[str, Histogram] = {}

    def counter(self, name: str) -> Counter:
        if name not in self.counters:
            self.counters[name] = Counter()
        return self.counters[name]

    def histogram(self, name: str) -> Histogram:
        if name not in self.histograms:
            self.histograms[name] = Histogram()
        return self.histograms[name]

    def render_prometheus(self) -> str:
        lines = []
        for name, counter in self.counters.items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {counter.value}")
        for name, hist in self.histograms.items():
            metric_name = name if name.endswith("_avg") else f"{name}_avg"
            lines.append(f"# TYPE {metric_name} gauge")
            lines.append(f"{metric_name} {hist.avg()}")
        return "\n".join(lines) + "\n"


REGISTRY = MetricsRegistry()


def track_latency(metric_name: str):
    """Декоратор для подсчета среднего времени исполнения."""

    def wrapper(func):
        async def inner(*args, **kwargs):
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            REGISTRY.histogram(metric_name).observe((time.perf_counter() - start) * 1000)
            return result

        return inner

    return wrapper
