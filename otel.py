"""
Обертка над OTEL с мягким фолбэком: если opentelemetry не установлена,
используем простой in-memory рекордер, чтобы тесты не падали.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:  # Реальная OTEL цепочка, если зависимость доступна.
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        ConsoleSpanExporter,
        InMemorySpanExporter,
        SimpleSpanProcessor,
    )

    resource = Resource.create({"service.name": "conversation-pipeline"})
    span_exporter = InMemorySpanExporter()
    console_exporter = ConsoleSpanExporter()

    provider = TracerProvider(resource=resource)
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    provider.add_span_processor(SimpleSpanProcessor(console_exporter))
    trace.set_tracer_provider(provider)

    def get_tracer():
        """Возвращает tracer для всего пайплайна."""
        return trace.get_tracer("conversation")

    def get_exported_spans():
        """Позволяет тестам читать завершенные спаны из in-memory экспортера."""
        return span_exporter.get_finished_spans()

    def reset_traces() -> None:
        """Очищает накопленные спаны между тестами."""
        span_exporter.clear()

except ImportError:
    # Минимальный заменитель, чтобы не тянуть тяжелые зависимости в окружении 3.9.
    class _Span:
        def __init__(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
            self.name = name
            self.attributes = attributes or {}

    class _SpanRecorder:
        def __init__(self) -> None:
            self._finished: List[_Span] = []

        def add(self, span: _Span) -> None:
            self._finished.append(span)

        def get_finished_spans(self) -> List[_Span]:
            return list(self._finished)

        def clear(self) -> None:
            self._finished.clear()

    class _SpanContext:
        def __init__(self, recorder: _SpanRecorder, span: _Span) -> None:
            self._recorder = recorder
            self._span = span

        def __enter__(self) -> _Span:
            return self._span

        def __exit__(self, exc_type, exc, tb) -> None:
            self._recorder.add(self._span)

    class _Tracer:
        def __init__(self, recorder: _SpanRecorder) -> None:
            self._recorder = recorder

        def start_as_current_span(
            self, name: str, attributes: Optional[Dict[str, Any]] = None
        ):
            return _SpanContext(self._recorder, _Span(name, attributes))

    _recorder = _SpanRecorder()
    _tracer = _Tracer(_recorder)

    def get_tracer() -> _Tracer:
        return _tracer

    def get_exported_spans():
        return _recorder.get_finished_spans()

    def reset_traces() -> None:
        _recorder.clear()
