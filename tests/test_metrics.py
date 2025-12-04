from __future__ import annotations

from metrics import REGISTRY


def test_metrics_registry_renders_prometheus():
    REGISTRY.counter("test_counter").inc()
    REGISTRY.histogram("test_hist").observe(10)
    output = REGISTRY.render_prometheus()
    assert "test_counter" in output
    assert "test_hist_avg" in output
