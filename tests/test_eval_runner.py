from __future__ import annotations

from tools import eval_runner


def test_eval_runner_passes_on_golden_set(monkeypatch):
    # Используем существующий golden set; должны пройти текущие эвристики.
    status = eval_runner.main()
    assert status == 0
