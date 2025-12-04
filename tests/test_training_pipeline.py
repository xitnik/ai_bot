from __future__ import annotations

from training.pipeline import compute_perplexity, run_toy_training
from training.sample_data import SAMPLE_DIALOGUES


def test_toy_training_produces_metrics_and_files(tmp_path):
    metrics = run_toy_training(SAMPLE_DIALOGUES, output_dir=tmp_path)
    assert metrics["perplexity_after"] >= 0
    assert (tmp_path / "lora_adapter.json").exists()
    assert (tmp_path / "training_metrics.json").exists()


def test_compute_perplexity_nonzero():
    ppl = compute_perplexity(["a b c a"])
    assert ppl > 0
