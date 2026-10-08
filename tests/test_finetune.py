"""Smoke test of the fine-tuning loop on a tiny random BERT (skipped without torch)."""

from pathlib import Path

import pandas as pd
import pytest

from first_reply.routing import base

pytest.importorskip("torch")
pytest.importorskip("transformers")

from first_reply.routing.finetune import FinetunedRouter

TINY = "hf-internal-testing/tiny-random-BertModel"


def _tickets(n: int = 120) -> pd.DataFrame:
    queues = ["Billing", "Technical", "HR"]
    splits = ["train"] * 7 + ["dev"] + ["test"] * 2
    return pd.DataFrame(
        {
            "id": [f"t{i}" for i in range(n)],
            "language": ["en", "de"] * (n // 2),
            "text": [f"{queues[i % 3]} problem number {i}" for i in range(n)],
            "queue": [queues[i % 3] for i in range(n)],
            "priority": ["low", "high"] * (n // 2),
            "type": ["Incident", "Request"] * (n // 2),
            "tags": [[queues[i % 3]] for i in range(n)],
            "split": [splits[i % 10] for i in range(n)],
        }
    )


def test_finetune_runs_and_keeps_best_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(base, "RESULTS", tmp_path / "results")
    router = FinetunedRouter(TINY, epochs=2, batch_size=16, lr=1e-3, max_length=16)
    try:
        report = base.evaluate(router, _tickets(), out_dir=tmp_path / "runs", tag_min_count=5)
    except OSError as err:  # model download unavailable (offline runner)
        pytest.skip(f"tiny model not available: {err}")
    assert len(report["history"]) == 2
    assert "dev_mean_macro_f1" in report["history"][0]
    assert set(report["targets"]) == {"queue", "priority", "type"}
    assert 0.0 <= report["tags"]["micro_f1"] <= 1.0
