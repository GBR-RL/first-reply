import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from first_reply import routing
from first_reply.routing import base, targets

WORDS = {
    "Billing and Payments": "invoice refund charge payment",
    "Technical Support": "server crash error install",
    "Human Resources": "vacation contract salary leave",
}

SPLITS = ["train"] * 7 + ["dev"] + ["test"] * 2


def _tickets(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    queues = list(WORDS)
    for i in range(n):
        q = queues[i % 3]
        words = WORDS[q].split()
        text = " ".join(rng.choice(words, 6)) + f" ticket {i}"
        rows.append(
            {
                "id": f"t{i}",
                "language": "en" if i % 2 else "de",
                "text": text,
                "queue": q,
                "priority": ["low", "medium", "high"][i % 3],
                "type": "Incident" if "crash" in text else "Request",
                "tags": [q.split()[0], "Common"],
                "split": SPLITS[i % 10],
            }
        )
    return pd.DataFrame(rows)


def test_tag_vocab_uses_min_count_and_matrix_marks_known_tags() -> None:
    df = pd.DataFrame({"tags": [["a", "b"], ["a"], ["a", "c"]]})
    vocab = targets.tag_vocab(df, min_count=2)
    assert vocab == ["a"]
    assert targets.tag_matrix(df, ["a", "c"]).tolist() == [[1, 0], [1, 0], [1, 1]]


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(KeyError):
        routing.make("nope")


def test_rounded_handles_nested_values() -> None:
    assert base.rounded({"a": [0.123456, (1.0, 2.55555)], "b": "x"}) == {
        "a": [0.1235, [1.0, 2.5556]],
        "b": "x",
    }


def test_tfidf_router_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base, "RESULTS", tmp_path / "results")
    df = _tickets()
    report = base.evaluate(
        routing.make("tfidf_lr"), df, out_dir=tmp_path / "runs", tag_min_count=10
    )
    assert report["targets"]["queue"]["macro_f1"] > 0.95
    assert set(report["targets"]["queue"]["by_group"]) == {"de", "en"}
    assert report["tags"]["micro_f1"] > 0.9
    saved = json.loads((tmp_path / "results" / "tfidf_lr.json").read_text())
    assert saved["method"] == "tfidf_lr"
    preds = pd.read_parquet(tmp_path / "runs" / "tfidf_lr" / "predictions_test.parquet")
    assert len(preds) == (df.split == "test").sum()
    assert len(preds.queue_proba.iloc[0]) == 3
