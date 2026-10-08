"""Common interface for routing methods and the evaluation loop around it."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from first_reply.config import ROOT, RUNS
from first_reply.eval.metrics import classification_report, multilabel_report
from first_reply.routing.targets import TAG_MIN_COUNT, TARGETS, tag_matrix, tag_vocab

RESULTS = ROOT / "docs" / "results" / "routing"


@dataclass
class Probabilities:
    classes: list[str]
    proba: NDArray[np.float64]  # (n, len(classes))

    def predict(self) -> NDArray[np.str_]:
        labels: NDArray[np.str_] = np.asarray(self.classes)[self.proba.argmax(axis=1)]
        return labels


@dataclass
class Prediction:
    targets: dict[str, Probabilities]
    tag_scores: NDArray[np.float64] | None = None  # (n, len(vocab)), higher = more likely
    extra: dict[str, Any] = field(default_factory=dict)


class Router(Protocol):
    name: str

    def fit(self, train: pd.DataFrame, tags: list[str]) -> None: ...

    def predict(self, df: pd.DataFrame) -> Prediction: ...


def best_threshold(y: NDArray[Any], scores: NDArray[np.float64]) -> float:
    """Single global tag threshold that maximises micro-F1 on the given split."""
    from sklearn.metrics import f1_score

    grid = np.quantile(scores, np.linspace(0.5, 0.995, 100))
    f1s = [f1_score(y, scores >= t, average="micro", zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def rounded(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: rounded(v, digits) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [rounded(v, digits) for v in value]
    return value


def _frame(df: pd.DataFrame, pred: Prediction) -> pd.DataFrame:
    out = df[["id", "language", *TARGETS]].reset_index(drop=True).copy()
    for target, p in pred.targets.items():
        out[f"{target}_pred"] = p.predict()
        out[f"{target}_proba"] = list(p.proba)
    return out


def evaluate(
    router: Router,
    df: pd.DataFrame,
    out_dir: Path = RUNS / "routing",
    tag_min_count: int = TAG_MIN_COUNT,
) -> dict[str, Any]:
    """Fit on train, predict dev and test, tune the tag threshold on dev, report test."""
    train, dev, test = (df[df.split == s] for s in ("train", "dev", "test"))
    vocab = tag_vocab(train, tag_min_count)
    t0 = time.perf_counter()
    router.fit(train, vocab)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    p_dev, p_test = router.predict(dev), router.predict(test)
    predict_s = time.perf_counter() - t0

    report: dict[str, Any] = {
        "method": router.name,
        "fit_seconds": round(fit_s, 1),
        "predict_ms_per_ticket": round(1000 * predict_s / (len(dev) + len(test)), 3),
        "targets": {},
    }
    for target in TARGETS:
        if target in p_test.targets:
            report["targets"][target] = classification_report(
                test[target].to_numpy(), p_test.targets[target].predict(), test.language.to_numpy()
            )
    if vocab and p_dev.tag_scores is not None and p_test.tag_scores is not None:
        threshold = best_threshold(tag_matrix(dev, vocab), p_dev.tag_scores)
        report["tags"] = {
            "vocab_size": len(vocab),
            "threshold_from_dev": threshold,
            **multilabel_report(tag_matrix(test, vocab), p_test.tag_scores >= threshold),
        }
    report.update(p_test.extra)

    run = out_dir / router.name
    run.mkdir(parents=True, exist_ok=True)
    for split, part, pred in (("dev", dev, p_dev), ("test", test, p_test)):
        frame = _frame(part, pred)
        frame.to_parquet(run / f"predictions_{split}.parquet", index=False)
    classes = {t: p.classes for t, p in p_test.targets.items()}
    (run / "classes.json").write_text(json.dumps(classes, indent=2))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{router.name}.json").write_text(json.dumps(rounded(report), indent=2) + "\n")
    return report
