"""Classification metrics with bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import accuracy_score, f1_score

N_BOOT = 1000


def macro_f1(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return float(accuracy_score(y_true, y_pred))


def bootstrap_ci(
    metric: Callable[[Any, Any], float],
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    n_boot: int = N_BOOT,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap interval over examples."""
    t, p = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(t)
    stats = [metric(t[idx], p[idx]) for idx in (rng.integers(0, n, n) for _ in range(n_boot))]
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def classification_report(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    groups: ArrayLike | None = None,
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    """Macro-F1 and accuracy with 95% intervals, overall and per group (e.g. language)."""
    t, p = np.asarray(y_true), np.asarray(y_pred)
    out: dict[str, Any] = {
        "n": len(t),
        "macro_f1": macro_f1(t, p),
        "macro_f1_ci": bootstrap_ci(macro_f1, t, p, n_boot=n_boot),
        "accuracy": accuracy(t, p),
        "accuracy_ci": bootstrap_ci(accuracy, t, p, n_boot=n_boot),
    }
    if groups is not None:
        g = np.asarray(groups)
        out["by_group"] = {
            str(k): {"n": int((g == k).sum()), "macro_f1": macro_f1(t[g == k], p[g == k])}
            for k in sorted(set(g.tolist()))
        }
    return out


def multilabel_report(y_true: NDArray[Any], y_pred: NDArray[Any]) -> dict[str, Any]:
    """Micro- and macro-F1 over a fixed tag vocabulary (binary indicator matrices)."""
    micro = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    rows = np.arange(len(y_true))

    def _micro(idx: NDArray[Any], _: Any) -> float:
        return float(f1_score(y_true[idx], y_pred[idx], average="micro", zero_division=0))

    return {
        "n": len(y_true),
        "micro_f1": micro,
        "micro_f1_ci": bootstrap_ci(_micro, rows, rows, n_boot=200),
        "macro_f1": macro,
    }
