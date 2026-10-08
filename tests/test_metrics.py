import numpy as np
import pytest

from first_reply.eval import metrics


def test_macro_f1_perfect_and_interval_degenerate() -> None:
    y = ["a", "b", "a", "c"] * 25
    report = metrics.classification_report(y, y, groups=["en", "de"] * 50, n_boot=50)
    assert report["macro_f1"] == 1.0
    assert report["macro_f1_ci"] == (1.0, 1.0)
    assert report["by_group"]["de"]["n"] == 50


def test_bootstrap_interval_contains_point_estimate() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 500)
    p = np.where(rng.random(500) < 0.7, y, rng.integers(0, 3, 500))
    point = metrics.macro_f1(y, p)
    lo, hi = metrics.bootstrap_ci(metrics.macro_f1, y, p, n_boot=200)
    assert lo < point < hi
    assert hi - lo < 0.15


def test_multilabel_report() -> None:
    y = np.array([[1, 0], [0, 1], [1, 1], [0, 0]] * 10)
    report = metrics.multilabel_report(y, y)
    assert report["micro_f1"] == pytest.approx(1.0)
    assert report["macro_f1"] == pytest.approx(1.0)
