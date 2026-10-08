import numpy as np
import pandas as pd

from first_reply.eval import reports


def test_coverage_and_precision_of_a_gate() -> None:
    take = np.array([True, True, False, False])
    right = np.array([True, False, True, True])
    assert reports._coverage(take, right) == {"coverage": 0.5, "precision": 0.5}
    assert reports._coverage(np.zeros(4, bool), right)["precision"] is None


def test_ops_summarises_tokens_latency_and_errors() -> None:
    runs = pd.DataFrame(
        {
            "prompt_tokens": [100, 200, None],
            "completion_tokens": [10, 30, None],
            "latency_s": [1.0, 3.0, None],
            "cost": [0.001, 0.003, None],
            "error": [None, None, "RuntimeError: boom"],
        }
    )
    out = reports.ops(runs)
    assert out["items"] == 3
    assert out["errors"] == 1
    assert out["prompt_tokens"]["mean"] == 150
    assert out["output_tokens_per_s"] == 10.0
    assert abs(out["chargeback_cost_per_1000_items"] - 2.0) < 1e-9
