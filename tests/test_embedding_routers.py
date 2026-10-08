from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from first_reply import embed, routing
from first_reply.routing import base


def _fake_cache(df: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Embeddings that separate queues cleanly, stored through the real cache format."""
    monkeypatch.setattr(embed, "CACHE", tmp_path / "cache")
    rng = np.random.default_rng(0)
    centres = {q: rng.normal(size=16) for q in df.queue.unique()}
    vectors = np.stack([centres[q] + 0.3 * rng.normal(size=16) for q in df.queue])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    path = embed._path("e5-small", "tickets")
    path.parent.mkdir(parents=True)
    np.savez(path, ids=df.id.to_numpy(dtype=str), vectors=vectors.astype(np.float32))


def _tickets(n: int = 600) -> pd.DataFrame:
    queues = ["Billing", "Technical", "HR"]
    splits = ["train"] * 7 + ["dev"] + ["test"] * 2
    return pd.DataFrame(
        {
            "id": [f"t{i}" for i in range(n)],
            "lang": ["en", "de"] * (n // 2),
            "queue": [queues[i % 3] for i in range(n)],
            "priority": ["low", "high"] * (n // 2),
            "type": ["Incident", "Request"] * (n // 2),
            "tags": [[queues[i % 3]] for i in range(n)],
            "split": [splits[i % 10] for i in range(n)],
        }
    )


@pytest.mark.parametrize("method", ["e5-small_lr", "e5-small_knn"])
def test_embedding_routers(method: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    df = _tickets()
    _fake_cache(df, tmp_path, monkeypatch)
    monkeypatch.setattr(base, "RESULTS", tmp_path / "results")
    report = base.evaluate(routing.make(method), df, out_dir=tmp_path / "runs", tag_min_count=10)
    assert report["targets"]["queue"]["macro_f1"] > 0.95
    assert report["tags"]["micro_f1"] > 0.9


def test_cache_load_keeps_order_and_rejects_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    df = _tickets(30)
    _fake_cache(df, tmp_path, monkeypatch)
    v = embed.load("e5-small", "tickets", ["t5", "t1"])
    full = embed.load("e5-small", "tickets", df.id.tolist())
    assert np.allclose(v, full[[5, 1]])
    with pytest.raises(KeyError):
        embed.load("e5-small", "tickets", ["nope"])


def test_shards_partition_the_ids() -> None:
    ids = [f"t{i}" for i in range(500)]
    parts = [embed.shard(ids, i, 4) for i in range(4)]
    assert sorted(i for p in parts for i in p) == sorted(ids)
    assert all(len(p) > 80 for p in parts)
