"""How much does the choice of split flatter a router?

Scores the TF-IDF baseline and the bge-m3 nearest-neighbour router (queue, priority, type
macro-F1 on test) under five splits:

- random split drawn before removing exact duplicates (v5 repeats 8,399 v4 tickets),
- random split after de-duplication,
- splits over paraphrase families at three similarity thresholds (0.90 is the one used).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from first_reply import embed
from first_reply.config import RAW, SEED
from first_reply.data import families, tickets
from first_reply.eval.metrics import macro_f1
from first_reply.routing.base import Router
from first_reply.routing.embedding import KnnRouter
from first_reply.routing.targets import TARGETS
from first_reply.routing.tfidf import TfidfRouter

THRESHOLDS = (0.95, 0.93, 0.90)


def _score(router: Router, df: pd.DataFrame) -> dict[str, float]:
    train, test = df[df.split == "train"], df[df.split == "test"]
    router.fit(train, [])
    pred = router.predict(test)
    return {t: macro_f1(test[t].to_numpy(), pred.targets[t].predict()) for t in TARGETS}


def _twins(df: pd.DataFrame, x: np.ndarray, index: dict[str, int]) -> float:
    """Share of test tickets with a training ticket at cosine >= 0.95."""
    tr = x[[index[i] for i in df[df.split == "train"].id]]
    te = x[[index[i] for i in df[df.split == "test"].id]]
    best = np.concatenate([(te[s : s + 2000] @ tr.T).max(1) for s in range(0, len(te), 2000)])
    return float((best >= 0.95).mean())


def check(seed: int = SEED) -> dict[str, Any]:
    raw = tickets.load_raw(RAW)
    clean = tickets.clean(raw)
    x = embed.load("bge-m3", "tickets", clean.id.tolist())
    index = {i: n for n, i in enumerate(clean.id)}
    nbr, sim = families.nearest(x)

    splits: dict[str, pd.DataFrame] = {
        "random, with exact duplicates": tickets.assign_splits(tickets.clean(raw, dedupe=False)),
        "random, de-duplicated": tickets.assign_splits(clean, seed),
    }
    for t in THRESHOLDS:
        groups = pd.Series(families.components(nbr, sim, t), index=clean.index)
        splits[f"families at cosine {t:.2f}"] = tickets.assign_splits(clean, seed, groups=groups)

    out: dict[str, Any] = {}
    for name, df in splits.items():
        row: dict[str, Any] = {"test_size": int((df.split == "test").sum())}
        row["tfidf_lr"] = _score(TfidfRouter(), df)
        if df.id.isin(index).all():
            row["bge-m3_knn"] = _score(KnnRouter("bge-m3"), df)
            row["test_with_train_twin_ge_0.95"] = _twins(df, x, index)
        out[name] = row
    return out
