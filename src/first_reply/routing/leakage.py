"""How much would a split drawn *before* de-duplication flatter the router?

v5 repeats 8,399 v4 tickets verbatim. This draws the same 70/10/20 split from the merged
releases without removing those repeats and scores the TF-IDF baseline on both versions.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from first_reply.config import RAW, SEED
from first_reply.data import tickets
from first_reply.eval.metrics import macro_f1
from first_reply.routing.targets import TARGETS
from first_reply.routing.tfidf import TfidfRouter


def _score(df: pd.DataFrame) -> dict[str, float]:
    train, test = df[df.split == "train"], df[df.split == "test"]
    router = TfidfRouter()
    router.fit(train, [])
    pred = router.predict(test)
    return {t: macro_f1(test[t].to_numpy(), pred.targets[t].predict()) for t in TARGETS}


def check(seed: int = SEED) -> dict[str, Any]:
    raw = tickets.load_raw(RAW)
    deduped = tickets.assign_splits(tickets.clean(raw), seed)
    # Same cleaning, but nothing is dropped as a duplicate.
    leaky = tickets.clean(raw, dedupe=False)
    leaky = tickets.assign_splits(leaky, seed)
    shared = len(set(leaky[leaky.split == "test"].body) & set(leaky[leaky.split == "train"].body))
    return {
        "test_tickets_also_in_train": shared,
        "test_size": int((leaky.split == "test").sum()),
        "macro_f1_with_duplicates": _score(leaky),
        "macro_f1_deduplicated": _score(deduped),
    }
