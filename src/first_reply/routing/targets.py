"""What the router predicts: three single-label targets and a fixed tag vocabulary."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from numpy.typing import NDArray

TARGETS = ("queue", "priority", "type")
# Tags seen at least this often in train form the vocabulary (the long tail is mostly
# one-off free text). Chosen on train only, so dev and test do not shape it.
TAG_MIN_COUNT = 500


def tag_vocab(train: pd.DataFrame, min_count: int = TAG_MIN_COUNT) -> list[str]:
    counts = Counter(t for tags in train.tags for t in tags)
    return sorted(t for t, c in counts.items() if c >= min_count)


def tag_matrix(df: pd.DataFrame, vocab: list[str]) -> NDArray[np.int8]:
    index = {t: i for i, t in enumerate(vocab)}
    m = np.zeros((len(df), len(vocab)), dtype=np.int8)
    for row, tags in enumerate(df.tags):
        for t in tags:
            if t in index:
                m[row, index[t]] = 1
    return m
