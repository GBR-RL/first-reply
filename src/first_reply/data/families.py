"""Paraphrase families: tickets generated from the same seed.

The generator wrote several rewordings and translations of each ticket. Exact duplicates are
removed during cleaning, but paraphrases are not, and a random split spreads them across train
and test. A nearest-neighbour router then mostly looks up a test ticket's twin, which says
little about routing a ticket the system has never seen.

A family is a connected component of the graph that links tickets whose bge-m3 embeddings have
cosine similarity >= THRESHOLD among each ticket's 20 nearest neighbours. Pairs down to about
0.90 are still recognisable paraphrases or translations of one another, so 0.90 is used: the
strictest setting, which also groups closely related topics. The split keeps every family in
one of train, dev or test. The assignment ships with the package (ticket_families.csv.gz) so
reproducing the split does not require encoding 40k tickets with bge-m3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

ASSET = Path(__file__).with_name("ticket_families.csv.gz")
THRESHOLD = 0.90
NEIGHBOURS = 20


def nearest(
    x: NDArray[np.float32], k: int = NEIGHBOURS, block: int = 2000
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """k nearest neighbours (cosine, excluding self) of every row of L2-normalised `x`."""
    n = len(x)
    nbr = np.zeros((n, k), dtype=np.int64)
    sim = np.zeros((n, k), dtype=np.float32)
    for start in range(0, n, block):
        s = x[start : start + block] @ x.T
        s[np.arange(len(s)), np.arange(start, start + len(s))] = -1.0
        idx = np.argpartition(-s, k, axis=1)[:, :k]
        nbr[start : start + len(s)] = idx
        sim[start : start + len(s)] = np.take_along_axis(s, idx, axis=1)
    return nbr, sim


def components(
    nbr: NDArray[np.int64], sim: NDArray[np.float32], threshold: float = THRESHOLD
) -> NDArray[np.int64]:
    """Union-find over edges above the threshold; returns a component id per row."""
    parent = np.arange(len(nbr))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    rows, cols = np.nonzero(sim >= threshold)
    for a, b in zip(rows, nbr[rows, cols], strict=True):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in range(len(nbr))])


def assign(ids: list[str], x: NDArray[np.float32], threshold: float = THRESHOLD) -> pd.DataFrame:
    nbr, sim = nearest(x)
    comp = components(nbr, sim, threshold)
    # Stable family names: the smallest ticket id in the family.
    frame = pd.DataFrame({"id": ids, "comp": comp})
    frame["family"] = frame.groupby("comp").id.transform("min")
    return frame[["id", "family"]].sort_values("id").reset_index(drop=True)


def load(path: Path = ASSET) -> pd.DataFrame:
    return pd.read_csv(path)
