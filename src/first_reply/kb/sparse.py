"""BM25 as Qdrant sparse vectors.

Documents store the BM25 term-frequency part, tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl));
Qdrant's IDF modifier multiplies in the inverse document frequency at query time, so a query
vector of ones over its terms scores exactly BM25. Term ids are stable hashes, so no vocabulary
file has to travel with the index.
"""

from __future__ import annotations

import re
import zlib
from collections import Counter

K1 = 1.2
B = 0.75
_TOKEN = re.compile(r"[a-z0-9]+(?:[._][a-z0-9]+)*")
# Small English + German stop list: the knowledge base is English, questions may be German.
STOP = frozenset(
    """a an and are as at be but by can do does for from has have how i if in into is it its
    me my no not of on or our so than that the their then there these they this to was we
    were what when where which while who why will with you your der die das und ist nicht
    ein eine mit von zu den im für auf wie ich es sie wir bei""".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOP]


def term_id(term: str) -> int:
    return zlib.crc32(term.encode()) & 0x7FFFFFFF


def _merge(pairs: dict[int, float]) -> tuple[list[int], list[float]]:
    items = sorted(pairs.items())
    return [i for i, _ in items], [v for _, v in items]


def doc_vector(tokens: list[str], avgdl: float) -> tuple[list[int], list[float]]:
    dl = len(tokens)
    weights: dict[int, float] = {}
    for term, tf in Counter(tokens).items():
        w = tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl / avgdl))
        tid = term_id(term)
        weights[tid] = weights.get(tid, 0.0) + w  # hash collisions just add up
    return _merge(weights)


def query_vector(text: str) -> tuple[list[int], list[float]]:
    return _merge(dict.fromkeys((term_id(t) for t in set(tokenize(text))), 1.0))


def corpus_vectors(texts: list[str]) -> list[tuple[list[int], list[float]]]:
    tokens = [tokenize(t) for t in texts]
    avgdl = sum(map(len, tokens)) / max(len(tokens), 1)
    return [doc_vector(t, avgdl) for t in tokens]
