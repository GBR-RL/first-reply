"""Cross-encoder reranking of the top retrieved chunks."""

from __future__ import annotations

import functools
from dataclasses import replace
from typing import Any

from first_reply.kb.retrieve import Hit

RERANKERS = {
    "minilm": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "bge-reranker": "BAAI/bge-reranker-v2-m3",
}
DEPTH = 20


@functools.cache
def _model(key: str) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKERS[key], device="cpu", max_length=512)


def rerank(query: str, hits: list[Hit], key: str, depth: int = DEPTH) -> list[Hit]:
    """Rescore the first `depth` hits with a cross-encoder; the rest keep their order after."""
    head, tail = hits[:depth], hits[depth:]
    if not head:
        return hits
    scores = _model(key).predict([(query, h.text) for h in head], batch_size=8)
    ranked = sorted(zip(head, scores, strict=True), key=lambda p: -float(p[1]))
    return [replace(h, score=float(s)) for h, s in ranked] + tail
