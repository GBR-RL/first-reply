"""Search the knowledge base: BM25, dense, or both fused with reciprocal rank fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from first_reply.kb import sparse
from first_reply.kb.access import role_filter
from first_reply.kb.index import SPARSE, collection, ensure

MODES = ("bm25", "dense", "hybrid")


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_id: str
    doc_type: str
    tier: int
    section: str
    text: str
    score: float


def _hit(point: Any) -> Hit:
    p = point.payload
    return Hit(
        p["chunk_id"], p["doc_id"], p["doc_type"], p["tier"], p["section"], p["text"], point.score
    )


class Retriever:
    def __init__(
        self,
        chunker: str = "sections",
        dense_model: str | None = None,
        qc: QdrantClient | None = None,
    ) -> None:
        self.collection = collection(chunker)
        self.dense_model = dense_model
        self.qc = ensure(chunker, qc)

    def search(
        self,
        query: str,
        *,
        mode: str = "hybrid",
        dense: list[float] | None = None,
        limit: int = 50,
        role: str | None = None,
    ) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode != "bm25" and (dense is None or self.dense_model is None):
            raise ValueError(f"mode {mode!r} needs a dense model and a query vector")
        flt = role_filter(role) if role else None
        idx, val = sparse.query_vector(query)
        sparse_q = qm.SparseVector(indices=idx, values=val)
        if mode == "bm25":
            res = self.qc.query_points(
                self.collection, query=sparse_q, using=SPARSE, query_filter=flt, limit=limit
            )
        elif mode == "dense":
            res = self.qc.query_points(
                self.collection, query=dense, using=self.dense_model, query_filter=flt, limit=limit
            )
        else:
            res = self.qc.query_points(
                self.collection,
                prefetch=[
                    qm.Prefetch(query=sparse_q, using=SPARSE, filter=flt, limit=limit),
                    qm.Prefetch(query=dense, using=self.dense_model, filter=flt, limit=limit),
                ],
                query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                query_filter=flt,
                limit=limit,
            )
        return [_hit(p) for p in res.points]


def doc_ranking(hits: list[Hit]) -> list[str]:
    """Documents in the order of their best chunk."""
    seen: dict[str, None] = {}
    for h in hits:
        seen.setdefault(h.doc_id, None)
    return list(seen)
