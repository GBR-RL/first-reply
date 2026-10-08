"""Qdrant index over knowledge-base chunks: BM25 sparse vectors plus any cached dense vectors.

The same collection serves the experiments and the service. With `QDRANT_URL` set it lives in a
Qdrant server (Docker Compose). Otherwise it is built in memory with Qdrant's embedded mode,
which takes seconds for this corpus; the embedded mode's on-disk storage commits every point
to SQLite separately and was ~500x slower to fill.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from first_reply import embed
from first_reply.config import DATA
from first_reply.kb import sparse
from first_reply.kb.access import tier

SPARSE = "bm25"


def collection(chunker: str) -> str:
    return f"kb_{chunker}"


@functools.cache
def _memory() -> QdrantClient:
    return QdrantClient(":memory:")


def client() -> QdrantClient:
    """The Qdrant server from QDRANT_URL, or one shared in-memory instance per process."""
    url = os.environ.get("QDRANT_URL")
    return QdrantClient(url=url, timeout=120) if url else _memory()


def ensure(chunker: str, qc: QdrantClient | None = None) -> QdrantClient:
    """Return a client whose collection for `chunker` exists, building it if needed."""
    qc = qc or client()
    if not qc.collection_exists(collection(chunker)):
        build(chunker, qc)
    return qc


def chunks_path(chunker: str) -> Path:
    return DATA / "processed" / f"kb_chunks_{chunker}.parquet"


def load_chunks(chunker: str) -> pd.DataFrame:
    return pd.read_parquet(chunks_path(chunker))


def dense_models(chunker: str) -> list[str]:
    """Embedding models whose vectors for this chunk corpus are in the cache."""
    corpus = f"kb-chunks-{chunker}"
    return [m for m in embed.MODELS if (embed.CACHE / m / f"{corpus}.npz").exists()]


def build(chunker: str, qc: QdrantClient | None = None, batch: int = 256) -> int:
    qc = qc or client()
    chunks = load_chunks(chunker)
    name = collection(chunker)
    models = dense_models(chunker)
    vectors = {m: embed.load(m, f"kb-chunks-{chunker}", chunks.chunk_id.tolist()) for m in models}
    if qc.collection_exists(name):
        qc.delete_collection(name)
    qc.create_collection(
        name,
        vectors_config={
            m: qm.VectorParams(size=v.shape[1], distance=qm.Distance.COSINE)
            for m, v in vectors.items()
        },
        sparse_vectors_config={SPARSE: qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
    )
    bm25 = sparse.corpus_vectors(chunks.text.tolist())
    records = chunks.to_dict("records")
    for start in range(0, len(chunks), batch):
        points = []
        for row in range(start, min(start + batch, len(chunks))):
            c = records[row]
            idx, val = bm25[row]
            vec: dict[str, list[float] | qm.SparseVector] = {
                SPARSE: qm.SparseVector(indices=idx, values=val)
            }
            for m, v in vectors.items():
                vec[m] = v[row].tolist()
            points.append(
                qm.PointStruct(
                    id=row,
                    vector=vec,  # type: ignore[arg-type]
                    payload={
                        "chunk_id": c["chunk_id"],
                        "doc_id": c["doc_id"],
                        "doc_type": c["doc_type"],
                        "tier": tier(c["doc_type"]),
                        "section": c["section"],
                        "text": c["text"],
                    },
                )
            )
        qc.upsert(name, points)
    if os.environ.get("QDRANT_URL"):  # payload indexes only exist on a server
        qc.create_payload_index(name, "tier", qm.PayloadSchemaType.INTEGER)
    return len(chunks)
