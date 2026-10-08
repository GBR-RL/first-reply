"""Sentence embeddings with an on-disk cache keyed by model and text id.

Encoding is the slow step on a CPU (multilingual-e5-small: ~30 tickets/s on a 6-core laptop;
bge-m3: ~3/s), so vectors are computed once, optionally split into shards that run as parallel
CI jobs, and reused by every experiment.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from first_reply.config import DATA

CACHE = DATA / "cache" / "embeddings"


@dataclass(frozen=True)
class EmbeddingModel:
    key: str
    hf_name: str
    query_prefix: str = ""
    passage_prefix: str = ""
    max_seq_length: int = 512


MODELS = {
    m.key: m
    for m in (
        EmbeddingModel("e5-small", "intfloat/multilingual-e5-small", "query: ", "passage: "),
        EmbeddingModel("bge-m3", "BAAI/bge-m3"),
    )
}


def _load(model: EmbeddingModel):  # type: ignore[no-untyped-def]
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(model.hf_name, device="cpu")
    st.max_seq_length = model.max_seq_length
    return st


def encode(
    texts: list[str], model_key: str, *, kind: str = "query", batch_size: int = 32
) -> NDArray[np.float32]:
    """L2-normalised embeddings. `kind` picks the model's query or passage prefix."""
    model = MODELS[model_key]
    prefix = model.query_prefix if kind == "query" else model.passage_prefix
    st = _load(model)
    vectors = st.encode(
        [prefix + t for t in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 1000,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def shard(ids: list[str], index: int, count: int) -> list[str]:
    """Deterministic shard of ids by hash, so shards can be computed independently."""
    return [i for i in ids if int(hashlib.md5(i.encode()).hexdigest(), 16) % count == index]


def _path(model_key: str, corpus: str, part: str = "") -> Path:
    return CACHE / model_key / f"{corpus}{part}.npz"


def compute(
    ids: list[str],
    texts: list[str],
    model_key: str,
    corpus: str,
    *,
    kind: str = "query",
    shard_index: int = 0,
    shard_count: int = 1,
) -> Path:
    """Encode one shard of a corpus and store it as ids + vectors."""
    wanted = set(shard(ids, shard_index, shard_count))
    pairs = [(i, t) for i, t in zip(ids, texts, strict=True) if i in wanted]
    vectors = encode([t for _, t in pairs], model_key, kind=kind)
    part = "" if shard_count == 1 else f".part{shard_index}of{shard_count}"
    path = _path(model_key, corpus, part)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.asarray([i for i, _ in pairs], dtype=str), vectors=vectors)
    return path


def merge_shards(model_key: str, corpus: str) -> Path:
    parts = sorted(_path(model_key, corpus).parent.glob(f"{corpus}.part*of*.npz"))
    if not parts:
        raise FileNotFoundError(f"no shards for {model_key}/{corpus}")
    loaded = [np.load(p) for p in parts]
    path = _path(model_key, corpus)
    np.savez(
        path,
        ids=np.concatenate([d["ids"] for d in loaded]),
        vectors=np.concatenate([d["vectors"] for d in loaded]),
    )
    return path


def load(model_key: str, corpus: str, ids: list[str]) -> NDArray[np.float32]:
    """Vectors for `ids`, in that order. Fails if any id is missing from the cache."""
    data = np.load(_path(model_key, corpus))
    index = {str(i): row for row, i in enumerate(data["ids"])}
    missing = [i for i in ids if i not in index]
    if missing:
        raise KeyError(f"{len(missing)} ids missing from {model_key}/{corpus} cache")
    vectors: NDArray[np.float32] = data["vectors"][[index[i] for i in ids]]
    return vectors
