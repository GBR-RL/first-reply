"""Document-level retrieval metrics for the TechQA questions."""

from __future__ import annotations

import json
import math
import time
from typing import Any

import numpy as np
import pandas as pd

from first_reply import embed
from first_reply.config import ROOT, RUNS
from first_reply.data import techqa
from first_reply.kb.retrieve import Retriever, doc_ranking
from first_reply.routing.base import rounded

RESULTS = ROOT / "docs" / "results" / "retrieval"
KS = (1, 3, 5, 10)


def metrics(ranked: list[str], gold: list[str]) -> dict[str, float]:
    gold_set = set(gold)
    out = {f"recall@{k}": len(gold_set & set(ranked[:k])) / len(gold_set) for k in KS}
    first = next((i for i, d in enumerate(ranked[:10]) if d in gold_set), None)
    out["mrr@10"] = 0.0 if first is None else 1.0 / (first + 1)
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:10]) if d in gold_set)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold_set), 10)))
    out["ndcg@10"] = dcg / idcg
    return out


def _ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n_boot)]
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def run_name(chunker: str, mode: str, model: str | None, suffix: str = "") -> str:
    parts = [chunker, mode] + ([model] if model and mode != "bm25" else [])
    return "_".join(parts) + suffix


def evaluate(
    chunker: str,
    mode: str,
    model: str | None = None,
    *,
    splits: tuple[str, ...] = ("test",),
    limit: int = 50,
    role: str | None = None,
    questions: pd.DataFrame | None = None,
    query_corpus: str = "kb-questions",
    suffix: str = "",
) -> dict[str, Any]:
    q = questions if questions is not None else techqa.load_questions()
    q = q[q.split.isin(splits) & q.answerable].reset_index(drop=True)
    vectors = None
    if mode != "bm25":
        assert model is not None
        vectors = embed.load(model, query_corpus, q.qid.tolist())
    retriever = Retriever(chunker, model if mode != "bm25" else None)
    rows = []
    t0 = time.perf_counter()
    for i, r in enumerate(q.to_dict("records")):
        dense = None if vectors is None else vectors[i].tolist()
        hits = retriever.search(str(r["question"]), mode=mode, dense=dense, limit=limit, role=role)
        ranked = doc_ranking(hits)
        gold = [str(g) for g in r["gold_doc_ids"]]
        rows.append({"qid": r["qid"], "ranked": ranked[:10], **metrics(ranked, gold)})
    ms = 1000 * (time.perf_counter() - t0) / max(len(q), 1)
    df = pd.DataFrame(rows)
    name = run_name(chunker, mode, model, suffix)
    summary: dict[str, Any] = {
        "run": name,
        "chunker": chunker,
        "mode": mode,
        "model": model,
        "splits": list(splits),
        "questions": len(df),
        "search_ms_per_query": ms,
    }
    for col in [c for c in df.columns if "@" in c]:
        summary[col] = float(df[col].mean())
    for col in ("recall@5", "ndcg@10"):
        summary[f"{col}_ci"] = _ci(df[col].to_numpy())
    out = RUNS / "retrieval"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"{name}.parquet", index=False)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(rounded(summary), indent=2) + "\n")
    return summary
