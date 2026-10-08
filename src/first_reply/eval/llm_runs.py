"""Sharded LLM runs: ticket triage, grounded drafting and question translation.

Each shard writes one JSON line per item, so a run can be split over parallel CI runners (each
with its own llama.cpp server behind the gateway) and resumed: items already in the output file
are skipped. `merge` joins the shard files.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from first_reply import embed
from first_reply.config import RUNS
from first_reply.data import techqa, tickets
from first_reply.llm import answer, judge, triage
from first_reply.llm.client import LLM
from first_reply.routing.targets import tag_vocab

OUT = RUNS / "llm"
FEW_SHOT_K = 5


def run_dir(name: str) -> Path:
    return OUT / name


def _done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines()}


def _loop(
    items: Iterable[tuple[str, Any]],
    fn: Callable[[Any], dict[str, Any]],
    path: Path,
    shard: int,
    shards: int,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    items = list(items)
    ids = embed.shard([i for i, _ in items], shard, shards)
    wanted, done = set(ids), _done(path)
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for item_id, item in items:
            if item_id not in wanted or item_id in done:
                continue
            t0 = time.perf_counter()
            try:
                row = {"id": item_id, **fn(item)}
            except Exception as err:  # keep going; failures are counted in the report
                row = {"id": item_id, "error": f"{type(err).__name__}: {err}"[:500]}
            row["wall_s"] = round(time.perf_counter() - t0, 2)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
    return n


def _usage(c: Any) -> dict[str, Any]:
    return {
        "prompt_tokens": c.prompt_tokens,
        "completion_tokens": c.completion_tokens,
        "latency_s": round(c.latency_s, 2),
        "cost": c.cost,
    }


def triage_run(
    llm: LLM, name: str, *, few_shot: bool, split: str = "test", shard: int = 0, shards: int = 1
) -> Path:
    df = tickets.load()
    train = df[df.split == "train"].reset_index(drop=True)
    tags = tag_vocab(train)
    items = df[(df.split == split) & df.llm_subset].reset_index(drop=True)
    neighbours: dict[str, list[dict[str, Any]]] = {}
    if few_shot:
        x_train = embed.load("bge-m3", "tickets", train.id.tolist())
        x_items = embed.load("bge-m3", "tickets", items.id.tolist())
        top = np.argsort(-(x_items @ x_train.T), axis=1)[:, :FEW_SHOT_K]
        records = train.to_dict("records")
        for row, item_id in enumerate(items.id):
            neighbours[item_id] = [records[j] for j in top[row]]

    def fn(t: dict[str, Any]) -> dict[str, Any]:
        out, c = triage.triage(llm, t["text"], tags, neighbours.get(t["id"]))
        return {"pred": out, **_usage(c)}

    path = run_dir(name) / f"shard{shard}of{shards}.jsonl"
    _loop(((r["id"], r) for r in items.to_dict("records")), fn, path, shard, shards)
    return path


def draft_run(
    llm: LLM,
    name: str,
    *,
    retrieve: Callable[[dict[str, Any]], list[Any]],
    splits: tuple[str, ...] = ("test",),
    questions: pd.DataFrame | None = None,
    shard: int = 0,
    shards: int = 1,
) -> Path:
    q = questions if questions is not None else techqa.load_questions()
    q = q[q.split.isin(splits)]

    def fn(r: dict[str, Any]) -> dict[str, Any]:
        hits = retrieve(r)
        out, c = answer.draft(llm, r["question"], hits)
        return {
            **out,
            "retrieved_doc_ids": [h.doc_id for h in hits],
            "retrieved_chunk_ids": [h.chunk_id for h in hits],
            **_usage(c),
        }

    path = run_dir(name) / f"shard{shard}of{shards}.jsonl"
    _loop(((r["qid"], r) for r in q.to_dict("records")), fn, path, shard, shards)
    return path


TRANSLATE = """Translate the user's message into German as a native speaker at a German company
would write it to IT support. Keep product names, error codes, commands and version numbers
exactly as they are. Reply with the translation only."""


def translate_run(llm: LLM, name: str, *, shard: int = 0, shards: int = 1) -> Path:
    q = techqa.load_questions()

    def fn(r: dict[str, Any]) -> dict[str, Any]:
        c = llm.chat(
            [{"role": "system", "content": TRANSLATE}, {"role": "user", "content": r["question"]}],
            max_tokens=600,
        )
        return {"question_de": c.text.strip(), **_usage(c)}

    path = run_dir(name) / f"shard{shard}of{shards}.jsonl"
    _loop(((r["qid"], r) for r in q.to_dict("records")), fn, path, shard, shards)
    return path


def merge(name: str) -> pd.DataFrame:
    rows = [
        json.loads(line)
        for p in sorted(run_dir(name).glob("shard*.jsonl"))
        for line in p.read_text(encoding="utf-8").splitlines()
    ]
    df = pd.DataFrame(rows).drop_duplicates("id", keep="last")
    df.to_json(run_dir(name) / "merged.jsonl", orient="records", lines=True, force_ascii=False)
    return df


JUDGE_SAMPLE = 300


def judge_ragbench_run(llm: LLM, name: str, *, shard: int = 0, shards: int = 1) -> Path:
    """Judge a balanced sample of RAGBench reference responses against their documents."""
    r = techqa.load_responses()
    half = JUDGE_SAMPLE // 2
    sample = pd.concat(
        [r[r.supported].sample(half, random_state=7), r[~r.supported].sample(half, random_state=7)]
    )
    q = techqa.load_questions().set_index("qid")
    docs = techqa.load_docs().set_index("doc_id").text

    def fn(row: dict[str, Any]) -> dict[str, Any]:
        question = q.loc[row["qid"]]
        texts = [str(docs[d]) for d in question.candidate_doc_ids]
        out, c = judge.judge(llm, str(question.question), row["response"], texts)
        return {"qid": row["qid"], "label_supported": bool(row["supported"]), **out, **_usage(c)}

    items = ((f"{x['qid']}|{x['generator']}", x) for x in sample.to_dict("records"))
    path = run_dir(name) / f"shard{shard}of{shards}.jsonl"
    _loop(items, fn, path, shard, shards)
    return path


def judge_drafts_run(
    llm: LLM, name: str, drafts: pd.DataFrame, chunks: pd.DataFrame, *, shard: int = 0,
    shards: int = 1,
) -> Path:  # fmt: skip
    """Judge our own drafts against exactly the excerpts the drafting model was shown."""
    text = dict(zip(chunks.chunk_id, chunks.text, strict=True))
    q = techqa.load_questions().set_index("qid").question
    answered = drafts[drafts["answerable"].fillna(False).astype(bool)]

    def fn(row: dict[str, Any]) -> dict[str, Any]:
        texts = [text[c] for c in row["retrieved_chunk_ids"]]
        out, c = judge.judge(llm, str(q[row["id"]]), row["reply"], texts)
        return {**out, **_usage(c)}

    path = run_dir(name) / f"shard{shard}of{shards}.jsonl"
    _loop(((r["id"], r) for r in answered.to_dict("records")), fn, path, shard, shards)
    return path
