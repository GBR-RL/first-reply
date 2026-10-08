"""Ticket set: merge the two labelled releases, remove duplicates, split.

Duplicates are removed on the normalised body *before* splitting. v5 repeats 8,399 v4
tickets verbatim with identical labels, so a split taken before de-duplication would put the
same ticket in train and test and inflate every routing score.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from first_reply.config import RAW, SEED, TICKETS

LABELS = ("queue", "priority", "type")
TAG_COLUMNS = tuple(f"tag_{i}" for i in range(1, 9))
SPLIT_SIZES = {"train": 0.7, "dev": 0.1, "test": 0.2}
# CPU-bound LLM runs use fixed, stratified subsets of dev and test.
LLM_SUBSET = {"dev": 300, "test": 1000}

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Key used for de-duplication: NFKC, case-folded, whitespace collapsed."""
    return _WS.sub(" ", unicodedata.normalize("NFKC", text)).strip().casefold()


def ticket_id(body: str) -> str:
    return "t" + hashlib.sha1(normalize(body).encode()).hexdigest()[:12]


def load_raw(raw: Path = RAW) -> pd.DataFrame:
    frames = []
    for release in ("v5", "v4"):  # v5 first: it wins when a ticket appears in both
        df = pd.read_csv(raw / f"tickets_{release}.csv")
        df["release"] = release
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame, dedupe: bool = True) -> pd.DataFrame:
    df = df.dropna(subset=["body", "answer", *LABELS]).copy()
    for col in ("subject", "body", "answer"):
        df[col] = df[col].fillna("").astype(str).str.strip()
    df = df[df.body.str.len() > 0]
    df["language"] = df.language.str.lower()
    df["tags"] = [
        sorted({str(t).strip() for t in row if isinstance(t, str) and t.strip()})
        for row in df[list(TAG_COLUMNS)].itertuples(index=False)
    ]
    df["key"] = df.body.map(normalize)
    if dedupe:
        df = df.drop_duplicates("key", keep="first")
        df["id"] = df.body.map(ticket_id)
    else:
        df["id"] = [f"{ticket_id(b)}-{i}" for i, b in enumerate(df.body)]
    df["text"] = np.where(df.subject.str.len() > 0, df.subject + "\n\n" + df.body, df.body)
    cols = ["id", "release", "language", "subject", "body", "text", "answer", *LABELS, "tags"]
    return df[cols].reset_index(drop=True)


def _strata(df: pd.DataFrame) -> pd.Series:
    return df.queue + "|" + df.language


def assign_splits(
    df: pd.DataFrame, seed: int = SEED, llm_subset: dict[str, int] | None = None
) -> pd.DataFrame:
    """70/10/20 split stratified on queue x language, plus fixed LLM subsets."""
    df = df.copy()
    rest, test = train_test_split(
        df.index, test_size=SPLIT_SIZES["test"], random_state=seed, stratify=_strata(df)
    )
    dev_share = SPLIT_SIZES["dev"] / (1 - SPLIT_SIZES["test"])
    train, dev = train_test_split(
        rest, test_size=dev_share, random_state=seed, stratify=_strata(df.loc[rest])
    )
    df["split"] = ""
    df.loc[train, "split"] = "train"
    df.loc[dev, "split"] = "dev"
    df.loc[test, "split"] = "test"
    df["llm_subset"] = False
    for split, n in (llm_subset or LLM_SUBSET).items():
        part = df[df.split == split]
        _, pick = train_test_split(
            part.index, test_size=n, random_state=seed, stratify=_strata(part)
        )
        df.loc[pick, "llm_subset"] = True
    return df


def build(raw: Path = RAW, out: Path = TICKETS, seed: int = SEED) -> pd.DataFrame:
    df = assign_splits(clean(load_raw(raw)), seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


def load(path: Path = TICKETS) -> pd.DataFrame:
    return pd.read_parquet(path)
