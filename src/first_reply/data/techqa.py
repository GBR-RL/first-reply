"""Knowledge base from RAGBench TechQA: IBM support documents and real forum questions.

Each question ships with five candidate documents. The documents of all questions are pooled
into one corpus, so for retrieval every other question's documents act as distractors. A
document counts as relevant to a question when RAGBench marks at least one of its sentences as
relevant. Those labels were produced by GPT-4, not by people. Questions with no relevant
document are kept: the right behaviour for them is to decline and hand over to a person.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pandas as pd

from first_reply.config import KB_DOCS, KB_QUESTIONS, KB_RESPONSES, RAW

SPLITS = {"train": "train", "validation": "dev", "test": "test"}
_SPLIT_ORDER = {"train": 0, "dev": 1, "test": 2}

# Ordered: the first pattern that matches the start of a document decides its type.
_DOC_TYPES = (
    ("security_bulletin", re.compile(r"SECURITY BULLETIN")),
    ("troubleshooting", re.compile(r"TECHNOTE \(TROUBLESHOOTING\)")),
    ("faq", re.compile(r"TECHNOTE \(FAQ\)")),
    ("alert", re.compile(r"FLASH \(ALERT\)")),
    ("apar", re.compile(r"\bAPAR\b|A FIX IS AVAILABLE|FIXES ARE AVAILABLE|LINKS TO FIXES")),
    ("documentation", re.compile(r"DOWNLOADABLE FILES|PRODUCT DOCUMENTATION|PRODUCT README")),
)
_HEAD_CHARS = 400


def doc_id(text: str) -> str:
    return "d" + hashlib.sha1(text.encode()).hexdigest()[:12]


def doc_type(text: str) -> str:
    head = text[:_HEAD_CHARS]
    for name, pattern in _DOC_TYPES:
        if pattern.search(head):
            return name
    return "other"


def _keys(value: Any) -> list[str]:
    return [str(k) for k in value] if value is not None else []


def _relevant_positions(keys: list[str]) -> set[int]:
    """Sentence keys look like '0b' or '3d': document position, then sentence letter."""
    return {int(m.group(1)) for k in keys if (m := re.match(r"(\d+)", k))}


def load_raw(raw: Path = RAW) -> pd.DataFrame:
    frames = []
    for name, split in SPLITS.items():
        df = pd.read_parquet(raw / f"techqa_{name}.parquet")
        df["split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_docs(raw_df: pd.DataFrame) -> pd.DataFrame:
    texts: dict[str, str] = {}
    for documents in raw_df.documents:
        for text in documents:
            texts.setdefault(doc_id(str(text)), str(text))
    docs = pd.DataFrame({"doc_id": list(texts), "text": list(texts.values())})
    docs["doc_type"] = docs.text.map(doc_type)
    docs["chars"] = docs.text.str.len()
    return docs.sort_values("doc_id").reset_index(drop=True)


def build_questions(raw_df: pd.DataFrame) -> pd.DataFrame:
    """One row per unique question.

    RAGBench lists every question twice, with the same documents and two different generated
    responses. Questions repeated across splits are kept in the earliest split only.
    """
    rows = []
    for r in raw_df.to_dict("records"):
        candidates = [doc_id(str(t)) for t in r["documents"]]
        relevant = _relevant_positions(_keys(r["all_relevant_sentence_keys"]))
        gold = [candidates[i] for i in sorted(relevant) if i < len(candidates)]
        rows.append(
            {
                "qid": str(r["id"]),
                "split": r["split"],
                "question": str(r["question"]).strip(),
                "candidate_doc_ids": candidates,
                "gold_doc_ids": gold,
                "answerable": bool(gold),
            }
        )
    q = pd.DataFrame(rows)
    q["_o"] = q.split.map(_SPLIT_ORDER)
    q = q.sort_values(["_o", "qid"]).drop_duplicates("question", keep="first")
    return q.drop(columns="_o").reset_index(drop=True)


def build_responses(raw_df: pd.DataFrame, questions: pd.DataFrame) -> pd.DataFrame:
    """RAGBench's reference responses with GPT-4's support labels, one row per response.

    Used only to validate our own faithfulness judge: a response is "supported" when every
    one of its sentences is backed by the documents.
    """
    keep = dict(zip(questions.question, questions.qid, strict=True))
    rows = []
    for r in raw_df.to_dict("records"):
        question = str(r["question"]).strip()
        if keep.get(question) != str(r["id"]):
            continue
        sentences = [(str(k), str(t).strip()) for k, t in r["response_sentences"]]
        rows.append(
            {
                "qid": str(r["id"]),
                "split": r["split"],
                "generator": str(r["generation_model_name"]),
                "response": str(r["response"]).strip(),
                "sentence_keys": [k for k, _ in sentences],
                "sentences": [t for _, t in sentences],
                "unsupported_keys": _keys(r["unsupported_response_sentence_keys"]),
                "supported": bool(r["adherence_score"]),
            }
        )
    return pd.DataFrame(rows)


def build(
    raw: Path = RAW,
    docs_out: Path = KB_DOCS,
    questions_out: Path = KB_QUESTIONS,
    responses_out: Path = KB_RESPONSES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_df = load_raw(raw)
    docs, questions = build_docs(raw_df), build_questions(raw_df)
    responses = build_responses(raw_df, questions)
    docs_out.parent.mkdir(parents=True, exist_ok=True)
    docs.to_parquet(docs_out, index=False)
    questions.to_parquet(questions_out, index=False)
    responses.to_parquet(responses_out, index=False)
    return docs, questions, responses


def load_docs(path: Path = KB_DOCS) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_questions(path: Path = KB_QUESTIONS) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_responses(path: Path = KB_RESPONSES) -> pd.DataFrame:
    return pd.read_parquet(path)
