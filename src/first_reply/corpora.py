"""Named text collections that can be embedded: (ids, texts, query-or-passage)."""

from __future__ import annotations

import pandas as pd


def load(corpus: str) -> tuple[list[str], list[str], str]:
    if corpus == "tickets":
        from first_reply.data import tickets

        df = tickets.load()
        return df.id.tolist(), df.text.tolist(), "query"
    if corpus == "kb-questions":
        from first_reply.data import techqa

        q = techqa.load_questions()
        return q.qid.tolist(), q.question.tolist(), "query"
    if corpus == "kb-questions-de":
        q = german_questions()
        return q.qid.tolist(), q.question.tolist(), "query"
    if corpus.startswith("kb-chunks-"):
        from first_reply.kb.index import load_chunks

        chunks = load_chunks(corpus.removeprefix("kb-chunks-"))
        return chunks.chunk_id.tolist(), chunks.text.tolist(), "passage"
    raise KeyError(f"unknown corpus {corpus!r}")


GERMAN_RUN = "translate-qwen3.5-4b"


def german_questions() -> pd.DataFrame:
    """TechQA questions machine-translated into German (Qwen3.5-4B), same ids and labels."""
    from first_reply.data import techqa
    from first_reply.eval.llm_runs import run_dir

    q = techqa.load_questions()
    de = pd.read_json(run_dir(GERMAN_RUN) / "merged.jsonl", lines=True, dtype={"id": str})
    q = q.merge(de[["id", "question_de"]], left_on="qid", right_on="id", how="inner")
    return q.assign(question=q.question_de).drop(columns=["id", "question_de"])
