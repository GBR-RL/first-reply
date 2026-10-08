"""Named text collections that can be embedded: (ids, texts, query-or-passage)."""

from __future__ import annotations


def load(corpus: str) -> tuple[list[str], list[str], str]:
    if corpus == "tickets":
        from first_reply.data import tickets

        df = tickets.load()
        return df.id.tolist(), df.text.tolist(), "query"
    if corpus == "kb-questions":
        from first_reply.data import techqa

        q = techqa.load_questions()
        return q.qid.tolist(), q.question.tolist(), "query"
    if corpus.startswith("kb-chunks-"):
        from first_reply.kb.index import load_chunks

        chunks = load_chunks(corpus.removeprefix("kb-chunks-"))
        return chunks.chunk_id.tolist(), chunks.text.tolist(), "passage"
    raise KeyError(f"unknown corpus {corpus!r}")
