"""Named text collections that can be embedded: (ids, texts, query-or-passage)."""

from __future__ import annotations


def load(corpus: str) -> tuple[list[str], list[str], str]:
    if corpus == "tickets":
        from first_reply.data import tickets

        df = tickets.load()
        return df.id.tolist(), df.text.tolist(), "query"
    raise KeyError(f"unknown corpus {corpus!r}")
