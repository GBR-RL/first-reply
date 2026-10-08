"""Draft a reply from retrieved knowledge-base excerpts, with citations, or decline.

The model sees numbered excerpts and must either answer from them, citing excerpt numbers, or
mark the question as not answerable from the knowledge base. A declined question goes to a
person instead of receiving a guessed answer.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from first_reply.kb.retrieve import Hit
from first_reply.llm.client import LLM, Completion

MAX_EXCERPT_WORDS = 350

SYSTEM = """You are a support engineer drafting a reply to a customer. Use only the numbered
knowledge-base excerpts below; do not use outside knowledge.

- If the excerpts answer the question, set "answerable" to true and write a short, practical
  reply (at most 150 words) in the language of the question. Put the excerpt number in square
  brackets after each statement it supports, e.g. [2]. List every cited number in "citations".
- If the excerpts do not contain the answer, set "answerable" to false, leave "reply" empty and
  "citations" empty. Do not guess.

Excerpts:
{excerpts}"""


class Draft(BaseModel):
    answerable: bool
    reply: str = Field(max_length=1500)
    citations: list[int] = Field(max_length=5)


def excerpts(hits: list[Hit]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        words = h.text.split()
        body = " ".join(words[:MAX_EXCERPT_WORDS]) + (
            " ..." if len(words) > MAX_EXCERPT_WORDS else ""
        )
        parts.append(f"[{i}] {body}")
    return "\n\n".join(parts)


def messages(question: str, hits: list[Hit]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM.format(excerpts=excerpts(hits))},
        {"role": "user", "content": question},
    ]


def inline_citations(reply: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", reply)})


def draft(llm: LLM, question: str, hits: list[Hit]) -> tuple[dict[str, Any], Completion]:
    parsed, completion = llm.parse(messages(question, hits), Draft, max_tokens=400)
    out: dict[str, Any] = json.loads(parsed.model_dump_json())
    # Keep only citations that point at an excerpt that was actually shown.
    cited = set(out["citations"]) | set(inline_citations(out["reply"]))
    out["citations"] = sorted(c for c in cited if 1 <= c <= len(hits))
    out["cited_doc_ids"] = sorted({hits[c - 1].doc_id for c in out["citations"]})
    return out, completion
