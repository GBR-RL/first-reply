"""LLM judge for faithfulness: is every statement of a reply supported by the excerpts?

The judge is checked against RAGBench's labels before it is used to score our own drafts:
agreement (Cohen's kappa) and the false-pass rate, i.e. how often it calls an unsupported
response supported. A judge that passes unsupported answers would make every number built on
it look better than it is.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from first_reply.llm.client import LLM, Completion

MAX_DOC_WORDS = 900

SYSTEM = """You check whether a support reply is fully supported by the documents it was based on.

Go through the reply statement by statement. A statement is supported only if the documents
state it or it follows directly from them. Advice, facts, version numbers or steps that are not
in the documents are unsupported, even if they are true. Saying that the documents do not
contain the answer counts as supported when that is correct.

Documents:
{documents}"""


class Verdict(BaseModel):
    unsupported_statements: list[str] = Field(max_length=10)
    supported: bool


def documents(texts: list[str]) -> str:
    parts = []
    for i, t in enumerate(texts, 1):
        words = t.split()
        parts.append(f"[{i}] " + " ".join(words[:MAX_DOC_WORDS]))
    return "\n\n".join(parts)


def judge(
    llm: LLM, question: str, reply: str, docs: list[str]
) -> tuple[dict[str, Any], Completion]:
    msgs = [
        {"role": "system", "content": SYSTEM.format(documents=documents(docs))},
        {"role": "user", "content": f"Question:\n{question}\n\nReply to check:\n{reply}"},
    ]
    parsed, completion = llm.parse(msgs, Verdict, max_tokens=400)
    out: dict[str, Any] = json.loads(parsed.model_dump_json())
    # A verdict that lists unsupported statements cannot be "supported".
    out["supported"] = bool(out["supported"]) and not out["unsupported_statements"]
    return out, completion
