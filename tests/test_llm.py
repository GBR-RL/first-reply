import json
from typing import Any

import httpx
import pytest

from first_reply.kb.retrieve import Hit
from first_reply.llm import answer, triage
from first_reply.llm.client import LLM

TAGS = ["Bug", "Network", "Security"]


def _llm(content: dict[str, Any], seen: list[dict[str, Any]]) -> LLM:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "choices": [{"message": {"content": json.dumps(content)}}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 30},
            },
        )

    return LLM("test-model", base_url="http://llm/v1", transport=httpx.MockTransport(handler))


def test_triage_sends_schema_and_parses_labels() -> None:
    seen: list[dict[str, Any]] = []
    reply = {
        "queue": "IT Support",
        "priority": "high",
        "type": "Incident",
        "tags": ["Network"],
        "summary": "VPN is down for the whole office.",
        "product": "VPN",
    }
    out, completion = triage.triage(_llm(reply, seen), "Our VPN is down.", TAGS)
    assert out == reply
    assert completion.prompt_tokens == 120
    body = seen[0]
    schema = body["response_format"]["json_schema"]["schema"]
    assert "IT Support" in json.dumps(schema)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_triage_rejects_labels_outside_the_schema() -> None:
    bad = {"queue": "Legal", "priority": "high", "type": "Incident", "tags": [], "summary": "",
           "product": ""}  # fmt: skip
    with pytest.raises(ValueError, match="queue"):
        triage.triage(_llm(bad, []), "x", TAGS)


def test_few_shot_examples_are_included() -> None:
    examples = [{"text": "Invoice wrong", "queue": "Billing and Payments", "priority": "low",
                 "type": "Request", "tags": ["Bug"]}]  # fmt: skip
    msgs = triage.messages("New ticket", TAGS, examples)
    assert "Billing and Payments" in msgs[1]["content"]
    assert msgs[-1]["content"].endswith("New ticket")


def _hits() -> list[Hit]:
    return [
        Hit("d1:0", "d1", "faq", 0, "", "Restart the queue manager.", 1.0),
        Hit("d2:0", "d2", "faq", 0, "", "Apply fix pack 9.1.0.4.", 0.9),
    ]


def test_draft_keeps_only_citations_to_shown_excerpts() -> None:
    reply = {"answerable": True, "reply": "Apply the fix pack [2] and restart [1] [7].",
             "citations": [2, 9]}  # fmt: skip
    out, _ = answer.draft(_llm(reply, []), "Queue manager hangs", _hits())
    assert out["citations"] == [1, 2]
    assert out["cited_doc_ids"] == ["d1", "d2"]


def test_excerpts_are_numbered_and_truncated() -> None:
    long = Hit("d3:0", "d3", "faq", 0, "", "word " * 1000, 0.5)
    text = answer.excerpts([*_hits(), long])
    assert text.startswith("[1] Restart")
    assert "[3] " in text
    assert text.endswith("...")


def test_client_retries_then_fails() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503)

    llm = LLM("m", base_url="http://llm/v1", retries=1, transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError):
        llm.chat([{"role": "user", "content": "hi"}])
    assert len(calls) == 2
