"""Structured triage of an incoming ticket with an LLM.

The model fills a fixed JSON schema: queue, priority, type, tags, a one-line summary and the
product the customer refers to. Labels are constrained to the known values by the schema, so
the output always maps onto the routing targets. Optionally, the most similar past tickets with
their labels are given as examples (retrieval-augmented few-shot).
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, create_model

from first_reply.llm.client import LLM, Completion

QUEUES = {
    "Technical Support": "faults and errors in software, platforms or integrations",
    "Product Support": "questions and problems about using a specific product or its features",
    "Customer Service": "general customer concerns, feedback, account and service questions",
    "IT Support": "internal IT: devices, networks, accounts, access, security incidents",
    "Billing and Payments": "invoices, charges, refunds, payment methods",
    "Returns and Exchanges": "returning or exchanging purchased goods",
    "Service Outages and Maintenance": "outages, downtime, planned maintenance",
    "Sales and Pre-Sales": "offers, pricing, product information before a purchase",
    "Human Resources": "employment, payroll, leave, HR processes",
    "General Inquiry": "anything that fits no other queue",
}
PRIORITIES = {
    "high": "business-critical: outage, data loss, security breach, many users blocked",
    "medium": "degraded service or a blocked task with a workaround",
    "low": "information requests, minor issues, feedback",
}
TYPES = {
    "Incident": "an unplanned interruption or degradation of a service",
    "Problem": "the underlying cause of recurring or widespread incidents needs investigating",
    "Request": "a request for information, access or a standard service",
    "Change": "a request to add, modify, upgrade or remove something",
}

SYSTEM = """You triage customer support emails for a service desk. Emails are in English or
German. Read the email and fill in every field.

Queues:
{queues}

Priority:
{priorities}

Type:
{types}

Tags: choose up to six from: {tags}.
summary: one sentence in English. product: the product or system named, or "" if none.
Answer with the JSON object only."""


def _lines(d: dict[str, str]) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in d.items())


def schema(tags: list[str]) -> type[BaseModel]:
    # The label sets are only known at run time, so the enums are built dynamically.
    make: Any = Enum
    queue = make("Queue", {f"q{i}": q for i, q in enumerate(QUEUES)})
    tag = make("Tag", {f"t{i}": t for i, t in enumerate(tags)})
    priority = make("Priority", {p: p for p in PRIORITIES})
    ticket_type = make("TicketType", {t: t for t in TYPES})
    tag_list: Any = list[tag]  # type: ignore[valid-type]
    model: type[BaseModel] = create_model(
        "Triage",
        queue=(queue, ...),
        priority=(priority, ...),
        type=(ticket_type, ...),
        tags=(tag_list, Field(max_length=6)),
        summary=(str, Field(max_length=300)),
        product=(str, Field(max_length=80)),
    )
    return model


def _labels(t: Any) -> dict[str, Any]:
    return {
        "queue": t["queue"],
        "priority": t["priority"],
        "type": t["type"],
        "tags": list(t["tags"]),
    }


def messages(
    text: str, tags: list[str], examples: list[dict[str, Any]] | None = None
) -> list[dict[str, str]]:
    system = SYSTEM.format(
        queues=_lines(QUEUES),
        priorities=_lines(PRIORITIES),
        types=_lines(TYPES),
        tags=", ".join(tags),
    )
    out = [{"role": "system", "content": system}]
    if examples:
        shots = "\n\n".join(
            f"Email:\n{e['text'][:1200]}\nLabels: {json.dumps(_labels(e))}" for e in examples
        )
        out.append(
            {
                "role": "user",
                "content": "Similar past tickets and how they were labelled:\n\n" + shots,
            }
        )
        out.append({"role": "assistant", "content": "Understood."})
    out.append({"role": "user", "content": f"Email:\n{text[:4000]}"})
    return out


def triage(
    llm: LLM, text: str, tags: list[str], examples: list[dict[str, Any]] | None = None
) -> tuple[dict[str, Any], Completion]:
    model = schema(tags)
    parsed, completion = llm.parse(messages(text, tags, examples), model, max_tokens=300)
    return json.loads(parsed.model_dump_json()), completion
