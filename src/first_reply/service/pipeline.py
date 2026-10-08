"""The triage pipeline behind POST /triage.

mask PII -> embed -> similar past tickets (kNN vote = routing confidence) -> LLM triage with
those tickets as examples -> hybrid retrieval under the caller's role -> cited draft or decline
-> gate. Routing is applied automatically only when the kNN confidence clears the threshold
chosen on dev for 95% precision *and* the LLM agrees; a reply is never sent without a person
approving it. Each step is an OpenTelemetry span.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from opentelemetry import trace

from first_reply.config import ROOT
from first_reply.data import language
from first_reply.kb.retrieve import Hit, Retriever
from first_reply.llm import answer, triage
from first_reply.llm.client import LLM
from first_reply.routing.targets import TARGETS
from first_reply.service import pii

tracer = trace.get_tracer("first_reply")

AUTOMATION = ROOT / "docs" / "results" / "automation" / "bge-m3_knn_p95.json"
K_NEIGHBOURS = 25
K_EXAMPLES = 5
TEMPERATURE = 0.05


@dataclass
class Settings:
    llm_model: str = "triage"
    chunker: str = "sections"
    dense_model: str = "bge-m3"
    excerpts: int = 5
    thresholds: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_results(cls, path: Path = AUTOMATION, **kwargs: Any) -> Settings:
        thresholds = {}
        if path.exists():
            data = json.loads(path.read_text())
            thresholds = {t: v["threshold_from_dev"] for t, v in data["targets"].items()}
        return cls(thresholds=thresholds, **kwargs)


class Neighbours:
    """Similarity-weighted vote over past (training) tickets."""

    def __init__(self, train: pd.DataFrame, vectors: NDArray[np.float32]) -> None:
        self.train = train.reset_index(drop=True)
        self.x = vectors

    def vote(self, q: NDArray[np.float32]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        sims = self.x @ q
        idx = np.argpartition(-sims, K_NEIGHBOURS)[:K_NEIGHBOURS]
        idx = idx[np.argsort(-sims[idx])]
        w = np.exp((sims[idx] - sims[idx].max()) / TEMPERATURE)
        w /= w.sum()
        votes: dict[str, Any] = {}
        for t in TARGETS:
            scores: dict[str, float] = {}
            for label, weight in zip(self.train[t].to_numpy()[idx], w, strict=True):
                scores[str(label)] = scores.get(str(label), 0.0) + float(weight)
            best = max(scores, key=scores.__getitem__)
            votes[t] = {"label": best, "confidence": scores[best]}
        cols = ["id", "text", *TARGETS, "tags"]
        examples = [
            {**self.train.iloc[i][cols].to_dict(), "similarity": float(sims[i])}
            for i in idx[:K_EXAMPLES]
        ]
        for e in examples:
            e["tags"] = list(e["tags"])
        return votes, examples


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        *,
        llm: LLM,
        retriever: Retriever,
        encode: Callable[[str], NDArray[np.float32]],
        neighbours: Neighbours,
        tags: list[str],
    ) -> None:
        self.s = settings
        self.llm = llm
        self.retriever = retriever
        self.encode = encode
        self.neighbours = neighbours
        self.tags = tags

    def run(self, subject: str, body: str, role: str = "customer") -> tuple[str, dict[str, Any]]:
        timings: dict[str, float] = {}
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}

        def step(name: str) -> Any:
            return _Step(name, timings)

        with tracer.start_as_current_span("triage_case") as case_span:
            with step("mask"):
                masked = pii.mask(f"{subject}\n\n{body}".strip())
            text = masked.text
            with step("embed"):
                q = self.encode(text)
            with step("similar_tickets"):
                votes, examples = self.neighbours.vote(q)
            with step("llm_triage"):
                fields, c1 = triage.triage(self.llm, text, self.tags, examples)
            with step("retrieve"):
                hits: list[Hit] = self.retriever.search(
                    text, mode="hybrid", dense=q.tolist(), role=role
                )[: self.s.excerpts]
            with step("draft"):
                draft, c2 = answer.draft(self.llm, text, hits)
            for c in (c1, c2):
                usage["prompt_tokens"] += c.prompt_tokens
                usage["completion_tokens"] += c.completion_tokens
                usage["cost"] += c.cost or 0.0

            gate = self.gate(votes, fields, draft)
            case_span.set_attribute("triage.queue", fields["queue"])
            case_span.set_attribute("triage.auto_route", gate["auto_route"])
            case_span.set_attribute("draft.answerable", bool(draft["answerable"]))
            case_span.set_attribute("llm.prompt_tokens", usage["prompt_tokens"])
            case_span.set_attribute("llm.completion_tokens", usage["completion_tokens"])

        proposal = {
            "language": language.detect(text),
            "pii_masked": masked.found,
            "routing": {"llm": fields, "similar_tickets_vote": votes},
            "similar_tickets": [
                {k: e[k] for k in ("id", "queue", "priority", "type", "similarity")}
                for e in examples
            ],
            "evidence": [
                {
                    "n": i,
                    "chunk_id": h.chunk_id,
                    "doc_id": h.doc_id,
                    "section": h.section,
                    "snippet": h.text[:400],
                }
                for i, h in enumerate(hits, 1)
            ],
            "draft": draft,
            "gate": gate,
            "usage": usage,
            "timings_s": timings,
        }
        return text, proposal

    def gate(
        self, votes: dict[str, Any], fields: dict[str, Any], draft: dict[str, Any]
    ) -> dict[str, Any]:
        reasons = []
        queue_vote = votes["queue"]
        threshold = self.s.thresholds.get("queue", float("inf"))
        confident = queue_vote["confidence"] >= threshold
        agree = queue_vote["label"] == fields["queue"]
        if not confident:
            reasons.append("routing confidence below the 95%-precision threshold")
        if not agree:
            reasons.append("LLM and similar tickets disagree on the queue")
        if not draft["answerable"]:
            reasons.append("knowledge base does not answer the question")
        return {
            "auto_route": confident and agree,
            "queue": fields["queue"],
            "queue_confidence": queue_vote["confidence"],
            "reply_needs_approval": True,
            "reasons": reasons,
        }


class _Step:
    def __init__(self, name: str, timings: dict[str, float]) -> None:
        self.name = name
        self.timings = timings

    def __enter__(self) -> _Step:
        self.span = tracer.start_span(self.name)
        self.ctx = trace.use_span(self.span, end_on_exit=True)
        self.ctx.__enter__()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.timings[self.name] = round(time.perf_counter() - self.t0, 3)
        self.ctx.__exit__(*exc)  # type: ignore[arg-type]
