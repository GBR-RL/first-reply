"""HTTP service: triage incoming emails, keep the review queue, record decisions.

n8n posts every incoming email to /triage together with the resume URL of its Wait node. A
reviewer approves, edits or rejects the proposal on /review (or via the API); the service then
calls the resume URL and n8n sends the reply. Nothing reaches a customer without that decision.
"""

from __future__ import annotations

import os
import time
from typing import Any, Literal, Protocol

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from first_reply.service.audit import Store
from first_reply.service.review import PAGE

CASES = Counter("first_reply_cases_total", "Triaged emails", ["auto_route", "answerable"])
DECISIONS = Counter("first_reply_decisions_total", "Reviewer decisions", ["action"])
LATENCY = Histogram(
    "first_reply_triage_seconds",
    "End-to-end triage latency",
    buckets=(1, 2, 5, 10, 20, 30, 60, 90, 120, 180, 300),
)


class Runner(Protocol):
    def run(
        self, subject: str, body: str, role: str = "customer"
    ) -> tuple[str, dict[str, Any]]: ...


class Email(BaseModel):
    subject: str = Field("", max_length=500)
    body: str = Field(..., min_length=1, max_length=20_000)
    sender: str = Field("", max_length=320)
    role: Literal["customer", "support_engineer", "security_team"] = "customer"
    resume_url: str | None = Field(None, max_length=2000)


class Decision(BaseModel):
    reviewer: str = Field(..., min_length=1, max_length=100)
    action: Literal["approve", "edit", "reject"]
    reply: str | None = Field(None, max_length=5000)
    queue: str | None = Field(None, max_length=100)
    note: str | None = Field(None, max_length=1000)


def create_app(pipeline: Runner, store: Store, http: httpx.Client | None = None) -> FastAPI:
    app = FastAPI(title="first-reply", version="0.1.0")
    client = http or httpx.Client(timeout=30)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/triage")
    def triage(email: Email) -> dict[str, Any]:
        t0 = time.perf_counter()
        masked, proposal = pipeline.run(email.subject, email.body, email.role)
        LATENCY.observe(time.perf_counter() - t0)
        case_id = store.create(
            sender=email.sender,
            subject=email.subject,
            body_masked=masked,
            proposal=proposal,
            resume_url=email.resume_url,
        )
        CASES.labels(
            str(proposal["gate"]["auto_route"]).lower(),
            str(bool(proposal["draft"]["answerable"])).lower(),
        ).inc()
        return {"case_id": case_id, **proposal}

    @app.get("/cases")
    def cases(status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return store.list(status, min(limit, 500))

    @app.get("/cases/{case_id}")
    def case(case_id: str) -> dict[str, Any]:
        found = store.get(case_id)
        if found is None:
            raise HTTPException(404, "case not found")
        return found

    @app.post("/cases/{case_id}/decision")
    def decide(case_id: str, d: Decision) -> dict[str, Any]:
        found = store.get(case_id)
        if found is None:
            raise HTTPException(404, "case not found")
        if d.action == "edit" and not d.reply:
            raise HTTPException(422, "an edit needs the edited reply")
        try:
            status = store.decide(
                case_id, reviewer=d.reviewer, action=d.action, reply=d.reply, queue=d.queue,
                note=d.note,
            )  # fmt: skip
        except LookupError as err:
            raise HTTPException(409, str(err)) from err
        DECISIONS.labels(d.action).inc()
        proposal = found["proposal"]
        reply = d.reply if d.action == "edit" else proposal["draft"]["reply"]
        resumed = False
        if found["resume_url"]:
            client.post(
                found["resume_url"],
                json={
                    "case_id": case_id,
                    "action": d.action,
                    "send": d.action in ("approve", "edit") and bool(reply),
                    "to": found["sender"],
                    "subject": f"Re: {found['subject']}",
                    "reply": reply,
                    "queue": d.queue or proposal["gate"]["queue"],
                },
            ).raise_for_status()
            resumed = True
        return {"case_id": case_id, "status": status, "workflow_resumed": resumed}

    @app.post("/cases/{case_id}/sent")
    def sent(case_id: str) -> dict[str, str]:
        store.mark_sent(case_id)
        return {"case_id": case_id, "status": "sent"}

    @app.get("/stats")
    def stats() -> dict[str, Any]:
        return store.stats()

    @app.get("/metrics")
    def metrics() -> PlainTextResponse:
        return PlainTextResponse(generate_latest().decode(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/review", response_class=HTMLResponse)
    def review() -> str:
        return PAGE

    return app


def setup_tracing(service: str = "first-reply") -> None:
    """Export spans over OTLP/HTTP when OTEL_EXPORTER_OTLP_ENDPOINT is set (e.g. Phoenix)."""
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)


def build() -> FastAPI:
    """Wire the real pipeline (used by `uvicorn first_reply.service.app:build --factory`)."""
    from first_reply import embed
    from first_reply.config import DATA
    from first_reply.data import tickets
    from first_reply.kb.retrieve import Retriever
    from first_reply.llm.client import LLM
    from first_reply.routing.targets import tag_vocab
    from first_reply.service.pipeline import Neighbours, Pipeline, Settings

    setup_tracing()
    settings = Settings.from_results()
    df = tickets.load()
    train = df[df.split == "train"].reset_index(drop=True)
    neighbours = Neighbours(train, embed.load(settings.dense_model, "tickets", train.id.tolist()))
    model = embed._load(embed.MODELS[settings.dense_model])

    def encode(text: str) -> Any:
        return model.encode(text, normalize_embeddings=True, convert_to_numpy=True)

    pipeline = Pipeline(
        settings,
        llm=LLM(settings.llm_model),
        retriever=Retriever(settings.chunker, settings.dense_model),
        encode=encode,
        neighbours=neighbours,
        tags=tag_vocab(train),
    )
    store = Store(os.environ.get("FIRST_REPLY_DB", str(DATA / "cases.db")))
    return create_app(pipeline, store)
