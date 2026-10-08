import json
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from first_reply.service import pii
from first_reply.service.app import create_app
from first_reply.service.audit import Store
from first_reply.service.pipeline import Neighbours, Pipeline, Settings


class FakePipeline:
    def __init__(self, answerable: bool = True) -> None:
        self.answerable = answerable

    def run(self, subject: str, body: str, role: str = "customer") -> tuple[str, dict[str, Any]]:
        masked = pii.mask(f"{subject}\n\n{body}").text
        return masked, {
            "language": "en",
            "routing": {
                "llm": {"queue": "IT Support", "priority": "high", "type": "Incident", "tags": []},
                "similar_tickets_vote": {"queue": {"label": "IT Support", "confidence": 0.9}},
            },
            "evidence": [],
            "draft": {"answerable": self.answerable, "reply": "Restart the VPN gateway [1].",
                      "citations": [1]},
            "gate": {"auto_route": True, "queue": "IT Support", "reasons": []},
        }  # fmt: skip


@pytest.fixture
def setup() -> tuple[TestClient, list[dict[str, Any]]]:
    resumed: list[dict[str, Any]] = []

    def n8n(request: httpx.Request) -> httpx.Response:
        resumed.append(json.loads(request.content))
        return httpx.Response(200, json={})

    http = httpx.Client(transport=httpx.MockTransport(n8n))
    app = create_app(FakePipeline(), Store(":memory:"), http)
    return TestClient(app), resumed


def _email(**kw: Any) -> dict[str, Any]:
    return {"subject": "VPN down", "body": "Call me at +49 711 1234567.", "sender": "a@b.de",
            "resume_url": "http://n8n/webhook-waiting/1", **kw}  # fmt: skip


def test_triage_creates_a_pending_case_with_masked_body(setup: Any) -> None:
    client, _ = setup
    case = client.post("/triage", json=_email()).json()
    stored = client.get(f"/cases/{case['case_id']}").json()
    assert stored["status"] == "pending"
    assert "<phone>" in stored["body_masked"]
    assert "1234567" not in stored["body_masked"]
    assert [c["id"] for c in client.get("/cases?status=pending").json()] == [case["case_id"]]


def test_approve_resumes_the_workflow_with_the_draft(setup: Any) -> None:
    client, resumed = setup
    case_id = client.post("/triage", json=_email()).json()["case_id"]
    r = client.post(f"/cases/{case_id}/decision", json={"reviewer": "kim", "action": "approve"})
    assert r.json() == {"case_id": case_id, "status": "approved", "workflow_resumed": True}
    assert resumed[0]["send"] is True
    assert resumed[0]["reply"].startswith("Restart the VPN")
    assert resumed[0]["to"] == "a@b.de"
    # A case can be decided once.
    again = client.post(f"/cases/{case_id}/decision", json={"reviewer": "kim", "action": "reject"})
    assert again.status_code == 409


def test_edit_sends_the_edited_reply_and_reject_sends_nothing(setup: Any) -> None:
    client, resumed = setup
    a = client.post("/triage", json=_email()).json()["case_id"]
    b = client.post("/triage", json=_email()).json()["case_id"]
    assert (
        client.post(f"/cases/{a}/decision", json={"reviewer": "k", "action": "edit"}).status_code
        == 422
    )
    client.post(f"/cases/{a}/decision", json={"reviewer": "k", "action": "edit", "reply": "Hi."})
    client.post(f"/cases/{b}/decision", json={"reviewer": "k", "action": "reject"})
    assert resumed[0]["reply"] == "Hi."
    assert resumed[0]["send"] is True
    assert resumed[1]["send"] is False
    client.post(f"/cases/{a}/sent")
    assert client.get(f"/cases/{a}").json()["status"] == "sent"
    assert client.get("/stats").json()["decisions_by_action"] == {"edit": 1, "reject": 1}


def test_review_page_and_metrics(setup: Any) -> None:
    client, _ = setup
    client.post("/triage", json=_email())
    assert "Review queue" in client.get("/review").text
    assert "first_reply_cases_total" in client.get("/metrics").text


def test_pii_masking_keeps_versions_and_codes() -> None:
    m = pii.mask(
        "Mail jan@firma.de, +49 711 123 4567, host 10.0.0.12, version 9.1.0.4, SQLCODE=-1585"
    )
    assert m.text == "Mail <email>, <phone>, host <ip>, version 9.1.0.4, SQLCODE=-1585"
    assert m.found == {"EMAIL": 1, "IP": 1, "PHONE": 1}


def test_gate_requires_confidence_and_agreement() -> None:
    p = Pipeline(
        Settings(thresholds={"queue": 0.8}),
        llm=None,  # type: ignore[arg-type]
        retriever=None,  # type: ignore[arg-type]
        encode=None,  # type: ignore[arg-type]
        neighbours=None,  # type: ignore[arg-type]
        tags=[],
    )
    vote = {"queue": {"label": "IT Support", "confidence": 0.9}}
    draft = {"answerable": True}
    assert p.gate(vote, {"queue": "IT Support"}, draft)["auto_route"] is True
    assert p.gate(vote, {"queue": "Billing and Payments"}, draft)["auto_route"] is False
    low = {"queue": {"label": "IT Support", "confidence": 0.5}}
    g = p.gate(low, {"queue": "IT Support"}, {"answerable": False})
    assert g["auto_route"] is False
    assert len(g["reasons"]) == 2
    assert g["reply_needs_approval"] is True


def test_neighbour_vote_is_weighted_by_similarity() -> None:
    train = pd.DataFrame({"id": [f"t{i}" for i in range(30)], "text": ["x"] * 30,
                          "queue": ["A"] * 25 + ["B"] * 5, "priority": ["low"] * 30,
                          "type": ["Request"] * 30, "tags": [["T"]] * 30})  # fmt: skip
    x = np.zeros((30, 2), dtype=np.float32)
    x[:25] = [1, 0]
    x[25:] = [0, 1]
    votes, examples = Neighbours(train, x).vote(np.array([0, 1], dtype=np.float32))
    assert votes["queue"]["label"] == "B"
    assert examples[0]["queue"] == "B"
