"""End-to-end check of the Compose stack, as a customer and a reviewer would use it.

1. A customer emails the support mailbox (SMTP).
2. n8n picks the email up over IMAP and posts it to the service; a pending case appears.
3. A reviewer approves the draft through the API.
4. n8n resumes, sends the reply over SMTP, and marks the case sent.
5. The reply is in the customer's mailbox.
6. The gateway logged the model calls, and a key with a spent budget is refused.

Usage: python -m first_reply.service.smoke [--question-id QID]
"""

from __future__ import annotations

import argparse
import email
import imaplib
import smtplib
import sys
import time
from email.message import EmailMessage
from typing import Any

import httpx

SERVICE = "http://127.0.0.1:8000"
GATEWAY = "http://127.0.0.1:4000"
PHOENIX = "http://127.0.0.1:6006"
MAIL_HOST = "127.0.0.1"
QUESTION = (
    "I receive the following during UDX compilation: Can't exec "
    '"/nz/kit/bin/adm/nzudxcompile": Argument list too long. How can I fix this?'
)


def wait_for(what: str, check: Any, timeout: float, every: float = 5.0) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = check()
        if result:
            print(f"ok: {what}")
            return result
        time.sleep(every)
    raise TimeoutError(f"timed out waiting for: {what}")


def send_email(subject: str, body: str) -> None:
    msg = EmailMessage()
    msg["From"] = "customer@example.com"
    msg["To"] = "support@first-reply.test"
    msg["Subject"] = subject
    msg.set_content(body)
    with smtplib.SMTP(MAIL_HOST, 3025) as smtp:
        smtp.send_message(msg)


def inbox(user: str, password: str) -> list[email.message.Message]:
    with imaplib.IMAP4(MAIL_HOST, 3143) as imap:
        imap.login(user, password)
        imap.select("INBOX")
        _, data = imap.search(None, "ALL")
        out = []
        for num in data[0].split():
            _, parts = imap.fetch(num, "(RFC822)")
            first = parts[0]
            raw = first[1] if isinstance(first, tuple) else b""
            out.append(email.message_from_bytes(raw))
        return out


def picked_up() -> bool:
    """The IMAP trigger marks a message as read as soon as it fetches it."""
    with imaplib.IMAP4(MAIL_HOST, 3143) as imap:
        imap.login("support", "support")
        imap.select("INBOX")
        _, unseen = imap.search(None, "UNSEEN")
        _, everything = imap.search(None, "ALL")
    return bool(everything[0].split()) and not unseen[0].split()


def deliver(subject: str) -> None:
    """Send the customer email; resend if n8n has not fetched it within a minute.

    The trigger ignores mail that arrived before it connected.
    """
    for attempt in range(3):
        send_email(subject, QUESTION)
        print(f"sent the customer email (attempt {attempt + 1})")
        try:
            wait_for("n8n fetched the email", picked_up, 60)
            return
        except TimeoutError:
            if attempt == 2:
                raise


def check_gateway(http: httpx.Client, master_key: str) -> None:
    headers = {"Authorization": f"Bearer {master_key}"}
    logs = http.get(f"{GATEWAY}/spend/logs", headers=headers).json()
    assert logs, "the gateway logged no spend"
    print(f"ok: the gateway logged {len(logs)} model calls")

    key = http.post(
        f"{GATEWAY}/key/generate", headers=headers, json={"max_budget": 0.0000001}
    ).json()["key"]
    body_json = {"model": "triage", "messages": [{"role": "user", "content": "Hi"}],
                 "max_tokens": 5}  # fmt: skip
    first = http.post(f"{GATEWAY}/v1/chat/completions", headers={"Authorization": f"Bearer {key}"},
                      json=body_json)  # fmt: skip
    first.raise_for_status()
    refused = wait_for(
        "a key over its budget is refused",
        lambda: (
            http.post(
                f"{GATEWAY}/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json=body_json,
            ).status_code
            in (400, 401, 429)
        ),
        240,
        every=15,
    )
    assert refused


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=1500)
    parser.add_argument("--master-key", default="sk-compose")
    args = parser.parse_args()
    http = httpx.Client(timeout=60)

    subject = "UDX compilation fails: argument list too long"

    def pending() -> list[dict[str, Any]]:
        cases: list[dict[str, Any]] = http.get(f"{SERVICE}/cases?status=pending").json()
        return [c for c in cases if c["subject"] == subject]

    deliver(subject)
    case = wait_for("the triage finished and a case is pending", pending, args.timeout)[0]
    case = http.get(f"{SERVICE}/cases/{case['id']}").json()
    proposal = case["proposal"]
    print("routing:", proposal["routing"]["llm"]["queue"], "| gate:", proposal["gate"])
    print("draft:", proposal["draft"]["reply"][:300])
    assert proposal["evidence"], "no knowledge-base evidence retrieved"
    assert proposal["draft"]["answerable"], "the draft declined a question the KB answers"

    r = http.post(
        f"{SERVICE}/cases/{case['id']}/decision", json={"reviewer": "smoke", "action": "approve"}
    )
    r.raise_for_status()
    assert r.json()["workflow_resumed"], "the n8n workflow was not resumed"

    def reply() -> list[email.message.Message]:
        return [m for m in inbox("customer", "customer") if m["Subject"] == f"Re: {subject}"]

    msg = wait_for("the approved reply reached the customer's mailbox", reply, 300)[0]
    body = msg.get_payload(decode=True)
    text = body.decode(errors="replace") if isinstance(body, bytes) else str(msg.get_payload())
    assert proposal["draft"]["reply"][:40] in text, "the sent text is not the approved draft"
    wait_for(
        "the case is marked sent",
        lambda: http.get(f"{SERVICE}/cases/{case['id']}").json()["status"] == "sent",
        120,
    )

    check_gateway(http, args.master_key)

    try:
        projects = http.get(f"{PHOENIX}/v1/projects").json()
        print("phoenix projects:", [p.get("name") for p in projects.get("data", [])])
    except (httpx.HTTPError, ValueError) as err:  # trace viewer API is informational only
        print("phoenix check skipped:", err)
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
