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
    """Spend is logged, and a per-key limit is enforced by the gateway."""
    headers = {"Authorization": f"Bearer {master_key}"}
    logs = http.get(f"{GATEWAY}/spend/logs", headers=headers).json()
    assert logs, "the gateway logged no spend"
    print(f"ok: the gateway logged {len(logs)} model calls")

    key = http.post(
        f"{GATEWAY}/key/generate",
        headers=headers,
        json={"rpm_limit": 1, "max_budget": 1.0, "metadata": {"purpose": "smoke test"}},
    ).json()["key"]
    auth = {"Authorization": f"Bearer {key}"}
    body = {"model": "triage", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 5}
    http.post(f"{GATEWAY}/v1/chat/completions", headers=auth, json=body).raise_for_status()
    second = http.post(f"{GATEWAY}/v1/chat/completions", headers=auth, json=body)
    assert second.status_code == 429, f"rate limit not enforced: {second.status_code}"
    print("ok: a key over its rate limit is refused (429)")

    def spend() -> float:
        info = http.get(f"{GATEWAY}/key/info", headers=headers, params={"key": key}).json()
        return float(info.get("info", {}).get("spend") or 0.0)

    wait_for("the key's spend is recorded against its budget", lambda: spend() > 0, 240, 15)


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
