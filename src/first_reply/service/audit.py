"""Case store and audit log (SQLite).

Every incoming email becomes a case. The system's proposal (routing, draft, evidence) and every
human decision are stored. The log is the audit trail and, with the reviewers' corrections,
the labelled data for the next model version.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id TEXT PRIMARY KEY,
    created REAL NOT NULL,
    sender TEXT,
    subject TEXT,
    body_masked TEXT NOT NULL,
    proposal TEXT NOT NULL,
    status TEXT NOT NULL,          -- pending | approved | edited | rejected | sent
    resume_url TEXT
);
CREATE TABLE IF NOT EXISTS decisions (
    case_id TEXT NOT NULL REFERENCES cases(id),
    at REAL NOT NULL,
    reviewer TEXT NOT NULL,
    action TEXT NOT NULL,          -- approve | edit | reject
    reply TEXT,
    queue TEXT,
    note TEXT
);
CREATE INDEX IF NOT EXISTS cases_status ON cases(status);
"""


class Store:
    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(SCHEMA)

    def create(
        self,
        *,
        sender: str,
        subject: str,
        body_masked: str,
        proposal: dict[str, Any],
        resume_url: str | None = None,
    ) -> str:
        case_id = uuid.uuid4().hex[:12]
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO cases VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)",
                (case_id, time.time(), sender, subject, body_masked, json.dumps(proposal),
                 resume_url),
            )  # fmt: skip
        return case_id

    def get(self, case_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
            if row is None:
                return None
            decisions = self._db.execute(
                "SELECT * FROM decisions WHERE case_id = ? ORDER BY at", (case_id,)
            ).fetchall()
        case = dict(row)
        case["proposal"] = json.loads(case["proposal"])
        case["decisions"] = [dict(d) for d in decisions]
        return case

    def list(self, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        query = "SELECT id, created, sender, subject, status, proposal FROM cases"
        args: tuple[Any, ...] = ()
        if status:
            query += " WHERE status = ?"
            args = (status,)
        query += " ORDER BY created DESC LIMIT ?"
        with self._lock:
            rows = self._db.execute(query, (*args, limit)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["proposal"] = json.loads(d["proposal"])
            out.append(d)
        return out

    def decide(
        self,
        case_id: str,
        *,
        reviewer: str,
        action: str,
        reply: str | None = None,
        queue: str | None = None,
        note: str | None = None,
    ) -> str:
        status = {"approve": "approved", "edit": "edited", "reject": "rejected"}[action]
        with self._lock, self._db:
            cur = self._db.execute(
                "UPDATE cases SET status = ? WHERE id = ? AND status = 'pending'",
                (status, case_id),
            )
            if cur.rowcount != 1:
                raise LookupError(f"case {case_id} is not pending")
            self._db.execute(
                "INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (case_id, time.time(), reviewer, action, reply, queue, note),
            )
        return status

    def mark_sent(self, case_id: str) -> None:
        with self._lock, self._db:
            self._db.execute(
                "UPDATE cases SET status = 'sent' "
                "WHERE id = ? AND status IN ('approved', 'edited')",
                (case_id,),
            )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            by_status = dict(
                self._db.execute("SELECT status, COUNT(*) FROM cases GROUP BY status").fetchall()
            )
            by_action = dict(
                self._db.execute(
                    "SELECT action, COUNT(*) FROM decisions GROUP BY action"
                ).fetchall()
            )
        return {"cases_by_status": by_status, "decisions_by_action": by_action}
