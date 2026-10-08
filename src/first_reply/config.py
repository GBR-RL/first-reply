"""Filesystem layout. Everything generated lives under one data root."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("FIRST_REPLY_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("FIRST_REPLY_DATA", ROOT / "data"))
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
RUNS = Path(os.environ.get("FIRST_REPLY_RUNS", ROOT / "runs"))

TICKETS = PROCESSED / "tickets.parquet"
KB_DOCS = PROCESSED / "kb_docs.parquet"
KB_QUESTIONS = PROCESSED / "kb_questions.parquet"
KB_RESPONSES = PROCESSED / "kb_responses.parquet"

SEED = 13
