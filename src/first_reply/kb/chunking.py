"""Two ways to cut support documents into retrievable chunks.

`fixed`: overlapping word windows over the raw text, the common default.
`sections`: split on the documents' own section headers (PROBLEM(ABSTRACT), CAUSE, RESOLVING
THE PROBLEM, ...), drop boilerplate sections, merge small sections and window long ones. Each
chunk is prefixed with the document type, its keywords and a one-line abstract, so a chunk
cut from the middle of a document still says what the document is about.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import pandas as pd

MAX_WORDS = 300
OVERLAP = 50
MIN_WORDS = 60

HEADERS = (
    "PROBLEM(ABSTRACT)", "PROBLEM", "PROBLEM DESCRIPTION", "PROBLEM SUMMARY",
    "PROBLEM CONCLUSION", "SYMPTOM", "CAUSE", "ENVIRONMENT", "DIAGNOSING THE PROBLEM",
    "RESOLVING THE PROBLEM", "QUESTION", "ANSWER", "ABSTRACT", "CONTENT", "SUMMARY",
    "VULNERABILITY DETAILS", "AFFECTED PRODUCTS AND VERSIONS", "REMEDIATION/FIXES",
    "WORKAROUNDS AND MITIGATIONS", "REFERENCES", "RELATED INFORMATION", "ERROR DESCRIPTION",
    "LOCAL FIX", "TEMPORARY FIX", "COMMENTS", "APAR INFORMATION", "APAR STATUS",
    "APPLICABLE COMPONENT LEVELS", "FIX INFORMATION", "USERS AFFECTED", "DOWNLOAD DESCRIPTION",
    "DOWNLOAD PACKAGE", "INSTALLATION INSTRUCTIONS", "PREREQUISITES", "IMPORTANT NOTE",
    "DIRECT LINKS TO FIXES", "MODULES/MACROS", "PRODUCT ALIAS/SYNONYM", "HISTORICAL NUMBER",
    "SUBSCRIBE", "DISCLAIMER", "GET NOTIFIED ABOUT FUTURE SECURITY BULLETINS",
    "CHANGE HISTORY", "ACKNOWLEDGEMENT", "TAB NAVIGATION",
)  # fmt: skip
BOILERPLATE = {
    "SUBSCRIBE", "DISCLAIMER", "GET NOTIFIED ABOUT FUTURE SECURITY BULLETINS",
    "CHANGE HISTORY", "ACKNOWLEDGEMENT", "TAB NAVIGATION", "PRODUCT ALIAS/SYNONYM",
    "HISTORICAL NUMBER", "MODULES/MACROS", "APPLICABLE COMPONENT LEVELS",
}  # fmt: skip
ABSTRACT_SECTIONS = ("PROBLEM(ABSTRACT)", "ABSTRACT", "QUESTION", "SUMMARY", "PROBLEM")
_HEADER = re.compile(
    r"^\s*(" + "|".join(re.escape(h) for h in sorted(HEADERS, key=len, reverse=True)) + r"):?\s*$"
)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    section: str
    text: str


def windows(words: list[str], size: int = MAX_WORDS, overlap: int = OVERLAP) -> Iterator[str]:
    step = size - overlap
    for start in range(0, max(len(words) - overlap, 1), step):
        yield " ".join(words[start : start + size])


def fixed(doc_id: str, text: str) -> list[Chunk]:
    return [Chunk(f"{doc_id}:{i}", doc_id, "", w) for i, w in enumerate(windows(text.split()))]


def split_sections(text: str) -> list[tuple[str, str]]:
    """(header, body) pairs; text before the first header gets the header ''."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in text.splitlines():
        m = _HEADER.match(line)
        if m:
            sections.append((m.group(1), []))
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(body).strip()) for h, body in sections if "\n".join(body).strip()]


def _context(doc_type: str, sections: list[tuple[str, str]]) -> str:
    preamble = next((b for h, b in sections if h == ""), "")
    keywords = preamble.split("\n")[0][:160] if ";" in preamble[:200] else ""
    found = dict(sections)
    abstract = next((found[h] for h in ABSTRACT_SECTIONS if found.get(h)), "")
    abstract = " ".join(abstract.split()[:40])
    parts = [f"[{doc_type}]", keywords, abstract]
    return " | ".join(p for p in parts if p)


def by_section(doc_id: str, text: str, doc_type: str = "") -> list[Chunk]:
    sections = [(h, b) for h, b in split_sections(text) if h not in BOILERPLATE]
    context = _context(doc_type, sections)
    pieces: list[tuple[str, str]] = []
    pending_h: list[str] = []
    pending: list[str] = []

    def flush() -> None:
        if pending:
            pieces.append((" / ".join(h for h in pending_h if h), " ".join(pending)))
            pending_h.clear()
            pending.clear()

    for header, body in sections:
        words = body.split()
        if len(words) > MAX_WORDS:
            # A short section waiting to be flushed opens the long section's first window.
            label = " / ".join(h for h in [*pending_h, header] if h)
            pieces.extend((label, w) for w in windows(pending + words))
            pending_h.clear()
            pending.clear()
            continue
        if len(pending) + len(words) > MAX_WORDS:
            flush()
        pending_h.append(header)
        pending.extend(words)
        if len(pending) >= MIN_WORDS:
            flush()
    # A short tail joins the previous piece when the result still fits.
    if pending and pieces and len(pieces[-1][1].split()) + len(pending) <= MAX_WORDS + MIN_WORDS:
        label, body = pieces.pop()
        pending_h.insert(0, label)
        pending[:0] = body.split()
    flush()
    return [
        Chunk(f"{doc_id}:{i}", doc_id, h, f"{context}\n{h}\n{body}" if h else f"{context}\n{body}")
        for i, (h, body) in enumerate(pieces)
    ]


CHUNKERS: dict[str, Callable[[str, str, str], list[Chunk]]] = {
    "fixed": lambda doc_id, text, _type: fixed(doc_id, text),
    "sections": by_section,
}


def chunk_corpus(docs: pd.DataFrame, chunker: str) -> pd.DataFrame:
    fn = CHUNKERS[chunker]
    rows = [
        c.__dict__
        for d in docs.to_dict("records")
        for c in fn(str(d["doc_id"]), str(d["text"]), str(d["doc_type"]))
    ]
    out = pd.DataFrame(rows)
    out["doc_type"] = out.doc_id.map(dict(zip(docs.doc_id, docs.doc_type, strict=True)))
    out["words"] = [len(t.split()) for t in out.text]
    return out
