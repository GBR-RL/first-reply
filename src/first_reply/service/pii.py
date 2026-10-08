"""Mask personal data before an email reaches a model or the logs.

Rule-based on purpose: e-mail addresses, phone numbers, IBANs, card numbers and IP addresses
have recognisable shapes, and a rule never sends text anywhere. Names are not detected; the
ticket set already replaces them with placeholders.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"

_PATTERNS = (
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    # IPv4, but not right after "version", "release", "fix pack", "v": support emails are full
    # of four-part version numbers (9.1.0.4) that must survive masking.
    (
        "IP",
        re.compile(
            r"(?<!version )(?<!release )(?<!fix pack )(?<!fixpack )(?<![\w.])"
            rf"(?:{_OCTET}\.){{3}}{_OCTET}(?![\w.])",
            re.IGNORECASE,
        ),
    ),
    # International (+49 ..., 0049 ...) or national (0711 ...) format, at least 8 digits.
    ("PHONE", re.compile(r"(?<![\w+])(?:\+|00|0)\d(?:[ ()/-]{0,2}\d){7,}(?!\w)")),
)


@dataclass
class Masked:
    text: str
    found: dict[str, int] = field(default_factory=dict)


def mask(text: str) -> Masked:
    found: dict[str, int] = {}
    for label, pattern in _PATTERNS:
        text, n = pattern.subn(f"<{label.lower()}>", text)
        if n:
            found[label] = n
    return Masked(text, found)
