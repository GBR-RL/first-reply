"""Document access tiers and the roles allowed to read them.

TechQA has no access metadata, so the policy below is assigned from the document type to
exercise enforcement: APAR defect records are visible to support engineers, security bulletins
only to the security team (as for bulletins under embargo), everything else is public. The
filter is applied inside the vector search, not after it, so a restricted chunk can never reach
the prompt or the reply.
"""

from __future__ import annotations

from qdrant_client import models as qm

PUBLIC, PARTNER, INTERNAL = 0, 1, 2
TIER_BY_TYPE = {"apar": PARTNER, "security_bulletin": INTERNAL}
ROLES = {"customer": PUBLIC, "support_engineer": PARTNER, "security_team": INTERNAL}


def tier(doc_type: str) -> int:
    return TIER_BY_TYPE.get(doc_type, PUBLIC)


def role_filter(role: str) -> qm.Filter:
    if role not in ROLES:
        raise KeyError(f"unknown role {role!r}; choose from {sorted(ROLES)}")
    return qm.Filter(must=[qm.FieldCondition(key="tier", range=qm.Range(lte=ROLES[role]))])
