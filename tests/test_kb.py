from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from qdrant_client import QdrantClient

from first_reply import embed
from first_reply.kb import access, chunking, index, sparse
from first_reply.kb.evaluate import metrics
from first_reply.kb.retrieve import Retriever, doc_ranking

TECHNOTE = (
    """db2; SQLCODE=-1585; temporary table space TECHNOTE (TROUBLESHOOTING)

PROBLEM(ABSTRACT)
The database transfer task fails with SQLCODE -1585.

CAUSE
No system temporary table space with a large enough page size exists.

RESOLVING THE PROBLEM
"""
    + " ".join(f"step{i} create a temporary table space with 32K pages" for i in range(80))
    + """

DISCLAIMER
All content is provided as is.
"""
)


def test_sections_drop_boilerplate_and_carry_context() -> None:
    chunks = chunking.by_section("d1", TECHNOTE, "troubleshooting")
    assert all("DISCLAIMER" not in c.section for c in chunks)
    assert all(c.text.startswith("[troubleshooting] | db2; SQLCODE=-1585") for c in chunks)
    assert "SQLCODE -1585" in chunks[0].text  # abstract travels with every chunk
    assert all(len(c.text.split()) < chunking.MAX_WORDS + 80 for c in chunks)
    assert len(chunks) >= 2


def test_fixed_windows_overlap() -> None:
    words = [f"w{i}" for i in range(700)]
    chunks = chunking.fixed("d1", " ".join(words))
    assert [len(c.text.split()) for c in chunks] == [300, 300, 200]
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-chunking.OVERLAP :] == second[: chunking.OVERLAP]


def test_bm25_vectors_favour_rare_terms_through_idf() -> None:
    idx, val = sparse.query_vector("The SQLCODE error")
    assert len(idx) == 2  # "the" is a stop word
    assert val == [1.0, 1.0]
    vecs = sparse.corpus_vectors(["sqlcode error error", "error"])
    assert len(vecs) == 2
    assert all(v > 0 for v in vecs[0][1])


def test_metrics() -> None:
    m = metrics(["a", "b", "c"], ["b", "x"])
    assert m["recall@1"] == 0.0
    assert m["recall@3"] == 0.5
    assert m["mrr@10"] == 0.5
    assert 0 < m["ndcg@10"] < 1


def test_role_filter_rejects_unknown_roles() -> None:
    with pytest.raises(KeyError):
        access.role_filter("admin")


@pytest.fixture
def kb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> QdrantClient:
    docs = pd.DataFrame(
        {
            "doc_id": ["pub", "apar", "sec"],
            "doc_type": ["troubleshooting", "apar", "security_bulletin"],
            "text": [
                "TECHNOTE\nPROBLEM\nServer hangs after failover of the queue manager.",
                "APAR STATUS\nERROR DESCRIPTION\nQueue manager hangs after failover, fix in 9.1.",
                "SECURITY BULLETIN\nSUMMARY\nQueue manager failover vulnerability CVE-2024-1.",
            ],
        }
    )
    chunks = chunking.chunk_corpus(docs, "sections")
    monkeypatch.setattr(index, "chunks_path", lambda chunker: tmp_path / f"{chunker}.parquet")
    chunks.to_parquet(tmp_path / "sections.parquet", index=False)
    monkeypatch.setattr(embed, "CACHE", tmp_path / "cache")
    rng = np.random.default_rng(0)
    v = rng.normal(size=(len(chunks), 8)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    path = embed.CACHE / "e5-small" / "kb-chunks-sections.npz"
    path.parent.mkdir(parents=True)
    np.savez(path, ids=chunks.chunk_id.to_numpy(dtype=str), vectors=v)
    qc = QdrantClient(":memory:")
    index.build("sections", qc)
    return qc


def test_access_filter_never_returns_restricted_documents(kb: QdrantClient) -> None:
    r = Retriever("sections", "e5-small", kb)
    query = "queue manager hangs after failover"
    allowed = {"customer": {"pub"}, "support_engineer": {"pub", "apar"}}
    for role, docs in allowed.items():
        for mode in ("bm25", "hybrid"):
            dense = [0.1] * 8 if mode == "hybrid" else None
            hits = r.search(query, mode=mode, dense=dense, role=role)
            assert {h.doc_id for h in hits} <= docs
            assert all(h.tier <= access.ROLES[role] for h in hits)
    everything = r.search(query, mode="bm25", role="security_team")
    assert {h.doc_id for h in everything} == {"pub", "apar", "sec"}


def test_bm25_ranks_the_matching_document_first(kb: QdrantClient) -> None:
    hits = Retriever("sections", None, kb).search("CVE-2024-1 vulnerability", mode="bm25")
    assert doc_ranking(hits)[0] == "sec"


def test_dense_mode_needs_a_model(kb: QdrantClient) -> None:
    with pytest.raises(ValueError, match="dense model"):
        Retriever("sections", None, kb).search("x", mode="dense")
