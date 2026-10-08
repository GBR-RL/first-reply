import numpy as np
import pandas as pd
import pytest

from first_reply.config import KB_DOCS, KB_QUESTIONS, KB_RESPONSES
from first_reply.data import techqa


def _raw() -> pd.DataFrame:
    a = "TECHNOTE (TROUBLESHOOTING)\nPROBLEM(ABSTRACT)\nThe server hangs."
    b = "SECURITY BULLETIN\nSUMMARY\nA vulnerability affects the product."
    c = "SUBSCRIBE TO THIS APAR\nAPAR STATUS\n * CLOSED AS PROGRAM ERROR."
    return pd.DataFrame(
        [
            {
                "id": "q1",
                "split": "train",
                "question": "Why does the server hang?",
                "documents": np.array([a, b]),
                "all_relevant_sentence_keys": np.array(["0b", "0c"]),
                "response": "Because of X.",
                "generation_model_name": "gen-a",
                "response_sentences": np.array([np.array(["a", "Because of X."])]),
                "unsupported_response_sentence_keys": np.array([], dtype=object),
                "adherence_score": True,
            },
            {
                "id": "q2",
                "split": "test",
                "question": "Is my product affected? ",
                "documents": np.array([b, c]),
                "all_relevant_sentence_keys": np.array([], dtype=object),
                "response": "Not mentioned.",
                "generation_model_name": "gen-a",
                "response_sentences": np.array([np.array(["a", "Not mentioned."])]),
                "unsupported_response_sentence_keys": np.array(["a"]),
                "adherence_score": False,
            },
            {
                "id": "q3",
                "split": "test",
                "question": "Why does the server hang?",
                "documents": np.array([a, c]),
                "all_relevant_sentence_keys": np.array(["1a"]),
                "response": "Dup.",
                "generation_model_name": "gen-b",
                "response_sentences": np.array([np.array(["a", "Dup."])]),
                "unsupported_response_sentence_keys": np.array([], dtype=object),
                "adherence_score": True,
            },
        ]
    )


def test_doc_type_uses_the_document_header() -> None:
    assert techqa.doc_type("TECHNOTE (FAQ)\nQUESTION\nHow do I") == "faq"
    assert techqa.doc_type("WMB IIB SECURITY BULLETIN\nSUMMARY") == "security_bulletin"
    assert techqa.doc_type("z/os A FIX IS AVAILABLE\nObtain the fix") == "apar"
    assert techqa.doc_type("Some unrelated page") == "other"


def test_docs_are_pooled_and_unique() -> None:
    docs = techqa.build_docs(_raw())
    assert len(docs) == 3
    assert docs.doc_id.is_unique
    assert set(docs.doc_type) == {"troubleshooting", "security_bulletin", "apar"}


def test_gold_documents_follow_sentence_keys_and_duplicates_are_dropped() -> None:
    q = techqa.build_questions(_raw()).set_index("qid")
    assert list(q.index) == ["q1", "q2"]  # q3 repeats q1's question in a later split
    assert q.loc["q1", "gold_doc_ids"] == [q.loc["q1", "candidate_doc_ids"][0]]
    assert q.loc["q1", "answerable"]
    assert not q.loc["q2", "answerable"]
    assert q.loc["q2", "question"] == "Is my product affected?"


def test_responses_keep_only_the_kept_question_ids() -> None:
    raw = _raw()
    q = techqa.build_questions(raw)
    r = techqa.build_responses(raw, q)
    assert list(r.qid) == ["q1", "q2"]
    assert r.set_index("qid").loc["q2", "unsupported_keys"] == ["a"]
    assert r.set_index("qid").loc["q1", "sentences"] == ["Because of X."]


@pytest.mark.data
def test_prepared_knowledge_base() -> None:
    if not (KB_DOCS.exists() and KB_QUESTIONS.exists() and KB_RESPONSES.exists()):
        pytest.skip("run `first-reply data` first")
    docs, q = techqa.load_docs(), techqa.load_questions()
    assert docs.doc_id.is_unique
    assert q.question.is_unique
    ids = set(docs.doc_id)
    assert all(set(g) <= ids for g in q.gold_doc_ids)
    assert q.answerable.mean() > 0.8
    r = techqa.load_responses()
    assert set(r.qid) == set(q.qid)
    assert r.groupby("qid").size().max() == 2
