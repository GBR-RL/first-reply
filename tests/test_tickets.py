import pandas as pd
import pytest

from first_reply.config import TICKETS
from first_reply.data import tickets

QUEUES = ["Technical Support", "Billing and Payments"]


def _raw(n: int = 200) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "subject": f"Subject {i}" if i % 3 else None,
                "body": f"Body number {i} with some text",
                "answer": f"Answer {i}",
                "type": "Incident",
                "queue": QUEUES[i % 2],
                "priority": "medium",
                "language": "EN" if i % 4 < 2 else "de",
                "release": "v5",
                **dict.fromkeys(tickets.TAG_COLUMNS),
                "tag_1": "Bug",
                "tag_2": " Network ",
            }
        )
    return pd.DataFrame(rows)


def test_normalize_collapses_case_and_whitespace() -> None:
    assert tickets.normalize("  Hello\n\tWORLD  ") == tickets.normalize("hello world")


def test_clean_removes_normalised_duplicates_keeping_first() -> None:
    raw = _raw(10)
    dup = raw.iloc[[0]].copy()
    dup["body"] = "  BODY number 0   with some text "
    dup["release"] = "v4"
    out = tickets.clean(pd.concat([raw, dup], ignore_index=True))
    assert len(out) == 10
    assert (out.release == "v5").all()
    assert out.id.is_unique


def test_clean_drops_unlabelled_rows_and_builds_fields() -> None:
    raw = _raw(6)
    raw.loc[1, "answer"] = None
    raw.loc[2, "queue"] = None
    out = tickets.clean(raw)
    assert len(out) == 4
    assert set(out.language) == {"en", "de"}
    assert out.tags.iloc[0] == ["Bug", "Network"]
    no_subject = out.subject == ""
    assert (out[no_subject].text == out[no_subject].body).all()
    assert out[~no_subject].text.str.startswith("Subject").all()
    assert no_subject.any()
    assert (~no_subject).any()


def test_splits_are_disjoint_stratified_and_reproducible() -> None:
    df = tickets.clean(_raw(4000))
    subset = {"dev": 40, "test": 100}
    a = tickets.assign_splits(df, seed=1, llm_subset=subset)
    b = tickets.assign_splits(df, seed=1, llm_subset=subset)
    assert a.split.tolist() == b.split.tolist()
    shares = a.split.value_counts(normalize=True)
    for split, share in tickets.SPLIT_SIZES.items():
        assert shares[split] == pytest.approx(share, abs=0.01)
    by_split = a.groupby("split").queue.value_counts(normalize=True)
    assert by_split["test"]["Technical Support"] == pytest.approx(0.5, abs=0.02)
    for split, n in subset.items():
        assert (a[a.llm_subset].split == split).sum() == n
    assert not a[a.llm_subset].split.eq("train").any()


@pytest.mark.data
def test_prepared_ticket_set() -> None:
    if not TICKETS.exists():
        pytest.skip("run `first-reply data` first")
    df = tickets.load()
    assert df.id.is_unique
    assert df.body.map(tickets.normalize).is_unique
    assert set(df.split) == {"train", "dev", "test"}
    assert set(df.lang) == {"en", "de"}
    assert df.queue.nunique() == 10
    assert df.groupby("family").split.nunique().max() == 1
    assert set(df.priority) == {"low", "medium", "high"}
