import numpy as np
import pandas as pd

from first_reply.data import families, language, tickets


def test_language_detection_uses_text_not_label() -> None:
    assert language.detect("Sehr geehrter Support, ich habe ein Problem mit der Rechnung.") == "de"
    assert language.detect("Dear support, I have a problem with the invoice and the app.") == "en"


def _vectors() -> np.ndarray:
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(5, 32))
    rows = [c + 0.02 * rng.normal(size=32) for c in centres for _ in range(4)]
    rows += list(rng.normal(size=(10, 32)))  # unrelated singletons
    x = np.asarray(rows, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_families_link_near_duplicates_only() -> None:
    x = _vectors()
    out = families.assign([f"t{i:02d}" for i in range(len(x))], x, threshold=0.9)
    sizes = out.family.value_counts()
    assert sorted(sizes.tolist(), reverse=True)[:5] == [4, 4, 4, 4, 4]
    assert (sizes == 1).sum() == 10
    # Family names are the smallest member id.
    assert set(out[out.id.isin(["t00", "t01", "t02", "t03"])].family) == {"t00"}


def test_group_split_keeps_families_together() -> None:
    n = 3000
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "id": [f"t{i}" for i in range(n)],
            "queue": rng.choice(["A", "B", "C"], n),
            "lang": rng.choice(["en", "de"], n),
        }
    )
    groups = pd.Series(rng.integers(0, 600, n), index=df.index)
    out = tickets.assign_splits(df, seed=3, llm_subset={"dev": 20, "test": 50}, groups=groups)
    assert out.assign(g=groups).groupby("g").split.nunique().max() == 1
    shares = out.split.value_counts(normalize=True)
    assert abs(shares["test"] - 0.2) < 0.03
    assert abs(shares["dev"] - 0.1) < 0.03
