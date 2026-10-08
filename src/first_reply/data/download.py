"""Pinned, checksummed downloads of the two public datasets."""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from first_reply.config import RAW

_TICKETS = (
    "https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets/resolve/"
    "ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb/"
)
_RAGBENCH = (
    "https://huggingface.co/datasets/galileo-ai/ragbench/resolve/"
    "97808f3e5fd16ede40bbff6c2949af8139b2eb7b/techqa/"
)


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    sha256: str


SOURCES = (
    # Ticket set v5 (English and German, 28,587 rows) and v4 (20,000 rows). v5 repeats
    # 8,399 v4 tickets verbatim; the German-only file has no answers or types and is unused.
    Source(
        "tickets_v5.csv",
        _TICKETS + "aa_dataset-tickets-multi-lang-5-2-50-version.csv",
        "f187c090e59581c2bbf3aa1377c8db4dd647464ecf2ae51bf8966e42e0ed6bc0",
    ),
    Source(
        "tickets_v4.csv",
        _TICKETS + "dataset-tickets-multi-lang-4-20k.csv",
        "9be3bf810584fe01e8e83383e83dfd33f4c3910938ecad03ef151da79d8f0635",
    ),
    Source(
        "techqa_train.parquet",
        _RAGBENCH + "train-00000-of-00001.parquet",
        "4a729452c82aabbe14ea8658c43bd23afd6cb8db66f26f00f3aa780b93e12a66",
    ),
    Source(
        "techqa_validation.parquet",
        _RAGBENCH + "validation-00000-of-00001.parquet",
        "0a0fe4de7152af123ff482035bd6e495e690cfff0a4843230e082127262599a6",
    ),
    Source(
        "techqa_test.parquet",
        _RAGBENCH + "test-00000-of-00001.parquet",
        "426f360d12b5d3589b66293c3417d3686a9d69eba3e82938e93d3e1545681264",
    ),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(source: Source, raw: Path = RAW) -> Path:
    """Download one file unless a copy with the right checksum is already there."""
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / source.name
    if path.exists() and sha256(path) == source.sha256:
        return path
    tmp = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(source.url, tmp)
    got = sha256(tmp)
    if got != source.sha256:
        tmp.unlink()
        raise RuntimeError(f"{source.name}: checksum {got} != {source.sha256}")
    tmp.replace(path)
    return path


def fetch_all(raw: Path = RAW) -> dict[str, Path]:
    return {s.name: fetch(s, raw) for s in SOURCES}
