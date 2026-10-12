import sys
import types

import pytest

from first_reply.device import pick_device


def fake_torch(cuda: bool, mps: bool) -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)  # type: ignore[attr-defined]
    torch.backends = types.SimpleNamespace(  # type: ignore[attr-defined]
        mps=types.SimpleNamespace(is_available=lambda: mps)
    )
    return torch


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, False, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_prefers_a_gpu(
    monkeypatch: pytest.MonkeyPatch, cuda: bool, mps: bool, expected: str
) -> None:
    monkeypatch.delenv("FIRST_REPLY_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda, mps))
    assert pick_device() == expected


def test_the_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", fake_torch(True, False))
    monkeypatch.setenv("FIRST_REPLY_DEVICE", "cpu")
    assert pick_device() == "cpu"
