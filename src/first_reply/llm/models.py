"""Open-weight models served by llama.cpp (4-bit GGUF, Apache 2.0)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Model:
    name: str
    repo: str  # Hugging Face repository
    file: str
    params_b: float

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/main/{self.file}"


MODELS = {
    m.name: m
    for m in (
        Model("qwen3.5-4b", "unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf", 4.2),
        Model(
            "granite-4.2-3b", "ibm-granite/granite-4.2-3b-GGUF", "granite-4.2-3b-Q4_K_M.gguf", 3.7
        ),
    )
}
