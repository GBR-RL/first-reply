"""OpenAI-compatible chat client for the LiteLLM gateway or a bare llama.cpp server.

`FIRST_REPLY_LLM_URL` points at the gateway (or llama-server for development) and
`FIRST_REPLY_LLM_KEY` carries the gateway key. Structured outputs use `response_format` with a
JSON schema, which llama.cpp turns into a grammar, so the reply always parses.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import BaseModel

DEFAULT_URL = "http://127.0.0.1:4000/v1"


@dataclass
class Completion:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cost: float | None = None  # the gateway's chargeback cost, when called through it
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def json(self) -> Any:
        return json.loads(self.text)


class LLM:
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        *,
        api_key: str | None = None,
        timeout: float = 600.0,
        retries: int = 2,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.environ.get("FIRST_REPLY_LLM_URL") or DEFAULT_URL).rstrip(
            "/"
        )
        key = api_key or os.environ.get("FIRST_REPLY_LLM_KEY", "none")
        self.http = httpx.Client(
            timeout=timeout, headers={"Authorization": f"Bearer {key}"}, transport=transport
        )
        self.retries = retries

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        schema: type[BaseModel] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        metadata: dict[str, str] | None = None,
    ) -> Completion:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": 0,
            # Qwen3.5 thinks before answering by default; triage needs short, direct output.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
            }
        if metadata:
            body["metadata"] = metadata  # shows up in the gateway's spend logs
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            t0 = time.perf_counter()
            try:
                r = self.http.post(f"{self.base_url}/chat/completions", json=body)
                r.raise_for_status()
            except httpx.HTTPError as err:
                last = err
                time.sleep(2**attempt)
                continue
            data = r.json()
            usage = data.get("usage") or {}
            cost = r.headers.get("x-litellm-response-cost")
            return Completion(
                text=data["choices"][0]["message"]["content"] or "",
                model=str(data.get("model", self.model)),
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                latency_s=time.perf_counter() - t0,
                cost=float(cost) if cost is not None else None,
                raw=data,
            )
        raise RuntimeError(f"LLM call failed after {self.retries + 1} attempts") from last

    def parse(
        self, messages: list[dict[str, str]], schema: type[BaseModel], **kwargs: Any
    ) -> tuple[BaseModel, Completion]:
        c = self.chat(messages, schema=schema, **kwargs)
        return schema.model_validate_json(c.text), c
