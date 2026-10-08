"""Ticket routing: queue, priority, type and tags."""

from __future__ import annotations

from collections.abc import Callable

from first_reply.routing.base import Router


def make(method: str) -> Router:
    from first_reply.embed import MODELS
    from first_reply.routing.embedding import EmbeddingLRRouter, KnnRouter
    from first_reply.routing.tfidf import TfidfRouter

    factories: dict[str, Callable[[], Router]] = {"tfidf_lr": TfidfRouter}
    for key in MODELS:
        factories[f"{key}_lr"] = lambda key=key: EmbeddingLRRouter(key)  # type: ignore[misc]
        factories[f"{key}_knn"] = lambda key=key: KnnRouter(key)  # type: ignore[misc]
    if method not in factories:
        raise KeyError(f"unknown routing method {method!r}; choose from {sorted(factories)}")
    return factories[method]()
