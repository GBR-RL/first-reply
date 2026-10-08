"""Ticket routing: queue, priority, type and tags."""

from __future__ import annotations

from collections.abc import Callable

from first_reply.routing.base import Router


def make(method: str) -> Router:
    from first_reply.routing.tfidf import TfidfRouter

    factories: dict[str, Callable[[], Router]] = {"tfidf_lr": TfidfRouter}
    if method not in factories:
        raise KeyError(f"unknown routing method {method!r}; choose from {sorted(factories)}")
    return factories[method]()
