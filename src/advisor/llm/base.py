"""The Brain interface every agent talks to.

A brain is asked to `decide(task, context, schema)` and must return a validated instance of
`schema`. Two implementations exist behind this one interface: Claude (Messages API) and a
deterministic rule brain. Agents cannot tell them apart, which is what lets the whole pipeline
and its tests run offline.
"""
from __future__ import annotations

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

TASKS = ("analyst", "risks", "opportunities", "contrarian", "editor")


class Brain(Protocol):
    name: str

    def decide(self, task: str, context: dict[str, Any], schema: type[T]) -> T: ...


def make_brain(kind: str, model: str = "claude-sonnet-5") -> Brain:
    if kind == "heuristic":
        from .heuristic import HeuristicBrain

        return HeuristicBrain()
    if kind == "claude":
        from .claude import ClaudeBrain

        return ClaudeBrain(model=model)
    raise ValueError(f"unknown brain {kind!r} (expected 'claude' or 'heuristic')")
