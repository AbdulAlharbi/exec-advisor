"""BaseAgent: a brain, an audit log, and nothing else. Agents never write state directly."""
from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from ..audit import AuditLog
from ..llm.base import Brain

T = TypeVar("T", bound=BaseModel)


class BaseAgent:
    task: str = ""

    def __init__(self, brain: Brain, audit: AuditLog, run_id: str):
        self.brain = brain
        self.audit = audit
        self.run_id = run_id

    def decide(self, context: dict[str, Any], schema: type[T]) -> T:
        result = self.brain.decide(self.task, context, schema)
        fallback = getattr(self.brain, "last_error", None)
        self.audit.record(
            "decision", self.run_id, agent=self.__class__.__name__, task=self.task,
            brain=self.brain.name, fallback_error=fallback, output_summary=_summarise(result),
        )
        return result

    @property
    def decided_by(self) -> str:
        err = getattr(self.brain, "last_error", None)
        return f"heuristic (fallback after {err.split(':')[0]})" if err else self.brain.name


def _summarise(result: BaseModel) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in result.model_dump().items():
        out[name] = len(value) if isinstance(value, list) else value
    return out
