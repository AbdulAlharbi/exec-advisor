"""EditorAgent: surviving findings + assumptions + computed changes → the brief's narrative and recommendations."""
from __future__ import annotations

from ..models import Assumption, Change, EditorDecision, Finding, Recommendation
from ..signals import Metric, Signals, metrics_to_dict
from .base import BaseAgent


class EditorAgent(BaseAgent):
    task = "editor"

    def run(self, risks: list[Finding], opportunities: list[Finding], assumptions: list[Assumption], changes: list[Change],
            metrics: dict[str, Metric], signals: Signals, context: str) -> EditorDecision:
        dump = {"id", "key", "kind", "title", "level", "likelihood", "horizon", "rationale", "action", "status", "objections", "evidence"}
        ctx = {
            "context": context,
            "risks": [f.model_dump(include=dump) for f in risks],
            "opportunities": [f.model_dump(include=dump) for f in opportunities],
            "assumptions": [a.model_dump() for a in assumptions],
            "changes": [c.model_dump() for c in changes],
            "metrics": metrics_to_dict(metrics),
            "signals": signals.hits,
        }
        decision = self.decide(ctx, EditorDecision)
        live = {f.id for f in risks + opportunities if f.status != "dropped"}
        for r in decision.recommendations:
            r.linked = [x for x in r.linked if x in live]
        return decision


def to_recommendations(decision: EditorDecision) -> list[Recommendation]:
    return [Recommendation(**r.model_dump()) for r in decision.recommendations]
