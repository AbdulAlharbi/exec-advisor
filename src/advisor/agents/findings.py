"""FindingsAgent: verified claims → risks or opportunities → grounding gate on evidence."""
from __future__ import annotations

from typing import Literal

from ..grounding import verify_findings
from ..models import Claim, Finding, FindingsDecision, short_id
from ..signals import Metric, Signals, metrics_to_dict
from .base import BaseAgent


class FindingsAgent(BaseAgent):
    def __init__(self, kind: Literal["risk", "opportunity"], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kind = kind
        self.task = "risks" if kind == "risk" else "opportunities"

    def run(self, claims: list[Claim], metrics: dict[str, Metric], signals: Signals, context: str) -> tuple[list[Finding], list[Finding]]:
        ctx = {
            "context": context,
            "claims": [c.model_dump(include={"id", "source_id", "quote", "statement", "metric", "value", "prior", "unit"}) for c in claims],
            "metrics": metrics_to_dict(metrics),
            "signals": signals.hits,
        }
        decision = self.decide(ctx, FindingsDecision)
        proposed = [
            Finding(id=short_id("R" if self.kind == "risk" else "O"), kind=self.kind, key=f.key, title=f.title, level=f.level,
                    likelihood=f.likelihood, horizon=f.horizon, evidence=f.evidence, rationale=f.rationale, action=f.action,
                    proposed_by=self.brain.name)
            for f in decision.findings
        ]
        kept, dropped = verify_findings(proposed, {c.id for c in claims})
        self.audit.record("grounding", self.run_id, stage=self.task, proposed=len(proposed), kept=len(kept),
                          dropped=[f.title for f in dropped])
        return kept, dropped
