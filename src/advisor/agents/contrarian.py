"""ContrarianAgent: may only attack. Applies verdicts to findings and surfaces leadership assumptions."""
from __future__ import annotations

from ..models import Assumption, Claim, ContrarianDecision, Finding, short_id
from ..signals import Metric, Signals, metrics_to_dict
from .base import BaseAgent


class ContrarianAgent(BaseAgent):
    task = "contrarian"

    def run(self, findings: list[Finding], claims: list[Claim], metrics: dict[str, Metric], signals: Signals, context: str) -> tuple[list[Finding], list[Assumption]]:
        ctx = {
            "context": context,
            "findings": [f.model_dump(include={"id", "kind", "key", "title", "level", "evidence", "rationale"}) for f in findings],
            "claims": [c.model_dump(include={"id", "source_id", "quote", "metric", "value", "prior", "unit"}) for c in claims],
            "metrics": metrics_to_dict(metrics),
            "signals": signals.hits,
        }
        decision = self.decide(ctx, ContrarianDecision)
        by_id = {f.id: f for f in findings}
        verdicts = {"sustain": 0, "weaken": 0, "drop": 0, "unknown_target": 0}
        for o in decision.objections:
            f = by_id.get(o.finding_id)
            if f is None:
                verdicts["unknown_target"] += 1
                continue
            verdicts[o.verdict] += 1
            f.objections.append(f"contrarian ({o.verdict}): {o.objection}")
            if o.verdict == "drop":
                f.status = "dropped"
            elif o.verdict == "weaken" and f.status != "dropped":
                f.status = "weakened"
            elif f.status == "proposed":
                f.status = "kept"
        for f in findings:
            if f.status == "proposed":
                f.status = "kept"
        claim_ids = {c.id for c in claims}
        assumptions = [
            Assumption(id=short_id("A"), statement=a.statement, why_questionable=a.why_questionable, test=a.test,
                       quote_claim=a.quote_claim if a.quote_claim in claim_ids else None)
            for a in decision.assumptions
        ]
        self.audit.record("contrarian", self.run_id, verdicts=verdicts, assumptions=len(assumptions))
        return findings, assumptions
