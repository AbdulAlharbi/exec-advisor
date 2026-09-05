"""AnalystAgent: sources → claims → grounding gate → metrics and signals."""
from __future__ import annotations

from ..grounding import verify_claims
from ..models import AnalystDecision, Claim, Source, short_id
from ..signals import Metric, Signals, detect_signals, extract_metrics
from .base import BaseAgent


class AnalystAgent(BaseAgent):
    task = "analyst"

    def run(self, sources: list[Source], context: str) -> tuple[list[Claim], list[Claim], dict[str, Metric], Signals]:
        ctx = {"context": context, "sources": [{"id": s.id, "title": s.title, "text": s.text} for s in sources if s.text]}
        decision = self.decide(ctx, AnalystDecision)
        claims = [
            Claim(id=short_id("C"), source_id=c.source_id, quote=c.quote, statement=c.statement,
                  metric=c.metric, value=c.value, prior=c.prior, unit=c.unit)
            for c in decision.claims
        ]
        verified, dropped = verify_claims(claims, sources)
        self.audit.record("grounding", self.run_id, stage="claims", extracted=len(claims), verified=len(verified),
                          dropped=[{"quote": c.quote[:120], "why": c.verify_note} for c in dropped])
        metrics = extract_metrics(verified)
        signals = detect_signals(verified)
        self.audit.record("signals", self.run_id, metrics=sorted(metrics), signals={k: len(v) for k, v in signals.hits.items()})
        return verified, dropped, metrics, signals
