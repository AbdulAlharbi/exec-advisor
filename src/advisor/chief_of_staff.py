"""ChiefOfStaff — the orchestrator.

Sequences ingest → analyst → risks ∥ opportunities → contrarian → diff → editor → publish, checkpoints
the brief after every phase, and owns the terminal artefacts (run JSON, report, site data). It never
calls a brain itself and it never bypasses the grounding gate: the agents apply it, the orchestrator
only records what survived.
"""
from __future__ import annotations

import json
from pathlib import Path

from .agents import AnalystAgent, ContrarianAgent, EditorAgent, FindingsAgent
from .audit import AuditLog
from .config import Settings
from .diff import diff
from .llm.base import Brain
from .models import Brief, Grounding, Recommendation, SourceInfo
from .report import write_report
from .sources import SourceManifest


class BriefStore:
    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def save(self, brief: Brief) -> Path:
        p = self.runs_dir / f"{brief.run_id}.json"
        p.write_text(brief.model_dump_json(indent=2))
        return p

    def load(self, run_id: str) -> Brief:
        return Brief.model_validate_json((self.runs_dir / f"{run_id}.json").read_text())

    def latest(self) -> Brief | None:
        runs = sorted(self.runs_dir.glob("RUN-*.json"), key=lambda p: p.stat().st_mtime)
        return Brief.model_validate_json(runs[-1].read_text()) if runs else None

    def list(self) -> list[Brief]:
        return sorted((Brief.model_validate_json(p.read_text()) for p in self.runs_dir.glob("RUN-*.json")), key=lambda b: b.at)


class ChiefOfStaff:
    def __init__(self, brain: Brain, settings: Settings, audit: AuditLog | None = None):
        self.brain = brain
        self.settings = settings
        self.audit = audit or AuditLog(settings.audit_path)
        self.store = BriefStore(settings.runs_dir)

    def _checkpoint(self, brief: Brief, phase: str) -> None:
        self.store.save(brief)
        self.audit.record("phase", brief.run_id, phase=phase)

    def run(self, manifest: SourceManifest, previous: Brief | None = None) -> Brief:
        brief = Brief(brain=self.brain.name, model=getattr(self.brain, "model", ""), context=manifest.context,
                      previous_run_id=previous.run_id if previous else None)
        rid = brief.run_id
        self.audit.record("run_start", rid, brain=self.brain.name, manifest=str(manifest.path), previous=brief.previous_run_id)

        # ---- ingest
        sources = manifest.materialise(self.settings.fetch_timeout, self.settings.max_source_chars)
        brief.sources = [SourceInfo(id=s.id, title=s.title, kind=s.kind, chars=s.chars, fetched_at=s.fetched_at, error=s.error) for s in sources]
        for s in sources:
            self.audit.record("source", rid, id=s.id, kind=s.kind, chars=s.chars, error=s.error)
        if not any(s.text for s in sources):
            brief.headline = "No readable sources."
            brief.notes.append("Every source was empty or failed to load; nothing to analyse.")
            self._checkpoint(brief, "ingest_failed")
            return brief
        self._checkpoint(brief, "ingest")

        # ---- analyst → grounding gate
        analyst = AnalystAgent(self.brain, self.audit, rid)
        claims, dropped_claims, metrics, signals = analyst.run(sources, manifest.context)
        brief.claims = claims
        brief.decided_by["analyst"] = analyst.decided_by
        g = Grounding(claims_extracted=len(claims) + len(dropped_claims), claims_verified=len(claims), claims_dropped=len(dropped_claims),
                      dropped_examples=[f"{c.quote[:90]} — {c.verify_note}" for c in dropped_claims[:5]])
        brief.grounding = g
        self._checkpoint(brief, "analyst")

        # ---- risks and opportunities (independent; could run in parallel) → grounding gate
        risk_agent = FindingsAgent("risk", self.brain, self.audit, rid)
        opp_agent = FindingsAgent("opportunity", self.brain, self.audit, rid)
        risks, r_dropped = risk_agent.run(claims, metrics, signals, manifest.context)
        opps, o_dropped = opp_agent.run(claims, metrics, signals, manifest.context)
        brief.decided_by["risks"] = risk_agent.decided_by
        brief.decided_by["opportunities"] = opp_agent.decided_by
        g.findings_proposed = len(risks) + len(opps) + len(r_dropped) + len(o_dropped)
        g.findings_dropped = len(r_dropped) + len(o_dropped)
        brief.risks, brief.opportunities = risks, opps
        self._checkpoint(brief, "findings")

        # ---- contrarian: may only weaken or drop
        contrarian = ContrarianAgent(self.brain, self.audit, rid)
        all_findings, assumptions = contrarian.run(risks + opps, claims, metrics, signals, manifest.context)
        brief.decided_by["contrarian"] = contrarian.decided_by
        brief.risks = [f for f in all_findings if f.kind == "risk"]
        brief.opportunities = [f for f in all_findings if f.kind == "opportunity"]
        brief.assumptions = assumptions
        g.findings_dropped += sum(1 for f in all_findings if f.status == "dropped")
        g.findings_weakened = sum(1 for f in all_findings if f.status == "weakened")
        self._checkpoint(brief, "contrarian")

        # ---- change detection: computed, never asked
        if previous is not None:
            brief.changes = diff(brief.risks + brief.opportunities, previous.risks + previous.opportunities)
            self.audit.record("diff", rid, against=previous.run_id, changes={t: sum(1 for c in brief.changes if c.type == t) for t in ("new", "escalated", "improved", "resolved", "unchanged")})
        self._checkpoint(brief, "diff")

        # ---- editor
        editor = EditorAgent(self.brain, self.audit, rid)
        decision = editor.run(brief.risks, brief.opportunities, assumptions, brief.changes, metrics, signals, manifest.context)
        brief.decided_by["editor"] = editor.decided_by
        brief.headline, brief.summary, brief.posture, brief.health_score = decision.headline, decision.summary, decision.posture, decision.health_score
        brief.recommendations = [Recommendation(**r.model_dump()) for r in decision.recommendations]
        brief.questions, brief.gaps = decision.questions, decision.gaps
        brief.risks.sort(key=lambda f: (f.status == "dropped", -{"low": 1, "medium": 2, "high": 3, "critical": 4}[f.level]))
        brief.opportunities.sort(key=lambda f: (f.status == "dropped", -{"low": 1, "medium": 2, "high": 3, "critical": 4}[f.level]))
        self._checkpoint(brief, "editor")

        # ---- report + publish
        self.settings.reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.settings.reports_dir / f"{rid}.md"
        write_report(brief, report_path)
        brief.notes.append(f"report: {report_path}")
        self._checkpoint(brief, "report")
        self.audit.record("run_end", rid, health=brief.health_score, posture=brief.posture, risks=len(brief.risks), opportunities=len(brief.opportunities))
        return brief

    def publish(self, brief: Brief, keep_history: int = 50) -> Path:
        """Write the site's data files: latest.json, history/<run>.json and index.json."""
        data = self.settings.data_dir
        (data / "history").mkdir(parents=True, exist_ok=True)
        payload = brief.model_dump(mode="json")
        (data / "history" / f"{brief.run_id}.json").write_text(json.dumps(payload, indent=1))
        (data / "latest.json").write_text(json.dumps(payload, indent=1))
        index_path = data / "index.json"
        index = json.loads(index_path.read_text()) if index_path.exists() else []
        index = [e for e in index if e.get("run_id") != brief.run_id]
        index.insert(0, {"run_id": brief.run_id, "at": payload["at"], "headline": brief.headline, "health_score": brief.health_score,
                         "posture": brief.posture, "brain": brief.brain, "risks": len([f for f in brief.risks if f.status != "dropped"]),
                         "opportunities": len([f for f in brief.opportunities if f.status != "dropped"])})
        index_path.write_text(json.dumps(index[:keep_history], indent=1))
        self.audit.record("publish", brief.run_id, path=str(data / "latest.json"))
        return data / "latest.json"
