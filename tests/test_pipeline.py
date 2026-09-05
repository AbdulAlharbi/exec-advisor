"""End-to-end runs against the fixture company on the heuristic brain — no network, no key."""
import json

from advisor.audit import AuditLog
from advisor.chief_of_staff import ChiefOfStaff
from advisor.grounding import normalise
from advisor.llm.heuristic import HeuristicBrain
from advisor.models import AnalystDecision, ClaimSpec, FindingsDecision, FindingSpec
from advisor.sources import SourceManifest

from conftest import COMPANY, COMPANY_WEEK2


def run(settings, manifest_path=COMPANY, previous=None, brain=None):
    cos = ChiefOfStaff(brain or HeuristicBrain(), settings)
    return cos, cos.run(SourceManifest.load(manifest_path), previous)


def test_week1_reads_the_company_correctly(settings):
    cos, b = run(settings)
    keys = {f.key: f.level for f in b.risks if f.status != "dropped"}
    assert keys["runway"] == "critical" and keys["concentration"] == "critical" and keys["reliability"] == "critical"
    assert b.posture == "alarm" and b.health_score < 40
    assert "runway" in b.headline.lower()
    assert len(b.assumptions) >= 5
    weakened = {f.key for f in b.opportunities if f.status == "weakened"}
    assert "expansion" in weakened  # verbal yes ≠ signed
    assert b.grounding.claims_dropped == 0 and b.grounding.findings_dropped == 0
    assert b.recommendations and all(r.owner and r.timeline for r in b.recommendations)


def test_every_evidence_id_resolves_to_a_quote_that_exists_in_its_source(settings):
    _, b = run(settings)
    sources = {s.id: s for s in SourceManifest.load(COMPANY).materialise()}
    for f in b.risks + b.opportunities:
        assert f.evidence, f.title
        for cid in f.evidence:
            c = b.claim(cid)
            assert c is not None and c.verified
            assert normalise(c.quote) in normalise(sources[c.source_id].text)


def test_artifacts_report_audit_and_publish(settings):
    cos, b = run(settings)
    assert (settings.runs_dir / f"{b.run_id}.json").exists()
    report = (settings.reports_dir / f"{b.run_id}.md").read_text()
    assert b.headline in report and "How this brief was made" in report
    ok, n, _ = AuditLog(settings.audit_path).verify()
    assert ok and n > 15
    cos.publish(b)
    latest = json.loads((settings.data_dir / "latest.json").read_text())
    index = json.loads((settings.data_dir / "index.json").read_text())
    assert latest["run_id"] == b.run_id and index[0]["run_id"] == b.run_id
    assert (settings.data_dir / "history" / f"{b.run_id}.json").exists()


def test_week2_changes_are_computed_not_narrated(settings):
    cos, b1 = run(settings)
    _, b2 = run(settings, COMPANY_WEEK2, previous=b1, brain=cos.brain)
    assert b2.previous_run_id == b1.run_id
    types = {(c.type, c.title) for c in b2.changes}
    assert ("escalated", "Collections slowing: DSO 79 days from 48") in types
    assert any(t == "resolved" and "renewal" in title for t, title in types)
    lifted = [c for c in b2.changes if c.kind == "opportunity" and "objection lifted" in c.note]
    assert lifted and "expansion" in lifted[0].title.lower()
    assert not any(c.type == "new" for c in b2.changes)  # nothing invented between snapshots


class FabricatingBrain(HeuristicBrain):
    """A brain that hallucinates: invents a quote, cites the wrong source, and cites a fake claim id."""

    name = "fabricator"

    def decide(self, task, context, schema):
        result = super().decide(task, context, schema)
        if task == "analyst":
            result.claims += [
                ClaimSpec(source_id="q2-financials", quote="Revenue: $50M (Q1: $2M)", statement="invented"),
                ClaimSpec(source_id="ceo-memo", quote="NPS: 31 (May: 47).", statement="real quote, wrong source"),
            ]
        if task == "risks":
            result.findings.append(FindingSpec(title="Revenue collapsed to $2M", level="critical", evidence=["C-DOESNOTEXIST"], rationale="made up", action="panic"))
            result.findings.append(FindingSpec(title="Half-grounded finding", level="high", evidence=["C-DOESNOTEXIST", context["claims"][0]["id"]], rationale="mixed", action="x"))
        return result


def test_grounding_gate_stops_a_hallucinating_brain(settings):
    _, b = run(settings, brain=FabricatingBrain())
    assert b.grounding.claims_dropped == 2
    assert "quote not found in source" in " ".join(b.grounding.dropped_examples)
    assert not any("collapsed" in f.title for f in b.risks if f.status != "dropped")
    assert not any(c.quote.startswith("Revenue: $50M") for c in b.claims)
    half = next(f for f in b.risks if f.title == "Half-grounded finding")
    assert half.status != "dropped" and "C-DOESNOTEXIST" not in half.evidence and half.evidence
    assert b.grounding.findings_dropped >= 1


def test_bad_source_is_recorded_not_fatal(settings, tmp_path):
    manifest = tmp_path / "sources.yaml"
    manifest.write_text(
        "context: test\nsources:\n"
        "  - id: dead\n    title: Dead URL\n    url: http://127.0.0.1:9/nothing\n"
        "  - id: fin\n    title: Financials\n    text: 'Net burn: $2.0M/month (Q1: $1.0M). Cash on hand: $10.0M.'\n"
    )
    _, b = run(settings, manifest)
    dead = next(s for s in b.sources if s.id == "dead")
    assert dead.error and dead.chars == 0
    assert any(f.key == "runway" for f in b.risks)


def test_all_sources_empty_terminates_cleanly(settings, tmp_path):
    manifest = tmp_path / "sources.yaml"
    manifest.write_text("sources:\n  - id: dead\n    title: Dead\n    url: http://127.0.0.1:9/x\n")
    _, b = run(settings, manifest)
    assert b.headline == "No readable sources." and not b.risks
