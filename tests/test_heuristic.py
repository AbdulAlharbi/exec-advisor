from advisor.llm.heuristic import HeuristicBrain
from advisor.models import ContrarianDecision, EditorDecision, FindingsDecision


def metrics(**kw):
    return {k: {"value": v[0], "prior": v[1], "unit": v[2], "claim_id": f"C-{k}"} for k, v in kw.items()}


def test_runway_rule_computes_months_and_cites_both_claims():
    d = HeuristicBrain().decide("risks", {"metrics": metrics(cash=(14.6, None, "M"), burn=(1.9, 1.3, "M")), "signals": {}}, FindingsDecision)
    runway = next(f for f in d.findings if f.key == "runway")
    assert runway.level == "critical" and "8 months" in runway.title
    assert set(runway.evidence) == {"C-cash", "C-burn"}


def test_runway_rule_silent_when_healthy():
    d = HeuristicBrain().decide("risks", {"metrics": metrics(cash=(40.0, None, "M"), burn=(1.0, 1.0, "M")), "signals": {}}, FindingsDecision)
    assert not [f for f in d.findings if f.key == "runway"]


def test_contrarian_weakens_verbal_expansion_and_drops_unevidenced():
    ctx = {
        "metrics": {}, "signals": {"verbal_only": ["C-1"]}, "claims": [{"id": "C-1", "quote": "verbal yes"}],
        "findings": [
            {"id": "O-1", "kind": "opportunity", "key": "expansion", "evidence": ["C-1"]},
            {"id": "R-1", "kind": "risk", "key": "x", "evidence": []},
        ],
    }
    d = HeuristicBrain().decide("contrarian", ctx, ContrarianDecision)
    verdict = {o.finding_id: o.verdict for o in d.objections}
    assert verdict == {"O-1": "weaken", "R-1": "drop"}
    assert any("just paperwork" in a.statement for a in d.assumptions)


def test_editor_score_is_bounded_and_posture_consistent():
    risks = [{"id": f"R{i}", "key": "runway", "kind": "risk", "title": "t", "level": "critical", "status": "kept", "rationale": "r"} for i in range(10)]
    d = HeuristicBrain().decide("editor", {"metrics": {}, "signals": {}, "risks": risks, "opportunities": [], "assumptions": []}, EditorDecision)
    assert d.health_score == 5 and d.posture == "alarm"
    d2 = HeuristicBrain().decide("editor", {"metrics": {}, "signals": {}, "risks": [], "opportunities": [], "assumptions": []}, EditorDecision)
    assert d2.health_score == 95 and d2.posture == "confident" and d2.headline.startswith("No material risks")
