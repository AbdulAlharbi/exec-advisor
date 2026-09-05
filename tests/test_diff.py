from advisor.diff import diff, similarity
from advisor.models import Finding


def f(title, level, key=None, kind="risk", status="kept"):
    return Finding(id=f"X-{abs(hash(title)) % 10000}", kind=kind, key=key, title=title, level=level, evidence=["C-1"], rationale="r", action="a", status=status)


def test_key_match_beats_title_wording():
    prev = [f("Cash runway is roughly 8 months at current burn", "critical", key="runway")]
    cur = [f("Cash runway is roughly 7 months at current burn", "critical", key="runway")]
    assert [c.type for c in diff(cur, prev)] == ["unchanged"]


def test_title_similarity_matches_when_no_keys():
    prev = [f("Customer concentration: top account is 34% of ARR", "high")]
    cur = [f("Top account concentration now 36% of ARR", "critical")]
    ch = diff(cur, prev)
    assert ch[0].type == "escalated" and ch[0].from_level == "high" and ch[0].to_level == "critical"


def test_new_resolved_improved():
    prev = [f("Key renewal at risk", "high", key="renewal"), f("Gross margin fell 5 points", "high", key="margin")]
    cur = [f("Gross margin fell 2 points", "medium", key="margin"), f("A competitor is targeting the installed base", "medium", key="competitor")]
    types = {c.title: c.type for c in diff(cur, prev)}
    assert types == {
        "Gross margin fell 2 points": "improved",
        "A competitor is targeting the installed base": "new",
        "Key renewal at risk": "resolved",
    }


def test_dropped_findings_are_ignored_on_both_sides():
    prev = [f("Something", "high", key="x", status="dropped")]
    cur = [f("Something", "high", key="x", status="dropped")]
    assert diff(cur, prev) == []


def test_kinds_never_match_each_other():
    prev = [f("Expansion on the table", "high", kind="opportunity")]
    cur = [f("Expansion on the table", "high", kind="risk")]
    assert sorted(c.type for c in diff(cur, prev)) == ["new", "resolved"]


def test_similarity_is_symmetric_and_bounded():
    a, b = "Burn rate up 46% quarter on quarter", "Burn rate up 50% quarter on quarter"
    assert similarity(a, b) == similarity(b, a) and 0 < similarity(a, b) <= 1
