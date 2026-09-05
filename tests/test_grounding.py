from advisor.grounding import verify_claims, verify_findings
from advisor.models import Claim, Finding, Source

SRC = Source(id="fin", title="Financials", kind="inline", text='Net burn: $1.9M/month (Q1: $1.3M). Board deck says runway is "comfortably above 12 months".')


def claim(quote: str, source_id: str = "fin") -> Claim:
    return Claim(id="C-1", source_id=source_id, quote=quote, statement=quote)


def test_exact_quote_is_verified():
    ok, bad = verify_claims([claim("Net burn: $1.9M/month (Q1: $1.3M).")], [SRC])
    assert len(ok) == 1 and ok[0].verified is True and not bad


def test_curly_quotes_and_whitespace_are_tolerated():
    ok, bad = verify_claims([claim("says runway is  “comfortably above 12   months”")], [SRC])
    assert len(ok) == 1 and not bad


def test_paraphrase_is_dropped():
    ok, bad = verify_claims([claim("Burn is $1.9M a month")], [SRC])
    assert not ok and bad[0].verify_note == "quote not found in source"


def test_altered_number_is_dropped():
    ok, bad = verify_claims([claim("Net burn: $1.4M/month")], [SRC])
    assert not ok and len(bad) == 1


def test_wrong_source_is_dropped():
    ok, bad = verify_claims([claim("Net burn: $1.9M/month", source_id="memo")], [SRC])
    assert not ok and "unknown source" in bad[0].verify_note


def test_short_quote_is_dropped():
    ok, bad = verify_claims([claim("$1.9M")], [SRC])
    assert not ok and "too short" in bad[0].verify_note


def test_finding_with_unverified_evidence_is_stripped_or_dropped():
    good = Finding(id="R-1", kind="risk", title="a", level="high", evidence=["C-1", "C-FAKE"], rationale="r", action="a")
    bad = Finding(id="R-2", kind="risk", title="b", level="high", evidence=["C-FAKE"], rationale="r", action="a")
    kept, dropped = verify_findings([good, bad], {"C-1"})
    assert [f.id for f in kept] == ["R-1"] and kept[0].evidence == ["C-1"]
    assert "removed 1 evidence" in kept[0].objections[0]
    assert dropped[0].id == "R-2" and dropped[0].status == "dropped"
