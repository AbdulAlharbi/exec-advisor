import pytest

from advisor.models import Claim
from advisor.signals import detect_signals, extract_line_metrics, extract_metrics


@pytest.mark.parametrize(
    "line, expected",
    [
        ("Net burn: $1.9M/month (Q1: $1.3M). Cash on hand: $14.6M.", [("burn", 1.9, 1.3, "M"), ("cash", 14.6, None, "M")]),
        ("Revenue: $8.4M (Q1: $7.1M, +18% QoQ). ARR now $31.2M.", [("arr", 31.2, None, "M"), ("revenue", 8.4, 7.1, "M")]),
        ("Deferred revenue: $4.1M, down from $5.3M.", [("deferred_revenue", 4.1, 5.3, "M")]),
        ("conversion down from 11% to 6%. Reps say the leads are lower quality.", [("lead_conversion", 6.0, 11.0, "%")]),
        ("SLA credits issued in August: $212k.", [("sla_credits", 0.212, None, "M")]),
        ("Top customer concentration: Kestrel Retail = 34% of ARR (was 19% in Q4).", [("concentration", 34.0, 19.0, "%")]),
        ("DSO: 71 days (Q1: 48).", [("dso", 71.0, 48.0, "days")]),
        ("NPS: 31 (May: 47). Enterprise segment NPS: 12.", [("nps", 31.0, 47.0, None)]),
        ("Nothing numeric here.", []),
    ],
)
def test_line_metrics(line, expected):
    assert extract_line_metrics(line) == expected


def test_metrics_and_signals_carry_claim_ids():
    claims = [
        Claim(id="C-A", source_id="s", quote="Net burn: $1.9M/month (Q1: $1.3M). Cash on hand: $14.6M.", statement="x"),
        Claim(id="C-B", source_id="s", quote="Two AEs resigned in August.", statement="x"),
        Claim(id="C-C", source_id="s", quote="The engineer has flagged burnout to his manager.", statement="x"),
    ]
    m = extract_metrics(claims)
    assert m["cash"].claim_id == "C-A" and m["burn"].prior == 1.3 and m["burn"].change_pct == pytest.approx(46.15, abs=0.1)
    s = detect_signals(claims)
    assert s.ids("resignation") == ["C-B"] and "burnout" in s and "burn" not in s.hits
