"""Deterministic metric and signal extraction over verified claims.

Both brains reason over the same named metrics and signals. For the rule brain they are the whole
input; for Claude they are hints alongside the claims. Every metric and signal carries the claim
id it came from, so anything built on them stays traceable to a verified quote.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import Claim

# Order matters: more specific labels first ("deferred revenue" before "revenue").
METRIC_LABELS: list[tuple[str, str]] = [
    ("deferred_revenue", r"\bdeferred revenue\b"),
    ("gross_margin", r"\bgross margin\b"),
    ("burn", r"\b(?:net )?burn\b"),
    ("cash", r"\bcash(?: on hand)?\b"),
    ("concentration", r"\bconcentration\b"),
    ("dso", r"\bdso\b"),
    ("nps", r"\bnps\b"),
    ("headcount", r"\bheadcount\b"),
    ("arr", r"(?<!of )\barr\b"),
    ("revenue", r"\brevenue\b"),
    ("sla_credits", r"\bsla credits?\b"),
    ("ticket_volume", r"\bticket volume\b"),
    ("lead_conversion", r"\bconversion\b"),
    ("first_response", r"\bfirst response time\b"),
    ("leads", r"\bleads\b"),
]

NUMBER = re.compile(r"\$?(\d+(?:,\d{3})*(?:\.\d+)?)\s*(M|m|k|K|%|days?|h|x)?(?!\w)")
FROM_TO = re.compile(r"from \$?(\d+(?:\.\d+)?)\s*(%|M|k)?\s*to \$?(\d+(?:\.\d+)?)\s*(%|M|k)?", re.I)
PRIOR = re.compile(r"(?:\(|\b)(?:q\d|was|may|april|prior|prev\w*|target|from|last quarter)[^\d$]{0,12}\$?(\d+(?:,\d{3})*(?:\.\d+)?)", re.I)

SIGNAL_PATTERNS: dict[str, str] = {
    "outage": r"\b(outage|degradation|downtime)\b",
    "termination_clause": r"termination[- ]for[- ]convenience",
    "resignation": r"\bresign\w*\b",
    "champion_left": r"champion left|has not returned|not returned (?:two|our) calls",
    "competitor": r"\bcompetit\w*\b|free migration tool",
    "early_termination": r"early termination|asked about (?:early )?terminat",
    "verbal_only": r"verbal yes|just paperwork|handshake",
    "budget_freeze": r"budget freeze|moved to q[1-4]",
    "burnout": r"\bburnout\b|same person for all",
    "unfilled_reqs": r"unfilled|open reqs",
    "promoters": r"\bpromoter\w*\b|saves us real money|fantastic",
    "leads_up": r"leads up|inbound .* up \d+%",
    "expansion": r"\bexpansion\b",
    "hedge_runway": r"comfortably above",
    "hedge_control": r"under control|growing pains",
    "hedge_expansion": r"make it up in expansion",
    "hedge_hiring": r"do not think we need|don't think we need",
    "hedge_forecast": r"still forecast at \d+%|forecast as commit",
    "hedge_best_quarter": r"best quarter ever",
    "hedge_leads": r"lower quality",
    "focus_shift": r"deprioriti[sz]e|built for one big customer",
}


@dataclass
class Metric:
    name: str
    value: float
    unit: str | None
    claim_id: str
    source_id: str
    prior: float | None = None

    @property
    def change_pct(self) -> float | None:
        if self.prior in (None, 0):
            return None
        return (self.value - self.prior) / self.prior * 100

    @property
    def delta(self) -> float | None:
        return None if self.prior is None else self.value - self.prior


@dataclass
class Signals:
    hits: dict[str, list[str]] = field(default_factory=dict)  # signal -> claim ids

    def __contains__(self, name: str) -> bool:
        return bool(self.hits.get(name))

    def ids(self, *names: str) -> list[str]:
        out: list[str] = []
        for n in names:
            for cid in self.hits.get(n, []):
                if cid not in out:
                    out.append(cid)
        return out

    def count(self, name: str) -> int:
        return len(self.hits.get(name, []))


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def extract_line_metrics(line: str) -> list[tuple[str, float, float | None, str | None]]:
    """Return [(metric, value, prior, unit)] for every label found in one line."""
    out: list[tuple[str, float, float | None, str | None]] = []
    seen: set[str] = set()
    low = line.lower()
    for name, pat in METRIC_LABELS:
        m = re.search(pat, low)
        if not m or name in seen:
            continue
        if name == "revenue" and "deferred_revenue" in seen:
            continue
        tail = line[m.end():]
        ft = FROM_TO.search(tail)
        if ft and ft.start() < 25:
            out.append((name, _to_float(ft.group(3)), _to_float(ft.group(1)), ft.group(4) or ft.group(2)))
            seen.add(name)
            continue
        num = NUMBER.search(tail)
        if not num:
            continue
        value, unit = _to_float(num.group(1)), num.group(2)
        if unit in ("k", "K"):
            value, unit = value / 1000, "M"
        if unit == "m":
            unit = "M"
        pri = PRIOR.search(tail[num.end():num.end() + 45])
        prior = _to_float(pri.group(1)) if pri else None
        out.append((name, value, prior, unit))
        seen.add(name)
    return out


def extract_metrics(claims: list[Claim]) -> dict[str, Metric]:
    """First occurrence of each metric wins; claims already carrying metric fields are honoured."""
    metrics: dict[str, Metric] = {}
    for c in claims:
        if c.metric and c.value is not None and c.metric not in metrics:
            metrics[c.metric] = Metric(c.metric, c.value, c.unit, c.id, c.source_id, c.prior)
        for name, value, prior, unit in extract_line_metrics(c.quote):
            if name not in metrics:
                metrics[name] = Metric(name, value, unit, c.id, c.source_id, prior)
    return metrics


def detect_signals(claims: list[Claim]) -> Signals:
    sig = Signals()
    for c in claims:
        text = c.quote.lower()
        for name, pat in SIGNAL_PATTERNS.items():
            if re.search(pat, text):
                sig.hits.setdefault(name, []).append(c.id)
    return sig


def metrics_to_dict(metrics: dict[str, Metric]) -> dict[str, dict]:
    return {
        k: {"value": m.value, "prior": m.prior, "unit": m.unit, "claim_id": m.claim_id, "change_pct": m.change_pct}
        for k, m in metrics.items()
    }
