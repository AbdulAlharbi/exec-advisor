"""The grounding gate.

Every claim a brain produces carries a verbatim quote and a source id. This module checks, in
code, that the quote really appears in that source. Findings must rest on at least one verified
claim; any evidence id that does not resolve to a verified claim is stripped, and a finding left
with no evidence is dropped. No prompt can bypass this: the agents call it after every brain
decision.

Matching is whitespace- and case-insensitive and tolerant of typographic quotes and dashes, so a
model that copies correctly is never punished for a curly apostrophe — but a paraphrase, an
invented number, or a quote attributed to the wrong source fails.
"""
from __future__ import annotations

import re

from .models import Claim, Finding, Source

MIN_QUOTE_CHARS = 12

_TRANSLATE = str.maketrans(
    {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u00a0": " ",
    }
)
_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    return _WS.sub(" ", text.translate(_TRANSLATE)).strip().lower()


class GroundingReport:
    def __init__(self) -> None:
        self.claims_verified: list[Claim] = []
        self.claims_dropped: list[Claim] = []
        self.findings_kept: list[Finding] = []
        self.findings_dropped: list[Finding] = []

    @property
    def verified_ids(self) -> set[str]:
        return {c.id for c in self.claims_verified}


def verify_claims(claims: list[Claim], sources: list[Source]) -> tuple[list[Claim], list[Claim]]:
    """Return (verified, dropped). Each returned claim has `verified` and `verify_note` set."""
    by_id = {s.id: normalise(s.text) for s in sources}
    verified: list[Claim] = []
    dropped: list[Claim] = []
    for c in claims:
        c = c.model_copy()
        q = normalise(c.quote)
        if c.source_id not in by_id:
            c.verified, c.verify_note = False, f"unknown source {c.source_id!r}"
        elif len(q) < MIN_QUOTE_CHARS:
            c.verified, c.verify_note = False, f"quote too short ({len(q)} chars)"
        elif q not in by_id[c.source_id]:
            c.verified, c.verify_note = False, "quote not found in source"
        else:
            c.verified, c.verify_note = True, "quote found in source"
        (verified if c.verified else dropped).append(c)
    return verified, dropped


def verify_findings(findings: list[Finding], verified_ids: set[str]) -> tuple[list[Finding], list[Finding]]:
    """Strip evidence ids that are not verified claims; drop findings left with none."""
    kept: list[Finding] = []
    dropped: list[Finding] = []
    for f in findings:
        f = f.model_copy()
        stripped = [e for e in f.evidence if e not in verified_ids]
        f.evidence = [e for e in f.evidence if e in verified_ids]
        if stripped:
            f.objections.append(f"grounding: removed {len(stripped)} evidence reference(s) that were not verified claims")
        if f.evidence:
            kept.append(f)
        else:
            f.status = "dropped"
            f.objections.append("grounding: no verified evidence — dropped")
            dropped.append(f)
    return kept, dropped
