"""Domain models (what the system stores) and decision schemas (what a brain is asked to return).

An LLM never mutates state. It returns one of the *Decision schemas at the bottom of this file,
which is validated by pydantic and then mapped into the domain objects by the agents.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

Level = Literal["low", "medium", "high", "critical"]
Likelihood = Literal["unlikely", "possible", "likely"]
Posture = Literal["confident", "steady", "caution", "alarm"]
FindingKind = Literal["risk", "opportunity"]
FindingStatus = Literal["proposed", "kept", "weakened", "dropped"]
ChangeType = Literal["new", "resolved", "escalated", "improved", "unchanged"]

LEVEL_RANK: dict[str, int] = {"low": 1, "medium": 2, "high": 3, "critical": 4}


def now() -> datetime:
    return datetime.now(timezone.utc)


def short_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"


# --------------------------------------------------------------------------- domain


class Source(BaseModel):
    id: str
    title: str
    kind: Literal["file", "url", "inline"] = "file"
    text: str = ""
    path: str | None = None
    url: str | None = None
    fetched_at: datetime | None = None
    error: str | None = None

    @property
    def chars(self) -> int:
        return len(self.text)


class SourceInfo(BaseModel):
    """What the published brief says about a source (never the full text)."""

    id: str
    title: str
    kind: str
    chars: int
    fetched_at: datetime | None = None
    error: str | None = None


class Claim(BaseModel):
    """One atomic statement lifted from a source, with the verbatim quote that supports it."""

    id: str
    source_id: str
    quote: str
    statement: str
    metric: str | None = None
    value: float | None = None
    prior: float | None = None
    unit: str | None = None
    verified: bool | None = None
    verify_note: str = ""


class Finding(BaseModel):
    id: str
    kind: FindingKind
    key: str | None = None  # stable rule id (heuristic) — used by the diff engine when present
    title: str
    level: Level
    likelihood: Likelihood | None = None
    horizon: str = ""
    evidence: list[str] = Field(default_factory=list)  # claim ids
    rationale: str
    action: str  # mitigation for a risk, next step for an opportunity
    status: FindingStatus = "proposed"
    objections: list[str] = Field(default_factory=list)
    proposed_by: str = ""


class Assumption(BaseModel):
    id: str
    statement: str
    quote_claim: str | None = None  # claim id carrying the leadership's own words
    why_questionable: str
    test: str


class Recommendation(BaseModel):
    action: str
    rationale: str
    owner: str
    timeline: str
    expected_outcome: str
    linked: list[str] = Field(default_factory=list)  # finding ids


class Change(BaseModel):
    type: ChangeType
    kind: FindingKind
    title: str
    note: str
    from_level: Level | None = None
    to_level: Level | None = None


class Grounding(BaseModel):
    claims_extracted: int = 0
    claims_verified: int = 0
    claims_dropped: int = 0
    findings_proposed: int = 0
    findings_dropped: int = 0
    findings_weakened: int = 0
    dropped_examples: list[str] = Field(default_factory=list)


class Brief(BaseModel):
    run_id: str = Field(default_factory=lambda: short_id("RUN"))
    at: datetime = Field(default_factory=now)
    brain: str = ""
    model: str = ""
    previous_run_id: str | None = None
    context: str = ""
    headline: str = ""
    summary: str = ""
    posture: Posture = "steady"
    health_score: int = 50
    risks: list[Finding] = Field(default_factory=list)
    opportunities: list[Finding] = Field(default_factory=list)
    assumptions: list[Assumption] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    changes: list[Change] = Field(default_factory=list)
    grounding: Grounding = Field(default_factory=Grounding)
    sources: list[SourceInfo] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)  # verified claims only
    decided_by: dict[str, str] = Field(default_factory=dict)  # phase -> brain name
    notes: list[str] = Field(default_factory=list)

    def finding(self, fid: str) -> Finding | None:
        for f in self.risks + self.opportunities:
            if f.id == fid:
                return f
        return None

    def claim(self, cid: str) -> Claim | None:
        for c in self.claims:
            if c.id == cid:
                return c
        return None


# --------------------------------------------------------------------------- decision schemas


class ClaimSpec(BaseModel):
    source_id: str
    quote: str = Field(description="Verbatim text copied from the source. Must appear in the source exactly.")
    statement: str = Field(description="The claim in plain words.")
    metric: str | None = Field(default=None, description="Canonical metric name if this is a number, e.g. revenue, gross_margin, cash, burn, nps")
    value: float | None = None
    prior: float | None = None
    unit: str | None = None


class AnalystDecision(BaseModel):
    claims: list[ClaimSpec]


class FindingSpec(BaseModel):
    key: str | None = Field(default=None, description="Leave null. Stable rule id, set only by the rule brain.")
    title: str
    level: Level
    likelihood: Likelihood = "possible"
    horizon: str = ""
    evidence: list[str] = Field(description="Ids of the claims this rests on. Only verified claim ids are allowed.")
    rationale: str
    action: str


class FindingsDecision(BaseModel):
    findings: list[FindingSpec]


class Objection(BaseModel):
    finding_id: str
    objection: str
    verdict: Literal["sustain", "weaken", "drop"]


class AssumptionSpec(BaseModel):
    statement: str
    quote_claim: str | None = Field(default=None, description="Claim id containing leadership's own words, if any")
    why_questionable: str
    test: str


class ContrarianDecision(BaseModel):
    objections: list[Objection]
    assumptions: list[AssumptionSpec]


class RecommendationSpec(BaseModel):
    action: str
    rationale: str
    owner: str
    timeline: str
    expected_outcome: str
    linked: list[str] = Field(default_factory=list)


class EditorDecision(BaseModel):
    headline: str
    summary: str
    posture: Posture
    health_score: int = Field(ge=0, le=100)
    recommendations: list[RecommendationSpec]
    questions: list[str]
    gaps: list[str]
