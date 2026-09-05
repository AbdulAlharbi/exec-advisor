"""Deterministic Markdown report. Every finding prints the verified quotes it rests on."""
from __future__ import annotations

from pathlib import Path

from .models import Brief, Finding

POSTURE = {"confident": "Confident", "steady": "Steady", "caution": "Proceed with caution", "alarm": "Act now"}


def _finding(b: Brief, f: Finding, label: str) -> list[str]:
    lines = [f"### {f.title}", f"*{label}: {f.level} · likelihood: {f.likelihood or 'n/a'} · horizon: {f.horizon or 'n/a'} · status: {f.status}*", "", f.rationale, "", f"**{'Reduce it' if f.kind == 'risk' else 'First step'}:** {f.action}", "", "Evidence:"]
    for cid in f.evidence:
        c = b.claim(cid)
        if c:
            src = next((s.title for s in b.sources if s.id == c.source_id), c.source_id)
            lines.append(f'- "{c.quote}" — *{src}* (`{cid}`)')
    for o in f.objections:
        lines.append(f"- ⚠ {o}")
    return lines + [""]


def render(b: Brief) -> str:
    L: list[str] = [f"# Executive brief — {b.at:%Y-%m-%d %H:%M} UTC", "", f"**{b.headline}**", "", b.summary, "",
                    f"Health {b.health_score}/100 · Posture: {POSTURE.get(b.posture, b.posture)} · Brain: {b.brain}{(' ' + b.model) if b.model else ''} · Run `{b.run_id}`", ""]
    if b.changes:
        L += [f"## What changed since `{b.previous_run_id}`", ""]
        for c in b.changes:
            if c.type != "unchanged":
                L.append(f"- **{c.type}** ({c.kind}) {c.title} — {c.note}")
        unchanged = sum(1 for c in b.changes if c.type == "unchanged")
        if unchanged:
            L.append(f"- {unchanged} finding(s) unchanged")
        L.append("")
    L += ["## Risks", ""]
    for f in b.risks:
        if f.status != "dropped":
            L += _finding(b, f, "severity")
    L += ["## Opportunities", ""]
    for f in b.opportunities:
        if f.status != "dropped":
            L += _finding(b, f, "impact")
    L += ["## Assumptions worth challenging", ""]
    for a in b.assumptions:
        L += [f'### "{a.statement}"']
        if a.quote_claim and (c := b.claim(a.quote_claim)):
            L.append(f'> "{c.quote}"')
        L += ["", a.why_questionable, "", f"**Test it:** {a.test}", ""]
    L += ["## Recommendations", ""]
    for i, r in enumerate(b.recommendations, 1):
        L += [f"{i}. **{r.action}**", f"   - Why: {r.rationale}", f"   - Owner: {r.owner} · When: {r.timeline}", f"   - Success looks like: {r.expected_outcome}", ""]
    L += ["## Ask your team this week", ""] + [f"- {q}" for q in b.questions] + ["", "## What the advisor cannot see", ""] + [f"- {g}" for g in b.gaps] + [""]
    dropped = [f for f in b.risks + b.opportunities if f.status == "dropped"]
    g = b.grounding
    L += ["## How this brief was made", "",
          f"- Claims extracted {g.claims_extracted}, verified against sources {g.claims_verified}, dropped {g.claims_dropped}",
          f"- Findings proposed {g.findings_proposed}, dropped {g.findings_dropped}, weakened by the contrarian {g.findings_weakened}",
          "- Decided by: " + ", ".join(f"{k}={v}" for k, v in b.decided_by.items()), ""]
    if g.dropped_examples:
        L += ["Dropped claims (quote not found in the named source):", ""] + [f"- {d}" for d in g.dropped_examples] + [""]
    if dropped:
        L += ["Dropped findings:", ""] + [f"- {f.title}: {f.objections[-1] if f.objections else ''}" for f in dropped] + [""]
    L += ["Sources:", ""] + [f"- {s.title} ({s.kind}, {s.chars:,} chars{', error: ' + s.error if s.error else ''})" for s in b.sources]
    return "\n".join(L) + "\n"


def write_report(b: Brief, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(b))
    return path
