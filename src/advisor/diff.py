"""Change detection between two briefs, computed — never asked of the model.

Findings are matched across runs by a stable `key` when both sides have one (the heuristic brain
sets rule ids), otherwise by Jaccard similarity of their title tokens. Matched pairs are compared
on level to produce escalated / improved / unchanged; unmatched current findings are new;
unmatched previous findings are resolved.
"""
from __future__ import annotations

import re

from .models import LEVEL_RANK, Change, Finding

STOP = {
    "the", "a", "an", "of", "to", "in", "on", "and", "or", "is", "are", "for", "with", "at", "by",
    "from", "that", "this", "its", "it", "as", "be", "has", "have", "into", "up", "down", "than",
    "vs", "over", "under", "our", "their", "risk", "opportunity",
}
_TOKEN = re.compile(r"[a-z0-9]+")

JACCARD_THRESHOLD = 0.4


def fingerprint(title: str) -> frozenset[str]:
    return frozenset(t for t in _TOKEN.findall(title.lower()) if t not in STOP and len(t) > 2)


def similarity(a: str, b: str) -> float:
    fa, fb = fingerprint(a), fingerprint(b)
    if not fa or not fb:
        return 0.0
    return len(fa & fb) / len(fa | fb)


def match(current: list[Finding], previous: list[Finding]) -> list[tuple[Finding, Finding | None]]:
    """Greedy best-match. Keys win outright; otherwise highest similarity above threshold."""
    unmatched = list(previous)
    pairs: list[tuple[Finding, Finding | None]] = []
    for c in current:
        best: Finding | None = None
        best_score = 0.0
        for p in unmatched:
            if p.kind != c.kind:
                continue
            if c.key and p.key and c.key == p.key:
                best, best_score = p, 1.0
                break
            s = similarity(c.title, p.title)
            if s >= JACCARD_THRESHOLD and s > best_score:
                best, best_score = p, s
        if best is not None:
            unmatched.remove(best)
        pairs.append((c, best))
    return pairs


def diff(current: list[Finding], previous: list[Finding]) -> list[Change]:
    """Compare live findings (status != dropped) and return an ordered change list."""
    cur = [f for f in current if f.status != "dropped"]
    prev = [f for f in previous if f.status != "dropped"]
    pairs = match(cur, prev)
    matched_prev = {p.id for _, p in pairs if p is not None}
    changes: list[Change] = []

    for c, p in pairs:
        if p is None:
            changes.append(Change(type="new", kind=c.kind, title=c.title, to_level=c.level, note=f"new {c.kind} rated {c.level}"))
            continue
        rc, rp = LEVEL_RANK[c.level], LEVEL_RANK[p.level]
        if rc > rp:
            changes.append(Change(type="escalated", kind=c.kind, title=c.title, from_level=p.level, to_level=c.level,
                                  note=f"{p.level} → {c.level}"))
        elif rc < rp:
            changes.append(Change(type="improved", kind=c.kind, title=c.title, from_level=p.level, to_level=c.level,
                                  note=f"{p.level} → {c.level}"))
        else:
            note = f"still {c.level}"
            if p.status == "weakened" and c.status == "kept":
                note += "; contrarian objection lifted"
            elif p.status == "kept" and c.status == "weakened":
                note += "; now contested by the contrarian"
            changes.append(Change(type="unchanged", kind=c.kind, title=c.title, from_level=p.level, to_level=c.level, note=note))

    for p in prev:
        if p.id not in matched_prev:
            changes.append(Change(type="resolved", kind=p.kind, title=p.title, from_level=p.level,
                                  note=f"no longer present (was {p.level})"))

    order = {"escalated": 0, "new": 1, "resolved": 2, "improved": 3, "unchanged": 4}
    changes.sort(key=lambda ch: (order[ch.type], ch.kind, ch.title))
    return changes
