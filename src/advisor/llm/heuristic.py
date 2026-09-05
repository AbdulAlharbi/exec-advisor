"""HeuristicBrain — explicit rules over the same metrics and signals Claude sees.

It is the floor the model has to beat, the CI back-end, the offline demo, and the production
fallback when the API is unavailable. Every finding it proposes carries a stable `key` (the rule
id), which the diff engine uses to match findings across runs.
"""
from __future__ import annotations

import re
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from ..models import (
    AnalystDecision,
    AssumptionSpec,
    ClaimSpec,
    ContrarianDecision,
    EditorDecision,
    FindingsDecision,
    FindingSpec,
    Objection,
    RecommendationSpec,
)
from ..signals import Metric, Signals, extract_line_metrics

T = TypeVar("T", bound=BaseModel)

CLAIM_KEYWORDS = re.compile(
    r"resign|churn|outage|terminat|escalat|verbal|freeze|competit|burnout|unfilled|promoter|"
    r"detractor|comfortably|under control|growing pains|deprioriti|forecast|renewal|champion|"
    r"lower quality|make it up|best quarter|do not think|paperwork|isolation|onboarding|isolated",
    re.I,
)

# ------------------------------------------------------------------ recommendation templates
# owner, timeline, expected outcome — keyed by rule id so the editor can link them to findings.
RULE_RECS: dict[str, dict[str, str]] = {
    "runway": {
        "action": "Rebuild the cash forecast on current burn with no unsigned expansion revenue, and set a burn ceiling",
        "owner": "CFO",
        "timeline": "Before the next board meeting",
        "expected_outcome": "A runway number the board can defend, and a trigger for a cost plan if it drops under 12 months",
    },
    "concentration": {
        "action": "Cap the single-customer share of ARR in the plan and put a second-account expansion programme behind it",
        "owner": "CRO with CFO",
        "timeline": "This quarter",
        "expected_outcome": "A written plan that keeps the top customer under 25% of ARR by year end",
    },
    "reliability": {
        "action": "Freeze feature work on the affected subsystem, fill the open platform reqs, and put a change-management gate on production migrations",
        "owner": "CTO",
        "timeline": "Start this week; reqs filled within 60 days",
        "expected_outcome": "Zero SLA breaches in the next 90-day window; on-call rotated across at least four engineers",
    },
    "renewal": {
        "action": "Get an executive sponsor in front of the new decision-maker at the at-risk account before the renewal date",
        "owner": "CEO with CRO",
        "timeline": "Within two weeks",
        "expected_outcome": "A confirmed meeting and a re-forecast of the renewal based on what is heard",
    },
    "sales_attrition": {
        "action": "Review the comp plan change with the remaining top-quartile reps and correct it before more leave",
        "owner": "CRO",
        "timeline": "Two weeks",
        "expected_outcome": "No further regretted attrition this quarter",
    },
    "nps": {
        "action": "Run a detractor call programme with the enterprise segment and publish a reliability commitment",
        "owner": "VP Customer Success",
        "timeline": "30 days",
        "expected_outcome": "Enterprise NPS back above 25 at the next survey",
    },
    "margin": {
        "action": "Separate one-off onboarding cost from structural margin and decide which it is",
        "owner": "CFO",
        "timeline": "This month",
        "expected_outcome": "A gross-margin bridge the board can read",
    },
    "burn": {
        "action": "Tie further hiring to signed, not verbal, revenue",
        "owner": "CEO",
        "timeline": "Immediately",
        "expected_outcome": "Burn flat quarter on quarter until the expansion is signed",
    },
    "dso": {
        "action": "Move the accounts that switched to monthly billing back to annual prepay at renewal, with a discount if needed",
        "owner": "CFO",
        "timeline": "Next renewal cycle",
        "expected_outcome": "DSO under 55 days",
    },
    "deferred": {
        "action": "Track deferred revenue as a board metric and explain any shift to monthly billing",
        "owner": "CFO",
        "timeline": "Next board pack",
        "expected_outcome": "Deferred revenue trend visible and understood",
    },
    "competitor": {
        "action": "Build a retention offer for the mid-market base the competitor is targeting",
        "owner": "CRO",
        "timeline": "30 days",
        "expected_outcome": "No mid-market logo churn attributed to the competitor",
    },
    "smb_churn": {
        "action": "Call every SMB account that asked about leaving, and reverse the SMB feature freeze until the enterprise expansion is signed",
        "owner": "CEO",
        "timeline": "Two weeks",
        "expected_outcome": "Termination requests withdrawn; SMB NPS holds above 40",
    },
    "key_person": {
        "action": "Rotate on-call this week and backfill the platform team before anything else",
        "owner": "CTO",
        "timeline": "Immediately",
        "expected_outcome": "No engineer on call more than one week in four",
    },
    "headcount": {
        "action": "Pause non-critical hiring until revenue growth exceeds headcount growth",
        "owner": "CEO with CFO",
        "timeline": "Immediately",
        "expected_outcome": "Revenue per head rising by next quarter",
    },
    "leads": {
        "action": "A/B test the pricing page change against the previous version and fix conversion before blaming lead quality",
        "owner": "VP Marketing",
        "timeline": "Two weeks",
        "expected_outcome": "Conversion back to at least 9%",
    },
    "core_value": {
        "action": "Turn the quantified savings customers report into a retention and renewal message",
        "owner": "VP Marketing with CS",
        "timeline": "30 days",
        "expected_outcome": "Savings evidence used in every renewal conversation",
    },
    "expansion": {
        "action": "Get the expansion order form signed before any plan depends on it",
        "owner": "CRO",
        "timeline": "Before quarter close",
        "expected_outcome": "Signed paper, or the expansion moved out of Commit",
    },
}

# canonical metrics an executive brief needs; missing ones become "gaps"
EXPECTED_METRICS = {
    "revenue": "revenue and growth",
    "gross_margin": "gross margin trend",
    "burn": "net burn",
    "cash": "cash on hand",
    "concentration": "customer concentration",
    "nps": "customer sentiment (NPS)",
    "headcount": "headcount",
}
ALWAYS_GAPS = [
    "Gross and net revenue retention by segment — the brief cannot tell whether growth is new logos or expansion",
    "Pipeline coverage and stage-weighted forecast, not verbal commitments",
    "A 12-month cash forecast that separates signed from unsigned revenue",
]


def _metrics(ctx: dict[str, Any]) -> dict[str, Metric]:
    out: dict[str, Metric] = {}
    for name, d in (ctx.get("metrics") or {}).items():
        out[name] = Metric(name, float(d["value"]), d.get("unit"), d["claim_id"], d.get("source_id", ""), d.get("prior"))
    return out


def _signals(ctx: dict[str, Any]) -> Signals:
    return Signals(hits={k: list(v) for k, v in (ctx.get("signals") or {}).items()})


def _fmt(v: float | None, unit: str | None) -> str:
    if v is None:
        return "?"
    if unit == "M":
        return f"${v * 1000:g}k" if v < 1 else f"${v:g}M"
    if unit == "%":
        return f"{v:g}%"
    if unit and unit.startswith("day"):
        return f"{v:g} days"
    return f"{v:g}"


class HeuristicBrain:
    name = "heuristic"

    def decide(self, task: str, context: dict[str, Any], schema: type[T]) -> T:
        handler: Callable[[dict[str, Any]], BaseModel] = {
            "analyst": self._analyst,
            "risks": self._risks,
            "opportunities": self._opportunities,
            "contrarian": self._contrarian,
            "editor": self._editor,
        }[task]
        result = handler(context)
        if not isinstance(result, schema):
            raise TypeError(f"heuristic returned {type(result).__name__} for task {task!r}")
        return result

    # ------------------------------------------------------------------ analyst
    def _analyst(self, ctx: dict[str, Any]) -> AnalystDecision:
        claims: list[ClaimSpec] = []
        for src in ctx.get("sources", []):
            for raw in str(src.get("text", "")).splitlines():
                line = raw.strip().lstrip("-•* ").strip()
                if len(line) < 12:
                    continue
                if not (re.search(r"\d", line) or CLAIM_KEYWORDS.search(line)):
                    continue
                ms = extract_line_metrics(line)
                spec = ClaimSpec(source_id=src["id"], quote=line, statement=line)
                if ms:
                    name, value, prior, unit = ms[0]
                    spec.metric, spec.value, spec.prior, spec.unit = name, value, prior, unit
                claims.append(spec)
        return AnalystDecision(claims=claims)

    # ------------------------------------------------------------------ risks
    def _risks(self, ctx: dict[str, Any]) -> FindingsDecision:
        m, s = _metrics(ctx), _signals(ctx)
        out: list[FindingSpec] = []

        def add(key: str, title: str, level: str, ev: list[str], rationale: str, likelihood: str = "likely", horizon: str = "this quarter") -> None:
            ev = [e for i, e in enumerate(ev) if e and e not in ev[:i]]
            if not ev:
                return
            out.append(FindingSpec(title=title, level=level, likelihood=likelihood, horizon=horizon, evidence=ev,
                                   rationale=rationale, action=RULE_RECS[key]["action"], key=key))

        if "cash" in m and "burn" in m and m["burn"].value > 0:
            months = m["cash"].value / m["burn"].value
            if months < 12:
                add("runway", f"Cash runway is roughly {months:.0f} months at current burn",
                    "critical" if months < 9 else "high", [m["cash"].claim_id, m["burn"].claim_id],
                    f"Cash {_fmt(m['cash'].value, m['cash'].unit)} against burn {_fmt(m['burn'].value, m['burn'].unit)} per month gives about {months:.1f} months — under the 12-month floor most boards require before a raise.",
                    horizon="next 6-9 months")

        if "concentration" in m and m["concentration"].value >= 25:
            c = m["concentration"]
            level = "critical" if (c.prior and c.value > c.prior * 1.3) else "high"
            add("concentration", f"Top customer is {c.value:g}% of ARR", level, [c.claim_id] + s.ids("termination_clause"),
                f"Single-customer share is {c.value:g}%" + (f", up from {c.prior:g}%" if c.prior else "") + ". Losing or renegotiating that account moves the whole plan." +
                (" The same customer holds a contractual exit right." if "termination_clause" in s else ""))

        if s.count("outage") >= 2:
            clause = "termination_clause" in s
            add("reliability", "Repeated outages on the core subsystem" + (" have triggered a customer's termination right" if clause else ""),
                "critical" if clause else "high", s.ids("outage", "termination_clause") + ([m["sla_credits"].claim_id] if "sla_credits" in m else []),
                f"{s.count('outage')} incidents recorded on the same subsystem" + (f"; SLA credits of {_fmt(m['sla_credits'].value, m['sla_credits'].unit)}." if "sla_credits" in m else ".") +
                (" The third breach in a rolling window activates termination for convenience." if clause else ""), horizon="now")

        if "champion_left" in s:
            add("renewal", "A key renewal is being forecast on a relationship that no longer exists", "high", s.ids("champion_left"),
                "The champion has left and the replacement is not engaging, yet the deal is still forecast at a high probability.", horizon="before the renewal date")

        if "resignation" in s:
            add("sales_attrition", "Top-quartile sales reps are leaving", "high", s.ids("resignation"),
                "Regretted attrition in the sales team, with a comp-plan change named as the cause.", likelihood="likely")

        if "nps" in m and m["nps"].prior and m["nps"].prior - m["nps"].value >= 10:
            n = m["nps"]
            add("nps", f"Customer sentiment falling: NPS {n.value:g} from {n.prior:g}", "high", [n.claim_id] + s.ids("focus_shift"),
                f"A {n.prior - n.value:g}-point NPS drop in one period, with detractors citing reliability and a perceived shift of focus to one customer.")

        if "gross_margin" in m and m["gross_margin"].prior and m["gross_margin"].prior - m["gross_margin"].value >= 3:
            g = m["gross_margin"]
            add("margin", f"Gross margin fell {g.prior - g.value:g} points", "high" if g.prior - g.value >= 5 else "medium", [g.claim_id],
                f"Margin moved from {g.prior:g}% to {g.value:g}%. If onboarding cost is structural rather than one-off, the enterprise model is less profitable than presented.", likelihood="possible")

        if "burn" in m and m["burn"].change_pct and m["burn"].change_pct >= 25:
            b = m["burn"]
            add("burn", f"Burn rate up {b.change_pct:.0f}% quarter on quarter", "high", [b.claim_id] + ([m["headcount"].claim_id] if "headcount" in m else []),
                f"Monthly burn rose from {_fmt(b.prior, b.unit)} to {_fmt(b.value, b.unit)}, ahead of any signed expansion revenue.")

        if "dso" in m and m["dso"].change_pct and m["dso"].change_pct >= 30:
            d = m["dso"]
            add("dso", f"Collections slowing: DSO {d.value:g} days from {d.prior:g}", "high" if d.change_pct >= 60 else "medium", [d.claim_id] + ([m["deferred_revenue"].claim_id] if "deferred_revenue" in m else []),
                "Cash is arriving later while burn is rising; combined with renewals moving to monthly billing this compounds the runway problem.", likelihood="likely")
        elif "deferred_revenue" in m and m["deferred_revenue"].prior and m["deferred_revenue"].value < m["deferred_revenue"].prior:
            d = m["deferred_revenue"]
            add("deferred", "Deferred revenue is shrinking", "medium", [d.claim_id], "Customers moving off annual prepay reduces committed cash and can signal wavering commitment.", likelihood="possible")

        if "early_termination" in s:
            add("smb_churn", "SMB customers are asking how to leave", "high" if "focus_shift" in s else "medium", s.ids("early_termination", "focus_shift"),
                "Annual-plan customers asking about early termination is a leading churn indicator, coinciding with a deliberate deprioritisation of that segment.")

        if "competitor" in s:
            add("competitor", "A competitor is targeting the installed base with a free migration tool", "medium", s.ids("competitor"),
                "Customers are raising the competitor unprompted on calls.", likelihood="possible", horizon="6 months")

        if "burnout" in s or "unfilled_reqs" in s:
            add("key_person", "The platform team is a single point of failure", "high" if "burnout" in s else "medium", s.ids("burnout", "unfilled_reqs"),
                "One engineer has carried every incident and has flagged burnout; open platform reqs have gone unfilled for months.")

        if "headcount" in m and "revenue" in m and m["headcount"].change_pct and m["revenue"].change_pct and m["headcount"].change_pct > m["revenue"].change_pct:
            h, r = m["headcount"], m["revenue"]
            add("headcount", f"Headcount grew {h.change_pct:.0f}% while revenue grew {r.change_pct:.0f}%", "medium", [h.claim_id, r.claim_id],
                "Cost is scaling faster than revenue; the productivity story has to be shown, not assumed.", likelihood="likely")

        return FindingsDecision(findings=out)

    # ------------------------------------------------------------------ opportunities
    def _opportunities(self, ctx: dict[str, Any]) -> FindingsDecision:
        m, s = _metrics(ctx), _signals(ctx)
        out: list[FindingSpec] = []
        if "leads_up" in s or "leads" in m:
            ev = s.ids("leads_up") + ([m["leads"].claim_id] if "leads" in m else []) + ([m["lead_conversion"].claim_id] if "lead_conversion" in m else [])
            conv = m.get("lead_conversion")
            out.append(FindingSpec(title="Inbound demand is rising; conversion is the fixable bottleneck", level="high", likelihood="likely", horizon="30 days",
                                   evidence=list(dict.fromkeys(ev)), key="leads",
                                   rationale="Lead volume is up sharply" + (f" while conversion fell from {conv.prior:g}% to {conv.value:g}%" if conv and conv.prior else "") + ". Recovering conversion on the higher volume is cheaper growth than any new channel.",
                                   action=RULE_RECS["leads"]["action"]))
        if "promoters" in s:
            out.append(FindingSpec(title="Customers can quantify the savings the product delivers", level="medium", likelihood="likely", horizon="this quarter",
                                   evidence=s.ids("promoters"), key="core_value",
                                   rationale="Promoters cite hard savings and a strong onboarding team — the retention message is already written by customers.",
                                   action=RULE_RECS["core_value"]["action"]))
        if "expansion" in s:
            out.append(FindingSpec(title="A large expansion is on the table", level="high", likelihood="possible", horizon="this quarter",
                                   evidence=s.ids("expansion"), key="expansion",
                                   rationale="An expansion of this size would materially lift ARR — if it closes.",
                                   action=RULE_RECS["expansion"]["action"]))
        return FindingsDecision(findings=out)

    # ------------------------------------------------------------------ contrarian
    def _contrarian(self, ctx: dict[str, Any]) -> ContrarianDecision:
        m, s = _metrics(ctx), _signals(ctx)
        objections: list[Objection] = []
        for f in ctx.get("findings", []):
            ev = f.get("evidence", [])
            if not ev:
                objections.append(Objection(finding_id=f["id"], objection="No verified evidence supports this.", verdict="drop"))
            elif f.get("key") == "expansion" and "verbal_only" in s:
                objections.append(Objection(finding_id=f["id"], verdict="weaken",
                                            objection="The expansion is a verbal commitment described as 'just paperwork'; nothing is signed and the plan already depends on it."))
            elif f.get("kind") == "opportunity" and len(ev) == 1 and not any(c.get("metric") for c in ctx.get("claims", []) if c["id"] == ev[0]):
                objections.append(Objection(finding_id=f["id"], verdict="weaken", objection="Rests on a single qualitative line of evidence; no number behind it yet."))
            else:
                objections.append(Objection(finding_id=f["id"], verdict="sustain", objection="Evidence is consistent across sources."))

        assumptions: list[AssumptionSpec] = []
        claims = {c["id"]: c for c in ctx.get("claims", [])}

        def first(*names: str) -> str | None:
            ids = s.ids(*names)
            return ids[0] if ids else None

        def q(cid: str | None) -> str:
            return claims.get(cid or "", {}).get("quote", "")

        if "hedge_runway" in s:
            cid = first("hedge_runway")
            why = "The runway claim is asserted, not computed."
            if "cash" in m and "burn" in m and m["burn"].value:
                why = f"On the numbers in the same document, {_fmt(m['cash'].value, m['cash'].unit)} of cash at {_fmt(m['burn'].value, m['burn'].unit)} per month is about {m['cash'].value / m['burn'].value:.1f} months — not 'comfortably above 12'."
            assumptions.append(AssumptionSpec(statement="Runway is comfortably above 12 months", quote_claim=cid, why_questionable=why,
                                              test="Rebuild the cash forecast on current burn with unsigned expansion excluded; compare to the board deck figure."))
        if "hedge_control" in s:
            assumptions.append(AssumptionSpec(statement="The reliability problems are growing pains that the platform team has under control", quote_claim=first("hedge_control"),
                                              why_questionable=f"{s.count('outage')} incidents on the same subsystem, one engineer on call for all of them flagging burnout, and open reqs unfilled for months is not 'under control'; it is a team at capacity.",
                                              test="Ask for the incident trend by week and the on-call roster for the last 90 days."))
        if "verbal_only" in s:
            assumptions.append(AssumptionSpec(statement="The expansion will close this quarter because procurement is 'just paperwork'", quote_claim=first("verbal_only"),
                                              why_questionable="A verbal yes from an operations VP is not a signature, and the same customer has just escalated a third SLA breach to the CEO and holds a termination right.",
                                              test="Ask procurement for the signed order form date; until then move the deal out of Commit."))
        if "hedge_forecast" in s and "champion_left" in s:
            assumptions.append(AssumptionSpec(statement="The at-risk renewal is still a 90% probability", quote_claim=first("hedge_forecast"),
                                              why_questionable="The champion left in July and the new decision-maker is not returning calls; a 90% forecast is inertia, not evidence.",
                                              test="Re-forecast at 50% until the new VP takes a meeting; get an executive sponsor to request one this week."))
        if "hedge_expansion" in s:
            assumptions.append(AssumptionSpec(statement="Enterprise customer-success hiring will be paid back by expansion revenue", quote_claim=first("hedge_expansion"),
                                              why_questionable="The expansion is unsigned and concentration is already " + (f"{m['concentration'].value:g}%" if "concentration" in m else "high") + "; the cost is certain and the revenue is not.",
                                              test="Model CS cost against signed expansion only and show the payback period."))
        if "hedge_hiring" in s:
            assumptions.append(AssumptionSpec(statement="Engineering headcount can wait until after the raise", quote_claim=first("hedge_hiring"),
                                              why_questionable="The reliability incidents are the thing most likely to derail the raise; deferring the fix until after it is circular.",
                                              test="Cost two platform hires against one more SLA breach with the top customer."))
        if "hedge_best_quarter" in s:
            assumptions.append(AssumptionSpec(statement="Q2 was the best quarter ever", quote_claim=first("hedge_best_quarter"),
                                              why_questionable="Revenue grew, but margin, burn, DSO, deferred revenue and NPS all moved the wrong way in the same quarter. Best by one metric, worst by five.",
                                              test="Put the six metrics side by side on one slide and ask the board which quarter it prefers."))
        if "hedge_leads" in s:
            assumptions.append(AssumptionSpec(statement="Inbound conversion fell because the leads are lower quality", quote_claim=first("hedge_leads"),
                                              why_questionable="Conversion halved immediately after a pricing page change; lead quality is the explanation that requires no one to have made a mistake.",
                                              test="Restore the previous pricing page for half the traffic for two weeks and compare conversion."))
        return ContrarianDecision(objections=objections, assumptions=assumptions)

    # ------------------------------------------------------------------ editor
    def _editor(self, ctx: dict[str, Any]) -> EditorDecision:
        m = _metrics(ctx)
        risks = [f for f in ctx.get("risks", []) if f.get("status") != "dropped"]
        opps = [f for f in ctx.get("opportunities", []) if f.get("status") != "dropped"]
        weight = {"critical": 14, "high": 6, "medium": 3, "low": 1}
        score = 100 - sum(weight[f["level"]] for f in risks) + sum({"high": 4, "medium": 2, "low": 1}[f["level"]] for f in opps if f.get("status") != "weakened")
        score = max(5, min(95, score))
        posture = "confident" if score >= 75 else "steady" if score >= 60 else "caution" if score >= 40 else "alarm"

        rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        risks_sorted = sorted(risks, key=lambda f: -rank[f["level"]])
        crit = [f for f in risks if f["level"] == "critical"]
        high = [f for f in risks if f["level"] == "high"]
        top = risks_sorted[0] if risks_sorted else None

        if top:
            headline = f"{top['title']}. " + (f"{len(crit)} critical and {len(high)} high risks are live" if crit else f"{len(high)} high risks are live") + f"; {len(opps)} opportunities."
        else:
            headline = "No material risks found in the sources provided."
        deteriorating = [n for n in ("gross_margin", "nps", "deferred_revenue") if n in m and m[n].prior and m[n].value < m[n].prior]
        worsening = [n for n in ("burn", "dso", "concentration") if n in m and m[n].prior and m[n].value > m[n].prior]
        wrong = ", ".join(deteriorating + worsening).replace("_", " ")
        summary = ""
        if "revenue" in m and m["revenue"].change_pct:
            summary = f"Revenue is growing ({m['revenue'].change_pct:.0f}% on the period)"
            summary += f", but {wrong} all moved the wrong way in the same period. " if wrong else ". "
        elif wrong:
            summary = f"{wrong.capitalize()} all moved the wrong way in the period. "
        summary += f"The advisor's reading: {'the growth story is masking a concentration and reliability problem that could unwind quickly' if crit else 'manageable, if the top risks are owned this quarter'}."
        ass = ctx.get("assumptions", [])
        if ass:
            summary += f" {len(ass)} assumptions in the leadership narrative do not survive the evidence."

        recs: list[RecommendationSpec] = []
        seen: set[str] = set()
        for f in risks_sorted + sorted(opps, key=lambda f: -rank[f["level"]]):
            key = f.get("key")
            if not key or key in seen or key not in RULE_RECS:
                continue
            seen.add(key)
            t = RULE_RECS[key]
            recs.append(RecommendationSpec(action=t["action"], rationale=f["rationale"], owner=t["owner"], timeline=t["timeline"],
                                           expected_outcome=t["expected_outcome"], linked=[f["id"]]))
            if len(recs) >= 6:
                break

        questions = [a["test"] for a in ass][:5]
        questions += ["What is net revenue retention by segment, and what was it last quarter?",
                      "Which of the forecast deals have signed paper today?"][: max(0, 6 - len(questions))]
        gaps = [f"No {label} in the sources" for name, label in EXPECTED_METRICS.items() if name not in m] + ALWAYS_GAPS
        return EditorDecision(headline=headline, summary=summary, posture=posture, health_score=int(score),
                              recommendations=recs, questions=questions, gaps=gaps[:6])
