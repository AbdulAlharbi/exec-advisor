"""ClaudeBrain — the Anthropic Messages API behind the same Brain interface as the rule brain.

The prompt embeds the pydantic JSON schema and demands JSON only. The reply is unwrapped from any
prose or code fences and validated. A validation failure is fed back to the model once for repair.
After two failures, or on any API error, the heuristic brain answers so the workflow never stalls,
and the fallback is recorded on the decision so it is visible in the brief and the audit log.
"""
from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .heuristic import HeuristicBrain

T = TypeVar("T", bound=BaseModel)

PERSONA = (
    "You are a seasoned executive advisor: part board member, part chief of staff, part devil's advocate. "
    "You are direct, specific and evidence-led. You never invent figures. You cite claims by their ids."
)

SYSTEM: dict[str, str] = {
    "analyst": PERSONA + """

Task: extract atomic claims from the sources. For each claim copy a VERBATIM quote of 12+ characters from the
source — copy exactly, character for character, including numbers and punctuation. Every quote will be checked
against the source text by a program; a paraphrase or an altered number will be discarded. Prefer one claim per
number or per concrete fact. Fill `metric`, `value`, `prior`, `unit` when the claim is a number
(metric names: revenue, arr, gross_margin, burn, cash, concentration, dso, nps, headcount, deferred_revenue,
sla_credits, ticket_volume, lead_conversion, first_response, leads). Extract 25 to 60 claims.""",
    "risks": PERSONA + """

Task: identify the material RISKS in the business. You are given verified claims (with ids), derived metrics and
signals. Each risk must cite 1-4 claim ids as evidence — only ids from the claims list. Rate severity defensibly:
critical means it can end the company or the plan within two quarters. Return 3 to 7 risks, most severe first.
Leave `key` null.""",
    "opportunities": PERSONA + """

Task: identify OPPORTUNITIES the evidence supports. Each must cite 1-4 claim ids. Rate `level` as impact.
Be honest about effort in `horizon`. Return 2 to 5. Leave `key` null.""",
    "contrarian": PERSONA + """

Task: you are the designated contrarian. You may NOT add findings. For every finding, return an objection with a
verdict: sustain (evidence holds), weaken (real but overstated — say why), or drop (the evidence does not support
it). Then list the assumptions leadership is treating as facts — quote their own words via `quote_claim` (a claim
id) where possible — with why each is questionable and the cheapest test to find out. 2 to 6 assumptions.""",
    "editor": PERSONA + """

Task: write the executive brief from the surviving findings, assumptions and computed changes. The headline is one
specific sentence, the single thing the executive must know. Summary: 2-4 sentences with a point of view.
health_score 0-100 must be consistent with the risk levels. 3-6 recommendations, ordered by priority, each with an
owner role, timeline and measurable outcome, and `linked` finding ids. 4-6 sharp questions. 1-5 information gaps.""",
}

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> str:
    m = _FENCE.search(text)
    if m:
        return m.group(1).strip()
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b == -1 or b < a:
        raise ValueError("no JSON object in reply")
    return text[a : b + 1]


class ClaudeBrain:
    name = "claude"

    def __init__(self, model: str = "claude-sonnet-5", client: Any | None = None, max_tokens: int = 6000):
        self.model = model
        self.max_tokens = max_tokens
        self.fallback = HeuristicBrain()
        self.last_error: str | None = None
        if client is None:
            import anthropic  # imported lazily so the package works without the SDK

            client = anthropic.Anthropic()
        self.client = client

    def _call(self, system: str, messages: list[dict[str, str]]) -> str:
        resp = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, temperature=0.1, system=system, messages=messages
        )
        return "".join(getattr(b, "text", "") for b in resp.content)

    def decide(self, task: str, context: dict[str, Any], schema: type[T]) -> T:
        self.last_error = None
        system = SYSTEM[task] + "\n\nRespond with ONLY a JSON object matching this schema:\n" + json.dumps(schema.model_json_schema())
        ctx = json.dumps(context, default=str)
        if len(ctx) > 120_000:
            ctx = ctx[:120_000] + "…[truncated]"
        messages = [{"role": "user", "content": f"CONTEXT:\n{ctx}\n\nReturn the JSON now."}]
        last_err = ""
        for attempt in range(2):
            try:
                text = self._call(system, messages)
                return schema.model_validate_json(extract_json(text))
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                last_err = f"{type(exc).__name__}: {str(exc)[:400]}"
                messages += [
                    {"role": "assistant", "content": text if "text" in locals() else ""},
                    {"role": "user", "content": f"That was not valid. Error: {last_err}\nReturn ONLY the corrected JSON object."},
                ]
            except Exception as exc:  # noqa: BLE001 — network, auth, rate limit: fall back, don't stall
                last_err = f"{type(exc).__name__}: {str(exc)[:300]}"
                break
        self.last_error = last_err
        return self.fallback.decide(task, context, schema)
