"""Contract tests for ClaudeBrain with a stubbed client — never touches the network."""
import pytest

pytest.importorskip("anthropic")

from advisor.llm.claude import ClaudeBrain, extract_json  # noqa: E402
from advisor.models import AnalystDecision  # noqa: E402


class _Block:
    def __init__(self, text):
        self.text = text


class _Resp:
    def __init__(self, text):
        self.content = [_Block(text)]


class StubClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.messages = self

    def create(self, **kw):
        self.calls += 1
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return _Resp(r)


GOOD = '{"claims": [{"source_id": "s", "quote": "Net burn: $1.9M/month", "statement": "burn"}]}'
CTX = {"context": "", "sources": [{"id": "s", "title": "t", "text": "Net burn: $1.9M/month"}]}


def test_parses_plain_json():
    b = ClaudeBrain(client=StubClient([GOOD]))
    d = b.decide("analyst", CTX, AnalystDecision)
    assert d.claims[0].quote == "Net burn: $1.9M/month" and b.client.calls == 1 and b.last_error is None


def test_strips_fences_and_prose():
    b = ClaudeBrain(client=StubClient(["Sure, here it is:\n```json\n" + GOOD + "\n```\nHope that helps."]))
    assert b.decide("analyst", CTX, AnalystDecision).claims


def test_repairs_invalid_output_once():
    b = ClaudeBrain(client=StubClient(['{"claims": "not a list"}', GOOD]))
    d = b.decide("analyst", CTX, AnalystDecision)
    assert d.claims and b.client.calls == 2 and b.last_error is None


def test_falls_back_to_heuristic_after_two_failures():
    b = ClaudeBrain(client=StubClient(["garbage", "still garbage"]))
    d = b.decide("analyst", CTX, AnalystDecision)
    assert b.client.calls == 2 and b.last_error and d.claims  # heuristic extracted the burn line


def test_falls_back_to_heuristic_on_api_error():
    b = ClaudeBrain(client=StubClient([RuntimeError("rate limited")]))
    d = b.decide("analyst", CTX, AnalystDecision)
    assert b.client.calls == 1 and "RuntimeError" in b.last_error and d.claims


def test_extract_json_rejects_non_json():
    with pytest.raises(ValueError):
        extract_json("no braces here")
