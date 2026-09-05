import json

from advisor.audit import AuditLog


def test_chain_verifies_and_detects_tampering(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    for i in range(6):
        log.record("event", "RUN-1", n=i)
    ok, n, _ = log.verify()
    assert ok and n == 6

    lines = (tmp_path / "audit.jsonl").read_text().splitlines()
    e = json.loads(lines[2])
    e["payload"]["n"] = 99
    lines[2] = json.dumps(e)
    (tmp_path / "audit.jsonl").write_text("\n".join(lines) + "\n")
    ok, at, msg = AuditLog(tmp_path / "audit.jsonl").verify()
    assert not ok and at == 3 and "broken" in msg


def test_reopened_log_continues_chain(tmp_path):
    AuditLog(tmp_path / "a.jsonl").record("a")
    AuditLog(tmp_path / "a.jsonl").record("b")
    assert AuditLog(tmp_path / "a.jsonl").verify()[0]
