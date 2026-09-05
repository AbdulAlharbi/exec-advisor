"""Append-only, hash-chained JSONL audit log.

Every entry's SHA-256 covers its own payload *and* the previous entry's hash, so editing any
line after the fact breaks every hash that follows it. `verify()` walks the chain.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def _digest(prev: str, entry: dict[str, Any]) -> str:
    body = json.dumps(entry, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(f"{prev}{body}".encode()).hexdigest()


class AuditLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last = self._read_last_hash()

    def _read_last_hash(self) -> str:
        if not self.path.exists():
            return GENESIS
        last = GENESIS
        with self.path.open() as fh:
            for line in fh:
                if line.strip():
                    last = json.loads(line)["hash"]
        return last

    def record(self, event: str, run_id: str | None = None, **payload: Any) -> dict[str, Any]:
        entry = {
            "at": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "run_id": run_id,
            "payload": _truncate(payload),
            "prev": self._last,
        }
        entry["hash"] = _digest(self._last, {k: v for k, v in entry.items() if k != "hash"})
        with self.path.open("a") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
        self._last = entry["hash"]
        return entry

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def verify(self) -> tuple[bool, int, str]:
        """Return (valid, entry_count, message)."""
        prev = GENESIS
        for i, e in enumerate(self.entries(), start=1):
            expected = _digest(prev, {k: v for k, v in e.items() if k != "hash"})
            if e.get("prev") != prev or e.get("hash") != expected:
                return False, i, f"chain broken at entry {i}"
            prev = e["hash"]
        n = len(self.entries())
        return True, n, f"chain valid · {n} entries"


def _truncate(obj: Any, limit: int = 600) -> Any:
    if isinstance(obj, str):
        return obj if len(obj) <= limit else obj[:limit] + "…"
    if isinstance(obj, dict):
        return {k: _truncate(v, limit) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate(v, limit) for v in obj[:50]]
    return obj
