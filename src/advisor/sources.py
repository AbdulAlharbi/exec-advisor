"""Load the company's sources from a YAML manifest.

    context: one paragraph the advisor should know about the company
    sources:
      - id: q2-financials
        title: Q2 financial summary
        path: q2-financials.md          # relative to the manifest
      - id: pipeline-sheet
        title: Live pipeline (published Google Sheet)
        url: https://docs.google.com/spreadsheets/d/e/…/pub?output=csv
      - id: note
        title: CEO voice note transcript
        text: |
          inline text works too

Files and inline text are read as-is. URLs are fetched fresh on every run — that is what makes
"continuous" real when the pipeline runs on a schedule.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml

from .models import Source


class SourceManifest:
    def __init__(self, context: str, sources: list[Source], path: Path | None = None):
        self.context = context
        self.sources = sources
        self.path = path

    @classmethod
    def load(cls, path: Path) -> "SourceManifest":
        raw = yaml.safe_load(path.read_text()) or {}
        base = path.parent
        sources: list[Source] = []
        for i, item in enumerate(raw.get("sources", [])):
            sid = str(item.get("id") or f"src-{i+1}")
            title = str(item.get("title") or sid)
            if "path" in item:
                p = (base / item["path"]).resolve()
                sources.append(Source(id=sid, title=title, kind="file", path=str(p)))
            elif "url" in item:
                sources.append(Source(id=sid, title=title, kind="url", url=str(item["url"])))
            elif "text" in item:
                sources.append(Source(id=sid, title=title, kind="inline", text=str(item["text"])))
            else:
                raise ValueError(f"source {sid!r} needs one of: path, url, text")
        return cls(context=str(raw.get("context") or "").strip(), sources=sources, path=path)

    def materialise(self, timeout: float = 20.0, max_chars: int = 60_000) -> list[Source]:
        """Read files and fetch URLs. Never raises for a single bad source — it records the error."""
        out: list[Source] = []
        for s in self.sources:
            s = s.model_copy()
            try:
                if s.kind == "file":
                    s.text = Path(s.path or "").read_text(encoding="utf-8")
                elif s.kind == "url":
                    r = httpx.get(s.url or "", timeout=timeout, follow_redirects=True, headers={"cache-control": "no-cache"})
                    r.raise_for_status()
                    s.text = r.text
                s.fetched_at = datetime.now(timezone.utc)
                if len(s.text) > max_chars:
                    s.text = s.text[:max_chars] + "\n…[truncated]"
                    s.error = f"truncated to {max_chars} chars"
            except Exception as exc:  # noqa: BLE001 — a bad source must not kill the run
                s.error = f"{type(exc).__name__}: {exc}"
                s.text = ""
            out.append(s)
        return out
