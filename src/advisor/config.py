"""Runtime settings resolved from the environment, with CLI overrides."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    brain: str = field(default_factory=lambda: os.getenv("ADVISOR_BRAIN", "auto"))
    model: str = field(default_factory=lambda: os.getenv("ADVISOR_MODEL", "claude-sonnet-5"))
    out_dir: Path = field(default_factory=lambda: Path(os.getenv("ADVISOR_OUT_DIR", REPO_ROOT / "out")))
    site_dir: Path = field(default_factory=lambda: Path(os.getenv("ADVISOR_SITE_DIR", REPO_ROOT / "docs")))
    sources_file: Path = field(
        default_factory=lambda: Path(os.getenv("ADVISOR_SOURCES", REPO_ROOT / "data" / "company" / "sources.yaml"))
    )
    fetch_timeout: float = field(default_factory=lambda: float(os.getenv("ADVISOR_FETCH_TIMEOUT", "20")))
    max_source_chars: int = field(default_factory=lambda: int(os.getenv("ADVISOR_MAX_SOURCE_CHARS", "60000")))

    @property
    def runs_dir(self) -> Path:
        return self.out_dir / "runs"

    @property
    def reports_dir(self) -> Path:
        return self.out_dir / "reports"

    @property
    def audit_path(self) -> Path:
        return self.out_dir / "audit.jsonl"

    @property
    def data_dir(self) -> Path:
        return self.site_dir / "data"

    def resolve_brain(self) -> str:
        if self.brain == "auto":
            return "claude" if os.getenv("ANTHROPIC_API_KEY") else "heuristic"
        return self.brain
