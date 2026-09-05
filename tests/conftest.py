from pathlib import Path

import pytest

from advisor.config import REPO_ROOT, Settings

COMPANY = REPO_ROOT / "data" / "company" / "sources.yaml"
COMPANY_WEEK2 = REPO_ROOT / "data" / "company-week2" / "sources.yaml"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings()
    s.brain = "heuristic"
    s.out_dir = tmp_path / "out"
    s.site_dir = tmp_path / "site"
    s.sources_file = COMPANY
    return s
