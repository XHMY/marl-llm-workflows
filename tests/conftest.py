"""Shared pytest setup: import paths, an isolated data directory, and repository fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Same order as scripts/env.sh, so the pinned submodules win over any installed rllm/verl
# even when pytest is started without sourcing env.sh.
for _p in (REPO_ROOT / "third_party" / "verl", REPO_ROOT / "third_party" / "rllm", REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(autouse=True)
def _isolated_marl_data_dir(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point MARL_DATA_DIR at a fresh temporary directory so no test writes to ~/.cache."""
    data_dir = tmp_path_factory.mktemp("marl_data")
    monkeypatch.setenv("MARL_DATA_DIR", str(data_dir))
    return data_dir
