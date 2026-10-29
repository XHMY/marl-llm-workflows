"""Upstream pins: submodule commits, verl's LoRA constants, and the dependency pin lists."""

from __future__ import annotations

import configparser
import os
import subprocess
import tomllib
from pathlib import Path

import pytest

from marl_workflows import contracts

REPO = Path(__file__).resolve().parent.parent
PINS = {
    "third_party/rllm": ("https://github.com/rllm-org/rllm.git", "186d0bbbb5bfe18f8f6b8e4eac6db9379cf27738"),
    "third_party/verl": ("https://github.com/volcengine/verl.git", "d62da4950573d7a4b7ef2362337952e7ab59e78d"),
}


@pytest.mark.stack
def test_lora_constants_match_verl() -> None:
    from verl.workers.rollout.vllm_rollout.utils import VLLM_LORA_INT_ID, VLLM_LORA_NAME, VLLM_LORA_PATH

    assert contracts.LORA_INT_ID_BASE == VLLM_LORA_INT_ID
    assert contracts.LORA_PATH_PLACEHOLDER == VLLM_LORA_PATH
    assert str(contracts.LORA_INT_ID_BASE) == VLLM_LORA_NAME


@pytest.mark.stack
@pytest.mark.parametrize("path", sorted(PINS))
def test_submodule_head_matches_pin(path: str) -> None:
    head = subprocess.run(
        ["git", "-C", str(REPO / path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    assert head == PINS[path][1]


@pytest.mark.stack
def test_imported_stack_comes_from_submodules() -> None:
    import rllm
    import verl

    for mod, sub in ((rllm, "third_party/rllm"), (verl, "third_party/verl")):
        assert os.path.realpath(mod.__file__).startswith(str((REPO / sub).resolve())), mod.__file__


def test_gitmodules_point_at_upstream() -> None:
    parser = configparser.ConfigParser()
    parser.read(REPO / ".gitmodules")
    found = {parser[s]["path"]: parser[s]["url"] for s in parser.sections()}
    assert found == {p: url for p, (url, _) in PINS.items()}
    for s in parser.sections():
        assert parser[s].get("ignore") == "untracked"


def _stack_requirements() -> list[str]:
    lines = (REPO / "requirements" / "stack.txt").read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def test_pyproject_dependencies_equal_stack_txt() -> None:
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    assert project["dependencies"] == _stack_requirements()


def test_core_stack_pins() -> None:
    reqs = set(_stack_requirements())
    for pin in (
        "torch==2.8.0",
        "vllm==0.11.0",
        "transformers==4.56.1",
        "peft==0.18.1",
        "ray[default]==2.54.0",
    ):
        assert pin in reqs
    assert not any(r.split("==")[0] in {"rllm", "verl"} for r in reqs)
