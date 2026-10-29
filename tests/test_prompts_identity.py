"""The packaged prompt files are byte-identical to the research fork's prompt.json files."""

from __future__ import annotations

import hashlib
import json
from importlib import resources

import pytest

# sha256 of fork examples/math_reasoning/prompt.json and examples/deepcoder/prompt.json
# @f0240732, hard-coded when the files were copied (2026-10-29).
EXPECTED_SHA256 = {
    "math.json": "5645398af6415a82948ec90ef6e3f6bf8fc403111e10dd40cc9ac20f1e9ee9e9",
    "deepcoder.json": "16d415a45ec75dbc504aa0d76b5f2949888ea0688926e7a241e90d43a3a3142c",
}

# Every section a workflow binding loads.
SECTIONS = {
    "math": [
        "multi_agent_math_prompts",
        "voting_prompts",
        "evaluator_optimizer_prompts",
        "orchestrator_workers_prompts",
    ],
    "deepcoder": [
        "deepcoder_single_agent_prompts",
        "deepcoder_voting_prompts",
        "deepcoder_evaluator_optimizer_prompts",
        "deepcoder_orchestrator_workers_prompts",
    ],
}


def _prompt_bytes(name: str) -> bytes:
    return resources.files("marl_workflows.workflows").joinpath("prompts", name).read_bytes()


@pytest.mark.parametrize("name", sorted(EXPECTED_SHA256))
def test_prompt_file_sha256(name: str) -> None:
    assert hashlib.sha256(_prompt_bytes(name)).hexdigest() == EXPECTED_SHA256[name]


@pytest.mark.parametrize("task", sorted(SECTIONS))
def test_load_prompts_sections(task: str) -> None:
    from marl_workflows.workflows.base import load_prompts

    data = json.loads(_prompt_bytes(f"{task}.json"))
    for section in SECTIONS[task]:
        loaded = load_prompts(task, section)
        assert loaded == data[section]
        assert all("template" in v for v in loaded.values())
        # A fresh dict per call, so a caller mutating its copy cannot leak into another workflow.
        assert load_prompts(task, section) is not loaded
    with pytest.raises(KeyError):
        load_prompts(task, "no_such_section")
    with pytest.raises(ValueError):
        load_prompts("code", "x")  # type: ignore[arg-type]
