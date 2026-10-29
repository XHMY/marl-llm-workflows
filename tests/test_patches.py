"""patches.py: documentation completeness, anchors against the pinned verl, scoping and idempotence."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.stack

fsdp_workers = pytest.importorskip("verl.workers.fsdp_workers")

import peft  # noqa: E402
from verl.workers.rollout import base as rollout_base  # noqa: E402
from verl.workers.rollout.replica import RolloutReplicaRegistry  # noqa: E402

import marl_workflows.verl_ext.workers  # noqa: E402,F401  (import must not patch anything)
from marl_workflows.verl_ext import patches  # noqa: E402


@pytest.fixture
def clean_patches():
    yield
    patches.revert_worker_patches()
    patches.revert_driver_patches()
    assert patches.applied_patches() == []


def test_nothing_patched_on_import():
    assert patches.applied_patches() == []
    assert fsdp_workers.get_peft_model is peft.get_peft_model
    assert rollout_base._ROLLOUT_REGISTRY[("vllm", "async")] == "verl.workers.rollout.vllm_rollout.vLLMAsyncRollout"
    assert RolloutReplicaRegistry._registry["vllm"].__module__ == "verl.workers.rollout.replica"


def test_patch_table_complete_and_anchored(repo_root: Path):
    assert [p.pid for p in patches.PATCHES] == ["P1", "P2", "H1"]
    verl_root = repo_root / "third_party" / "verl"
    for p in patches.PATCHES:
        for field in ("upstream_symbol", "upstream_location", "reason_no_hook", "behaviour_change", "applied_by"):
            assert getattr(p, field).strip(), (p.pid, field)
        assert p.anchors, p.pid
        for rel, line, text in p.anchors:
            lines = (verl_root / rel).read_text().splitlines()
            assert text in lines[line - 1], f"{p.pid}: {rel}:{line} no longer contains {text!r}"
            assert rel in p.upstream_location, (p.pid, rel)


def test_docs_list_every_patch(repo_root: Path):
    doc = (repo_root / "docs" / "upstream-patches.md").read_text()
    assert patches.render_markdown() in doc, "docs/upstream-patches.md is stale: paste patches.render_markdown()"


def test_scoped_p1_calls_hook_with_pristine_config_and_restores():
    original = fsdp_workers.get_peft_model
    seen = []

    def hook(peft_model, cfg):
        seen.append((peft_model, cfg))

    class FakeConfig:
        def __init__(self):
            self.mutated = False

    def fake_original(model, cfg):
        cfg.mutated = True  # PEFT mutates the config it wraps with
        return ("peft", model)

    fsdp_workers.get_peft_model = fake_original
    try:
        with patches.scoped_multi_adapter_peft(hook):
            assert patches.applied_patches() == ["P1"]
            cfg = FakeConfig()
            out = fsdp_workers.get_peft_model("model", cfg)
            with pytest.raises(RuntimeError, match="re-entrant"):
                with patches.scoped_multi_adapter_peft(hook):
                    pass
        assert out == ("peft", "model")
        assert seen[0][0] == out
        assert seen[0][1] is not cfg and seen[0][1].mutated is False
        assert fsdp_workers.get_peft_model is fake_original
        assert patches.applied_patches() == []
    finally:
        fsdp_workers.get_peft_model = original


def test_scoped_p1_restores_on_exception():
    original = fsdp_workers.get_peft_model
    with pytest.raises(KeyError):
        with patches.scoped_multi_adapter_peft(lambda m, c: None):
            assert fsdp_workers.get_peft_model is not original
            raise KeyError("boom")
    assert fsdp_workers.get_peft_model is original
    assert patches.applied_patches() == []


def test_worker_patch_idempotent_and_selects_rollout(clean_patches):
    from verl.workers.rollout import get_rollout_class

    from marl_workflows.verl_ext.vllm_rollout import MultiLoRAvLLMAsyncRollout

    patches.apply_worker_patches()
    patches.apply_worker_patches()
    assert patches.applied_patches() == ["P2"]
    assert get_rollout_class("vllm", "async") is MultiLoRAvLLMAsyncRollout
    patches.revert_worker_patches()
    assert rollout_base._ROLLOUT_REGISTRY[("vllm", "async")] == "verl.workers.rollout.vllm_rollout.vLLMAsyncRollout"


def test_driver_patch_idempotent_and_selects_replica(clean_patches):
    from verl.workers.rollout.replica import get_rollout_replica_class

    from marl_workflows.verl_ext.vllm_server import MultiLoRAvLLMReplica

    upstream_loader = RolloutReplicaRegistry._registry["vllm"]
    patches.apply_driver_patches()
    patches.apply_driver_patches()
    assert patches.applied_patches() == ["H1"]
    assert get_rollout_replica_class("vllm") is MultiLoRAvLLMReplica
    patches.revert_driver_patches()
    assert RolloutReplicaRegistry._registry["vllm"] is upstream_loader
