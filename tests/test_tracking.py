"""ResumableTracking: wandb resume through WANDB_RUN_ID / WANDB_RESUME (wandb mocked)."""

from __future__ import annotations

import os
import sys
import types
from typing import Any

import pytest

pytestmark = pytest.mark.stack  # imports rllm.utils.tracking

ENV_KEYS = ("WANDB_RUN_ID", "WANDB_RESUME")


class _Run:
    def __init__(self, run_id: str) -> None:
        self.id = run_id


@pytest.fixture
def fake_wandb(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    """A stand-in ``wandb`` module that records the environment seen by ``init``."""
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    state = types.SimpleNamespace(init_env=None, init_kwargs=None, api_runs=[], api_error=None, api_query=None)
    mod = types.ModuleType("wandb")

    def init(**kwargs: Any) -> None:
        state.init_env = {k: os.environ.get(k) for k in ENV_KEYS}
        state.init_kwargs = kwargs
        mod.run = _Run(os.environ.get("WANDB_RUN_ID") or "fresh123")

    class Api:
        def runs(self, path: str, filters: dict, order: str) -> list[_Run]:
            if state.api_error:
                raise state.api_error
            state.api_query = (path, filters, order)
            return list(state.api_runs)

    mod.init = init
    mod.finish = lambda **kw: None
    mod.run = None
    mod.Api = Api
    mod.Settings = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "wandb", mod)
    return state


def _make(**kwargs: Any):
    from marl_workflows.trainer.tracking import ResumableTracking

    return ResumableTracking(
        project_name="proj", experiment_name="exp", default_backend=["wandb"], config={"trainer": {}}, **kwargs
    )


def test_resume_by_id_sets_and_pops_env(fake_wandb) -> None:
    tracker = _make(resume_run_id="abc", resume_mode="allow")
    assert fake_wandb.init_env == {"WANDB_RUN_ID": "abc", "WANDB_RESUME": "allow"}
    assert fake_wandb.init_kwargs["name"] == "exp" and fake_wandb.init_kwargs["project"] == "proj"
    assert tracker.wandb_run_id == "abc"
    assert all(k not in os.environ for k in ENV_KEYS)


def test_new_run_leaves_env_untouched(fake_wandb) -> None:
    tracker = _make()
    assert fake_wandb.init_env == {"WANDB_RUN_ID": None, "WANDB_RESUME": None}
    assert tracker.wandb_run_id == "fresh123"


def test_allow_without_id_finds_run_by_name(fake_wandb) -> None:
    fake_wandb.api_runs = [_Run("newest"), _Run("older")]
    tracker = _make(resume_mode="allow")
    assert fake_wandb.api_query == ("proj", {"display_name": "exp"}, "-created_at")
    assert fake_wandb.init_env == {"WANDB_RUN_ID": "newest", "WANDB_RESUME": "allow"}
    assert tracker.wandb_run_id == "newest"


def test_allow_without_match_or_api_error_creates_new_run(fake_wandb) -> None:
    assert _make(resume_mode="allow").wandb_run_id == "fresh123"
    fake_wandb.api_error = RuntimeError("offline")
    assert _make(resume_mode="allow").wandb_run_id == "fresh123"
    assert fake_wandb.init_env == {"WANDB_RUN_ID": None, "WANDB_RESUME": None}


def test_preexisting_env_is_restored(fake_wandb, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_RUN_ID", "user-set")
    _make(resume_run_id="abc", resume_mode="must")
    assert fake_wandb.init_env == {"WANDB_RUN_ID": "abc", "WANDB_RESUME": "must"}
    assert os.environ["WANDB_RUN_ID"] == "user-set" and "WANDB_RESUME" not in os.environ


def test_console_backend_never_touches_wandb(fake_wandb) -> None:
    from marl_workflows.trainer.tracking import ResumableTracking

    tracker = ResumableTracking("proj", "exp", default_backend="console", resume_run_id="abc", resume_mode="allow")
    assert fake_wandb.init_env is None
    assert tracker.wandb_run_id is None


# ---------------------------------------------------------------------- checkpoint metadata (resume source)


def _trainer_stub(tmp_path, *, share_policy: bool):
    """A MultiAgentWorkflowTrainer with only the attributes the checkpoint methods read."""
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    from marl_workflows.trainer.trainer import MultiAgentWorkflowTrainer

    trainer = object.__new__(MultiAgentWorkflowTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "default_local_dir": str(tmp_path / "ckpt"),
                "agent_names": ["generator", "aggregator"],
                "share_policy": share_policy,
                "experiment_name": "voting-qwen3_1.7b_s300-multi_lora-math",
                "project_name": "proj",
                "base_model_tag": "qwen3_1.7b_s300",
            },
            "actor_rollout_ref": {"model": {"path": "/init/qwen3_1.7b_s300"}},
        }
    )
    trainer.global_steps = 5
    trainer.total_training_steps = 300
    trainer._last_saved_step = 0
    trainer._tracking_logger = SimpleNamespace(wandb_run_id="rid42")
    trainer.saved_adapters = []
    trainer.actor_rollout_wg = SimpleNamespace(save_single_lora_adapter=lambda **kw: trainer.saved_adapters.append(kw))
    return trainer


def _stub_save(monkeypatch: pytest.MonkeyPatch, *, shared_dir_under_ip: bool) -> None:
    """Upstream save through the worker: the shared adapter directory under SP only (the IP worker skips it)."""
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    def fake_save(self) -> None:
        actor = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}", "actor")
        os.makedirs(actor)
        if self._share_policy or shared_dir_under_ip:
            os.makedirs(os.path.join(actor, "lora_adapter"))

    monkeypatch.setattr(RayPPOTrainer, "_save_checkpoint", fake_save)


@pytest.fixture
def stub_upstream_save(monkeypatch: pytest.MonkeyPatch):
    _stub_save(monkeypatch, shared_dir_under_ip=False)


def test_ip_checkpoint_writes_per_role_adapters_and_metadata(tmp_path, stub_upstream_save) -> None:
    import json

    trainer = _trainer_stub(tmp_path, share_policy=False)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "training_metadata.json").write_text(json.dumps({"slurm_config": "h100", "global_steps": 0}))

    trainer._save_checkpoint()
    actor = ckpt / "global_step_5" / "actor"
    assert not (actor / "lora_adapter").exists()
    assert trainer.saved_adapters == [
        {"agent_name": "generator", "save_path": str(actor / "lora_adapter_generator"), "global_step": 5},
        {"agent_name": "aggregator", "save_path": str(actor / "lora_adapter_aggregator"), "global_step": 5},
    ]
    assert trainer._last_saved_step == 5
    meta = json.loads((ckpt / "training_metadata.json").read_text())
    assert meta["slurm_config"] == "h100"  # fields written by others survive
    assert meta["global_steps"] == 5 and meta["total_training_steps"] == 300
    assert meta["wandb_run_id"] == "rid42"
    assert meta["base_model"] == "qwen3_1.7b_s300" and meta["actor_model_path"] == "/init/qwen3_1.7b_s300"
    assert trainer._load_training_metadata()["wandb_run_id"] == "rid42"


def test_sp_checkpoint_keeps_shared_adapter(tmp_path, stub_upstream_save) -> None:
    trainer = _trainer_stub(tmp_path, share_policy=True)
    trainer._save_checkpoint()
    assert (tmp_path / "ckpt" / "global_step_5" / "actor" / "lora_adapter").is_dir()
    assert trainer.saved_adapters == []
    assert (tmp_path / "ckpt" / "training_metadata.json").exists()


def test_ip_checkpoint_refuses_shared_adapter_dir(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_save(monkeypatch, shared_dir_under_ip=True)
    trainer = _trainer_stub(tmp_path, share_policy=False)
    with pytest.raises(RuntimeError, match="must not contain the SP adapter directory"):
        trainer._save_checkpoint()
    assert trainer.saved_adapters == []
