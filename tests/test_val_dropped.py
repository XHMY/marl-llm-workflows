"""ValDropAccounting: dropped validation episodes and the optional corrected pass@k."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.stack


def _input_batch(sources, uids):
    nt = {"task_ids": np.array(uids, dtype=object)}
    if sources is not None:
        nt["data_source"] = np.array(sources, dtype=object)
    return SimpleNamespace(non_tensor_batch=nt)


@pytest.fixture
def acct():
    from marl_workflows.trainer.trainer import ValDropAccounting

    a = ValDropAccounting()
    # n_val = 2 (interleaved repeats share task_ids). Source "a": uid1 (both seen), uid2 (one
    # dropped), uid3 (both dropped). Source "b": uid4 (both seen).
    a.record(
        _input_batch(["a"] * 6 + ["b"] * 2, ["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4"]),
        [2, 1, 3, 0, 0, 0, 1, 1],
    )
    return a


def test_default_off_only_adds_dropped_counts(acct) -> None:
    upstream = {"val/a/pass@1": 0.5, "val/a/pass@2": 0.5, "val/b/pass@1": 1.0, "val/b/pass@2": 1.0}
    out = acct.metrics(upstream, n_val=2, correct=False)
    assert out == {"val/a/dropped_episodes": 3.0, "val/b/dropped_episodes": 0.0}
    merged = {**upstream, **out}
    assert {k: merged[k] for k in upstream} == upstream  # upstream metrics unchanged


def test_corrected_pass_at_k(acct) -> None:
    # Upstream over seen episodes of "a": 3 episodes (u1 x2, u2 x1), say 2 correct -> pass@1 = 2/3;
    # seen uids u1, u2, one solved -> pass@2 = 1/2.
    upstream = {"val/a/pass@1": 2 / 3, "val/a/pass@2": 0.5, "val/b/pass@1": 0.5, "val/b/pass@2": 1.0}
    out = acct.metrics(upstream, n_val=2, correct=True)
    assert out["val/a/pass@1"] == pytest.approx(2 / 6)  # 2 correct of 6 attempted episodes
    assert out["val/a/pass@2"] == pytest.approx(1 / 3)  # 1 solved uid of 3 (u3 dropped entirely = failure)
    assert out["val/b/pass@1"] == pytest.approx(0.5)
    assert out["val/b/pass@2"] == pytest.approx(1.0)
    assert out["val/a/dropped_episodes"] == 3.0


def test_n_val_one_corrects_once() -> None:
    from marl_workflows.trainer.trainer import ValDropAccounting

    a = ValDropAccounting()
    a.record(_input_batch(None, ["u1", "u2", "u3", "u4"]), [1, 0, 2, 1])
    out = a.metrics({"val/unknown/pass@1": 2 / 3}, n_val=1, correct=True)
    assert out == {"val/unknown/dropped_episodes": 1.0, "val/unknown/pass@1": pytest.approx(0.5)}


def test_all_dropped_source_reports_zero_when_corrected() -> None:
    from marl_workflows.trainer.trainer import ValDropAccounting

    a = ValDropAccounting()
    a.record(_input_batch(["c", "c"], ["u1", "u1"]), [0, 0])
    assert a.metrics({}, n_val=2, correct=True) == {
        "val/c/dropped_episodes": 2.0,
        "val/c/pass@2": 0.0,
        "val/c/pass@1": 0.0,
    }
    assert a.metrics({}, n_val=2, correct=False) == {"val/c/dropped_episodes": 2.0}


def test_reset_and_accumulate_over_batches() -> None:
    from marl_workflows.trainer.trainer import ValDropAccounting

    a = ValDropAccounting()
    a.record(_input_batch(["a"], ["u1"]), [0])
    a.reset()
    a.record(_input_batch(["a"], ["u1"]), [0])
    a.record(_input_batch(["a"], ["u2"]), [0])
    assert a.metrics({}, n_val=1, correct=False) == {"val/a/dropped_episodes": 2.0}


# ---------------------------------------------------------------------- trainer wrappers


def _trainer_stub(probe_states=None, *, correct: bool = False, share_policy: bool = False):
    from omegaconf import OmegaConf

    from marl_workflows.trainer.trainer import MultiAgentWorkflowTrainer, ValDropAccounting

    trainer = object.__new__(MultiAgentWorkflowTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "agent_names": ["generator", "aggregator"],
                "share_policy": share_policy,
                "check_lora_registration": True,
                "val_count_dropped_episodes": correct,
            },
            "actor_rollout_ref": {"model": {"lora_rank": 64}, "rollout": {"val_kwargs": {"n": 1}}},
        }
    )
    trainer._lora_state_checked = False
    trainer._val_acct = ValDropAccounting()
    trainer.probe_calls = 0

    def probe():
        trainer.probe_calls += 1
        return probe_states

    trainer.actor_rollout_wg = SimpleNamespace(probe_rollout_lora_state=probe)
    return trainer


@pytest.fixture
def stub_upstream_generate(monkeypatch: pytest.MonkeyPatch):
    from rllm.trainer.verl.agent_workflow_trainer import AgentWorkflowPPOTrainer

    def fake_generate(self, batch, timing_raw=None, **kwargs):
        return SimpleNamespace(meta_info={"repeat_counts": list(batch.repeat_counts)})

    monkeypatch.setattr(AgentWorkflowPPOTrainer, "generate_trajectories", fake_generate)


def _gen_batch(validate: bool, counts):
    batch = _input_batch(["a"] * len(counts), [f"u{i}" for i in range(len(counts))])
    batch.meta_info = {"validate": True} if validate else {}
    batch.repeat_counts = counts
    return batch


def test_lora_registration_checked_once_after_first_training_rollout(stub_upstream_generate) -> None:
    ok = {"rollout_lora_registered": True, "lora_int_ids": [123, 124]}
    trainer = _trainer_stub([ok, dict(ok)])
    trainer.generate_trajectories(_gen_batch(True, [1]))  # validation does not probe
    assert trainer.probe_calls == 0
    trainer.generate_trajectories(_gen_batch(False, [1]))
    trainer.generate_trajectories(_gen_batch(False, [1]))
    assert trainer.probe_calls == 1


@pytest.mark.parametrize(
    "state",
    [
        {"rollout_lora_registered": False, "lora_int_ids": [123, 124]},
        {"rollout_lora_registered": True, "lora_int_ids": [123]},
    ],
)
def test_lora_registration_mismatch_raises(stub_upstream_generate, state) -> None:
    trainer = _trainer_stub([{"rollout_lora_registered": True, "lora_int_ids": [123, 124]}, state])
    with pytest.raises(RuntimeError, match="LoRA registration mismatch"):
        trainer.generate_trajectories(_gen_batch(False, [1]))


def test_sp_expects_single_id(stub_upstream_generate) -> None:
    trainer = _trainer_stub([{"rollout_lora_registered": True, "lora_int_ids": [123]}], share_policy=True)
    trainer.generate_trajectories(_gen_batch(False, [1]))
    assert trainer.probe_calls == 1


@pytest.mark.parametrize("correct, expected_pass1", [(False, 1.0), (True, 0.5)])
def test_validate_agent_wrapper(monkeypatch: pytest.MonkeyPatch, stub_upstream_generate, correct, expected_pass1):
    from rllm.trainer.verl.agent_workflow_trainer import AgentWorkflowPPOTrainer

    trainer = _trainer_stub(correct=correct)

    def fake_validate(self):
        # one validation batch: two episodes, one dropped; upstream sees only the solved one
        self.generate_trajectories(_gen_batch(True, [1, 0]))
        return {"val/a/pass@1": 1.0}

    monkeypatch.setattr(AgentWorkflowPPOTrainer, "_validate_agent", fake_validate)
    metrics = trainer._validate_agent()
    assert metrics["val/a/dropped_episodes"] == 1.0
    assert metrics["val/a/pass@1"] == pytest.approx(expected_pass1)
    # a second validation pass starts from zero
    assert trainer._validate_agent()["val/a/dropped_episodes"] == 1.0
