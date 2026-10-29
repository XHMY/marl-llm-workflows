"""MultiAgentDPActor: optimizer-step counters and the per-role KL coefficient override."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

pytestmark = pytest.mark.stack

dp_actor = pytest.importorskip("verl.workers.actor.dp_actor")

from verl import DataProto  # noqa: E402
from verl.workers.config import FSDPActorConfig  # noqa: E402

from marl_workflows import contracts  # noqa: E402
from marl_workflows.verl_ext.actor import MultiAgentDPActor  # noqa: E402

BASE = dp_actor.DataParallelPPOActor


def make_actor(mini_batch: int = 8):
    actor = object.__new__(MultiAgentDPActor)
    actor.config = FSDPActorConfig(
        strategy="fsdp",
        ppo_mini_batch_size=mini_batch,
        ppo_micro_batch_size_per_gpu=2,
        use_kl_loss=True,
        kl_loss_coef=0.001,
    )
    active = torch.nn.Parameter(torch.zeros(2))
    frozen = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
    actor.actor_optimizer = torch.optim.AdamW([active, frozen], lr=0.1)
    actor._params = (active, frozen)
    return actor


def batch(n: int, **meta) -> DataProto:
    return DataProto.from_dict(tensors={"x": torch.zeros(n, 1)}, meta_info={"temperature": 1.0, **meta})


def fake_update_policy(n_steps: int, seen: list):
    """Stands in for upstream update_policy: n optimizer steps, records the KL coefficient it reads."""

    def update_policy(self, data):
        seen.append(self.config.kl_loss_coef)
        metrics = {"actor/kl_coef": [self.config.kl_loss_coef]}
        for _ in range(n_steps):
            self._optimizer_step()
        return metrics

    return update_policy


def fake_optimizer_step(self):
    active, _ = self._params
    active.grad = torch.ones_like(active)
    self.actor_optimizer.step()
    self.actor_optimizer.zero_grad()
    return torch.tensor(1.0)


@pytest.mark.parametrize(("rows", "mini_batch", "steps"), [(24, 8, 3), (8, 8, 1), (20, 8, 3)])
def test_counters(rows, mini_batch, steps):
    actor = make_actor(mini_batch)
    seen: list = []
    with (
        patch.object(BASE, "update_policy", fake_update_policy(steps, seen)),
        patch.object(BASE, "_optimizer_step", fake_optimizer_step),
    ):
        m1 = actor.update_policy(batch(rows))
        m2 = actor.update_policy(batch(rows))
    num_key, calls_key, count_key = contracts.OPTIM_COUNTER_KEYS
    assert m1[num_key] == [-(-rows // mini_batch)]
    assert m1[calls_key] == [steps] and m2[calls_key] == [steps]  # reset per call
    assert m1[count_key] == [float(steps)] and m2[count_key] == [float(2 * steps)]


def test_optim_state_step_prefers_trainable_param():
    actor = make_actor()
    active, frozen = actor._params
    assert actor._optim_state_step() == 0.0
    # give the frozen parameter more steps: it is only the fallback
    frozen.requires_grad_(True)
    for _ in range(3):
        frozen.grad = torch.ones_like(frozen)
        actor.actor_optimizer.step()
    frozen.grad = None
    frozen.requires_grad_(False)
    assert actor._optim_state_step() == 3.0
    active.grad = torch.ones_like(active)
    actor.actor_optimizer.step()
    assert actor._optim_state_step() == 1.0


def test_kl_override_applied_and_restored():
    actor = make_actor()
    seen: list = []
    with (
        patch.object(BASE, "update_policy", fake_update_policy(1, seen)),
        patch.object(BASE, "_optimizer_step", fake_optimizer_step),
    ):
        metrics = actor.update_policy(batch(8, **{contracts.META_KL_OVERRIDE: 0.05}))
        actor.update_policy(batch(8))
    assert seen == [0.05, 0.001]
    assert metrics["actor/kl_coef"] == [0.05]
    assert actor.config.kl_loss_coef == 0.001


def test_kl_override_restored_on_exception():
    actor = make_actor()

    def failing(self, data):
        assert self.config.kl_loss_coef == 0.2
        raise RuntimeError("boom")

    with patch.object(BASE, "update_policy", failing), pytest.raises(RuntimeError):
        actor.update_policy(batch(8, **{contracts.META_KL_OVERRIDE: 0.2}))
    assert actor.config.kl_loss_coef == 0.001
