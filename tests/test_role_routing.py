"""RoleRoutedWorkerGroup and split_batch_by_role against a fake worker group and real DataProtos."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.stack

RESP_LEN = 4
# Interleaved roles: first appearance order is generator, aggregator, evaluator.
TRAJ_IDS = [
    "u-1_generator0",
    "u-1_aggregator",
    "u-2_generator1",
    "u_3_generator2",  # task uuid containing the separator
    "u-2_aggregator",
    "u-4_evaluator",
]


def _batch(traj_ids: list[str] = TRAJ_IDS, meta: dict | None = None):
    from verl import DataProto

    n = len(traj_ids)
    lengths = torch.tensor([4, 2, 3, 1, 2, 4])[:n]
    mask = (torch.arange(RESP_LEN)[None, :] < lengths[:, None]).long()
    return DataProto.from_dict(
        tensors={
            "row": torch.arange(n, dtype=torch.float32)[:, None].repeat(1, RESP_LEN),
            "responses": torch.zeros(n, RESP_LEN, dtype=torch.long),
            "attention_mask": torch.ones(n, 2 + RESP_LEN, dtype=torch.long),
            "response_mask": mask,
            "advantages": mask.float(),
        },
        non_tensors={"trajectory_ids": np.array(traj_ids, dtype=object)},
        meta_info=dict(meta or {}),
    )


class FakeWG:
    """Records calls; outputs encode the input row and the active adapter."""

    ROLE_OFFSET = {"generator": 100.0, "aggregator": 200.0, "evaluator": 300.0, None: 0.0}

    def __init__(self, world_size: int = 4) -> None:
        self.world_size = world_size
        self.active: str | None = None
        self.calls: list[tuple[str, Any]] = []

    def set_active_lora(self, agent_role: str, lora_config: dict | None = None) -> None:
        assert lora_config == {}
        self.active = agent_role
        self.calls.append(("set_active_lora", agent_role))

    def _check_divisible(self, data) -> None:
        assert len(data) % self.world_size == 0, f"sub-batch of {len(data)} rows not padded to {self.world_size}"

    def compute_log_prob(self, data):
        from verl import DataProto

        self._check_divisible(data)
        self.calls.append(("compute_log_prob", (self.active, len(data))))
        row = data.batch["row"]
        return DataProto.from_dict(
            tensors={"old_log_probs": -(row + self.ROLE_OFFSET[self.active]), "entropys": row + 0.5},
            meta_info={"temperature": 0.7},
        )

    def compute_ref_log_prob(self, data):
        from verl import DataProto

        self._check_divisible(data)
        self.calls.append(("compute_ref_log_prob", (self.active, len(data))))
        return DataProto.from_dict(tensors={"ref_log_prob": data.batch["row"] * 10})

    def update_actor(self, data):
        from verl import DataProto

        self._check_divisible(data)
        self.calls.append(("update_actor", (self.active, len(data), dict(data.meta_info))))
        metrics = {
            "actor/pg_loss": [1.0, 3.0],
            "actor/kl": [0.1],
            "rollout_corr/kl": [0.9],  # same leaf as actor/kl: the later key wins
            "actor/grad_norm": [float(len(data))],
        }
        return DataProto(meta_info={"metrics": metrics})

    def save_checkpoint(self, *args: Any, **kwargs: Any) -> str:
        return "delegated"


def _proxy(wg: FakeWG, *, share_policy: bool = False, trainer_config: dict | None = None, names=None):
    from omegaconf import OmegaConf

    from marl_workflows.trainer.role_routing import RoleRoutedWorkerGroup

    return RoleRoutedWorkerGroup(
        wg,
        agent_names=names or ["generator", "aggregator", "evaluator"],
        share_policy=share_policy,
        trainer_config=OmegaConf.create(trainer_config or {}),
        rollout_temperature=0.7,
    )


def test_split_order_padding_and_indices() -> None:
    from marl_workflows.trainer.role_routing import split_batch_by_role

    batch = _batch(meta={"k": 1})
    parts = split_batch_by_role(batch, world_size=4)
    assert list(parts) == ["generator", "aggregator", "evaluator"]
    sub, idx = parts["generator"]
    assert idx.tolist() == [0, 2, 3]
    assert len(sub) == 4  # padded by duplicating the first row
    assert sub.batch["row"][:, 0].tolist() == [0.0, 2.0, 3.0, 0.0]
    assert len(parts["evaluator"][0]) == 4 and parts["evaluator"][1].tolist() == [5]
    # meta_info is copied, never shared with the parent
    sub.meta_info["x"] = 1
    assert "x" not in batch.meta_info
    assert sub.meta_info["k"] == 1


def test_split_without_padding() -> None:
    from marl_workflows.trainer.role_routing import split_batch_by_role

    parts = split_batch_by_role(_batch(), world_size=1)
    assert [len(s) for s, _ in parts.values()] == [3, 2, 1]


def test_ip_compute_log_prob_scatters_per_role() -> None:
    wg = FakeWG()
    out = _proxy(wg).compute_log_prob(_batch())
    roles = ["generator", "aggregator", "generator", "generator", "aggregator", "evaluator"]
    expected = torch.stack([-(torch.full((RESP_LEN,), float(i)) + FakeWG.ROLE_OFFSET[r]) for i, r in enumerate(roles)])
    assert torch.equal(out.batch["old_log_probs"], expected)
    assert torch.equal(out.batch["entropys"][:, 0], torch.arange(6, dtype=torch.float32) + 0.5)
    assert out.meta_info == {"temperature": 0.7}
    switches = [c[1] for c in wg.calls if c[0] == "set_active_lora"]
    assert switches == ["generator", "aggregator", "evaluator"]


def test_ip_compute_ref_log_prob_does_not_switch_adapter() -> None:
    wg = FakeWG()
    out = _proxy(wg).compute_ref_log_prob(_batch())
    assert torch.equal(out.batch["ref_log_prob"][:, 0], torch.arange(6, dtype=torch.float32) * 10)
    assert not [c for c in wg.calls if c[0] == "set_active_lora"]
    assert len([c for c in wg.calls if c[0] == "compute_ref_log_prob"]) == 3


def test_ip_update_actor_order_overrides_and_metrics() -> None:
    wg = FakeWG()
    cfg = {"kl_loss_coef_per_role": {"evaluator": 0.01, "generator": 0.0}}
    parent = _batch(meta={"temperature": 0.7, "global_token_num": [1]})
    out = _proxy(wg, trainer_config=cfg).update_actor(parent)

    order = [c[1] for c in wg.calls if c[0] in ("set_active_lora",)]
    assert order == ["generator", "aggregator", "evaluator"]
    updates = {c[1][0]: c[1][2] for c in wg.calls if c[0] == "update_actor"}
    assert updates["generator"]["kl_loss_coef_override"] == 0.0
    assert updates["evaluator"]["kl_loss_coef_override"] == 0.01
    assert "kl_loss_coef_override" not in updates["aggregator"]
    assert all(m["temperature"] == 0.7 for m in updates.values())
    assert parent.meta_info == {"temperature": 0.7, "global_token_num": [1]}

    metrics = out.meta_info["metrics"]
    assert metrics["actor/generator/pg_loss"] == pytest.approx(2.0)
    assert metrics["actor/aggregator/grad_norm"] == pytest.approx(4.0)
    assert metrics["actor/evaluator/kl"] == pytest.approx(0.9)  # collision: rollout_corr/kl came later
    assert not any(k.startswith("actor/pg_loss") for k in metrics)


def test_ip_update_sets_temperature_when_missing() -> None:
    wg = FakeWG()
    _proxy(wg).update_actor(_batch())
    assert all(c[1][2]["temperature"] == 0.7 for c in wg.calls if c[0] == "update_actor")


def test_unknown_role_raises() -> None:
    with pytest.raises(ValueError, match="not in trainer.agent_names"):
        _proxy(FakeWG(), names=["generator", "aggregator"]).compute_log_prob(_batch())


def test_sp_passthrough_and_t1_metrics() -> None:
    wg = FakeWG(world_size=2)
    proxy = _proxy(wg, share_policy=True)
    batch = _batch()
    proxy.compute_log_prob(batch)
    proxy.compute_ref_log_prob(batch)
    out = proxy.update_actor(batch)
    assert [c[0] for c in wg.calls] == ["compute_log_prob", "compute_ref_log_prob", "update_actor"]
    assert all(c[1][1] == 6 for c in wg.calls)  # whole batch, no split
    assert not any(k.startswith("actor/loss_weight/") for k in out.meta_info["metrics"])

    wg = FakeWG(world_size=2)
    batch = _batch()
    out = _proxy(wg, share_policy=True, trainer_config={"loss_weight_per_role": {"mode": "length"}}).update_actor(batch)
    metrics = out.meta_info["metrics"]
    assert metrics["actor/loss_weight/row_mean"] == pytest.approx(1.0)
    assert {"actor/loss_weight/generator", "actor/loss_weight/aggregator", "actor/pg_loss"} <= set(metrics)
    # the update saw the reweighted advantages (aggregator rows are shorter, so weighted down)
    assert batch.batch["advantages"][1, 0] < batch.batch["advantages"][0, 0]


def test_delegation_and_world_size() -> None:
    proxy = _proxy(FakeWG(world_size=8))
    assert proxy.world_size == 8
    assert proxy.save_checkpoint("x") == "delegated"
    with pytest.raises(AttributeError):
        proxy.does_not_exist  # noqa: B018
