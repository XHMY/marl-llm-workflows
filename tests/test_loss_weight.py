"""Per-role loss weight of the SP update (T1); port of the fork's tests/trainer/test_loss_weight_per_role.py."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from marl_workflows.trainer.loss_weight import apply_per_role_loss_weight

pytestmark = pytest.mark.stack

TRAJ_IDS = ["t1_generator0", "t1_generator1", "t1_generator2", "t1_aggregator", "t2_generator0", "t2_aggregator"]
LENGTHS = [4000, 3000, 5000, 500, 4000, 300]
MAX_LEN = 5120


def _batch(lengths=LENGTHS, traj_ids=TRAJ_IDS, adv_dtype=torch.float32):
    n = len(lengths)
    mask = torch.zeros(n, MAX_LEN, dtype=torch.int64)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    adv = torch.ones(n, MAX_LEN, dtype=adv_dtype) * mask.to(adv_dtype)
    returns = adv.clone()
    return SimpleNamespace(
        batch={"response_mask": mask, "advantages": adv, "returns": returns},
        non_tensor_batch={"trajectory_ids": np.array(traj_ids, dtype=object)},
    )


def _apply(batch, cfg):
    return apply_per_role_loss_weight(batch, OmegaConf.create(cfg))


def _row_scale(batch):
    # advantages were ones on response tokens; the first token holds the row weight
    return batch.batch["advantages"][:, 0].double()


def test_length_mode_weights_and_scaling():
    batch = _batch()
    returns_before = batch.batch["returns"].clone()
    batch, metrics = _apply(batch, {"mode": "length"})
    gen, agg = 4000 / 2800, 400 / 2800
    assert metrics["actor/loss_weight/generator"] == pytest.approx(gen, rel=1e-6)
    assert metrics["actor/loss_weight/aggregator"] == pytest.approx(agg, rel=1e-6)
    assert metrics["actor/loss_weight/generator"] == pytest.approx(1.4286, abs=1e-4)
    assert metrics["actor/loss_weight/aggregator"] == pytest.approx(0.1429, abs=1e-4)
    assert metrics["actor/loss_weight/generator_mean_resp_len"] == pytest.approx(4000)
    assert metrics["actor/loss_weight/aggregator_mean_resp_len"] == pytest.approx(400)
    assert metrics["actor/loss_weight/generator_rows"] == 4
    assert metrics["actor/loss_weight/aggregator_rows"] == 2
    assert metrics["actor/loss_weight/row_mean"] == pytest.approx(1.0, abs=1e-9)
    expected = torch.tensor([gen, gen, gen, agg, gen, agg], dtype=torch.float64)
    assert torch.allclose(_row_scale(batch), expected, atol=1e-6)
    # whole row scaled, masked tokens stay zero, dtype preserved, returns untouched
    adv = batch.batch["advantages"]
    assert adv.dtype == torch.float32
    assert torch.allclose(adv[0, :4000], torch.full((4000,), gen, dtype=torch.float32))
    assert torch.all(adv[3, 500:] == 0)
    assert torch.equal(batch.batch["returns"], returns_before)


def test_zero_length_row_gets_weight_one_and_is_excluded():
    traj = TRAJ_IDS + ["t3_generator0"]
    lengths = LENGTHS + [0]
    batch = _batch(lengths=lengths, traj_ids=traj)
    batch.batch["advantages"][-1, :] = 1.0  # empty response: mark the row so its scale is visible
    batch, metrics = _apply(batch, {"mode": "length"})
    # means are unchanged by the empty row
    assert metrics["actor/loss_weight/generator"] == pytest.approx(4000 / 2800, rel=1e-6)
    assert metrics["actor/loss_weight/aggregator"] == pytest.approx(400 / 2800, rel=1e-6)
    assert metrics["actor/loss_weight/generator_rows"] == 4
    assert metrics["actor/loss_weight/generator_mean_resp_len"] == pytest.approx(4000)
    assert metrics["actor/loss_weight/row_mean"] == pytest.approx(1.0, abs=1e-9)
    assert torch.all(batch.batch["advantages"][-1] == 1.0)


def test_static_mode():
    batch = _batch()
    batch, metrics = _apply(batch, {"mode": "static", "weights": {"generator": 3, "aggregator": 1}})
    # raw row weights [3,3,3,1,3,1], mean 14/6
    norm = 14 / 6
    gen, agg = 3 / norm, 1 / norm
    assert metrics["actor/loss_weight/generator"] == pytest.approx(gen, rel=1e-6)
    assert metrics["actor/loss_weight/aggregator"] == pytest.approx(agg, rel=1e-6)
    assert metrics["actor/loss_weight/row_mean"] == pytest.approx(1.0, abs=1e-9)
    expected = torch.tensor([gen, gen, gen, agg, gen, agg], dtype=torch.float64)
    assert torch.allclose(_row_scale(batch), expected, atol=1e-6)


def test_static_mode_missing_role_defaults_to_one():
    batch = _batch()
    batch, metrics = _apply(batch, {"mode": "static", "weights": {"generator": 2}})
    norm = (4 * 2 + 2 * 1) / 6
    assert metrics["actor/loss_weight/aggregator"] == pytest.approx(1 / norm, rel=1e-6)
    assert metrics["actor/loss_weight/row_mean"] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("mode", ["bogus", "none"])
def test_bad_mode_raises(mode):
    with pytest.raises(ValueError):
        _apply(_batch(), {"mode": mode})
