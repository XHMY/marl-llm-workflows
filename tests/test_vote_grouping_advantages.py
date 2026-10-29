"""Per-voter vs pooled GRPO groups for Voting under the broadcast episode reward.

Every Voting run broadcasts the episode outcome to each voter, so the three voters of an
episode carry identical rewards. rLLM groups a sample by ``{task_id}_{trajectory name}_step{k}``:
with ``separate_vote_groups=True`` the voters are ``generator0/1/2`` (one group of ``rollout.n``
per voter), with ``False`` they are all ``generator`` (one pooled group of ``3 * rollout.n``).
With identical voter rewards both groupings give the same group mean, hence the same
zero-advantage mask, and advantages that differ only by the ratio of the unbiased stds,
``sqrt(3 (n - 1) / (3 n - 1))`` (``sqrt(21 / 23) ~ 0.9555`` for ``n = 8``). With per-voter
rewards the groupings differ, including which samples get zero advantage.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

core_algos = pytest.importorskip("verl.trainer.ppo.core_algos")
from marl_workflows.workflows.voting import VotingWorkflow  # noqa: E402

N_ROLLOUTS = 8
N_VOTES = 3
RESPONSE_LEN = 4
EPS = 1e-6


def _voter_names(separate: bool) -> list[str]:
    owner = SimpleNamespace(separate_vote_groups=separate, GENERATOR_NAME=VotingWorkflow.GENERATOR_NAME)
    return [VotingWorkflow.generator_trajectory_name(owner, i) for i in range(N_VOTES)]


def _batch(rewards: np.ndarray, separate: bool) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """``rewards[task, rollout, voter]`` -> (token_level_rewards, response_mask, uid index)."""
    names = _voter_names(separate)
    uids, scores = [], []
    for t in range(rewards.shape[0]):
        for r in range(rewards.shape[1]):
            for v in range(rewards.shape[2]):
                uids.append(f"task{t}_{names[v]}_step0")
                scores.append(float(rewards[t, r, v]))
    bsz = len(scores)
    token_rewards = torch.zeros(bsz, RESPONSE_LEN)
    token_rewards[:, -1] = torch.tensor(scores)  # outcome reward on the last response token
    mask = torch.ones(bsz, RESPONSE_LEN)
    return token_rewards, mask, np.array(uids, dtype=object)


def _advantages(rewards: np.ndarray, separate: bool) -> torch.Tensor:
    token_rewards, mask, index = _batch(rewards, separate)
    adv, _ = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=token_rewards,
        response_mask=mask,
        index=index,
        epsilon=EPS,
        norm_adv_by_std_in_grpo=True,
    )
    return adv[:, 0]  # constant across the response tokens


def _episode_rewards() -> np.ndarray:
    """Broadcast reward: 6 tasks x 8 rollouts, the episode outcome copied to all 3 voters."""
    rng = np.random.default_rng(0)
    episode = np.stack(
        [
            np.ones(N_ROLLOUTS),  # all correct
            np.zeros(N_ROLLOUTS),  # all wrong
            np.array([1, 0, 0, 0, 0, 0, 0, 0]),
            np.array([1, 1, 1, 1, 0, 0, 0, 0]),
            np.array([1, 1, 1, 1, 1, 1, 1, 0]),
            rng.integers(0, 2, N_ROLLOUTS),
        ]
    ).astype(float)
    assert 0 < episode[5].sum() < N_ROLLOUTS
    return np.repeat(episode[:, :, None], N_VOTES, axis=2)


def test_uid_namings_match_the_workflow() -> None:
    assert _voter_names(True) == ["generator0", "generator1", "generator2"]
    assert _voter_names(False) == ["generator"] * N_VOTES


def test_broadcast_reward_grouping_only_rescales() -> None:
    rewards = _episode_rewards()
    per_voter = _advantages(rewards, separate=True)
    pooled = _advantages(rewards, separate=False)

    zero_per_voter = per_voter.abs() < 1e-8
    zero_pooled = pooled.abs() < 1e-8
    assert torch.equal(zero_per_voter, zero_pooled)
    # The all-correct and all-wrong tasks (first 2 * 8 * 3 rows) carry zero advantage.
    assert zero_pooled[: 2 * N_ROLLOUTS * N_VOTES].all()
    assert not zero_pooled[2 * N_ROLLOUTS * N_VOTES :].all()

    nz = ~zero_pooled
    ratio = per_voter[nz] / pooled[nz]
    expected = math.sqrt(N_VOTES * (N_ROLLOUTS - 1) / (N_VOTES * N_ROLLOUTS - 1))  # sqrt(21/23)
    assert expected == pytest.approx(0.9555, abs=1e-4)
    assert ratio.min().item() == pytest.approx(expected, abs=1e-3)
    assert ratio.max().item() == pytest.approx(expected, abs=1e-3)
    assert torch.equal(torch.sign(per_voter), torch.sign(pooled))


def test_per_voter_rewards_change_the_groups() -> None:
    """With per-voter rewards (voters differ) the grouping changes which samples get zero advantage."""
    rewards = np.zeros((2, N_ROLLOUTS, N_VOTES))
    rewards[0, :, 0] = 1.0  # task 0: voter 0 always right, voters 1 and 2 always wrong
    # Task 1: every voter mixed, with different success rates (4/8, 6/8, 1/8).
    rewards[1, :, 0] = [1, 0, 1, 0, 1, 0, 1, 0]
    rewards[1, :, 1] = [1, 1, 1, 0, 1, 1, 0, 1]
    rewards[1, :, 2] = [0, 0, 0, 1, 0, 0, 0, 0]
    per_voter = _advantages(rewards, separate=True)
    pooled = _advantages(rewards, separate=False)

    zero_per_voter = per_voter.abs() < 1e-8
    zero_pooled = pooled.abs() < 1e-8
    assert not torch.equal(zero_per_voter, zero_pooled)
    # Task 0: each voter's own group is constant, so per-voter GRPO gives zero advantage
    # everywhere; the pooled group rewards voter 0 and penalizes voters 1 and 2.
    task0 = slice(0, N_ROLLOUTS * N_VOTES)
    assert zero_per_voter[task0].all()
    assert not zero_pooled[task0].any()
    nz = ~zero_per_voter & ~zero_pooled
    ratio = per_voter[nz] / pooled[nz]
    assert (ratio.max() - ratio.min()).item() > 1e-2  # not a constant rescaling
