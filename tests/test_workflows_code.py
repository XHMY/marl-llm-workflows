"""Code workflows on a scripted rollout engine with a stub code-reward executor.

``code_reward_fn`` is monkeypatched in every code module with a fake that marks a response
correct iff it contains ``PASS``; it runs through ``run_in_code_executor`` on a
``ThreadPoolExecutor`` standing in for the spawn process pool.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from fakes.fake_rollout import FakeRolloutEngine
from rllm.agents.agent import Episode
from rllm.rewards.reward_types import RewardOutput

from marl_workflows import contracts
from marl_workflows.recipes import build_workflow_args, get_recipe, load_object
from marl_workflows.workflows.base import load_prompts
from marl_workflows.workflows.code.evaluator_optimizer import DeepcodeEvaluatorOptimizerWorkflow
from marl_workflows.workflows.code.single_agent import SingleAgentDeepcodeWorkflow
from marl_workflows.workflows.code.voting import DeepcodeVotingWorkflow
from marl_workflows.workflows.code.voting_v2 import DeepcodeVotingV2Workflow

pytestmark = pytest.mark.stack

UID = "7d1e0c2a-0000-4abc-8def-0123456789ab:1"
PASS = "```python\nprint('PASS')\n```"
FAIL = "```python\nprint('FAIL')\n```"
NO_CODE = "I could not finish"
CODE_MODULES = ("single_agent", "voting", "voting_v2", "evaluator_optimizer", "orchestrator_workers")


def fake_code_reward_fn(task_info: dict, action: str) -> RewardOutput:
    ok = "PASS" in (action or "")
    return RewardOutput(
        reward=1.0 if ok else 0.0,
        is_correct=ok,
        metadata={"passed_tests": 2 if ok else 0, "total_tests": 2, "test_results": [{"passed": ok}]},
    )


@pytest.fixture
def reward_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace code_reward_fn in every code module; record the graded responses."""
    calls: list[str] = []

    def _fn(task_info: dict, action: str) -> RewardOutput:
        calls.append(action)
        return fake_code_reward_fn(task_info, action)

    for name in CODE_MODULES:
        monkeypatch.setattr(importlib.import_module(f"marl_workflows.workflows.code.{name}"), "code_reward_fn", _fn)
    return calls


@pytest.fixture
def executor() -> Iterator[ThreadPoolExecutor]:
    pool = ThreadPoolExecutor(max_workers=4)
    yield pool
    pool.shutdown(wait=True)


def _task() -> dict[str, Any]:
    return {"question": "Print PASS.", "ground_truth": "[]", "data_source": "primeintellect"}


def _build(workflow: str, engine: FakeRolloutEngine, executor: ThreadPoolExecutor, **overrides: Any):
    recipe = get_recipe("deepcoder", workflow)
    args = build_workflow_args(recipe, None)
    assert "reward_function" not in args  # the code workflows call code_reward_fn themselves
    args.update(overrides)
    return load_object(recipe.workflow_cls)(rollout_engine=engine, code_reward_executor=executor, **args)


def _names(episode: Episode) -> list[str]:
    return [t.name for t in episode.trajectories]


def _rewards(episode: Episode) -> list[float]:
    return [t.reward for t in episode.trajectories]


def _check_contract(workflow: str, episode: Episode, engine: FakeRolloutEngine) -> None:
    recipe = get_recipe("deepcoder", workflow)
    for traj in episode.trajectories:
        contracts.validate_trajectory_name(traj.name)
        assert contracts.role_from_name(traj.name) in recipe.agent_names
        assert [s.reward for s in traj.steps] == [traj.reward]
    assert set(engine.agent_names) == set(recipe.agent_names)


async def test_single_agent(executor, reward_calls) -> None:
    engine = FakeRolloutEngine({"generator": FAIL})
    wf = _build("single_agent", engine, executor)
    episode = await wf.run(_task(), UID)
    _check_contract("single_agent", episode, engine)
    assert _names(episode) == ["generator"] and _rewards(episode) == [0.0]
    assert reward_calls == [FAIL]
    assert "application_id" not in engine.calls[0].kwargs
    assert episode.metrics["pass_rate"] == 0.0 and episode.info == {"reward_test_details": [{"passed": False}]}
    template = load_prompts("deepcoder", "deepcoder_single_agent_prompts")["generator_initial"]["template"]
    assert engine.calls[0].prompt == template.format(problem="Print PASS.")


async def test_run_in_code_executor_requires_an_executor(reward_calls) -> None:
    wf = SingleAgentDeepcodeWorkflow(rollout_engine=FakeRolloutEngine({"generator": PASS}))
    with pytest.raises(ValueError, match="No executor"):
        await wf.run(_task(), UID)
    # The general executor is the fallback when no code executor is given.
    with ThreadPoolExecutor(max_workers=1) as pool:
        wf = SingleAgentDeepcodeWorkflow(rollout_engine=FakeRolloutEngine({"generator": PASS}), executor=pool)
        episode = await wf.run(_task(), UID)
    assert episode.is_correct and _rewards(episode) == [1.0]


@pytest.mark.parametrize("ufor", [True, False])
async def test_voting_pooled_names_and_unconditional_broadcast(executor, reward_calls, ufor: bool) -> None:
    engine = FakeRolloutEngine({"generator": [FAIL, PASS, NO_CODE], "aggregator": "\\boxed{2}"})
    wf = _build("voting", engine, executor, use_final_outcome_reward=ufor)
    episode = await wf.run(_task(), UID)
    _check_contract("voting", episode, engine)
    # All voters share one trajectory name (one pooled GRPO group).
    assert _names(episode) == ["generator", "generator", "generator", "aggregator"]
    # No sticky-session id on any Code Voting call.
    assert all("application_id" not in c.kwargs for c in engine.calls)
    # Only the selected solution is tested.
    assert reward_calls == [PASS]
    # The single pass broadcasts the final outcome whatever use_final_outcome_reward says.
    assert _rewards(episode) == [1.0, 1.0, 1.0, 1.0]
    assert episode.metrics == {
        "aggregator_acc": 1.0,
        "n_votes": 3,
        "test_rounds": 1,
        "success": 1,
        "generator_attempts": 3,
    }


async def test_voting_wrong_selection(executor, reward_calls) -> None:
    engine = FakeRolloutEngine({"generator": [FAIL, PASS, PASS], "aggregator": "\\boxed{1}"})
    episode = await _build("voting", engine, executor).run(_task(), UID)
    assert not episode.is_correct and _rewards(episode) == [0.0] * 4


@pytest.mark.parametrize("ufor", [True, False])
async def test_voting_v2_default_broadcast(executor, reward_calls, ufor: bool) -> None:
    engine = FakeRolloutEngine({"voterA": FAIL, "voterB": PASS, "voterC": NO_CODE, "aggregator": "\\boxed{2}"})
    wf = _build("voting_v2", engine, executor, use_final_outcome_reward=ufor)
    episode = await wf.run(_task(), UID)
    _check_contract("voting_v2", episode, engine)
    assert _names(episode) == ["voterA", "voterB", "voterC", "aggregator"]
    assert engine.agent_names == ["voterA", "voterB", "voterC", "aggregator"]
    assert all("application_id" not in c.kwargs for c in engine.calls)
    assert reward_calls == [PASS]
    assert _rewards(episode) == [1.0] * 4
    assert "agg_rubric_reward" not in episode.metrics


async def test_voting_v2_rubric(executor, reward_calls) -> None:
    engine = FakeRolloutEngine({"voterA": FAIL, "voterB": PASS, "voterC": NO_CODE, "aggregator": "\\boxed{2}"})
    recipe = get_recipe("deepcoder", "voting_v2")
    wf = load_object(recipe.workflow_cls)(
        rollout_engine=engine, code_reward_executor=executor, **dict(recipe.eval_params)
    )
    episode = await wf.run(_task(), UID)
    # Every voter is tested, then the selection.
    assert sorted(reward_calls) == sorted([FAIL, PASS, NO_CODE, PASS])
    # voterA: wrong with a code block -> 0.5 - 1.0; voterB: 0.5 + 0.5; voterC: no code block -> 0.5.
    assert _rewards(episode) == [-0.5, 1.0, 0.5, 1.0]
    assert episode.metrics["voter_parseable_count"] == 2 and episode.metrics["voter_acc"] == 1 / 3


@pytest.mark.parametrize("ufor", [True, False])
async def test_evaluator_optimizer(executor, reward_calls, ufor: bool) -> None:
    engine = FakeRolloutEngine(
        {"generator": [FAIL, PASS], "evaluator": ["\\boxed{Incorrect}", "\\boxed{Correct}"]},
    )
    wf = _build("evaluator_optimizer", engine, executor, use_final_outcome_reward=ufor)
    assert wf.max_iterations == 2
    episode = await wf.run(_task(), UID)
    _check_contract("evaluator_optimizer", episode, engine)
    assert _names(episode) == ["generator", "evaluator", "generator", "evaluator"]
    assert all("application_id" not in c.kwargs for c in engine.calls)
    # Only the final code is tested.
    assert reward_calls == [PASS]
    if ufor:
        assert _rewards(episode) == [1.0] * 4
    else:
        # Generators get the final test result; evaluators get verdict == final test result.
        assert _rewards(episode) == [1.0, 0.0, 1.0, 1.0]
    assert episode.metrics["evaluator_predictions"] == 2 and episode.metrics["test_passed"] == 1
    prompts = load_prompts("deepcoder", "deepcoder_evaluator_optimizer_prompts")
    assert engine.calls[0].prompt == prompts["generator_initial"]["template"].format(problem="Print PASS.")
    assert engine.calls[1].prompt == prompts["evaluator_critique"]["template"]


async def test_orchestrator_workers(executor, reward_calls) -> None:
    engine = FakeRolloutEngine(
        {
            "orchestrator": "STRATEGY 1: brute force\nSTRATEGY 2: greedy",
            "worker": [FAIL, PASS],
            "synthesizer": PASS,
        }
    )
    wf = _build("orchestrator_workers_propose", engine, executor)
    episode = await wf.run(_task(), UID)
    _check_contract("orchestrator_workers_propose", episode, engine)
    assert _names(episode) == ["orchestrator", "worker", "worker", "synthesizer"]
    # The generic orchestrator workflow sends the sticky-session id on every call.
    assert all(c.kwargs.get("application_id") == UID for c in engine.calls)
    assert reward_calls == [PASS]
    assert _rewards(episode) == [1.0] * 4


@pytest.mark.parametrize(
    "cls",
    [SingleAgentDeepcodeWorkflow, DeepcodeVotingWorkflow, DeepcodeVotingV2Workflow, DeepcodeEvaluatorOptimizerWorkflow],
)
def test_enable_test_loop_raises(cls) -> None:
    with pytest.raises(ValueError, match="enable_test_loop"):
        cls(rollout_engine=FakeRolloutEngine(), enable_test_loop=True)
    wf = cls(rollout_engine=FakeRolloutEngine(), enable_test_loop=False, max_test_rounds=5, public_test_only=True)
    assert wf.enable_test_loop is False and wf.max_test_rounds == 5
