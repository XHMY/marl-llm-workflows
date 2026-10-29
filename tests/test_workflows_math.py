"""Math workflows on a scripted rollout engine: names, routing, rewards, prompts.

Also checks that every class path in ``marl_workflows.recipes.RECIPES`` resolves (WP1's
``tests/test_recipes.py`` checks the paths only syntactically).
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any

import pytest
from fakes.fake_rollout import FakeRolloutEngine
from rllm.agents.agent import Episode

from marl_workflows import contracts
from marl_workflows.recipes import RECIPES, build_workflow_args, get_recipe, load_object
from marl_workflows.rewards.math_reward import math_reward_fn
from marl_workflows.workflows.base import MultiAgentWorkflow, broadcast_final_reward, load_prompts
from marl_workflows.workflows.math.controlled_sa_hga import ControlledSAHGAMathWorkflow
from marl_workflows.workflows.math.evaluator_optimizer import EvaluatorOptimizerMathWorkflow
from marl_workflows.workflows.math.orchestrator_workers import OrchestratorWorkersMathWorkflow
from marl_workflows.workflows.math.single_agent import SingleAgentMathWorkflow
from marl_workflows.workflows.math.voting import VotingMathWorkflow
from marl_workflows.workflows.math.voting_prompt_het import STYLE_SUFFIXES, VotingMathPromptHetWorkflow
from marl_workflows.workflows.math.voting_v2 import VotingV2MathWorkflow

pytestmark = pytest.mark.stack

UID = "3f2c9a8e-1b2d-4c5e-9f00-123456789abc:0"
RIGHT = "<think>2 + 2 = 4</think>\nThe answer is \\boxed{4}."
WRONG = "<think>2 + 2 = 5</think>\nThe answer is \\boxed{5}."
UNBOXED = "<think>still thinking about it"


def _task() -> dict[str, Any]:
    return {"question": "What is 2 + 2?", "final_answer": "4"}


def _build(task: str, workflow: str, engine: FakeRolloutEngine, **overrides: Any) -> MultiAgentWorkflow:
    """Instantiate a recipe's class the way the trainer does (recipe params + reward_function)."""
    recipe = get_recipe(task, workflow)
    args = build_workflow_args(recipe, None)
    args.update(overrides)
    return load_object(recipe.workflow_cls)(rollout_engine=engine, **args)


def _names(episode: Episode) -> list[str]:
    return [t.name for t in episode.trajectories]


def _rewards(episode: Episode) -> list[float]:
    return [t.reward for t in episode.trajectories]


def _check_contract(task: str, workflow: str, episode: Episode, engine: FakeRolloutEngine) -> None:
    """Trajectory names follow C1, their roles and every agent_name are recipe agent names (C5)."""
    recipe = get_recipe(task, workflow)
    for traj in episode.trajectories:
        contracts.validate_trajectory_name(traj.name)
        assert contracts.role_from_name(traj.name) in recipe.agent_names, traj.name
        assert contracts.role_from_trajectory_id(f"{UID}_{traj.name}") == contracts.role_from_name(traj.name)
        assert len(traj.steps) == 1
    for name in engine.agent_names:
        assert name in recipe.agent_names, name
    # Every adapter that the recipe trains is actually sampled.
    assert set(engine.agent_names) == set(recipe.agent_names)


# --------------------------------------------------------------------------------------
# Recipes, base class and helpers
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(RECIPES))
def test_recipe_class_paths_resolve(key: tuple[str, str]) -> None:
    recipe = RECIPES[key]
    cls = load_object(recipe.workflow_cls)
    assert inspect.isclass(cls) and issubclass(cls, MultiAgentWorkflow)
    assert not inspect.isabstract(cls)
    params = inspect.signature(cls.__init__).parameters
    accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    for name, _ in (*recipe.params, *recipe.eval_params):
        assert name in params or accepts_kwargs, (key, name)
    if recipe.reward_fn is not None:
        assert load_object(recipe.reward_fn) is math_reward_fn


def test_executor_is_optional() -> None:
    wf = VotingMathWorkflow(rollout_engine=FakeRolloutEngine(), reward_function=math_reward_fn)
    assert wf.executor is None and wf.code_reward_executor is None

    async def _call() -> None:
        await wf.run_in_executor(len, "abc")

    import asyncio

    with pytest.raises(ValueError, match="no executor"):
        asyncio.run(_call())


def test_broadcast_final_reward_sets_every_step() -> None:
    from rllm.agents.agent import Step, Trajectory

    trajs = [Trajectory(name="a", steps=[Step(reward=0.3), Step(reward=0.1)], reward=0.2), Trajectory(name="b")]
    broadcast_final_reward(trajs, 1.0)
    assert [t.reward for t in trajs] == [1.0, 1.0]
    assert [s.reward for s in trajs[0].steps] == [1.0, 1.0]


# --------------------------------------------------------------------------------------
# Single agent
# --------------------------------------------------------------------------------------


async def test_single_agent() -> None:
    engine = FakeRolloutEngine({"generator": RIGHT})
    wf = _build("math", "single_agent", engine)
    task = _task()
    episode = await wf.run(task, UID)
    _check_contract("math", "single_agent", episode, engine)
    assert _names(episode) == ["generator"]
    assert episode.is_correct and episode.metrics == {"generator_acc": 1.0, "success": 1}
    assert episode.trajectories[0].steps[0].reward == 1.0
    # The fork sets only the step reward here; the trajectory keeps its default.
    assert episode.trajectories[0].reward == 0.0
    assert task["ground_truth"] == "4"
    # No sticky-session id on the single-agent call.
    assert "application_id" not in engine.calls[0].kwargs
    template = load_prompts("math", "multi_agent_math_prompts")["generator_initial"]["template"]
    assert engine.calls[0].prompt == template.format(problem=task["question"])


# --------------------------------------------------------------------------------------
# Voting
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("ufor", [True, False])
async def test_voting(ufor: bool) -> None:
    engine = FakeRolloutEngine({"generator": [WRONG, RIGHT, UNBOXED], "aggregator": "I pick \\boxed{2}"})
    wf = _build("math", "voting", engine, use_final_outcome_reward=ufor)
    episode = await wf.run(_task(), UID)
    _check_contract("math", "voting", episode, engine)

    # One GRPO group per voter index: generator0/1/2; three calls routed to the "generator" adapter.
    assert _names(episode) == ["generator0", "generator1", "generator2", "aggregator"]
    assert engine.agent_names == ["generator", "generator", "generator", "aggregator"]
    assert all(c.kwargs.get("application_id") == UID for c in engine.calls)
    assert episode.is_correct
    assert episode.metrics == {
        "generator_acc": 1 / 3,
        "aggregator_acc": 1.0,
        "n_votes": 3,
        "any_correct": 1,
        "success": 1,
        "generator_attempts": 3,
    }
    # The aggregator prompt lists the three voter outputs in vote order.
    agg_prompt = engine.calls_for("aggregator")[0].prompt
    assert agg_prompt.index(WRONG) < agg_prompt.index(RIGHT) < agg_prompt.index(UNBOXED)
    assert episode.trajectories[3].steps[0].action == RIGHT
    if ufor:
        assert _rewards(episode) == [1.0, 1.0, 1.0, 1.0]
    else:
        assert _rewards(episode) == [0.0, 1.0, 0.0, 1.0]
    for traj in episode.trajectories:
        assert [s.reward for s in traj.steps] == [traj.reward]


async def test_voting_broadcast_of_wrong_selection() -> None:
    engine = FakeRolloutEngine({"generator": [WRONG, RIGHT, RIGHT], "aggregator": "\\boxed{1}"})
    wf = _build("math", "voting", engine)
    episode = await wf.run(_task(), UID)
    assert not episode.is_correct
    # The broadcast reward gives correct voters 0 as well.
    assert _rewards(episode) == [0.0, 0.0, 0.0, 0.0]
    assert episode.metrics["any_correct"] == 1


def _without_names(episode: Episode) -> dict[str, Any]:
    """The episode as plain data, without trajectory names and the random trajectory uids."""
    data = dataclasses.asdict(episode)
    for traj in data["trajectories"]:
        del traj["uid"], traj["name"]
    return data


@pytest.mark.parametrize("workflow", ["voting", "voting_prompt_het"])
async def test_voting_separate_vote_groups(workflow: str) -> None:
    """Math default: one GRPO group per voter; the flag changes the voter names and nothing else."""
    script = {"generator": [WRONG, RIGHT, UNBOXED], "aggregator": "I pick \\boxed{2}"}
    episodes, engines = {}, {}
    for separate in (True, False):
        engines[separate] = FakeRolloutEngine(dict(script))
        overrides = {} if separate else {"separate_vote_groups": False}
        wf = _build("math", workflow, engines[separate], **overrides)
        assert wf.separate_vote_groups is separate
        episodes[separate] = await wf.run(_task(), UID)
        _check_contract("math", workflow, episodes[separate], engines[separate])
    assert _names(episodes[True]) == ["generator0", "generator1", "generator2", "aggregator"]
    assert _names(episodes[False]) == ["generator", "generator", "generator", "aggregator"]
    assert _without_names(episodes[True]) == _without_names(episodes[False])
    calls = {k: [(c.messages, c.kwargs) for c in e.calls] for k, e in engines.items()}
    assert calls[True] == calls[False]


def test_voting_aggregator_parsing() -> None:
    wf = VotingMathWorkflow(rollout_engine=FakeRolloutEngine(), reward_function=math_reward_fn)
    cands = ["a", "b", "c"]
    assert wf.parse_aggregator_response("\\boxed{3}", cands) == "c"
    assert wf.parse_aggregator_response("\\boxed{B}, i.e. solution 2", cands) == "b"
    # The digit fallback takes the first digit, even an out-of-range one, then gives up.
    assert wf.parse_aggregator_response("\\boxed{9} but maybe 2", cands) == "a"
    assert wf.parse_aggregator_response("none of them", cands) == "a"
    assert wf.parse_aggregator_response("x", []) == ""


async def test_voting_prompt_het_suffixes() -> None:
    engine = FakeRolloutEngine({"generator": RIGHT, "aggregator": "\\boxed{1}"})
    wf = _build("math", "voting_prompt_het", engine)
    assert isinstance(wf, VotingMathPromptHetWorkflow)
    episode = await wf.run(_task(), UID)
    _check_contract("math", "voting_prompt_het", episode, engine)
    assert _names(episode) == ["generator0", "generator1", "generator2", "aggregator"]
    base_prompt = wf.build_generator_prompt(_task())
    gen_prompts = [c.prompt for c in engine.calls_for("generator")]
    assert gen_prompts == [base_prompt + s for s in STYLE_SUFFIXES]
    assert all(c.kwargs.get("application_id") == UID for c in engine.calls)
    assert len(STYLE_SUFFIXES) == 3 and STYLE_SUFFIXES[0].startswith("\n\nApproach: algebraic manipulation.")
    assert wf._style_for_index(4) == STYLE_SUFFIXES[1]


# --------------------------------------------------------------------------------------
# voting_v2 (one adapter per voter)
# --------------------------------------------------------------------------------------


async def test_voting_v2_broadcast() -> None:
    engine = FakeRolloutEngine(
        {"voterA": WRONG, "voterB": RIGHT, "voterC": UNBOXED, "aggregator": "\\boxed{2}"},
    )
    wf = _build("math", "voting_v2", engine)  # training defaults: broadcast on, rubric off
    episode = await wf.run(_task(), UID)
    _check_contract("math", "voting_v2", episode, engine)
    assert _names(episode) == ["voterA", "voterB", "voterC", "aggregator"]
    assert all(c.kwargs.get("application_id") == UID for c in engine.calls)
    assert _rewards(episode) == [1.0, 1.0, 1.0, 1.0]
    assert episode.metrics["voterA_acc"] == 0.0 and episode.metrics["voterB_acc"] == 1.0
    assert episode.metrics["voter_parseable_count"] == 2
    assert "agg_rubric_reward" not in episode.metrics


async def test_voting_v2_rubric() -> None:
    engine = FakeRolloutEngine(
        {"voterA": WRONG, "voterB": RIGHT, "voterC": UNBOXED, "aggregator": "\\boxed{2}"},
    )
    recipe = get_recipe("math", "voting_v2")
    wf = load_object(recipe.workflow_cls)(
        rollout_engine=engine, reward_function=math_reward_fn, **dict(recipe.eval_params)
    )
    episode = await wf.run(_task(), UID)
    # voterA: wrong and boxed -> +0.5 (final) - 1.0; voterB: +0.5 + 0.5; voterC: unboxed -> +0.5.
    assert _rewards(episode) == [-0.5, 1.0, 0.5, 1.0]
    for traj in episode.trajectories:
        assert traj.steps[0].reward == traj.reward
    assert episode.metrics["agg_rubric_reward"] == 1.0


async def test_voting_v2_rubric_aggregator_not_blamed_when_all_wrong() -> None:
    engine = FakeRolloutEngine({"voterA": WRONG, "voterB": WRONG, "voterC": WRONG, "aggregator": "\\boxed{1}"})
    wf = VotingV2MathWorkflow(rollout_engine=engine, reward_function=math_reward_fn, use_rubric_reward=True)
    episode = await wf.run(_task(), UID)
    assert _rewards(episode) == [-1.0, -1.0, -1.0, 0.0]


def test_voting_v2_rubric_xor_broadcast() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        VotingV2MathWorkflow(
            rollout_engine=FakeRolloutEngine(),
            reward_function=math_reward_fn,
            use_rubric_reward=True,
            use_final_outcome_reward=True,
        )
    with pytest.raises(AssertionError):
        VotingV2MathWorkflow(rollout_engine=FakeRolloutEngine(), reward_function=math_reward_fn, voter_names=["a_b"])
    wf = VotingV2MathWorkflow(rollout_engine=FakeRolloutEngine(), reward_function=math_reward_fn, n_votes=4)
    assert wf.voter_names == ["voterA", "voterB", "voterC", "voterD"]


# --------------------------------------------------------------------------------------
# Evaluator-Optimizer
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("ufor", [True, False])
async def test_evaluator_optimizer_loop(ufor: bool) -> None:
    engine = FakeRolloutEngine(
        {
            "generator": [UNBOXED, RIGHT],
            "evaluator": ["It is cut off. \\boxed{Incorrect}", "Looks right. \\boxed{Correct}"],
        }
    )
    wf = _build("math", "evaluator_optimizer", engine, use_final_outcome_reward=ufor)
    episode = await wf.run(_task(), UID)
    _check_contract("math", "evaluator_optimizer", episode, engine)

    # Every turn is its own trajectory; generator turns share the name "generator".
    assert _names(episode) == ["generator", "evaluator", "generator", "evaluator"]
    assert engine.agent_names == _names(episode)
    assert all(c.kwargs.get("application_id") == UID for c in engine.calls)
    assert episode.is_correct
    assert episode.metrics == {
        "generator_acc": 0.5,
        "evaluator_acc": 1.0,
        "total_iterations": 2,
        "success": 1,
        "generator_attempts": 2,
        "evaluator_predictions": 2,
    }
    if ufor:
        assert _rewards(episode) == [1.0, 1.0, 1.0, 1.0]
    else:
        # Per-role rewards: generator turn correctness; evaluator verdict matches correctness.
        assert _rewards(episode) == [0.0, 1.0, 1.0, 1.0]
    # Shared history: the evaluator sees the generator turn, the refinement sees both replies.
    prompts = load_prompts("math", "evaluator_optimizer_prompts")
    eval_call, refine_call = engine.calls[1], engine.calls[2]
    assert [m["role"] for m in eval_call.messages] == ["user", "assistant", "user"]
    assert eval_call.prompt == prompts["evaluator_critique"]["template"]
    # The fork appends the evaluator reply but not the evaluator prompt to the history.
    assert [m["role"] for m in refine_call.messages] == ["user", "assistant", "assistant", "user"]
    assert refine_call.prompt == prompts["generator_refinement"]["template"]
    assert episode.trajectories[1].steps[0].action == {
        "verdict": "incorrect",
        "feedback": "It is cut off. \\boxed{Incorrect}",
    }


async def test_evaluator_optimizer_reward_wrong_verdicts() -> None:
    engine = FakeRolloutEngine(
        {"generator": [RIGHT, WRONG, WRONG], "evaluator": ["\\boxed{Incorrect}", "no verdict", "\\boxed{Correct}"]}
    )
    wf = _build("math", "evaluator_optimizer", engine, use_final_outcome_reward=False)
    episode = await wf.run(_task(), UID)
    # max_iterations=3: gen, eval(wrong verdict), gen, eval(unparsed), gen, eval(wrong verdict).
    assert _names(episode) == ["generator", "evaluator"] * 3
    assert _rewards(episode) == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert not episode.is_correct and episode.metrics["total_iterations"] == 3


async def test_evaluator_optimizer_stops_on_correct_verdict() -> None:
    engine = FakeRolloutEngine({"generator": WRONG, "evaluator": "\\boxed{Correct}"})
    wf = _build("math", "evaluator_optimizer", engine)
    episode = await wf.run(_task(), UID)
    assert _names(episode) == ["generator", "evaluator"]
    assert _rewards(episode) == [0.0, 0.0]


# --------------------------------------------------------------------------------------
# Orchestrator-Workers
# --------------------------------------------------------------------------------------


PROPOSAL = "STRATEGY 1: add directly\nSTRATEGY 2: count on fingers\nSTRATEGY 3: use a number line\nSTRATEGY 4: extra"


@pytest.mark.parametrize("ufor", [True, False])
async def test_orchestrator_workers(ufor: bool) -> None:
    engine = FakeRolloutEngine(
        {"orchestrator": PROPOSAL, "worker": [RIGHT, WRONG, UNBOXED], "synthesizer": RIGHT},
    )
    wf = _build("math", "orchestrator_workers_propose", engine, use_final_outcome_reward=ufor)
    assert wf.max_subtasks == 3
    episode = await wf.run(_task(), UID)
    _check_contract("math", "orchestrator_workers_propose", episode, engine)
    assert _names(episode) == ["orchestrator", "worker", "worker", "worker", "synthesizer"]
    assert all(c.kwargs.get("application_id") == UID for c in engine.calls)
    # The fourth strategy is truncated by max_subtasks=3.
    assert episode.trajectories[0].steps[0].action["strategies"] == [
        "add directly",
        "count on fingers",
        "use a number line",
    ]
    worker_prompts = [c.prompt for c in engine.calls_for("worker")]
    op = load_prompts("math", "orchestrator_workers_prompts")
    assert worker_prompts == [
        op["worker_full_solve"]["template"].format(problem="What is 2 + 2?", strategy=s)
        for s in ("add directly", "count on fingers", "use a number line")
    ]
    assert episode.is_correct
    assert episode.metrics["n_workers"] == 3 and episode.metrics["total_trajectories"] == 5
    if ufor:
        assert _rewards(episode) == [1.0] * 5
    else:
        assert _rewards(episode) == [0.0, 0.0, 0.0, 0.0, 1.0]


def test_orchestrator_default_max_subtasks() -> None:
    wf = OrchestratorWorkersMathWorkflow(rollout_engine=FakeRolloutEngine(), reward_function=math_reward_fn)
    assert wf.max_subtasks == 3
    proposal = wf.parse_proposals("1. first idea here\n2. second idea here")
    assert proposal.strategies == ["first idea here", "second idea here"]
    assert wf.parse_proposals("").strategies == [""]


# --------------------------------------------------------------------------------------
# Single-role control (controlled_sa_hga)
# --------------------------------------------------------------------------------------


async def test_controlled_sa_hga() -> None:
    engine = FakeRolloutEngine({"generator": [RIGHT, WRONG, RIGHT]})
    # use_final_outcome_reward is ignored: each virtual generator is rewarded on its own answer.
    wf = _build("math", "controlled_sa_hga", engine, use_final_outcome_reward=True)
    assert isinstance(wf, ControlledSAHGAMathWorkflow)
    episode = await wf.run(_task(), UID)
    _check_contract("math", "controlled_sa_hga", episode, engine)
    assert _names(episode) == ["generator0", "generator1", "generator2"]
    assert engine.agent_names == ["generator"] * 3
    assert all(c.kwargs.get("application_id") == UID for c in engine.calls)
    assert _rewards(episode) == [1.0, 0.0, 1.0]
    assert episode.is_correct and episode.metrics == {
        "generator_acc": 2 / 3,
        "generator_attempts": 3,
        "success": 1,
    }
    template = load_prompts("math", "multi_agent_math_prompts")["generator_initial"]["template"]
    assert all(c.prompt == template.format(problem="What is 2 + 2?") for c in engine.calls)


# --------------------------------------------------------------------------------------
# Prompts come from the packaged files
# --------------------------------------------------------------------------------------


def test_math_prompts_from_packaged_sections() -> None:
    engine = FakeRolloutEngine()
    task = _task()
    voting = VotingMathWorkflow(rollout_engine=engine, reward_function=math_reward_fn)
    vp = load_prompts("math", "voting_prompts")
    assert voting.build_generator_prompt(task) == vp["generator"]["template"].format(problem=task["question"])
    eo = EvaluatorOptimizerMathWorkflow(rollout_engine=engine, reward_function=math_reward_fn)
    ep = load_prompts("math", "evaluator_optimizer_prompts")
    assert eo.build_generator_prompt(task) == ep["generator_initial"]["template"].format(problem=task["question"])
    orch = OrchestratorWorkersMathWorkflow(rollout_engine=engine, reward_function=math_reward_fn)
    op = load_prompts("math", "orchestrator_workers_prompts")
    assert orch.build_proposal_prompt(task, 3) == op["orchestrator_propose"]["template"].format(
        problem=task["question"], max_subtasks=3
    )
    # An explicit prompts dict overrides the packaged file.
    custom = VotingMathWorkflow(
        rollout_engine=engine, reward_function=math_reward_fn, prompts={"generator": {"template": "Q: {problem}"}}
    )
    assert custom.build_generator_prompt(task) == "Q: What is 2 + 2?"
    sa = SingleAgentMathWorkflow(rollout_engine=engine, reward_function=math_reward_fn)
    assert sa.executor is None
