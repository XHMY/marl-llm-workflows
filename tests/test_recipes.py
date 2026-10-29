"""Tests for marl_workflows.recipes: the B1 table, experiment-name parsing, import lightness."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from marl_workflows import contracts
from marl_workflows.recipes import (
    RECIPES,
    Recipe,
    build_workflow_args,
    get_recipe,
    infer_task,
    load_object,
    recipe_for_experiment,
    validate_recipe,
)

FIXTURES = Path(__file__).parent / "fixtures"
MATH_REWARD = "marl_workflows.rewards.math_reward.math_reward_fn"
UFOR = ("use_final_outcome_reward", True)
MATH_VOTING = (("n_votes", 3), UFOR, ("separate_vote_groups", True))
CODE_VOTING = (("n_votes", 3), UFOR, ("separate_vote_groups", False))

# Copy of DESIGN.md B1, checked value by value against fork dashboard/task_configs.py,
# dashboard/launch_experiment.sh and examples/*/train_*.py.
# (task, workflow): (prefix, suffix, cls, agent_names, sp_allowed, params, train, eval, eval_params, remap, gen_roles)
EXPECTED = {
    ("math", "single_agent"): (
        "single_agent", None, "marl_workflows.workflows.math.single_agent.SingleAgentMathWorkflow",
        ("generator",), (False,), (), (15360, 5120), (15360, 8192), (), (), ("generator",),
    ),
    ("math", "voting"): (
        "voting", None, "marl_workflows.workflows.math.voting.VotingMathWorkflow",
        ("generator", "aggregator"), (False, True), MATH_VOTING, (20480, 5120), (25600, 8192),
        MATH_VOTING, (), ("generator",),
    ),
    ("math", "voting_prompt_het"): (
        "voting", "prompt_het", "marl_workflows.workflows.math.voting_prompt_het.VotingMathPromptHetWorkflow",
        ("generator", "aggregator"), (False, True), MATH_VOTING, (20480, 5120), (25600, 8192),
        MATH_VOTING, (), ("generator",),
    ),
    ("math", "evaluator_optimizer"): (
        "evaluator_optimizer", None, "marl_workflows.workflows.math.evaluator_optimizer.EvaluatorOptimizerMathWorkflow",
        ("generator", "evaluator"), (False, True), (("max_iterations", 3), UFOR), (30720, 5120), (30720, 8192),
        (("max_iterations", 3), UFOR), (), ("generator",),
    ),
    ("math", "orchestrator_workers_propose"): (
        "orchestrator_workers_propose", None,
        "marl_workflows.workflows.math.orchestrator_workers.OrchestratorWorkersMathWorkflow",
        ("orchestrator", "worker", "synthesizer"), (False, True), (("max_subtasks", 3), UFOR), (20480, 5120),
        (25600, 8192), (("max_subtasks", 3), UFOR), (("generator", "worker"),), ("worker",),
    ),
    ("math", "voting_v2"): (
        "voting_v2", None, "marl_workflows.workflows.math.voting_v2.VotingV2MathWorkflow",
        ("voterA", "voterB", "voterC", "aggregator"), (False, True),
        (("n_votes", 3), ("voter_names", None), UFOR, ("use_rubric_reward", False)), (20480, 5120), (25600, 8192),
        (("n_votes", 3), ("use_rubric_reward", True), ("use_final_outcome_reward", False)), (),
        ("voterA", "voterB", "voterC"),
    ),
    ("math", "controlled_sa_hga"): (
        "controlled_sa_hga", None, "marl_workflows.workflows.math.controlled_sa_hga.ControlledSAHGAMathWorkflow",
        ("generator",), (False,), (("n_virtual_generators", 3),), (15360, 5120), (15360, 8192), (), (),
        ("generator",),
    ),
    ("deepcoder", "single_agent"): (
        "single_agent", None, "marl_workflows.workflows.code.single_agent.SingleAgentDeepcodeWorkflow",
        ("generator",), (False,), (), (4096, 2048), (4096, 5120), (), (), ("generator",),
    ),
    ("deepcoder", "voting"): (
        "voting", None, "marl_workflows.workflows.code.voting.DeepcodeVotingWorkflow",
        ("generator", "aggregator"), (False, True), CODE_VOTING, (10240, 2048), (20480, 5120),
        CODE_VOTING, (), ("generator",),
    ),
    ("deepcoder", "evaluator_optimizer"): (
        "evaluator_optimizer", None,
        "marl_workflows.workflows.code.evaluator_optimizer.DeepcodeEvaluatorOptimizerWorkflow",
        ("generator", "evaluator"), (False, True), (("max_iterations", 2), UFOR), (10240, 2048), (15360, 5120),
        (("max_iterations", 2), UFOR), (), ("generator",),
    ),
    ("deepcoder", "orchestrator_workers_propose"): (
        "orchestrator_workers_propose", None,
        "marl_workflows.workflows.code.orchestrator_workers.DeepcodeOrchestratorWorkersWorkflow",
        ("orchestrator", "worker", "synthesizer"), (False, True), (("max_subtasks", 3), UFOR), (10240, 2048),
        (20480, 5120), (("max_subtasks", 3), UFOR), (("generator", "worker"),), ("worker",),
    ),
    ("deepcoder", "voting_v2"): (
        "voting_v2", None, "marl_workflows.workflows.code.voting_v2.DeepcodeVotingV2Workflow",
        ("voterA", "voterB", "voterC", "aggregator"), (False, True),
        (("n_votes", 3), ("voter_names", None), UFOR, ("use_rubric_reward", False)), (10240, 2048), (20480, 5120),
        (("n_votes", 3), ("use_rubric_reward", True), ("use_final_outcome_reward", False)), (),
        ("voterA", "voterB", "voterC"),
    ),
}  # fmt: skip


def test_recipe_keys_match_table() -> None:
    assert set(RECIPES) == set(EXPECTED)
    for (task, wf), r in RECIPES.items():
        assert (r.task, r.workflow) == (task, wf)


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_recipe_values(key: tuple[str, str]) -> None:
    r = RECIPES[key]
    prefix, suffix, cls, names, sp, params, train, evl, eval_params, remap, gen_roles = EXPECTED[key]
    assert r.experiment_prefix == prefix
    assert r.default_name_suffix == suffix
    assert r.workflow_cls == cls
    assert r.agent_names == names
    assert r.allowed_share_policy == sp
    assert r.params == params
    assert r.train_lengths == train
    assert r.eval_lengths == evl
    assert r.eval_params == eval_params
    assert r.initial_lora_remap == remap
    assert r.generator_roles == gen_roles
    if r.task == "math":
        assert (r.train_dataset, r.val_dataset, r.reward_fn, r.uses_code_executor) == (
            "dapo_math",
            "aime2025",
            MATH_REWARD,
            False,
        )
    else:
        assert (r.train_dataset, r.val_dataset, r.reward_fn, r.uses_code_executor) == (
            "deepcoder_primeintellect",
            "deepcoder_primeintellect",
            None,
            True,
        )


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_every_recipe_validates(key: tuple[str, str]) -> None:
    r = RECIPES[key]
    validate_recipe(r)
    for sp in r.allowed_share_policy:
        contracts.validate_agent_names(r.agent_names, sp)
    # The paper's experiment names come out of the recipe exactly as the fork launcher built them.
    name = contracts.experiment_name(r.experiment_prefix, "qwen3_1.7b", False, r.task, r.default_name_suffix)
    assert name.startswith(f"{r.experiment_prefix}-qwen3_1.7b-multi_lora-{contracts.TASK_SUFFIX[r.task]}")


def test_dotted_paths_are_syntactically_valid() -> None:
    import re

    dotted = re.compile(r"^marl_workflows(\.[A-Za-z_][A-Za-z0-9_]*)+\.[A-Z][A-Za-z0-9_]*$")
    for r in RECIPES.values():
        assert dotted.match(r.workflow_cls), r.workflow_cls
        module = r.workflow_cls.rsplit(".", 1)[0]
        expected_pkg = "marl_workflows.workflows.math." if r.task == "math" else "marl_workflows.workflows.code."
        assert module.startswith(expected_pkg)
        if r.reward_fn is not None:
            assert re.match(r"^marl_workflows(\.[a-z_]+)+$", r.reward_fn)


def test_invalid_recipe_rejected() -> None:
    base = RECIPES[("math", "voting")]
    kwargs = {f: getattr(base, f) for f in base.__dataclass_fields__}
    with pytest.raises(ValueError):
        Recipe(**{**kwargs, "agent_names": ("generator", "default")})
    with pytest.raises(ValueError):
        Recipe(**{**kwargs, "generator_roles": ("voterA",)})
    with pytest.raises(ValueError):
        Recipe(**{**kwargs, "workflow_cls": "not a path"})
    with pytest.raises(ValueError):
        Recipe(**{**kwargs, "task": "code"})


def test_get_recipe_error_lists_pairs() -> None:
    assert get_recipe("math", "voting") is RECIPES[("math", "voting")]
    with pytest.raises(KeyError, match="math/voting"):
        get_recipe("math", "nope")


# ----------------------------------------------------------------------------- experiment names

# Names that do not follow `{workflow}-{model_tag}-...`, with the recipe they evaluate as.
_SPECIAL = {
    "qwen3_1.7b-math_single_agent-length5120": "single_agent",
    "controlled_sa_hga-qwen3_1.7b-multi_lora-math": "single_agent",
}


def _fixture_names() -> list[str]:
    lines = (FIXTURES / "experiment_names.txt").read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def test_fixture_is_populated() -> None:
    assert len(_fixture_names()) >= 61


@pytest.mark.parametrize("name", _fixture_names())
def test_recipe_for_experiment_maps_fork_names(name: str) -> None:
    task = infer_task(name)
    recipe = recipe_for_experiment(name, task)
    assert recipe is not None, name
    expected_wf = _SPECIAL.get(name, name.split("-", 1)[0])
    assert (recipe.task, recipe.workflow) == (task, expected_wf)


def test_recipe_for_experiment_rules() -> None:
    assert recipe_for_experiment("voting-qwen3_1.7b-multi_lora-math-prompt_het", "math").workflow == "voting"
    assert recipe_for_experiment("voting_v2-qwen3_1.7b-multi_lora-math", "math").workflow == "voting_v2"
    legacy = recipe_for_experiment("orchestrator_workers-qwen3_0.6b-math", "math")
    assert legacy is not None and legacy.workflow == "orchestrator_workers_propose"
    assert recipe_for_experiment("voting-qwen3_8b-multi_lora-math", "math") is None
    assert recipe_for_experiment("voting-llama-multi_lora-math", "math") is None
    assert recipe_for_experiment("debug_codeexec", "deepcoder") is None
    with pytest.warns(UserWarning, match="evaluator_optimizer_v2"):
        assert recipe_for_experiment("evaluator_optimizer_v2-qwen3_1.7b-multi_lora-math", "math") is None


def test_infer_task() -> None:
    assert infer_task("voting-qwen3_1.7b-multi_lora-deepcoder_primeintellect") == "deepcoder"
    assert infer_task("voting-qwen3_1.7b-multi_lora-math") == "math"


# ----------------------------------------------------------------------------- runtime helpers


def test_build_workflow_args_defaults_and_overrides() -> None:
    code_voting = RECIPES[("deepcoder", "voting")]
    assert build_workflow_args(code_voting, None) == {
        "n_votes": 3,
        "use_final_outcome_reward": True,
        "separate_vote_groups": False,
    }
    cfg = {"rllm": {"workflow": {"n_votes": 5, "use_final_outcome_reward": False, "unrelated": 1}}}
    assert build_workflow_args(code_voting, cfg) == {
        "n_votes": 5,
        "use_final_outcome_reward": False,
        "separate_vote_groups": False,
    }
    cfg = {"rllm": {"workflow": {"separate_vote_groups": True}}}
    assert build_workflow_args(code_voting, cfg)["separate_vote_groups"] is True
    math_voting = RECIPES[("math", "voting")]
    assert build_workflow_args(math_voting, None)["separate_vote_groups"] is True


def test_build_workflow_args_with_omegaconf() -> None:
    omegaconf = pytest.importorskip("omegaconf")
    cfg = omegaconf.OmegaConf.create(
        {"rllm": {"workflow": {"voter_names": ["a", "b", "c"], "use_rubric_reward": True}}}
    )
    args = build_workflow_args(RECIPES[("deepcoder", "voting_v2")], cfg)
    assert args == {
        "n_votes": 3,
        "voter_names": ["a", "b", "c"],
        "use_final_outcome_reward": True,
        "use_rubric_reward": True,
    }
    assert type(args["voter_names"]) is list


def test_load_object() -> None:
    assert load_object("marl_workflows.contracts.role_from_name") is contracts.role_from_name
    with pytest.raises(ImportError):
        load_object("marl_workflows.contracts.does_not_exist")
    with pytest.raises(ValueError):
        load_object("nodots")


def test_light_imports() -> None:
    code = (
        "import sys\n"
        "import marl_workflows, marl_workflows.contracts, marl_workflows.recipes, marl_workflows.utils\n"
        "heavy = sorted(m for m in ('torch', 'rllm', 'verl', 'vllm', 'ray', 'omegaconf') if m in sys.modules)\n"
        "assert not heavy, heavy\n"
    )
    repo = Path(__file__).resolve().parent.parent
    subprocess.run([sys.executable, "-c", code], check=True, cwd=repo)
