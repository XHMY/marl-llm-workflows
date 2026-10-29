"""The (task, workflow) recipe table: agent names, lengths, workflow kwargs and class paths.

Pure data plus lazily resolved dotted paths; importing this module imports neither torch
nor rllm nor verl. Every value is taken from the research fork (`dashboard/task_configs.py`,
`dashboard/launch_experiment.sh` and the `examples/*/train_*.py` entry scripts). Two
defaults deliberately differ from the fork's entry scripts: the orchestrator's
``max_subtasks`` default is 3 (the value every paper run passed; the script default was 4),
and ``initial_lora_weights`` is honoured for every recipe by the trainer, so it is not a
workflow kwarg here.
"""

from __future__ import annotations

import importlib
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from marl_workflows.contracts import TASKS, validate_agent_names

__all__ = [
    "RECIPES",
    "Recipe",
    "build_workflow_args",
    "get_recipe",
    "infer_task",
    "load_object",
    "recipe_for_experiment",
    "validate_recipe",
]

_DOTTED_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+$")


@dataclass(frozen=True)
class Recipe:
    """One trainable (task, workflow) combination."""

    task: str  # "math" | "deepcoder"
    workflow: str  # recipe id (CLI --workflow / marl.workflow)
    experiment_prefix: str  # first token of the experiment name
    default_name_suffix: str | None
    workflow_cls: str  # dotted path, resolved lazily
    agent_names: tuple[str, ...]
    allowed_share_policy: tuple[bool, ...]
    train_dataset: str  # registry names; splits "train" / "test"
    val_dataset: str
    reward_fn: str | None  # dotted path passed as workflow_args["reward_function"]
    params: tuple[tuple[str, Any], ...]  # (name, default) read as config.rllm.workflow.get(name, default)
    train_lengths: tuple[int, int]  # (max_prompt_length, max_response_length)
    eval_lengths: tuple[int, int]
    eval_params: tuple[tuple[str, Any], ...]  # workflow kwargs used at evaluation
    initial_lora_remap: tuple[tuple[str, str], ...] = ()
    uses_code_executor: bool = False
    generator_roles: tuple[str, ...] = ("generator",)  # single_agent_generator_transfer eval mode

    def __post_init__(self) -> None:
        validate_recipe(self)


def validate_recipe(recipe: Recipe) -> None:
    """Raise ``ValueError`` if a recipe is internally inconsistent."""
    where = f"recipe ({recipe.task}, {recipe.workflow})"
    if recipe.task not in TASKS:
        raise ValueError(f"{where}: unknown task {recipe.task!r}")
    if not recipe.allowed_share_policy:
        raise ValueError(f"{where}: allowed_share_policy is empty")
    for sp in recipe.allowed_share_policy:
        validate_agent_names(recipe.agent_names, sp)
    for path in (recipe.workflow_cls, recipe.reward_fn):
        if path is not None and not _DOTTED_RE.match(path):
            raise ValueError(f"{where}: {path!r} is not a dotted path")
    for lengths in (recipe.train_lengths, recipe.eval_lengths):
        if len(lengths) != 2 or not all(isinstance(x, int) and x > 0 for x in lengths):
            raise ValueError(f"{where}: lengths must be two positive ints, got {lengths}")
    for table in (recipe.params, recipe.eval_params):
        keys = [k for k, _ in table]
        if len(set(keys)) != len(keys):
            raise ValueError(f"{where}: duplicate workflow kwarg in {keys}")
        if "reward_function" in keys:
            raise ValueError(f"{where}: reward_function is set from reward_fn, not params")
    missing = [r for r in recipe.generator_roles if r not in recipe.agent_names]
    if missing:
        raise ValueError(f"{where}: generator_roles {missing} not in agent_names")
    for _src, dst in recipe.initial_lora_remap:
        if dst not in recipe.agent_names:
            raise ValueError(f"{where}: initial_lora_remap target {dst!r} not in agent_names")


# --------------------------------------------------------------------------------------
# Table
# --------------------------------------------------------------------------------------

_MATH = "marl_workflows.workflows.math"
_CODE = "marl_workflows.workflows.code"
_MATH_REWARD = "marl_workflows.rewards.math_reward.math_reward_fn"
_IP_SP = (False, True)
_IP_ONLY = (False,)

_UFOR = ("use_final_outcome_reward", True)
# separate_vote_groups: voter trajectory names generator0/1/2 (one GRPO group per voter, the
# paper's Math runs) or generator x3 (one pooled group, the paper's Code runs).
_MATH_VOTING_PARAMS = (("n_votes", 3), _UFOR, ("separate_vote_groups", True))
_CODE_VOTING_PARAMS = (("n_votes", 3), _UFOR, ("separate_vote_groups", False))
# Training defaults of the voting_v2 entry scripts; the yaml declares use_final_outcome_reward=True.
_VOTING_V2_PARAMS = (("n_votes", 3), ("voter_names", None), _UFOR, ("use_rubric_reward", False))
# dashboard/task_configs.py workflow_params, which the evaluator passes to voting_v2.
_VOTING_V2_EVAL = (("n_votes", 3), ("use_rubric_reward", True), ("use_final_outcome_reward", False))
_VOTING_V2_NAMES = ("voterA", "voterB", "voterC", "aggregator")
_ORCH_NAMES = ("orchestrator", "worker", "synthesizer")
# SA-RL adapters are saved under "generator"; the orchestrator workflow trains them as "worker".
_ORCH_REMAP = (("generator", "worker"),)


def _math(workflow: str, **kw: Any) -> Recipe:
    return Recipe(
        task="math",
        workflow=workflow,
        train_dataset="dapo_math",
        val_dataset="aime2025",
        reward_fn=_MATH_REWARD,
        **kw,
    )


def _code(workflow: str, **kw: Any) -> Recipe:
    return Recipe(
        task="deepcoder",
        workflow=workflow,
        train_dataset="deepcoder_primeintellect",
        val_dataset="deepcoder_primeintellect",
        # The code workflows call code_reward_fn themselves through the code executor.
        reward_fn=None,
        uses_code_executor=True,
        **kw,
    )


_RECIPE_LIST: tuple[Recipe, ...] = (
    # ---------------------------------------------------------------- math
    _math(
        "single_agent",
        experiment_prefix="single_agent",
        default_name_suffix=None,
        workflow_cls=f"{_MATH}.single_agent.SingleAgentMathWorkflow",
        agent_names=("generator",),
        allowed_share_policy=_IP_ONLY,
        params=(),
        train_lengths=(15360, 5120),
        eval_lengths=(15360, 8192),
        eval_params=(),
    ),
    _math(
        "voting",
        experiment_prefix="voting",
        default_name_suffix=None,
        workflow_cls=f"{_MATH}.voting.VotingMathWorkflow",
        agent_names=("generator", "aggregator"),
        allowed_share_policy=_IP_SP,
        params=_MATH_VOTING_PARAMS,
        train_lengths=(20480, 5120),
        eval_lengths=(25600, 8192),
        eval_params=_MATH_VOTING_PARAMS,
    ),
    _math(
        "voting_prompt_het",
        experiment_prefix="voting",
        default_name_suffix="prompt_het",
        workflow_cls=f"{_MATH}.voting_prompt_het.VotingMathPromptHetWorkflow",
        agent_names=("generator", "aggregator"),
        allowed_share_policy=_IP_SP,
        params=_MATH_VOTING_PARAMS,
        train_lengths=(20480, 5120),
        eval_lengths=(25600, 8192),
        eval_params=_MATH_VOTING_PARAMS,
    ),
    _math(
        "evaluator_optimizer",
        experiment_prefix="evaluator_optimizer",
        default_name_suffix=None,
        workflow_cls=f"{_MATH}.evaluator_optimizer.EvaluatorOptimizerMathWorkflow",
        agent_names=("generator", "evaluator"),
        allowed_share_policy=_IP_SP,
        params=(("max_iterations", 3), _UFOR),
        train_lengths=(30720, 5120),
        eval_lengths=(30720, 8192),
        eval_params=(("max_iterations", 3), _UFOR),
    ),
    _math(
        "orchestrator_workers_propose",
        experiment_prefix="orchestrator_workers_propose",
        default_name_suffix=None,
        workflow_cls=f"{_MATH}.orchestrator_workers.OrchestratorWorkersMathWorkflow",
        agent_names=_ORCH_NAMES,
        allowed_share_policy=_IP_SP,
        params=(("max_subtasks", 3), _UFOR),
        train_lengths=(20480, 5120),
        eval_lengths=(25600, 8192),
        eval_params=(("max_subtasks", 3), _UFOR),
        initial_lora_remap=_ORCH_REMAP,
        generator_roles=("worker",),
    ),
    _math(
        "voting_v2",
        experiment_prefix="voting_v2",
        default_name_suffix=None,
        workflow_cls=f"{_MATH}.voting_v2.VotingV2MathWorkflow",
        agent_names=_VOTING_V2_NAMES,
        allowed_share_policy=_IP_SP,
        params=_VOTING_V2_PARAMS,
        train_lengths=(20480, 5120),
        eval_lengths=(25600, 8192),
        eval_params=_VOTING_V2_EVAL,
        generator_roles=("voterA", "voterB", "voterC"),
    ),
    _math(
        "controlled_sa_hga",
        experiment_prefix="controlled_sa_hga",
        default_name_suffix=None,
        workflow_cls=f"{_MATH}.controlled_sa_hga.ControlledSAHGAMathWorkflow",
        agent_names=("generator",),
        allowed_share_policy=_IP_ONLY,
        params=(("n_virtual_generators", 3),),
        train_lengths=(15360, 5120),
        # Evaluated as single_agent (recipe_for_experiment maps the name there).
        eval_lengths=(15360, 8192),
        eval_params=(),
    ),
    # ---------------------------------------------------------------- deepcoder
    _code(
        "single_agent",
        experiment_prefix="single_agent",
        default_name_suffix=None,
        workflow_cls=f"{_CODE}.single_agent.SingleAgentDeepcodeWorkflow",
        agent_names=("generator",),
        allowed_share_policy=_IP_ONLY,
        params=(),
        train_lengths=(4096, 2048),
        eval_lengths=(4096, 5120),
        eval_params=(),
    ),
    _code(
        "voting",
        experiment_prefix="voting",
        default_name_suffix=None,
        workflow_cls=f"{_CODE}.voting.DeepcodeVotingWorkflow",
        agent_names=("generator", "aggregator"),
        allowed_share_policy=_IP_SP,
        params=_CODE_VOTING_PARAMS,
        train_lengths=(10240, 2048),
        eval_lengths=(20480, 5120),
        eval_params=_CODE_VOTING_PARAMS,
    ),
    _code(
        "evaluator_optimizer",
        experiment_prefix="evaluator_optimizer",
        default_name_suffix=None,
        workflow_cls=f"{_CODE}.evaluator_optimizer.DeepcodeEvaluatorOptimizerWorkflow",
        agent_names=("generator", "evaluator"),
        allowed_share_policy=_IP_SP,
        params=(("max_iterations", 2), _UFOR),
        train_lengths=(10240, 2048),
        eval_lengths=(15360, 5120),
        eval_params=(("max_iterations", 2), _UFOR),
    ),
    _code(
        "orchestrator_workers_propose",
        experiment_prefix="orchestrator_workers_propose",
        default_name_suffix=None,
        workflow_cls=f"{_CODE}.orchestrator_workers.DeepcodeOrchestratorWorkersWorkflow",
        agent_names=_ORCH_NAMES,
        allowed_share_policy=_IP_SP,
        params=(("max_subtasks", 3), _UFOR),
        train_lengths=(10240, 2048),
        eval_lengths=(20480, 5120),
        eval_params=(("max_subtasks", 3), _UFOR),
        initial_lora_remap=_ORCH_REMAP,
        generator_roles=("worker",),
    ),
    _code(
        "voting_v2",
        experiment_prefix="voting_v2",
        default_name_suffix=None,
        workflow_cls=f"{_CODE}.voting_v2.DeepcodeVotingV2Workflow",
        agent_names=_VOTING_V2_NAMES,
        allowed_share_policy=_IP_SP,
        params=_VOTING_V2_PARAMS,
        train_lengths=(10240, 2048),
        eval_lengths=(20480, 5120),
        eval_params=_VOTING_V2_EVAL,
        generator_roles=("voterA", "voterB", "voterC"),
    ),
)

RECIPES: dict[tuple[str, str], Recipe] = {(r.task, r.workflow): r for r in _RECIPE_LIST}


def get_recipe(task: str, workflow: str) -> Recipe:
    """Look up a recipe; the ``KeyError`` lists every valid pair."""
    try:
        return RECIPES[(task, workflow)]
    except KeyError:
        valid = ", ".join(f"{t}/{w}" for t, w in RECIPES)
        raise KeyError(f"No recipe for task={task!r}, workflow={workflow!r}. Valid: {valid}") from None


# --------------------------------------------------------------------------------------
# Experiment-name parsing (fork dashboard/evaluate_checkpoints.py parse_experiment_name)
# --------------------------------------------------------------------------------------

# Substring order matters: voting_v2 must be tried before voting.
_EXPERIMENT_WORKFLOW_ORDER = (
    "single_agent",
    "voting_v2",
    "voting",
    "evaluator_optimizer",
    "orchestrator_workers_propose",
)
_MODEL_SIZE_RE = re.compile(r"qwen3_(\d+\.?\d*b)", re.IGNORECASE)
_KNOWN_SIZES = ("0.6b", "1.7b", "4b")


def infer_task(experiment_name: str) -> str:
    """``deepcoder`` if the name mentions it, else ``math`` (fork ``infer_task_type``)."""
    return "deepcoder" if "deepcoder" in experiment_name.lower() else "math"


def _workflow_from_experiment(experiment_name: str) -> str | None:
    for wf in _EXPERIMENT_WORKFLOW_ORDER:
        if wf in experiment_name:
            return wf
    # Legacy runs were named "orchestrator_workers" before the propose variant became the only one.
    if "orchestrator_workers" in experiment_name:
        return "orchestrator_workers_propose"
    # The single-role control trains one "generator" adapter and is evaluated as single_agent.
    if "controlled_sa_hga" in experiment_name:
        return "single_agent"
    return None


def recipe_for_experiment(experiment_name: str, task: str) -> Recipe | None:
    """Return the recipe used to evaluate a checkpoint directory name, or ``None``.

    Same matching as the fork's ``parse_experiment_name``: the first workflow id contained
    in the name wins, legacy ``orchestrator_workers`` maps to the propose variant,
    ``controlled_sa_hga`` maps to ``single_agent``, and a name without a known
    ``qwen3_<size>`` token is not an experiment. ``evaluator_optimizer_v2`` runs are out of
    scope and are skipped with a warning.
    """
    if "evaluator_optimizer_v2" in experiment_name:
        warnings.warn(f"Skipping {experiment_name!r}: evaluator_optimizer_v2 is not supported", stacklevel=2)
        return None
    workflow = _workflow_from_experiment(experiment_name)
    if workflow is None:
        return None
    m = _MODEL_SIZE_RE.search(experiment_name)
    if m is None or m.group(1).lower() not in _KNOWN_SIZES:
        return None
    return RECIPES.get((task, workflow))


# --------------------------------------------------------------------------------------
# Runtime helpers
# --------------------------------------------------------------------------------------


def load_object(dotted: str) -> Any:
    """Import ``package.module.Name`` and return ``Name``."""
    if not _DOTTED_RE.match(dotted):
        raise ValueError(f"{dotted!r} is not a dotted path")
    module_name, attr = dotted.rsplit(".", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError:
        raise ImportError(f"module {module_name!r} has no attribute {attr!r}") from None


def _lookup(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Mapping) or hasattr(obj, "get"):
        return obj.get(key)
    return getattr(obj, key, None)


def _to_plain(value: Any) -> Any:
    # Workflows expect plain lists/dicts (the fork's entry scripts called list()/dict() on them);
    # omegaconf is imported lazily so that this module stays light.
    if type(value).__module__.startswith("omegaconf"):
        from omegaconf import OmegaConf

        return OmegaConf.to_container(value, resolve=True)
    return value


def build_workflow_args(recipe: Recipe, config: Any) -> dict[str, Any]:
    """Workflow kwargs for training: each param read as ``config.rllm.workflow.get(name, default)``.

    ``reward_function`` is added (resolved) when the recipe names one. ``config`` may be an
    omegaconf ``DictConfig``, a plain dict or ``None``.
    """
    workflow_cfg = _lookup(_lookup(config, "rllm"), "workflow")
    args: dict[str, Any] = {}
    for name, default in recipe.params:
        value = _lookup(workflow_cfg, name)
        args[name] = default if value is None else _to_plain(value)
    if recipe.reward_fn is not None:
        args["reward_function"] = load_object(recipe.reward_fn)
    return args
