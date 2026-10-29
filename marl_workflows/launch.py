"""Launcher: map a cell (task, workflow, policy, scale) plus an optional control to a training command.

The pure part (:func:`build_overrides`, :func:`build_command`, :func:`render_sbatch`) only
builds strings and never imports torch, rllm or verl. :func:`cli` adds printing, SLURM
submission and segment chains. The generated command runs the Hydra entry point
``python -m marl_workflows.trainer.train`` with the overrides listed in
``docs/reproducing.md``.

Override semantics: overrides are merged by key in order, so a later override of the same
key (a control, a paper-run entry, ``--extra``) replaces the earlier value in place and the
key keeps its first position. A leading ``+`` from the later override wins; otherwise the
earlier prefix is kept (keys absent from the Hydra yaml need ``+``).
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shlex
import string
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from marl_workflows.contracts import POLICY_TAG, TASK_SUFFIX, TASKS, experiment_name
from marl_workflows.recipes import Recipe, get_recipe
from marl_workflows.utils import init_weight_dir, repo_root

__all__ = [
    "CONTROLS",
    "DEFAULT_PROJECT",
    "ENTRY_MODULE",
    "PAPER_PROJECT",
    "Control",
    "LaunchSpec",
    "apply_slurm_meta",
    "build_command",
    "build_overrides",
    "check_gpu_policy",
    "cli",
    "load_paper_runs",
    "merge_overrides",
    "normalize_gpu_type",
    "normalize_scale",
    "parse_override",
    "read_slurm_conf",
    "render_eval_sbatch",
    "render_sbatch",
    "resolve_resources",
    "spec_from_paper_run",
]

ENTRY_MODULE = "marl_workflows.trainer.train"
EVAL_MODULE = "marl_workflows.eval.evaluate_checkpoints"
DEFAULT_PROJECT = "marl-llm-workflows"
# wandb project of the paper runs; --paper-run reuses it so that resumed runs and eval discovery line up.
PAPER_PROJECT = "rllm-workflow-MARL-v2"

# --------------------------------------------------------------------------------------
# Scales, GPUs, resources
# --------------------------------------------------------------------------------------

# --scale -> (Hugging Face model id, model tag used in experiment names)
SCALES: dict[str, tuple[str, str]] = {
    "0.6B": ("Qwen/Qwen3-0.6B", "qwen3_0.6b"),
    "1.7B": ("Qwen/Qwen3-1.7B", "qwen3_1.7b"),
    "4B": ("Qwen/Qwen3-4B", "qwen3_4b"),
}

GPU_TYPES = ("H100", "H200", "L40S", "A40", "RTX8000")
# 80 GB-class GPUs use the "large" token budget; 48 GB-class GPUs the "small" one.
_GPU_CLASS = {"H100": "large", "H200": "large", "L40S": "small", "A40": "small", "RTX8000": "small"}

# actor_rollout_ref.actor.ppo_max_token_len_per_gpu by (scale, GPU class); fork dashboard/launch_experiment.sh.
PPO_MAX_TOKEN_LEN: dict[tuple[str, str], int] = {
    ("0.6B", "small"): 28672,
    ("0.6B", "large"): 51712,
    ("1.7B", "small"): 23554,
    ("1.7B", "large"): 40960,
    ("4B", "small"): 10240,
    ("4B", "large"): 40960,
}

# Per-GPU CPU and host-memory defaults when the SLURM conf gives none. Code needs many CPUs
# for the code-execution pool (code_executor_workers = n_gpus * cpus_per_gpu; a paper run that pins
# the pool size gets cpus_per_gpu = ceil(code_executor_workers / n_gpus) instead).
_RESOURCE_DEFAULTS: dict[tuple[str, str], tuple[int, str]] = {
    ("math", "large"): (8, "64G"),
    ("math", "small"): (8, "48G"),
    ("deepcoder", "large"): (16, "80G"),
    ("deepcoder", "small"): (16, "64G"),
}

# GPUs allowed for training per scale: 4B needs 80 GB GPUs (it runs out of memory in the actor
# update on 48 GB cards); A40 and RTX8000 are not used for training.
_ALLOWED_TRAIN_GPUS: dict[str, tuple[str, ...]] = {
    "0.6B": ("H100", "H200", "L40S"),
    "1.7B": ("H100", "H200", "L40S"),
    "4B": ("H100", "H200"),
}


def normalize_scale(scale: str) -> str:
    """Accept ``0.6B``, ``0.6b``, ``qwen3_0.6b`` (and the ``_sNNN`` tags) and return ``0.6B``."""
    s = scale.strip().lower().removeprefix("qwen3_")
    s = re.sub(r"_s\d+$", "", s)
    for key in SCALES:
        if s == key.lower():
            return key
    raise ValueError(f"Unknown scale {scale!r}; expected one of {list(SCALES)}")


def normalize_gpu_type(gpu_type: str) -> str:
    """Case-insensitive GPU type (``L40s`` -> ``L40S``)."""
    g = gpu_type.strip().upper()
    if g not in GPU_TYPES:
        raise ValueError(f"Unknown GPU type {gpu_type!r}; expected one of {list(GPU_TYPES)}")
    return g


def normalize_task(task: str) -> str:
    """``code`` is an alias of ``deepcoder``."""
    t = "deepcoder" if task == "code" else task
    if t not in TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {list(TASKS)} (or 'code')")
    return t


def check_gpu_policy(scale: str, gpu_type: str) -> None:
    """Raise ``ValueError`` if ``gpu_type`` is not used for training at ``scale``."""
    scale, gpu_type = normalize_scale(scale), normalize_gpu_type(gpu_type)
    allowed = _ALLOWED_TRAIN_GPUS[scale]
    if gpu_type not in allowed:
        raise ValueError(
            f"{scale} training is not run on {gpu_type}; use one of {list(allowed)} (or pass --skip-gpu-check)"
        )


# --------------------------------------------------------------------------------------
# Workflow parameters per recipe (the values every paper run passed)
# --------------------------------------------------------------------------------------

_UFOR_TRUE = "rllm.workflow.use_final_outcome_reward=true"
_VOTING = ("+rllm.workflow.n_votes=3", _UFOR_TRUE)
WORKFLOW_PARAMS: dict[tuple[str, str], tuple[str, ...]] = {
    ("math", "single_agent"): (),
    ("math", "voting"): _VOTING,
    ("math", "voting_prompt_het"): _VOTING,
    ("math", "evaluator_optimizer"): ("+rllm.workflow.max_iterations=3", _UFOR_TRUE),
    ("math", "orchestrator_workers_propose"): ("+rllm.workflow.max_subtasks=3", _UFOR_TRUE),
    ("math", "voting_v2"): _VOTING,
    ("math", "controlled_sa_hga"): ("+rllm.workflow.n_virtual_generators=3",),
    ("deepcoder", "single_agent"): (),
    ("deepcoder", "voting"): _VOTING,
    ("deepcoder", "evaluator_optimizer"): ("+rllm.workflow.max_iterations=2", _UFOR_TRUE),
    ("deepcoder", "orchestrator_workers_propose"): ("+rllm.workflow.max_subtasks=3", _UFOR_TRUE),
    ("deepcoder", "voting_v2"): _VOTING,
}

# --------------------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------------------

_LOGPROBS = "+actor_rollout_ref.rollout.engine_kwargs.vllm.logprobs_mode=processed_logprobs"
_WALLCLOCK = "+trainer.wallclock_stop.enable=true"


@dataclass(frozen=True)
class Control:
    """A named variation applied on top of a base cell."""

    name: str
    description: str
    overrides: tuple[str, ...] = ()
    task: str | None = None  # required task (None: any)
    workflow: str | None = None  # required --workflow (None: any)
    share_policy: bool | None = None  # required policy (None: any)
    name_suffix: str | None = None
    # Full experiment-name template; fields: prefix, model_tag, policy, task_suffix.
    name_template: str | None = None
    base_model_tag: str | None = None  # e.g. "qwen3_1.7b_s300" (scale-specific tags use "{size}")
    needs_resume_from: bool = False
    needs_init_adapters: bool = False
    n_gpus: int | None = None
    example: str = ""  # base cell used in the docs and golden tests: "task workflow policy scale"


CONTROLS: dict[str, Control] = {
    c.name: c
    for c in (
        Control(
            "per_voter_reward",
            "Each voter is rewarded for its own answer instead of the episode outcome.",
            ("rllm.workflow.use_final_outcome_reward=false",),
            task="math",
            workflow="voting",
            share_policy=False,
            name_suffix="per_agent_reward",
            example="math voting ip 1.7B",
        ),
        Control(
            "evaluator_accuracy_reward",
            "Each role is rewarded for its own correctness instead of the episode outcome.",
            ("rllm.workflow.use_final_outcome_reward=false",),
            task="math",
            workflow="evaluator_optimizer",
            share_policy=False,
            name_suffix="per_agent_reward",
            example="math evaluator_optimizer ip 1.7B",
        ),
        Control(
            "prompt_het",
            "Voting with a different prompt per voter (workflow voting_prompt_het).",
            ("actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960",),
            task="math",
            workflow="voting_prompt_het",
            share_policy=False,
            example="math voting_prompt_het ip 1.7B",
        ),
        Control(
            "single_role",
            "Three independent generator samples per problem on one adapter, no aggregator.",
            ("+rllm.workflow.n_virtual_generators=3",),
            task="math",
            workflow="controlled_sa_hga",
            share_policy=False,
            example="math controlled_sa_hga ip 1.7B",
        ),
        Control(
            "voter_adapters_rubric",
            "One adapter per voter (workflow voting_v2), each voter rewarded by a rubric.",
            (
                "+rllm.workflow.n_votes=3",
                "+rllm.workflow.use_rubric_reward=true",
                "rllm.workflow.use_final_outcome_reward=false",
            ),
            task="math",
            workflow="voting_v2",
            share_policy=False,
            example="math voting_v2 ip 1.7B",
        ),
        Control(
            "voter_adapters_broadcast_sainit",
            "One adapter per voter with the outcome reward, from the SA-RL step-300 merged weights.",
            ("+rllm.workflow.n_votes=3", _UFOR_TRUE),
            task="math",
            workflow="voting_v2",
            share_policy=False,
            base_model_tag="qwen3_{size}_s300",
            example="math voting_v2 ip 1.7B",
        ),
        Control(
            "t1_length_weight",
            "Shared policy with each sequence's loss scaled by its role's mean response length.",
            (
                "+trainer.loss_weight_per_role.mode=length",
                "actor_rollout_ref.actor.optim.lr_warmup_steps=0",
                _LOGPROBS,
                _WALLCLOCK,
                "trainer.total_training_steps=300",
            ),
            task="math",
            workflow="voting",
            share_policy=True,
            name_suffix="role_len_weight",
            example="math voting sp 1.7B",
        ),
        Control(
            "t2_merged_minibatch",
            "Per-role adapters with one optimizer step per role per training step (mini-batch 192).",
            (
                "actor_rollout_ref.actor.ppo_mini_batch_size=192",
                "actor_rollout_ref.actor.optim.lr_warmup_steps=0",
                _LOGPROBS,
                _WALLCLOCK,
                "trainer.total_training_steps=500",
            ),
            task="math",
            workflow="voting",
            share_policy=False,
            name_suffix="merged_minibatch",
            example="math voting ip 1.7B",
        ),
        Control(
            "kl_anchor_evaluator",
            "KL penalty to the base model on the evaluator adapter only (coefficient 0.01).",
            (
                "actor_rollout_ref.actor.use_kl_loss=true",
                "actor_rollout_ref.actor.kl_loss_coef=0.0",
                "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
                "+trainer.kl_loss_coef_per_role.evaluator=0.01",
                "+trainer.kl_loss_coef_per_role.generator=0.0",
            ),
            task="math",
            workflow="evaluator_optimizer",
            share_policy=False,
            name_suffix="i1_kl_evaluator_1",
            example="math evaluator_optimizer ip 1.7B",
        ),
        Control(
            "cap_8192",
            "Continue an Eval-Opt run from a checkpoint with an 8192-token response cap.",
            (
                "data.max_response_length=8192",
                "trainer.resume_mode=resume_path",
                "trainer.total_training_steps=400",
            ),
            task="math",
            workflow="evaluator_optimizer",
            share_policy=False,
            name_template="{prefix}-{model_tag}-{policy}-l8192-{task_suffix}",
            needs_resume_from=True,
            n_gpus=4,
            example="math evaluator_optimizer ip 1.7B",
        ),
        Control(
            "sp_extension",
            "Extend a shared-policy run to 450 steps (same experiment name, so it resumes).",
            ("trainer.total_training_steps=450", _WALLCLOCK),
            share_policy=True,
            example="math voting sp 1.7B",
        ),
        Control(
            "init_from_sa_adapters",
            "Initialize role adapters from given LoRA adapters (pass --init-adapter role=path).",
            needs_init_adapters=True,
            share_policy=False,
            example="math voting ip 1.7B",
        ),
    )
}

# --------------------------------------------------------------------------------------
# Spec
# --------------------------------------------------------------------------------------


@dataclass
class LaunchSpec:
    """Everything that determines one training command."""

    task: str
    workflow: str
    share_policy: bool
    scale: str
    control: str | None = None
    paper_run: str | None = None
    n_gpus: int | None = None  # None: the control's GPU count, else 2
    gpu_type: str = "H100"
    gpu_type_explicit: bool = False  # True: a SLURM conf's GPU_TYPE does not replace gpu_type
    cpus_per_gpu: int | None = None  # None: SLURM conf META, else the task/GPU default
    mem_per_gpu: str | None = None
    base_model_tag: str | None = None
    init_adapters: dict[str, str] = field(default_factory=dict)
    project: str = DEFAULT_PROJECT
    name_suffix: str | None = None
    experiment_name: str | None = None
    processed_logprobs: bool = False
    resume_from: str | None = None
    # Values a paper-run entry pins (None: launcher defaults).
    ppo_max_token_len_per_gpu: int | None = None
    code_executor_workers: int | None = None
    total_training_steps: int | None = None
    lr_warmup_steps: int | None = None
    paper_overrides: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.task = normalize_task(self.task)
        self.scale = normalize_scale(self.scale)
        self.gpu_type = normalize_gpu_type(self.gpu_type)


def _recipe(spec: LaunchSpec) -> Recipe:
    recipe = get_recipe(spec.task, spec.workflow)
    if spec.share_policy not in recipe.allowed_share_policy:
        raise ValueError(f"{spec.task}/{spec.workflow} does not support share_policy={spec.share_policy}")
    return recipe


def _control(spec: LaunchSpec) -> Control | None:
    if spec.control is None:
        return None
    try:
        ctrl = CONTROLS[spec.control]
    except KeyError:
        raise ValueError(f"Unknown control {spec.control!r}; expected one of {list(CONTROLS)}") from None
    if ctrl.task is not None and ctrl.task != spec.task:
        raise ValueError(f"control {ctrl.name} needs --task {ctrl.task}")
    if ctrl.workflow is not None and ctrl.workflow != spec.workflow:
        raise ValueError(f"control {ctrl.name} needs --workflow {ctrl.workflow}")
    if ctrl.share_policy is not None and ctrl.share_policy != spec.share_policy:
        raise ValueError(f"control {ctrl.name} needs --policy {'sp' if ctrl.share_policy else 'ip'}")
    if ctrl.needs_resume_from and not spec.resume_from:
        raise ValueError(f"control {ctrl.name} needs --resume-from <checkpoint dir>/global_step_<N>")
    if ctrl.needs_init_adapters and not spec.init_adapters:
        raise ValueError(f"control {ctrl.name} needs at least one --init-adapter role=path")
    return ctrl


def _scale_size(scale: str) -> str:
    return SCALES[scale][1].removeprefix("qwen3_")


def _base_model_tag(spec: LaunchSpec, ctrl: Control | None) -> str | None:
    if spec.base_model_tag:
        return spec.base_model_tag
    if ctrl is not None and ctrl.base_model_tag:
        return ctrl.base_model_tag.format(size=_scale_size(spec.scale))
    return None


def _experiment_name(spec: LaunchSpec, recipe: Recipe, ctrl: Control | None, model_tag: str) -> str:
    if spec.experiment_name:
        return spec.experiment_name
    if ctrl is not None and ctrl.name_template and not spec.name_suffix:
        return ctrl.name_template.format(
            prefix=recipe.experiment_prefix,
            model_tag=model_tag,
            policy=POLICY_TAG[spec.share_policy],
            task_suffix=TASK_SUFFIX[spec.task],
        )
    suffix = spec.name_suffix or (ctrl.name_suffix if ctrl else None) or recipe.default_name_suffix
    return experiment_name(recipe.experiment_prefix, model_tag, spec.share_policy, spec.task, suffix)


def resolve_resources(spec: LaunchSpec, conf_meta: Mapping[str, str] | None = None) -> tuple[int, str]:
    """(cpus_per_gpu, mem_per_gpu): explicit spec values, else SLURM conf META, else defaults."""
    meta = conf_meta or {}
    tag = "CODE" if spec.task == "deepcoder" else "MATH"
    default_cpus, default_mem = _RESOURCE_DEFAULTS[(spec.task, _GPU_CLASS[spec.gpu_type])]
    cpus = spec.cpus_per_gpu or int(meta.get(f"CPUS_PER_GPU_{tag}", default_cpus))
    mem = spec.mem_per_gpu or meta.get(f"MEM_PER_GPU_{tag}", default_mem)
    return cpus, mem


def apply_slurm_meta(spec: LaunchSpec, meta: Mapping[str, str]) -> LaunchSpec:
    """Take the GPU type and the CPU count from a SLURM conf's META lines (explicit values win)."""
    if "GPU_TYPE" in meta and not spec.gpu_type_explicit:
        spec = replace(spec, gpu_type=normalize_gpu_type(meta["GPU_TYPE"]))
    if spec.cpus_per_gpu is None:
        spec = replace(spec, cpus_per_gpu=resolve_resources(spec, meta)[0])
    return spec


# --------------------------------------------------------------------------------------
# Overrides
# --------------------------------------------------------------------------------------

_OVERRIDE_RE = re.compile(r"^(?P<prefix>\+\+|\+|~)?(?P<key>[^=]+?)(=(?P<value>.*))?$", re.DOTALL)


def parse_override(override: str) -> tuple[str, str, str | None]:
    """Split ``[+|++|~]key=value`` into ``(prefix, key, value)``."""
    m = _OVERRIDE_RE.match(override.strip())
    if m is None or not m.group("key"):
        raise ValueError(f"Malformed override {override!r}")
    return m.group("prefix") or "", m.group("key").strip(), m.group("value")


def merge_overrides(*groups: Iterable[str]) -> list[str]:
    """Merge override lists by key; a later value replaces the earlier one in place."""
    merged: dict[str, tuple[str, str | None]] = {}
    for group in groups:
        for ov in group:
            prefix, key, value = parse_override(ov)
            if key in merged and not prefix:
                prefix = merged[key][0]
            merged[key] = (prefix, value)
    return [f"{p}{k}" if v is None else f"{p}{k}={v}" for k, (p, v) in merged.items()]


def _bool(value: bool) -> str:
    return "True" if value else "False"


def build_overrides(spec: LaunchSpec) -> list[str]:
    """Ordered Hydra overrides for ``python -m marl_workflows.trainer.train`` (golden-tested)."""
    recipe = _recipe(spec)
    ctrl = _control(spec)
    n_gpus = spec.n_gpus or (ctrl.n_gpus if ctrl is not None else None) or 2
    tag = _base_model_tag(spec, ctrl)
    model_path, model_tag = SCALES[spec.scale]
    if tag:
        model_tag = tag
        model_path = os.path.join(init_weight_dir(), spec.task, tag)
    prompt_len, response_len = recipe.train_lengths
    tok = spec.ppo_max_token_len_per_gpu or PPO_MAX_TOKEN_LEN[(spec.scale, _GPU_CLASS[spec.gpu_type])]
    names = ",".join(recipe.agent_names)

    base = [
        f"marl.task={spec.task}",
        f"marl.workflow={spec.workflow}",
        f"data.max_prompt_length={prompt_len}",
        f"data.max_response_length={response_len}",
        f"actor_rollout_ref.model.path={model_path}",
        f"trainer.project_name={spec.project}",
        f"trainer.experiment_name={_experiment_name(spec, recipe, ctrl, model_tag)}",
        f"trainer.n_gpus_per_node={n_gpus}",
        f"trainer.agent_names=[{names}]",
        f"trainer.share_policy={_bool(spec.share_policy)}",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={tok}",
        *WORKFLOW_PARAMS[(spec.task, spec.workflow)],
    ]
    if spec.task == "deepcoder":
        if spec.code_executor_workers is not None:
            workers = spec.code_executor_workers
        else:
            workers = n_gpus * resolve_resources(spec)[0]
        base += [f"rllm.workflow.code_executor_workers={workers}", "+data.dataset_name=deepcoder_primeintellect"]
    if tag:
        base.append(f"trainer.base_model_tag={tag}")

    pinned = []
    if spec.total_training_steps is not None:
        pinned.append(f"trainer.total_training_steps={spec.total_training_steps}")
    if spec.lr_warmup_steps is not None:
        pinned.append(f"actor_rollout_ref.actor.optim.lr_warmup_steps={spec.lr_warmup_steps}")

    control = list(ctrl.overrides) if ctrl is not None else []
    if spec.resume_from:
        control += ["trainer.resume_mode=resume_path", f"trainer.resume_from_path={spec.resume_from}"]
    init = [f"+rllm.workflow.initial_lora_weights.{role}={path}" for role, path in spec.init_adapters.items()]
    logprobs = [_LOGPROBS] if spec.processed_logprobs else []
    return merge_overrides(base, pinned, control, init, logprobs, spec.paper_overrides, spec.extra)


def _quote(arg: str) -> str:
    return shlex.quote(arg)


def build_command(spec: LaunchSpec, python: str = "python") -> str:
    """The full shell command line (one line, shell-quoted)."""
    return " ".join([python, "-m", ENTRY_MODULE, *(_quote(o) for o in build_overrides(spec))])


# --------------------------------------------------------------------------------------
# Paper runs
# --------------------------------------------------------------------------------------


def default_paper_runs_path() -> Path:
    return repo_root() / "configs" / "paper_runs.yaml"


def load_paper_runs(path: str | os.PathLike[str] | None = None) -> dict[str, dict[str, Any]]:
    """Load ``configs/paper_runs.yaml`` (``runs:`` mapping run id -> entry)."""
    import yaml

    p = Path(path) if path is not None else default_paper_runs_path()
    if not p.is_file():
        raise FileNotFoundError(f"{p} not found; generate it with scripts/export_paper_runs.py")
    data = yaml.safe_load(p.read_text()) or {}
    return dict(data.get("runs", {}))


def spec_from_paper_run(run_id: str, runs: Mapping[str, Mapping[str, Any]] | None = None, **changes: Any) -> LaunchSpec:
    """Build the :class:`LaunchSpec` that reproduces a paper run; ``changes`` override fields."""
    runs = load_paper_runs() if runs is None else runs
    if run_id not in runs:
        raise KeyError(f"Unknown paper run {run_id!r}; {len(runs)} runs in paper_runs.yaml")
    e = runs[run_id]
    spec = LaunchSpec(
        task=e["task"],
        workflow=e["workflow"],
        share_policy=bool(e["share_policy"]),
        scale=e["scale"],
        control=e.get("control"),
        paper_run=run_id,
        n_gpus=int(e["n_gpus"]),
        gpu_type=e.get("gpu_type") or "H100",
        base_model_tag=e.get("base_model_tag"),
        project=PAPER_PROJECT,
        experiment_name=e["experiment_name"],
        resume_from=(e.get("resumed_from") or {}).get("path") if e.get("control") == "cap_8192" else None,
        ppo_max_token_len_per_gpu=e.get("ppo_max_token_len_per_gpu"),
        code_executor_workers=e.get("code_executor_workers"),
        total_training_steps=e.get("total_training_steps"),
        lr_warmup_steps=e.get("lr_warmup_steps"),
        paper_overrides=list(e.get("overrides") or []),
    )
    spec = replace(spec, **changes) if changes else spec
    if spec.task == "deepcoder" and spec.code_executor_workers and spec.cpus_per_gpu is None:
        # The fork sized the code-execution pool to the job's CPUs; a pinned pool size sets the CPUs.
        n_gpus = spec.n_gpus or 2
        spec = replace(spec, cpus_per_gpu=math.ceil(spec.code_executor_workers / n_gpus))
    return spec


# --------------------------------------------------------------------------------------
# SLURM
# --------------------------------------------------------------------------------------

_META_RE = re.compile(r"^#\s*META:\s*([A-Z0-9_]+)=(\S+)")
_SKIP_DIRECTIVES = re.compile(r"--(job-name|output|error|gres=gpu|cpus-per-gpu|mem-per-gpu)\b")


def read_slurm_conf(path: str | os.PathLike[str]) -> tuple[dict[str, str], list[str]]:
    """Return ``(META key -> value, #SBATCH lines)`` of a ``configs/slurm/*.conf`` file.

    Resource lines (gres, cpus, mem, job name, logs) are dropped: the launcher writes them.
    """
    meta: dict[str, str] = {}
    directives: list[str] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        m = _META_RE.match(line)
        if m:
            meta[m.group(1)] = m.group(2)
        elif line.startswith("#SBATCH") and not _SKIP_DIRECTIVES.search(line):
            directives.append(line)
    return meta, directives


def _with_time(directives: list[str], time_limit: str | None) -> list[str]:
    if not time_limit:
        return directives
    return [d for d in directives if "--time=" not in d] + [f"#SBATCH --time={time_limit}"]


def _template(name: str) -> string.Template:
    path = repo_root() / "scripts" / "slurm" / name
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found (SLURM rendering needs a repository checkout)")
    return string.Template(path.read_text())


def render_sbatch(
    spec: LaunchSpec,
    slurm_conf: str | os.PathLike[str],
    env_activate: str = "source scripts/env.sh",
    *,
    time_limit: str | None = None,
    root: str | None = None,
) -> str:
    """Render ``scripts/slurm/train.sbatch.tmpl`` for ``spec`` and a SLURM conf."""
    meta, directives = read_slurm_conf(slurm_conf)
    spec = apply_slurm_meta(spec, meta)
    cpus, mem = resolve_resources(spec, meta)
    overrides = build_overrides(spec)
    n_gpus = next(int(o.split("=", 1)[1]) for o in overrides if o.startswith("trainer.n_gpus_per_node="))
    exp = next(o.split("=", 1)[1] for o in overrides if o.startswith("trainer.experiment_name="))
    command = " \\\n    ".join([f"python -m {ENTRY_MODULE}", *(_quote(o) for o in overrides)])
    return _template("train.sbatch.tmpl").substitute(
        JOB_NAME=exp,
        N_GPUS=n_gpus,
        CPUS_PER_GPU=cpus,
        MEM_PER_GPU=mem,
        SBATCH_DIRECTIVES="\n".join(_with_time(directives, time_limit)),
        REPO_ROOT=root or str(repo_root()),
        ENV_ACTIVATE=env_activate,
        ULIMIT="ulimit -n 1048576" if spec.task == "deepcoder" else "",
        COMMAND=command,
    )


def render_eval_sbatch(
    eval_args: Sequence[str],
    slurm_conf: str | os.PathLike[str],
    env_activate: str = "source scripts/env.sh",
    *,
    job_name: str = "marl-eval",
    n_gpus: int = 1,
    task: str = "math",
    cpus_per_gpu: int | None = None,
    mem_per_gpu: str | None = None,
    time_limit: str | None = None,
    root: str | None = None,
) -> str:
    """Render ``scripts/slurm/eval.sbatch.tmpl`` around ``evaluate_checkpoints`` arguments."""
    meta, directives = read_slurm_conf(slurm_conf)
    probe = LaunchSpec(
        task=task,
        workflow="single_agent",
        share_policy=False,
        scale="0.6B",
        gpu_type=meta.get("GPU_TYPE", "H100"),
        cpus_per_gpu=cpus_per_gpu,
        mem_per_gpu=mem_per_gpu,
    )
    cpus, mem = resolve_resources(probe, meta)
    command = " ".join([f"python -m {EVAL_MODULE}", *(_quote(a) for a in eval_args)])
    return _template("eval.sbatch.tmpl").substitute(
        JOB_NAME=job_name,
        N_GPUS=n_gpus,
        CPUS_PER_GPU=cpus,
        MEM_PER_GPU=mem,
        SBATCH_DIRECTIVES="\n".join(_with_time(directives, time_limit)),
        REPO_ROOT=root or str(repo_root()),
        ENV_ACTIVATE=env_activate,
        ULIMIT="ulimit -n 1048576" if normalize_task(task) == "deepcoder" else "",
        COMMAND=command,
    )


def _submit(script: str, name: str, dependency: str | None = None) -> str:
    """Write ``script`` under ``logs/`` and ``sbatch`` it; return the job id."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    path = log_dir / f"{name}.{time.strftime('%Y%m%d-%H%M%S')}.sbatch.generated"
    path.write_text(script)
    cmd = ["sbatch", "--parsable"]
    if dependency:
        cmd.append(f"--dependency=afterany:{dependency}")
    out = subprocess.run([*cmd, str(path)], check=True, capture_output=True, text=True).stdout.strip()
    return out.split(";")[0]


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _parse_policy(value: str) -> bool:
    v = value.lower()
    if v in ("ip", "multi_lora", "false"):
        return False
    if v in ("sp", "share_policy", "true"):
        return True
    raise argparse.ArgumentTypeError(f"--policy must be ip or sp, got {value!r}")


def _parse_init_adapter(value: str) -> tuple[str, str]:
    role, sep, path = value.partition("=")
    if not sep or not role or not path:
        raise argparse.ArgumentTypeError(f"--init-adapter expects role=path, got {value!r}")
    return role, path


def _train_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="launch.py",
        description="Build (and optionally submit) the training command of one cell. "
        "Use 'launch.py eval ...' to render an evaluation job.",
    )
    p.add_argument("--task", choices=["math", "deepcoder", "code"])
    p.add_argument("--workflow", help="recipe id, e.g. voting, evaluator_optimizer, orchestrator_workers_propose")
    p.add_argument("--policy", type=_parse_policy, help="ip (one adapter per role) or sp (one shared adapter)")
    p.add_argument("--scale", help="0.6B, 1.7B or 4B")
    p.add_argument("--control", choices=sorted(CONTROLS))
    p.add_argument("--paper-run", help="run id from configs/paper_runs.yaml; reproduces that run")
    p.add_argument("--paper-runs-file", help="alternative paper_runs.yaml")
    p.add_argument("--gpus", type=int, help="GPUs per node (default 2)")
    p.add_argument("--gpu-type", help="H100, H200 or L40S (sets the token budget; default H100 or the conf META)")
    p.add_argument("--cpus-per-gpu", type=int)
    p.add_argument("--mem-per-gpu")
    p.add_argument("--base-model", help="init-weight tag, e.g. qwen3_1.7b_s300 ($MARL_INIT_WEIGHT_DIR/<task>/<tag>)")
    p.add_argument("--init-adapter", action="append", type=_parse_init_adapter, default=[], metavar="ROLE=PATH")
    p.add_argument("--resume-from", help="checkpoint directory global_step_<N> to resume from")
    p.add_argument("--project", help=f"wandb project (default {DEFAULT_PROJECT}; {PAPER_PROJECT} with --paper-run)")
    p.add_argument("--name-suffix")
    p.add_argument("--experiment-name")
    p.add_argument("--processed-logprobs", action="store_true", help="log vLLM processed log-probs")
    p.add_argument("--extra", action="append", default=[], help='extra Hydra overrides, e.g. --extra "a=1 b=2"')
    p.add_argument("--slurm-config", help="configs/slurm/<name>.conf; prints the sbatch script")
    p.add_argument("--time", help="SLURM time limit (replaces the conf's --time)")
    p.add_argument("--env-activate", default="source scripts/env.sh", help="shell line run before training")
    p.add_argument("--submit", action="store_true", help="submit with sbatch instead of printing")
    p.add_argument("--chain", type=int, default=1, help="submit K segments chained with afterany")
    p.add_argument("--skip-gpu-check", action="store_true")
    p.add_argument("--print", dest="print_cmd", action="store_true", help="print the training command line")
    p.add_argument("overrides", nargs="*", help="extra Hydra overrides (same as --extra)")
    return p


def _spec_from_args(args: argparse.Namespace) -> LaunchSpec:
    extra = [tok for chunk in args.extra for tok in shlex.split(chunk)] + list(args.overrides)
    if args.paper_run:
        runs = load_paper_runs(args.paper_runs_file)
        changes: dict[str, Any] = {"extra": extra}
        for attr, value in (
            ("n_gpus", args.gpus),
            ("cpus_per_gpu", args.cpus_per_gpu),
            ("mem_per_gpu", args.mem_per_gpu),
            ("project", args.project),
        ):
            if value is not None:
                changes[attr] = value
        if args.gpu_type:
            changes["gpu_type"] = normalize_gpu_type(args.gpu_type)
            changes["gpu_type_explicit"] = True
        return spec_from_paper_run(args.paper_run, runs, **changes)
    missing = [f"--{n}" for n in ("task", "workflow", "policy", "scale") if getattr(args, n) is None]
    if missing:
        raise SystemExit(f"launch.py: {', '.join(missing)} required (or --paper-run)")
    return LaunchSpec(
        task=args.task,
        workflow=args.workflow,
        share_policy=args.policy,
        scale=args.scale,
        control=args.control,
        n_gpus=args.gpus,
        gpu_type=args.gpu_type or "H100",
        gpu_type_explicit=args.gpu_type is not None,
        cpus_per_gpu=args.cpus_per_gpu,
        mem_per_gpu=args.mem_per_gpu,
        base_model_tag=args.base_model,
        init_adapters=dict(args.init_adapter),
        project=args.project or DEFAULT_PROJECT,
        name_suffix=args.name_suffix,
        experiment_name=args.experiment_name,
        processed_logprobs=args.processed_logprobs,
        resume_from=args.resume_from,
        extra=extra,
    )


def _eval_cli(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="launch.py eval",
        description="Render (or submit) a SLURM job running marl_workflows.eval.evaluate_checkpoints. "
        "Arguments after '--' go to the evaluator unchanged.",
    )
    p.add_argument("--slurm-config", required=True)
    p.add_argument("--task", default="math", choices=["math", "deepcoder", "code"])
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--cpus-per-gpu", type=int)
    p.add_argument("--mem-per-gpu")
    p.add_argument("--time")
    p.add_argument("--job-name", default="marl-eval")
    p.add_argument("--env-activate", default="source scripts/env.sh")
    p.add_argument("--submit", action="store_true")
    p.add_argument("eval_args", nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    eval_args = args.eval_args[1:] if args.eval_args[:1] == ["--"] else args.eval_args
    script = render_eval_sbatch(
        eval_args,
        args.slurm_config,
        args.env_activate,
        job_name=args.job_name,
        n_gpus=args.gpus,
        task=args.task,
        cpus_per_gpu=args.cpus_per_gpu,
        mem_per_gpu=args.mem_per_gpu,
        time_limit=args.time,
    )
    if args.submit:
        print(f"Submitted {args.job_name} -> job {_submit(script, args.job_name)}")
    else:
        print(script, end="")
    return 0


def cli(argv: list[str] | None = None) -> int:
    """Entry point of ``scripts/launch.py`` and the ``marl-launch`` console script."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["eval"]:
        return _eval_cli(argv[1:])
    args = _train_parser().parse_args(argv)
    try:
        spec = _spec_from_args(args)
        if args.chain > 1 and not any(o.lstrip("+").startswith("trainer.wallclock_stop.enable") for o in spec.extra):
            # Segments save before the time limit and the next one resumes from the same directory.
            spec = replace(spec, extra=[*spec.extra, _WALLCLOCK])
        if not args.slurm_config:
            if args.submit or args.chain > 1:
                raise ValueError("--submit and --chain need --slurm-config")
            print(build_command(spec))
            return 0
        meta, _ = read_slurm_conf(args.slurm_config)
        spec = apply_slurm_meta(spec, meta)
        if not args.skip_gpu_check:
            check_gpu_policy(spec.scale, spec.gpu_type)
        script = render_sbatch(spec, args.slurm_config, args.env_activate, time_limit=args.time)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"launch.py: error: {exc}", file=sys.stderr)
        return 2
    if args.print_cmd and not args.submit:
        print(build_command(spec))
        return 0
    if not args.submit:
        print(script, end="")
        return 0
    name = next(o.split("=", 1)[1] for o in build_overrides(spec) if o.startswith("trainer.experiment_name="))
    job: str | None = None
    for i in range(args.chain):
        job = _submit(script, name, dependency=job)
        print(f"Submitted {name} segment {i + 1}/{args.chain} -> job {job}")
    return 0
