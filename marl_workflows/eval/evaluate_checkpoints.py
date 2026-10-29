"""Evaluate trained checkpoints (or the base model) of the Math and DeepCoder workflows.

Port of the research fork's ``dashboard/evaluate_checkpoints.py``, trimmed to the core. For
each base model it starts one LoRA-enabled vLLM OpenAI server, loads the checkpoint's role
adapters, runs the recipe's workflow on every problem of the evaluation set through
:class:`~marl_workflows.engine.openai_engine.LoRAOpenAIEngine`, and appends one JSON line per
(checkpoint, dataset) to ``--output-json`` with the fork's ``eval_results.jsonl`` schema
(C19). With ``--trajectory-output-dir`` every episode is also written to
``{dir}/{experiment}/{dataset}/step_{N}/{episode_id}.json``.

Modes (``--eval-mode``): ``trained_checkpoint`` (the checkpoint's adapters), ``base_model``
(no adapter), ``single_agent_transfer`` (one SA-RL adapter for every role) and
``single_agent_generator_transfer`` (the SA-RL adapter for the recipe's solving roles only).

Example (the paper's Math validation, training-length caps)::

    python -m marl_workflows.eval.evaluate_checkpoints --task-type math \\
        --checkpoints-dir checkpoints/rllm-workflow-MARL-v2 \\
        --experiment-filter '^voting-qwen3_0.6b-multi_lora-math$' --step-filter 300 \\
        --dataset dapo_math --use-training-lengths --trajectory-output-dir evaluation_trajectories

Differences from the fork: workflow classes, kwargs and lengths come from ``recipes``;
``--max-samples N`` only truncates to the first N problems (results are still written,
unless ``--no-write``); the already-evaluated check also matches the number of problems, so a
truncated run never masks a full one; ``--eval-opt-controller`` is not ported.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import multiprocessing
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from marl_workflows.contracts import TASKS
from marl_workflows.eval.checkpoints import (
    CheckpointInfo,
    EvalMode,
    create_synthetic_checkpoints,
    discover_checkpoints,
    filter_last_checkpoints,
    group_checkpoints_by_model,
)
from marl_workflows.recipes import RECIPES, Recipe, get_recipe, load_object
from marl_workflows.utils import allocated_cpu_count

__all__ = [
    "GPU_TYPE_CHOICES",
    "EvalMode",
    "EvalResult",
    "build_workflow_kwargs",
    "cli",
    "compute_n_parallel",
    "compute_rollout_metrics",
    "detect_gpu_type",
    "evaluate_checkpoint",
    "generator_roles_for",
    "get_checkpoint_key",
    "load_eval_dataset",
    "load_existing_results",
    "main",
    "parse_args",
    "resolve_lengths",
    "save_results_to_json",
    "save_trajectory_to_json",
    "setup_lora_for_eval",
]

PASS_AT_K_VALUES = (1, 2, 4, 8, 16, 32, 64)

# ======================================================================================
# Result record (C19)
# ======================================================================================


@dataclass
class EvalResult:
    """Evaluation of one checkpoint on one dataset; ``to_dict`` is the fork's JSONL schema."""

    experiment_name: str
    checkpoint_step: int
    dataset: str
    workflow_type: str
    model_size: str
    model_name: str
    share_policy: bool
    accuracy: float
    num_correct: int
    num_total: int
    eval_duration_seconds: float
    eval_mode: str = "trained_checkpoint"
    length_mode: str = "eval"  # "eval" = recipe eval_lengths, "training" = recipe train_lengths
    max_prompt_length: int = 0
    max_response_length: int = 0
    problem_results: list[dict] = field(default_factory=list)
    hostname: str = field(default_factory=socket.gethostname)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    n_rollouts: int = 1
    mean_accuracy: float = 0.0
    std_accuracy: float = 0.0
    pass_at_n: float = 0.0
    per_problem_n_correct: list[int] = field(default_factory=list)
    pass_at_k: dict[int, float] = field(default_factory=dict)
    eval_variant: str = ""  # non-default evaluation variant; empty = standard eval

    # COPIED FROM fork dashboard/evaluate_checkpoints.py:185-216 (EvalResult.to_dict), unchanged.
    def to_dict(self) -> dict[str, Any]:
        """The JSONL record (key order and rounding as the fork)."""
        d: dict[str, Any] = {
            "timestamp": self.timestamp,
            "hostname": self.hostname,
            "dataset": self.dataset,
            "experiment_name": self.experiment_name,
            "checkpoint_step": int(self.checkpoint_step),
            "workflow_type": self.workflow_type,
            "model_size": self.model_size,
            "model_name": self.model_name,
            "share_policy": bool(self.share_policy),
            "accuracy": round(float(self.accuracy), 4),
            "num_correct": int(self.num_correct),
            "num_total": int(self.num_total),
            "eval_duration_seconds": round(self.eval_duration_seconds, 1),
            "eval_mode": self.eval_mode,
            "length_mode": self.length_mode,
            "max_prompt_length": self.max_prompt_length,
            "max_response_length": self.max_response_length,
            "n_rollouts": self.n_rollouts,
            "mean_accuracy": round(self.mean_accuracy, 4),
            "std_accuracy": round(self.std_accuracy, 4),
            "pass_at_n": round(self.pass_at_n, 4),
        }
        if self.per_problem_n_correct:
            d["per_problem_n_correct"] = [int(x) for x in self.per_problem_n_correct]
        if self.pass_at_k:
            d["pass_at_k"] = {str(k): round(v, 4) for k, v in self.pass_at_k.items()}
        if self.eval_variant:
            d["eval_variant"] = self.eval_variant
        return d


# ======================================================================================
# Concurrency sizing (fork :233-296)
# ======================================================================================

_GPU_BASE = {"RTX8000": 256, "A40": 384, "L40s": 512, "H100": 1024, "H200": 1280}
_MODEL_SCALE = {"0.6b": 1.5, "1.7b": 1.0, "4b": 0.5}
# Voting runs N parallel candidates per problem.
_WORKFLOW_SCALE = {"voting": 0.5, "voting_v2": 0.5}
GPU_TYPE_CHOICES = list(_GPU_BASE)


def detect_gpu_type() -> str | None:
    """GPU family of device 0 from ``nvidia-smi`` (``None`` if unknown or absent)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    name = result.stdout.strip().split("\n")[0].upper()
    for token, gpu in (("H200", "H200"), ("H100", "H100"), ("L40S", "L40s"), ("A40", "A40")):
        if token in name:
            return gpu
    if "RTX 8000" in name or "RTX8000" in name:
        return "RTX8000"
    print(f"Warning: Unrecognized GPU '{name}', defaulting to L40s scaling")
    return None


def compute_n_parallel(gpu_type: str | None, model_size: str, workflow_type: str) -> int:
    """``base[gpu] * scale[model] * scale[workflow]``, at least 32 (L40s + 1.7B + non-voting = 512)."""
    resolved = gpu_type or "L40s"
    n = int(
        _GPU_BASE.get(resolved, 512)
        * _MODEL_SCALE.get(model_size.lower(), 1.0)
        * _WORKFLOW_SCALE.get(workflow_type, 1.0)
    )
    n = max(n, 32)
    print(f"  Auto n_parallel={n} (gpu={resolved}, model={model_size}, workflow={workflow_type})")
    return n


# ======================================================================================
# Lengths, workflow kwargs, adapters
# ======================================================================================


def resolve_lengths(task: str, use_training_lengths: bool) -> dict[str, tuple[int, int]]:
    """Workflow id -> (max_prompt_length, max_response_length) for every recipe of ``task``."""
    return {
        wf: (r.train_lengths if use_training_lengths else r.eval_lengths) for (t, wf), r in RECIPES.items() if t == task
    }


def build_workflow_kwargs(recipe: Recipe) -> dict[str, Any]:
    """Evaluation kwargs: the recipe's ``eval_params`` plus the resolved ``reward_function`` (if any)."""
    kwargs: dict[str, Any] = dict(recipe.eval_params)
    if recipe.reward_fn is not None:
        kwargs["reward_function"] = load_object(recipe.reward_fn)
    return kwargs


def generator_roles_for(recipe: Recipe) -> list[str]:
    """Solving roles that take the SA-RL adapter under ``single_agent_generator_transfer``."""
    return [r for r in recipe.generator_roles if r in recipe.agent_names]


def setup_lora_for_eval(
    checkpoint: CheckpointInfo,
    server: Any,
    engine: Any,
    eval_mode: EvalMode,
    recipe: Recipe,
    single_agent_lora_path: str | None = None,
) -> dict[str, str]:
    """Load the adapters ``eval_mode`` needs and install the role -> served-name map on ``engine``.

    Served names are ``{experiment}_step{N}_{suffix}`` (fork :807-898), unique per checkpoint, so
    a server can hold several checkpoints. Returns the installed map.
    """
    prefix = f"{checkpoint.experiment_name}_step{checkpoint.checkpoint_step}"
    agent_names = list(recipe.agent_names)

    if eval_mode == EvalMode.BASE_MODEL:
        mapping: dict[str, str] = {}
    elif eval_mode in (EvalMode.SINGLE_AGENT_TRANSFER, EvalMode.SINGLE_AGENT_GENERATOR_TRANSFER):
        if not single_agent_lora_path:
            raise ValueError(f"single_agent_lora_path is required for {eval_mode.value}")
        generator_only = eval_mode == EvalMode.SINGLE_AGENT_GENERATOR_TRANSFER
        roles = generator_roles_for(recipe) if generator_only else agent_names
        if not roles:
            raise ValueError(f"No generator/worker role defined for workflow '{checkpoint.workflow_type}'")
        lora_name = f"{prefix}_{'gen_transfer' if generator_only else 'transfer'}"
        server.load_lora(lora_name, single_agent_lora_path)
        # Roles left out of the map run on the base model (LoRAOpenAIEngine.model_for).
        mapping = {role: lora_name for role in roles}
    elif checkpoint.share_policy:
        lora_name = f"{prefix}_shared"
        server.load_lora(lora_name, os.path.join(checkpoint.actor_path or "", "lora_adapter"))
        mapping = {role: lora_name for role in agent_names}
    else:
        mapping = {}
        for role in agent_names:
            path = os.path.join(checkpoint.actor_path or "", f"lora_adapter_{role}")
            if not os.path.exists(path):
                raise FileNotFoundError(f"LoRA adapter not found for agent '{role}' at: {path}")
            server.load_lora(f"{prefix}_{role}", path)
            mapping[role] = f"{prefix}_{role}"
    engine.set_lora_names(mapping)
    print(f"Role -> model map: {mapping or '(base model for every role)'}")
    return mapping


# ======================================================================================
# Metrics
# ======================================================================================


def compute_rollout_metrics(per_problem_runs: Sequence[Sequence[bool]], uid_offset: int = 0) -> dict[str, Any]:
    """Mean/std accuracy over rollouts, pass@N and unbiased pass@k (fork :1061-1111)."""
    n_problems = len(per_problem_runs)
    n_rollouts = len(per_problem_runs[0]) if n_problems else 0
    problem_results = []
    for i, runs in enumerate(per_problem_runs):
        n_correct = int(sum(bool(r) for r in runs))
        problem_results.append(
            {
                "uid": f"eval_{i + uid_offset}",
                "n_correct": n_correct,
                "n_rollouts": len(runs),
                "pass": any(runs),
                "per_run": [bool(r) for r in runs],
            }
        )
    per_rollout = [
        (sum(1 for pr in problem_results if pr["per_run"][j]) / n_problems if n_problems else 0.0)
        for j in range(n_rollouts)
    ]
    mean_accuracy = sum(per_rollout) / len(per_rollout) if per_rollout else 0.0
    std_accuracy = math.sqrt(sum((a - mean_accuracy) ** 2 for a in per_rollout) / n_rollouts) if n_rollouts > 1 else 0.0
    pass_at_n = sum(1 for pr in problem_results if pr["pass"]) / n_problems if n_problems else 0.0
    pass_at_k: dict[int, float] = {}
    for k in PASS_AT_K_VALUES:
        if k > n_rollouts:
            break
        vals = []
        for pr in problem_results:
            n, c = pr["n_rollouts"], pr["n_correct"]
            vals.append(1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k))
        pass_at_k[k] = sum(vals) / len(vals) if vals else 0.0
    return {
        "problem_results": problem_results,
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "pass_at_n": pass_at_n,
        "per_problem_n_correct": [pr["n_correct"] for pr in problem_results],
        "pass_at_k": pass_at_k,
        # Kept for the schema: derived from the mean accuracy, as in the fork.
        "num_correct": round(mean_accuracy * n_problems),
    }


# ======================================================================================
# Evaluation of one checkpoint
# ======================================================================================


async def evaluate_checkpoint(
    checkpoint: CheckpointInfo,
    server: Any,
    engine: Any,
    dataset: Sequence[dict],
    *,
    task: str,
    n_parallel: int = 32,
    eval_mode: EvalMode = EvalMode.TRAINED_CHECKPOINT,
    single_agent_lora_path: str | None = None,
    trajectory_output_dir: str | None = None,
    dataset_name: str = "",
    n_rollouts: int = 1,
    uid_offset: int = 0,
    length_mode: str = "eval",
    save_token_logprobs: bool = False,
    workflow_cls: type | None = None,
    workflow_kwargs: Mapping[str, Any] | None = None,
) -> EvalResult:
    """Run the recipe's workflow on every problem (``n_rollouts`` each) and score the episodes.

    ``workflow_cls`` / ``workflow_kwargs`` override the recipe (tests). Aborts with
    ``RuntimeError`` if any episode ended with ``TerminationReason.ERROR`` (inference service
    failure), so that no result line is written for a broken run.
    """
    from rllm.agents.agent import Episode
    from rllm.workflows.workflow import TerminationReason

    recipe = get_recipe(task, checkpoint.workflow_type)
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {checkpoint.experiment_name} step {checkpoint.checkpoint_step}")
    print(f"  Workflow: {checkpoint.workflow_type}  Actor Path: {checkpoint.actor_path}")
    print(f"  Share Policy: {checkpoint.share_policy}  Eval Mode: {eval_mode.value}  N Rollouts: {n_rollouts}")
    print(f"{'=' * 60}")
    start_time = time.time()

    setup_lora_for_eval(checkpoint, server, engine, eval_mode, recipe, single_agent_lora_path)

    cls = workflow_cls if workflow_cls is not None else load_object(recipe.workflow_cls)
    kwargs = dict(workflow_kwargs) if workflow_kwargs is not None else build_workflow_kwargs(recipe)

    # DeepCoder grades in a spawn ProcessPool sized to the allocation with direct execution,
    # the training configuration (code_executor_workers = CPUs, max_concurrent_code_execs = 0).
    code_reward_executor: ProcessPoolExecutor | None = None
    if recipe.uses_code_executor and "code_reward_executor" not in kwargs:
        from marl_workflows.rewards.code_reward import code_executor_initializer

        workers = allocated_cpu_count()
        print(f"Using code reward ProcessPoolExecutor with {workers} workers")
        code_reward_executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=code_executor_initializer,
        )
        kwargs["code_reward_executor"] = code_reward_executor

    from tqdm import tqdm

    semaphore = asyncio.Semaphore(n_parallel)
    progress = tqdm(total=len(dataset) * n_rollouts, desc="Evaluating")
    checkpoint_name = (
        f"{checkpoint.experiment_name}/{dataset_name}/step_{checkpoint.checkpoint_step}"
        if trajectory_output_dir
        else None
    )

    async def evaluate_single(task_row: dict, uid: str) -> Any:
        async with semaphore:
            try:
                workflow = cls(rollout_engine=engine, **kwargs)
                result = await workflow.run_with_termination_handling(task_row, uid)
                # Written immediately so that partial results survive a later failure.
                if trajectory_output_dir and checkpoint_name and isinstance(result, Episode):
                    save_trajectory_to_json(result, trajectory_output_dir, checkpoint_name, save_token_logprobs)
                return result
            finally:
                progress.update(1)

    coros = []
    for i, row in enumerate(dataset):
        for j in range(n_rollouts):
            uid = f"eval_{i + uid_offset}" if n_rollouts == 1 else f"eval_{i + uid_offset}_run_{j}"
            coros.append(evaluate_single(row, uid))
    try:
        episodes = await asyncio.gather(*coros)
    finally:
        progress.close()
        if code_reward_executor is not None:
            code_reward_executor.shutdown(wait=True)

    errors = [ep for ep in episodes if isinstance(ep, Episode) and ep.termination_reason == TerminationReason.ERROR]
    if errors:
        sample = (errors[0].info or {}).get("error", {}).get("error_message", "unknown")
        raise RuntimeError(
            f"{len(errors)}/{len(episodes)} rollouts failed with errors. Inference service may be down. "
            f"Not writing results. Sample error: {sample}"
        )

    per_problem_runs = [
        [
            bool(ep.is_correct) if isinstance(ep, Episode) else False
            for ep in episodes[i * n_rollouts : (i + 1) * n_rollouts]
        ]
        for i in range(len(dataset))
    ]
    m = compute_rollout_metrics(per_problem_runs, uid_offset)
    duration = time.time() - start_time
    print(f"\nResults (N={n_rollouts}): mean accuracy {m['mean_accuracy']:.2%}  duration {duration:.1f}s")
    if n_rollouts > 1:
        print(f"  std {m['std_accuracy']:.2%}  pass@{n_rollouts} {m['pass_at_n']:.2%}")

    return EvalResult(
        experiment_name=checkpoint.experiment_name,
        checkpoint_step=checkpoint.checkpoint_step,
        dataset=dataset_name,
        workflow_type=checkpoint.workflow_type,
        model_size=checkpoint.model_size,
        model_name=checkpoint.model_name,
        share_policy=checkpoint.share_policy,
        accuracy=m["mean_accuracy"],
        num_correct=m["num_correct"],
        num_total=len(dataset),
        eval_duration_seconds=duration,
        eval_mode=eval_mode.value,
        length_mode=length_mode,
        max_prompt_length=engine.max_prompt_length,
        max_response_length=engine.max_response_length,
        problem_results=m["problem_results"],
        n_rollouts=n_rollouts,
        mean_accuracy=m["mean_accuracy"],
        std_accuracy=m["std_accuracy"],
        pass_at_n=m["pass_at_n"],
        per_problem_n_correct=m["per_problem_n_correct"],
        pass_at_k=m["pass_at_k"],
    )


# ======================================================================================
# Output
# ======================================================================================


def save_trajectory_to_json(episode: Any, output_dir: str, checkpoint_name: str, keep_token_ids: bool = False) -> str:
    """Write ``episode.to_dict()`` to ``{output_dir}/{checkpoint_name}/{episode.id}.json`` (fork :1157-1193).

    Token-id lists and log-probs are dropped to keep files small unless ``keep_token_ids``
    (``--save-token-logprobs``), which keeps them inside ``model_output``.
    """
    directory = os.path.join(output_dir, checkpoint_name)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{episode.id}.json")
    episode_dict = episode.to_dict()
    for traj in episode_dict.get("trajectories", []):
        for step in traj.get("steps", []):
            step.pop("prompt_ids", None)
            step.pop("response_ids", None)
            step.pop("logprobs", None)
            model_output = step.get("model_output")
            if keep_token_ids:
                continue
            if isinstance(model_output, dict):
                model_output.pop("prompt_ids", None)
                model_output.pop("response_ids", None)
                model_output.pop("completion_ids", None)
    with open(path, "w") as f:
        json.dump(episode_dict, f, indent=2)
    return path


def save_results_to_json(results: Sequence[EvalResult], output_path: str) -> None:
    """Append one JSON line per result."""
    if not results:
        return
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(output_path, "a") as f:
        for result in results:
            f.write(json.dumps(result.to_dict()) + "\n")
    print(f"\nResults appended to: {output_path}")


def get_checkpoint_key(
    checkpoint: CheckpointInfo, dataset: str, eval_mode: str, n_rollouts: int, length_mode: str, num_total: int
) -> tuple:
    """Dedup key: the fork's six fields plus the number of problems evaluated."""
    return (
        checkpoint.experiment_name,
        checkpoint.checkpoint_step,
        dataset,
        eval_mode,
        n_rollouts,
        length_mode,
        num_total,
    )


def load_existing_results(output_path: str) -> set[tuple]:
    """Keys (as :func:`get_checkpoint_key`) of the lines already in ``output_path``."""
    existing: set[tuple] = set()
    if not os.path.exists(output_path):
        return existing
    with open(output_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: malformed JSON on line {line_num} of {output_path}, skipping")
                continue
            existing.add(
                (
                    obj.get("experiment_name", ""),
                    obj.get("checkpoint_step", 0),
                    obj.get("dataset", ""),
                    obj.get("eval_mode", ""),
                    obj.get("n_rollouts", 1),
                    obj.get("length_mode", "eval"),
                    obj.get("num_total", 0),
                )
            )
    return existing


def load_eval_dataset(
    name: str, max_samples: int | None = None, problem_range: Sequence[int] | None = None
) -> list[dict]:
    """The registered ``test`` split as dicts with both ``ground_truth`` and ``final_answer``."""
    from marl_workflows.data.registry import PREPARE_MODULES, MarlDatasetRegistry

    dataset = MarlDatasetRegistry.load_dataset(name, "test")
    if dataset is None:
        module = PREPARE_MODULES.get(name, "marl_workflows.data.prepare_*")
        raise ValueError(f"Dataset not found: {name} (run python -m {module})")
    rows = []
    for i in range(len(dataset)):
        row = dict(dataset[i])
        if "ground_truth" not in row and "final_answer" in row:
            row["ground_truth"] = row["final_answer"]
        elif "final_answer" not in row and "ground_truth" in row:
            row["final_answer"] = row["ground_truth"]
        rows.append(row)
    if max_samples is not None and max_samples < len(rows):
        print(f"Limiting to the first {max_samples} problems")
        rows = rows[:max_samples]
    if problem_range is not None:
        start, end = problem_range
        rows = rows[start:end]
        print(f"Restricting to problems [{start}, {end}) (uids keep global indices)")
    return rows


# ======================================================================================
# Driver
# ======================================================================================


def _select_checkpoints(args: argparse.Namespace, eval_mode: EvalMode) -> list[CheckpointInfo]:
    if eval_mode == EvalMode.TRAINED_CHECKPOINT:
        print(f"\nDiscovering checkpoints in: {args.checkpoints_dir}")
        cps = discover_checkpoints(
            args.checkpoints_dir,
            experiment_filter=args.experiment_filter,
            step_filter=args.step_filter or None,
            task_type=args.task_type,
        )
        if args.eval_interval and cps:
            before = len(cps)
            cps = [cp for cp in cps if cp.checkpoint_step % args.eval_interval == 0]
            print(f"Eval interval filter ({args.eval_interval}): {before} -> {len(cps)} checkpoints")
        if args.last_checkpoint_only and cps:
            cps = filter_last_checkpoints(cps)
        if args.base_model:
            print(f"Overriding base model to: {args.base_model}")
            for cp in cps:
                cp.base_model = args.base_model
        if args.step_filter and len(args.step_filter) > 1 and len({cp.experiment_name for cp in cps}) > 1:
            raise ValueError(
                "Multi-step --step-filter runs must be scoped to a single experiment "
                "(pass an --experiment-filter that matches exactly one experiment)."
            )
        return cps
    return create_synthetic_checkpoints(
        eval_mode, args.task_type, args.base_model, args.workflow_types, args.single_agent_lora_path
    )


def _validate_args(args: argparse.Namespace, eval_mode: EvalMode) -> None:
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be a positive integer")
    if eval_mode == EvalMode.TRAINED_CHECKPOINT:
        if not args.checkpoints_dir:
            raise ValueError("--checkpoints-dir is required for trained_checkpoint mode")
        return
    if not args.base_model:
        raise ValueError(f"--base-model is required for {eval_mode.value} mode")
    if not args.workflow_types:
        raise ValueError(f"--workflow-types is required for {eval_mode.value} mode")
    if eval_mode != EvalMode.BASE_MODEL and not args.single_agent_lora_path:
        raise ValueError(f"--single-agent-lora-path is required for {eval_mode.value} mode")


def main(args: argparse.Namespace) -> list[EvalResult]:
    """Evaluate every selected checkpoint; returns the results (also appended to ``--output-json``)."""
    from transformers import AutoTokenizer

    from marl_workflows.engine.openai_engine import LoRAOpenAIEngine
    from marl_workflows.eval.vllm_server import VLLMServerManager

    task = args.task_type
    if args.dataset is None:
        # The recipes' validation set (fork default_eval_dataset): aime2025 / deepcoder_primeintellect.
        args.dataset = next(r.val_dataset for (t, _), r in RECIPES.items() if t == task)
    length_mode = "training" if args.use_training_lengths else "eval"
    lengths = resolve_lengths(task, args.use_training_lengths)
    user_prompt_len = args.max_prompt_length is not None
    user_max_tokens = args.max_tokens is not None
    # The server is sized for the largest caps across the task's workflows (fork behaviour).
    max_prompt = args.max_prompt_length if user_prompt_len else max(p for p, _ in lengths.values())
    max_tokens = args.max_tokens if user_max_tokens else max(r for _, r in lengths.values())

    eval_mode = EvalMode(args.eval_mode)
    _validate_args(args, eval_mode)
    print(f"Checkpoint evaluation ({task}, {length_mode} lengths), mode {eval_mode.value}")

    checkpoints = _select_checkpoints(args, eval_mode)
    if not checkpoints:
        print("No checkpoints found!")
        return []
    for cp in checkpoints:
        print(f"  - {cp.experiment_name} step {cp.checkpoint_step}")

    dataset = load_eval_dataset(args.dataset, args.max_samples, args.problem_range)
    print(f"Dataset {args.dataset}: {len(dataset)} problems")

    if not args.no_write:
        existing = load_existing_results(args.output_json)
        before = len(checkpoints)
        checkpoints = [
            cp
            for cp in checkpoints
            if get_checkpoint_key(cp, args.dataset, eval_mode.value, args.n_rollouts, length_mode, len(dataset))
            not in existing
        ]
        if before != len(checkpoints):
            print(f"Skipping {before - len(checkpoints)} already evaluated checkpoint(s)")
        if not checkpoints:
            print("All checkpoints already evaluated. Nothing to do.")
            return []

    if args.gpu_type is None and args.n_parallel is None:
        args.gpu_type = detect_gpu_type()

    results: list[EvalResult] = []
    server: VLLMServerManager | None = None

    def _on_signal(_sig: int, _frame: object) -> None:
        print("\nReceived interrupt signal, cleaning up...")
        if server is not None:
            server.stop()
        sys.exit(1)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    for base_model, group in group_checkpoints_by_model(checkpoints).items():
        print(f"\n{'#' * 60}\nProcessing model: {base_model} ({len(group)} checkpoints)\n{'#' * 60}")
        try:
            server = VLLMServerManager(
                model=base_model,
                tensor_parallel_size=args.tensor_parallel,
                data_parallel_size=args.data_parallel,
                gpu_memory_utilization=args.gpu_memory_utilization,
                port=args.port,
                max_loras=args.max_loras,
                max_lora_rank=args.max_lora_rank,
                max_model_len=max_prompt + max_tokens,
                enable_lora=eval_mode != EvalMode.BASE_MODEL,
            )
            server.start()
            sampling: dict[str, Any] = {"temperature": args.temperature}
            if args.save_token_logprobs:
                sampling.update({"logprobs": 1, "extra_body": {"return_token_ids": True}})
            engine = LoRAOpenAIEngine(
                base_model=base_model,
                tokenizer=AutoTokenizer.from_pretrained(base_model, trust_remote_code=True),
                base_url=f"http://localhost:{args.port}/v1",
                max_prompt_length=max_prompt,
                max_response_length=max_tokens,
                sampling_params=sampling,
                disable_thinking=True,
                api_retries=1,  # surface server errors immediately (fork setting)
            )
            for cp in sorted(group, key=lambda c: c.checkpoint_step):
                wf_lengths = lengths.get(cp.workflow_type)
                if wf_lengths and not user_prompt_len:
                    engine.max_prompt_length = wf_lengths[0]
                if wf_lengths and not user_max_tokens:
                    engine.max_response_length = wf_lengths[1]
                engine.max_model_length = engine.max_prompt_length + engine.max_response_length - 1
                n_parallel = args.n_parallel or compute_n_parallel(args.gpu_type, cp.model_size, cp.workflow_type)
                try:
                    result = asyncio.run(
                        evaluate_checkpoint(
                            cp,
                            server,
                            engine,
                            dataset,
                            task=task,
                            n_parallel=n_parallel,
                            eval_mode=eval_mode,
                            single_agent_lora_path=args.single_agent_lora_path,
                            trajectory_output_dir=args.trajectory_output_dir,
                            dataset_name=args.dataset,
                            n_rollouts=args.n_rollouts,
                            uid_offset=args.problem_range[0] if args.problem_range else 0,
                            length_mode=length_mode,
                            save_token_logprobs=args.save_token_logprobs,
                        )
                    )
                    results.append(result)
                    if not args.no_write:
                        save_results_to_json([result], args.output_json)
                except Exception as e:  # one failed checkpoint must not stop the others
                    print(f"Error evaluating {cp.experiment_name} step {cp.checkpoint_step}: {e}")
                    traceback.print_exc()
                server.unload_all_loras()
        finally:
            if server is not None:
                server.stop()
                server = None

    print(f"\nEVALUATION COMPLETE: {len(results)} checkpoint(s) evaluated ({eval_mode.value})")
    for r in sorted(results, key=lambda x: (x.experiment_name, x.checkpoint_step)):
        print(f"{r.experiment_name:<55} {r.checkpoint_step:>6} {r.accuracy:>8.2%} n={r.num_total}")
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command line (fork flags, minus ``--eval-opt-controller``, plus ``--no-write``)."""
    p = argparse.ArgumentParser(description="Evaluate checkpoints with a LoRA-enabled vLLM server (math or deepcoder)")
    p.add_argument("--task-type", choices=list(TASKS), default="math")
    p.add_argument("--eval-mode", choices=[m.value for m in EvalMode], default=EvalMode.TRAINED_CHECKPOINT.value)
    p.add_argument("--checkpoints-dir", default=None, help="directory of experiment dirs (trained_checkpoint mode)")
    p.add_argument("--base-model", default=None, help="base model (other modes); override in trained_checkpoint mode")
    p.add_argument("--workflow-types", nargs="+", default=None, help="recipe workflow ids (base/transfer modes)")
    p.add_argument("--single-agent-lora-path", default=None, help="SA-RL adapter dir (transfer modes)")
    p.add_argument("--last-checkpoint-only", action="store_true")
    p.add_argument("--output-json", default="eval_results.jsonl", help="JSON Lines file; results are appended")
    p.add_argument("--no-write", action="store_true", help="do not append results to --output-json")
    p.add_argument("--dataset", default=None, help="registered dataset (test split); default: the recipe val set")
    p.add_argument("--experiment-filter", default=None, help="regex on experiment directory names")
    p.add_argument("--step-filter", type=int, nargs="+", default=None)
    p.add_argument("--eval-interval", type=int, default=10, help="only steps divisible by this (default 10)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=None, help="response cap (default: per recipe)")
    p.add_argument("--max-prompt-length", type=int, default=None, help="prompt cap (default: per recipe)")
    p.add_argument("--use-training-lengths", action="store_true", help="recipe training caps instead of eval caps")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--n-parallel", type=int, default=None, help="concurrent workflow instances (default: auto)")
    p.add_argument("--gpu-type", choices=GPU_TYPE_CHOICES, default=None)
    p.add_argument("--tensor-parallel", type=int, default=1)
    p.add_argument("--data-parallel", type=int, default=1)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-loras", type=int, default=5)
    p.add_argument("--max-lora-rank", type=int, default=64)
    p.add_argument("--trajectory-output-dir", default=None, help="write one JSON per episode under this dir")
    p.add_argument("--problem-range", type=int, nargs=2, default=None, metavar=("START", "END"))
    p.add_argument("--n-rollouts", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=None, help="evaluate only the first N problems")
    p.add_argument("--save-token-logprobs", action="store_true", help="keep token ids / log-probs in trajectories")
    return p.parse_args(argv)


def cli() -> None:
    """Console entry point (``marl-eval``)."""
    main(parse_args())


if __name__ == "__main__":
    cli()
