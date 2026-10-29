"""Checkpoint discovery for evaluation: experiment-name parsing and the adapter layout of C14.

Port of the research fork's ``dashboard/evaluate_checkpoints.py`` (``CheckpointInfo``
:142-153, ``parse_experiment_name`` :329-381, ``discover_checkpoints`` :384-486,
``group_checkpoints_by_model`` / ``filter_last_checkpoints`` / ``create_synthetic_checkpoints``
:489-598), with workflow types, agent names and task filters taken from ``recipes`` and the
directory names from ``contracts``.

Layout (C14): ``{checkpoints_dir}/{experiment}/global_step_{N}/actor/`` holding either
``lora_adapter/`` (SP, one shared adapter) or ``lora_adapter_{role}/`` for every role of the
recipe (IP). Older checkpoints without the ``actor/`` level are also recognized. A
``training_metadata.json`` whose ``base_model`` is a merged init weight (``qwen3_1.7b_s300``)
points the base model at ``$MARL_INIT_WEIGHT_DIR/<task>/<tag>``.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from marl_workflows.contracts import ACTOR_SUBDIR, TASK_SUFFIX, TRAINING_METADATA_FILE, adapter_dirname
from marl_workflows.recipes import RECIPES, Recipe, infer_task, recipe_for_experiment
from marl_workflows.utils import init_weight_dir

__all__ = [
    "MODEL_MAP",
    "CheckpointInfo",
    "EvalMode",
    "create_synthetic_checkpoints",
    "detect_adapter_layout",
    "discover_checkpoints",
    "extract_model_size",
    "filter_last_checkpoints",
    "group_checkpoints_by_model",
    "parse_experiment_name",
]

# Model size token of an experiment name -> HF base model id. Evaluation is the only consumer,
# so the table lives here rather than in contracts.
MODEL_MAP: dict[str, str] = {
    "0.6b": "Qwen/Qwen3-0.6B",
    "1.7b": "Qwen/Qwen3-1.7B",
    "4b": "Qwen/Qwen3-4B",
}

# Fork task_configs.py experiment_filter_include / _exclude: a directory is evaluated for a
# task only if its name contains the include token and not the exclude token.
_TASK_FILTERS: dict[str, tuple[str, str | None]] = {
    "math": ("math", "deepcoder"),
    "deepcoder": ("deepcoder", None),
}

_STEP_RE = re.compile(r"^global_step_(\d+)$")
_MODEL_SIZE_RE = re.compile(r"qwen3_(\d+\.?\d*b)", re.IGNORECASE)
_MODEL_NAME_RE = re.compile(r"qwen3_\d+\.?\d*b[^-]*", re.IGNORECASE)


class EvalMode(Enum):
    """Evaluation mode (fork :114-119)."""

    TRAINED_CHECKPOINT = "trained_checkpoint"  # the checkpoint's own adapters
    BASE_MODEL = "base_model"  # no adapter for any role
    SINGLE_AGENT_TRANSFER = "single_agent_transfer"  # one SA-RL adapter for every role
    SINGLE_AGENT_GENERATOR_TRANSFER = "single_agent_generator_transfer"  # SA-RL adapter for the solving roles only


@dataclass
class CheckpointInfo:
    """One checkpoint (or synthetic step-0 entry) to evaluate."""

    experiment_name: str  # e.g. "voting-qwen3_0.6b-multi_lora-math"
    workflow_type: str  # recipe workflow id, e.g. "voting"
    model_size: str  # "0.6b", "1.7b", "4b"
    model_name: str  # "qwen3_0.6b", "qwen3_1.7b_s300"
    base_model: str  # HF id or a merged init-weight directory
    checkpoint_step: int
    actor_path: str | None  # directory holding lora_adapter{,_<role>} (actor/ or the step dir)
    share_policy: bool


def extract_model_size(base_model: str) -> str:
    """``Qwen/Qwen3-1.7B`` -> ``1.7b`` (``unknown`` if no size token)."""
    match = re.search(r"(\d+\.?\d*)[Bb]", base_model)
    return match.group(1).lower() + "b" if match else "unknown"


def parse_experiment_name(experiment_name: str, task: str | None = None) -> dict[str, Any] | None:
    """Workflow, model size / name, base model and SP hint of an experiment directory name.

    The workflow is resolved by ``recipes.recipe_for_experiment`` (first workflow id contained
    in the name; legacy ``orchestrator_workers`` -> propose; ``controlled_sa_hga`` ->
    ``single_agent``; ``evaluator_optimizer_v2`` skipped with a warning). ``None`` if the name
    is not an experiment of ``task`` (default: inferred from the name).
    """
    task = task or infer_task(experiment_name)
    recipe = recipe_for_experiment(experiment_name, task)
    if recipe is None:
        return None
    size_match = _MODEL_SIZE_RE.search(experiment_name)
    if size_match is None:
        return None
    model_size = size_match.group(1).lower()
    base_model = MODEL_MAP.get(model_size)
    if base_model is None:
        return None
    name_match = _MODEL_NAME_RE.search(experiment_name)
    model_name = name_match.group(0).lower() if name_match else f"qwen3_{model_size}"
    return {
        "workflow_type": recipe.workflow,
        "model_size": model_size,
        "model_name": model_name,
        "base_model": base_model,
        # A hint only; discovery decides SP vs IP from the adapter directories.
        "share_policy": "share_policy" in experiment_name,
        "recipe": recipe,
    }


def detect_adapter_layout(step_dir: Path, agent_names: Sequence[str]) -> tuple[Path, bool] | None:
    """``(actor_path, share_policy)`` for a step directory, or ``None`` if no complete adapter set.

    ``actor/`` is tried before the step directory itself; ``lora_adapter/`` means SP, a
    ``lora_adapter_{role}/`` for every role means IP.
    """
    for candidate in (step_dir / ACTOR_SUBDIR, step_dir):
        if (candidate / adapter_dirname(None)).exists():
            return candidate, True
        if all((candidate / adapter_dirname(a)).exists() for a in agent_names):
            return candidate, False
    return None


def _resolve_init_weight(parsed: dict[str, Any], experiment_dir: Path, task: str) -> None:
    metadata_path = experiment_dir / TRAINING_METADATA_FILE
    if not metadata_path.exists():
        return
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    meta_base_model = metadata.get("base_model", "")
    if not meta_base_model:
        return
    parsed["model_name"] = meta_base_model
    # A tag other than the plain size (e.g. qwen3_1.7b_s300) names a merged init weight.
    if meta_base_model != f"qwen3_{parsed['model_size']}":
        candidate = Path(init_weight_dir()) / task / meta_base_model
        if candidate.exists():
            parsed["base_model"] = str(candidate)


def discover_checkpoints(
    checkpoints_dir: str | Path,
    experiment_filter: str | None = None,
    step_filter: Sequence[int] | None = None,
    task_type: str = "math",
) -> list[CheckpointInfo]:
    """All evaluable checkpoints of ``task_type`` under ``checkpoints_dir`` (one level of experiments)."""
    root = Path(checkpoints_dir)
    if not root.exists():
        raise ValueError(f"Checkpoints directory not found: {checkpoints_dir}")
    include, exclude = _TASK_FILTERS[task_type]
    steps = set(step_filter) if step_filter else None

    checkpoints: list[CheckpointInfo] = []
    for experiment_dir in sorted(root.iterdir()):
        if not experiment_dir.is_dir():
            continue
        name = experiment_dir.name
        lowered = name.lower()
        if include not in lowered or (exclude and exclude in lowered):
            continue
        if experiment_filter and not re.search(experiment_filter, name):
            continue
        parsed = parse_experiment_name(name, task_type)
        if parsed is None:
            print(f"Warning: Could not parse experiment name: {name}")
            continue
        _resolve_init_weight(parsed, experiment_dir, task_type)
        recipe: Recipe = parsed["recipe"]

        for step_dir in sorted(experiment_dir.iterdir()):
            m = _STEP_RE.match(step_dir.name)
            if not step_dir.is_dir() or m is None:
                continue
            step = int(m.group(1))
            if steps is not None and step not in steps:
                continue
            layout = detect_adapter_layout(step_dir, recipe.agent_names)
            if layout is None:
                print(f"Warning: No valid LoRA adapter found in: {step_dir}")
                continue
            actor_path, share_policy = layout
            checkpoints.append(
                CheckpointInfo(
                    experiment_name=name,
                    workflow_type=parsed["workflow_type"],
                    model_size=parsed["model_size"],
                    model_name=parsed["model_name"],
                    base_model=parsed["base_model"],
                    checkpoint_step=step,
                    actor_path=str(actor_path),
                    share_policy=share_policy,
                )
            )
    return checkpoints


def group_checkpoints_by_model(checkpoints: Sequence[CheckpointInfo]) -> dict[str, list[CheckpointInfo]]:
    """Group by base model, so one vLLM server serves each group."""
    grouped: dict[str, list[CheckpointInfo]] = defaultdict(list)
    for cp in checkpoints:
        grouped[cp.base_model].append(cp)
    return dict(grouped)


def filter_last_checkpoints(checkpoints: Sequence[CheckpointInfo]) -> list[CheckpointInfo]:
    """Keep the highest step per experiment."""
    last: dict[str, CheckpointInfo] = {}
    for cp in checkpoints:
        if cp.experiment_name not in last or cp.checkpoint_step > last[cp.experiment_name].checkpoint_step:
            last[cp.experiment_name] = cp
    return list(last.values())


def create_synthetic_checkpoints(
    mode: EvalMode,
    task: str,
    base_model: str,
    workflow_types: Sequence[str],
    single_agent_lora_path: str | None = None,
) -> list[CheckpointInfo]:
    """Step-0 entries for the base-model and transfer modes (fork :541-593), one per workflow.

    Experiment names: ``{workflow}-qwen3_{size}-{mode tag}-{task suffix}`` with the mode tag
    ``base_model`` / ``single_agent_transfer`` / ``single_agent_generator_transfer``.
    """
    if mode == EvalMode.TRAINED_CHECKPOINT:
        raise ValueError("synthetic checkpoints are for the base-model and transfer modes")
    model_size = extract_model_size(base_model)
    suffix = TASK_SUFFIX[task]
    out: list[CheckpointInfo] = []
    for workflow in workflow_types:
        if (task, workflow) not in RECIPES:
            print(f"Warning: Unknown workflow type: {workflow}")
            continue
        out.append(
            CheckpointInfo(
                experiment_name=f"{workflow}-qwen3_{model_size}-{mode.value}-{suffix}",
                workflow_type=workflow,
                model_size=model_size,
                model_name=f"qwen3_{model_size}",
                base_model=base_model,
                checkpoint_step=0,
                actor_path=None if mode == EvalMode.BASE_MODEL else single_agent_lora_path,
                # One adapter (or none) for every mapped role, as in the fork.
                share_policy=True,
            )
        )
    return out
