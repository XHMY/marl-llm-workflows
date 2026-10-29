"""Names and rules shared across module boundaries (the single source of truth).

Every literal that one package produces and another consumes lives here: trajectory and
role grammar, LoRA ids, checkpoint layout, metric names, internal carrier keys, stdout
markers, experiment names and environment variables. The module is pure Python and must
import neither torch nor rllm nor verl, so that the launcher, the evaluator and the tests
can use it without the training stack.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

__all__ = [
    "ACTOR_SUBDIR",
    "ENV_DATA_DIR",
    "ENV_INIT_WEIGHT_DIR",
    "ENV_STRICT_LORA",
    "LATEST_ITER_FILE",
    "LORA_INT_ID_BASE",
    "LORA_PATH_PLACEHOLDER",
    "LOSS_WEIGHT_PREFIX",
    "META_KL_OVERRIDE",
    "META_TEMPERATURE",
    "OPENAI_MODEL_KEY",
    "OPTIM_COUNTER_KEYS",
    "POLICY_TAG",
    "ROLE_RE",
    "RUN_COMPLETE_FMT",
    "SAMPLING_LORA_KEY",
    "SHARED_ADAPTER",
    "STEP_DIR_FMT",
    "TASKS",
    "TASK_SUFFIX",
    "TRAINING_METADATA_FILE",
    "TRAJ_ID_SEP",
    "TRAJ_NAME_RE",
    "WALLCLOCK_STOP_FMT",
    "actor_ckpt_dir",
    "adapter_dirname",
    "expected_lora_int_ids",
    "experiment_name",
    "lora_int_id_for",
    "lora_strict_from_env",
    "rename_role_metrics",
    "role_from_name",
    "role_from_trajectory_id",
    "role_metric_key",
    "roles_from_trajectory_ids",
    "validate_agent_names",
    "validate_trajectory_name",
]

# --------------------------------------------------------------------------------------
# LoRA ids (C6, C7)
# --------------------------------------------------------------------------------------

# Must equal verl@v0.6.1 verl/workers/rollout/vllm_rollout/utils.py VLLM_LORA_INT_ID: the
# stock server builds LoRARequest(123) for the SP adapter, and IP ids extend from it.
LORA_INT_ID_BASE: Final[int] = 123
# Must equal verl's VLLM_LORA_PATH; vLLM needs a path string even for tensor-loaded adapters.
LORA_PATH_PLACEHOLDER: Final[str] = "simon_lora_path"
# PEFT adapter name used under SP. Under IP it still exists (dead) to keep the FSDP layout.
SHARED_ADAPTER: Final[str] = "default"

# --------------------------------------------------------------------------------------
# Trajectory and role grammar (C1, C2)
# --------------------------------------------------------------------------------------

# rLLM builds trajectory_id = f"{task_uuid}_{Trajectory.name}" (engine/agent_workflow_engine.py:268),
# so the role is recoverable only if the name itself never contains the separator.
TRAJ_ID_SEP: Final[str] = "_"
TRAJ_NAME_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z]+[0-9]*$")
ROLE_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z]+$")

_TRAILING_DIGITS = re.compile(r"\d+$")


def role_from_name(traj_name: str) -> str:
    """Strip trailing digits from a trajectory name (``generator0`` -> ``generator``)."""
    return _TRAILING_DIGITS.sub("", traj_name)


def role_from_trajectory_id(trajectory_id: str) -> str:
    """Return the role encoded in ``f"{task_uuid}_{name}"``.

    The task uuid may itself contain ``_`` or ``-``; only the last separator counts.
    """
    if TRAJ_ID_SEP not in trajectory_id:
        raise ValueError(f"trajectory_id {trajectory_id!r} has no {TRAJ_ID_SEP!r} separator")
    return role_from_name(trajectory_id.rsplit(TRAJ_ID_SEP, 1)[1])


def roles_from_trajectory_ids(ids: Iterable[str]) -> list[str]:
    """Vectorised :func:`role_from_trajectory_id` (accepts lists and numpy object arrays)."""
    return [role_from_trajectory_id(str(i)) for i in ids]


def validate_trajectory_name(name: str) -> None:
    """Raise ``ValueError`` unless ``name`` matches :data:`TRAJ_NAME_RE`."""
    if not isinstance(name, str) or not TRAJ_NAME_RE.match(name):
        raise ValueError(
            f"Invalid trajectory name {name!r}: must match {TRAJ_NAME_RE.pattern} "
            "(letters, then optional digits; no '_' or ':')"
        )


def validate_agent_names(agent_names: Iterable[str], share_policy: bool) -> list[str]:
    """Check the role list and return it as a plain list.

    Always: non-empty, unique, each name letters only. Under IP additionally: ``default`` is
    forbidden (the dead shared adapter would be selected and nothing would train), and no
    name may be a substring of another, because the worker sets ``requires_grad`` by a
    substring match of the role on parameter names.
    """
    names = [str(n) for n in agent_names]
    if not names:
        raise ValueError("agent_names must be non-empty")
    if len(set(names)) != len(names):
        raise ValueError(f"agent_names must be unique, got {names}")
    for n in names:
        if not ROLE_RE.match(n):
            raise ValueError(f"Invalid agent name {n!r}: must match {ROLE_RE.pattern} (letters only)")
    if not share_policy:
        if SHARED_ADAPTER in names:
            raise ValueError(f"agent name {SHARED_ADAPTER!r} is reserved under IP (share_policy=False)")
        for a in names:
            for b in names:
                if a != b and a in b:
                    raise ValueError(
                        f"agent name {a!r} is a substring of {b!r}; IP adapter selection would be ambiguous"
                    )
    return names


# --------------------------------------------------------------------------------------
# LoRA id mapping (C5, C6, C7)
# --------------------------------------------------------------------------------------


def lora_int_id_for(
    agent_name: str | None,
    agent_names: Sequence[str],
    share_policy: bool,
    *,
    strict: bool = True,
) -> int | None:
    """Map a role to its vLLM LoRA id.

    SP returns ``None`` (the server falls back to the single id 123). IP returns
    ``agent_names.index(agent_name) + 123``. Under IP a missing ``agent_name`` raises when
    ``strict`` and returns ``None`` otherwise (the fork's silent base-model fallback). An
    unknown name always raises.
    """
    if share_policy:
        return None
    names = list(agent_names)
    if agent_name is None:
        if strict:
            raise ValueError("agent_name is required under IP (share_policy=False); set MARL_LORA_STRICT=0 to allow")
        return None
    if agent_name not in names:
        raise ValueError(f"Unknown agent_name {agent_name!r}; expected one of {names}")
    return names.index(agent_name) + LORA_INT_ID_BASE


def expected_lora_int_ids(agent_names: Sequence[str], share_policy: bool) -> list[int]:
    """Ids that must be registered in vLLM after every weight sync."""
    if share_policy:
        return [LORA_INT_ID_BASE]
    return [LORA_INT_ID_BASE + i for i in range(len(list(agent_names)))]


# --------------------------------------------------------------------------------------
# Checkpoint layout (C14)
# --------------------------------------------------------------------------------------

STEP_DIR_FMT: Final[str] = "global_step_{step}"
ACTOR_SUBDIR: Final[str] = "actor"
TRAINING_METADATA_FILE: Final[str] = "training_metadata.json"
LATEST_ITER_FILE: Final[str] = "latest_checkpointed_iteration.txt"


def adapter_dirname(role: str | None) -> str:
    """``None`` -> ``lora_adapter`` (SP); a role -> ``lora_adapter_{role}`` (IP)."""
    return "lora_adapter" if role is None else f"lora_adapter_{role}"


def actor_ckpt_dir(default_local_dir: str, step: int) -> str:
    """``{default_local_dir}/global_step_{step}/actor``."""
    return os.path.join(default_local_dir, STEP_DIR_FMT.format(step=step), ACTOR_SUBDIR)


# --------------------------------------------------------------------------------------
# Metric names (C13)
# --------------------------------------------------------------------------------------


def role_metric_key(role: str, verl_key: str) -> str:
    """``actor/pg_loss`` -> ``actor/{role}/pg_loss`` (only the last path component is kept)."""
    return f"actor/{role}/{verl_key.split('/')[-1]}"


def rename_role_metrics(role: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Apply :func:`role_metric_key` to every key.

    Keys that share a leaf collide; the later key in iteration order wins, which is the
    fork's behaviour (it assigns into one dict in the same order).
    """
    out: dict[str, Any] = {}
    for k, v in metrics.items():
        out[role_metric_key(role, k)] = v
    return out


OPTIM_COUNTER_KEYS: Final[tuple[str, str, str]] = (
    "actor/num_mini_batches",
    "actor/optim_steps_this_call",
    "actor/optim_step_count",
)
LOSS_WEIGHT_PREFIX: Final[str] = "actor/loss_weight/"

# --------------------------------------------------------------------------------------
# DataProto meta_info keys and internal carriers (C11, C20)
# --------------------------------------------------------------------------------------

META_TEMPERATURE: Final[str] = "temperature"
META_KL_OVERRIDE: Final[str] = "kl_loss_coef_override"
# Carried inside sampling_params from MultiLoRAVerlEngine to MultiLoRAServerManager, which
# pops it; it must never reach vLLM.
SAMPLING_LORA_KEY: Final[str] = "__marl_lora_int_id__"
# Carried inside sampling_params from LoRAOpenAIEngine to its client proxy, which pops it.
OPENAI_MODEL_KEY: Final[str] = "__marl_model__"

# --------------------------------------------------------------------------------------
# Stdout markers read by SLURM chain scripts (C15)
# --------------------------------------------------------------------------------------

RUN_COMPLETE_FMT: Final[str] = "RUN_COMPLETE global_step={step}"
WALLCLOCK_STOP_FMT: Final[str] = "WALLCLOCK_STOP completed_step={step} saved={saved}"

# --------------------------------------------------------------------------------------
# Experiment names (C16)
# --------------------------------------------------------------------------------------

POLICY_TAG: Final[dict[bool, str]] = {False: "multi_lora", True: "share_policy"}
TASKS: Final[tuple[str, str]] = ("math", "deepcoder")
TASK_SUFFIX: Final[dict[str, str]] = {"math": "math", "deepcoder": "deepcoder_primeintellect"}


def experiment_name(
    prefix: str,
    model_tag: str,
    share_policy: bool,
    task: str,
    suffix: str | None = None,
) -> str:
    """``{prefix}-{model_tag}-{multi_lora|share_policy}-{math|deepcoder_primeintellect}[-{suffix}]``."""
    if task not in TASK_SUFFIX:
        raise ValueError(f"Unknown task {task!r}; expected one of {list(TASK_SUFFIX)}")
    name = f"{prefix}-{model_tag}-{POLICY_TAG[bool(share_policy)]}-{TASK_SUFFIX[task]}"
    return f"{name}-{suffix}" if suffix else name


# --------------------------------------------------------------------------------------
# Environment variables (C21)
# --------------------------------------------------------------------------------------

# "0" restores the fork's silent base-model fallback for unknown / missing LoRA ids.
ENV_STRICT_LORA: Final[str] = "MARL_LORA_STRICT"
ENV_DATA_DIR: Final[str] = "MARL_DATA_DIR"
ENV_INIT_WEIGHT_DIR: Final[str] = "MARL_INIT_WEIGHT_DIR"


def lora_strict_from_env(env: Mapping[str, str] | None = None) -> bool:
    """Strict LoRA routing unless ``MARL_LORA_STRICT`` is ``"0"`` (default strict)."""
    env = os.environ if env is None else env
    return env.get(ENV_STRICT_LORA, "1") != "0"
