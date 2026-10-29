"""Derive the multi-agent keys that verl's workers and the vLLM server read (contract C10).

Runs on the driver before the config is shipped to the Ray task runner. It never resolves
interpolations (``OmegaConf.resolve`` happens in the task runner, as upstream). Replaces the
research fork's ``rllm/trainer/verl/train_agent_ppo.py:84-89, 186-197`` and the dataset and
executor defaults of its Code entry scripts.

Optional vLLM log-prob mode: ``+actor_rollout_ref.rollout.engine_kwargs.vllm.logprobs_mode=processed_logprobs``
makes vLLM return log-probs after temperature scaling. It is off by default, which is the
paper's behaviour; with it, the logged per-role ``chi2_token``/``kl``/``training_ppl`` compare
the trainer's and vLLM's log-probs at the same temperature instead of T=0.7 against T=1, so
those metrics change meaning (they stop tracking sampling entropy).
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf, open_dict

from marl_workflows.contracts import validate_agent_names
from marl_workflows.recipes import Recipe
from marl_workflows.utils import allocated_cpu_count

__all__ = ["VLLM_LOGPROBS_MODES", "inject_multi_agent_config", "resolve_initial_lora_weights"]

# vLLM 0.11 ``logprobs_mode`` values.
VLLM_LOGPROBS_MODES = ("raw_logprobs", "raw_logits", "processed_logprobs", "processed_logits")


def _to_plain(value: Any) -> Any:
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def resolve_initial_lora_weights(config: DictConfig, recipe: Recipe) -> dict[str, str]:
    """``rllm.workflow.initial_lora_weights`` as a plain dict, with the recipe's role remap applied.

    The remap renames a source role absent from the workflow (SA-RL adapters are saved as
    ``generator``) to the role that should start from it (``worker`` in Orchestrator-Workers).
    """
    raw = config.rllm.workflow.get("initial_lora_weights") or {}
    init = {str(k): str(v) for k, v in (_to_plain(raw) or {}).items() if v is not None}
    names = list(config.trainer.agent_names)
    for src, dst in recipe.initial_lora_remap:
        if src in init and src not in names:
            if dst in init:
                raise ValueError(f"initial_lora_weights sets both {src!r} and {dst!r}, but {src!r} maps to {dst!r}")
            init[dst] = init.pop(src)
    return init


def _deepcoder_defaults(config: DictConfig) -> None:
    workflow = config.rllm.workflow
    if int(workflow.get("code_executor_workers", 0) or 0) <= 0:
        # Mode 1 with one process per allocated CPU, as the paper launcher set it.
        workflow.code_executor_workers = allocated_cpu_count()


def _resolve_dataset_names(config: DictConfig, recipe: Recipe) -> None:
    marl = config.marl
    dataset_name = config.data.get("dataset_name")
    if recipe.task == "deepcoder" and dataset_name:
        # A user-set data.dataset_name names both splits, as in the fork's Code entry scripts.
        train_name = val_name = str(dataset_name)
    else:
        train_name = marl.get("train_dataset") or recipe.train_dataset
        val_name = marl.get("val_dataset") or recipe.val_dataset
    marl.train_dataset = str(train_name)
    marl.val_dataset = str(val_name)
    if recipe.task == "deepcoder" and not dataset_name:
        config.data.dataset_name = str(train_name)


def _check_unsupported(config: DictConfig) -> None:
    workflow = config.rllm.workflow
    if bool(workflow.get("enable_test_loop", False)):
        raise ValueError("rllm.workflow.enable_test_loop=True is not supported (the paper runs disable the test loop)")
    if int(workflow.get("max_concurrent_code_execs", 0) or 0) > 0:
        raise ValueError(
            "rllm.workflow.max_concurrent_code_execs>0 (batch test scheduler) is not supported; "
            "use rllm.workflow.code_executor_workers"
        )
    vllm_kwargs = config.actor_rollout_ref.rollout.get("engine_kwargs", {}).get("vllm", {}) or {}
    mode = vllm_kwargs.get("logprobs_mode")
    if mode is not None and mode not in VLLM_LOGPROBS_MODES:
        raise ValueError(f"engine_kwargs.vllm.logprobs_mode must be one of {VLLM_LOGPROBS_MODES}, got {mode!r}")


def inject_multi_agent_config(config: DictConfig, recipe: Recipe) -> None:
    """Fill in agent names, policy mode, ``max_loras``, warm-start adapters and dataset names in place."""
    with open_dict(config):
        sp = bool(config.trainer.share_policy)
        if [str(n) for n in config.trainer.agent_names] == ["default"]:
            config.trainer.agent_names = list(recipe.agent_names)
        names = validate_agent_names(config.trainer.agent_names, sp)
        if sp not in recipe.allowed_share_policy:
            raise ValueError(
                f"recipe ({recipe.task}, {recipe.workflow}) does not allow share_policy={sp}; "
                f"allowed: {list(recipe.allowed_share_policy)}"
            )
        _check_unsupported(config)

        arr = config.actor_rollout_ref
        arr.agent_names = names
        arr.share_policy = sp
        if int(arr.model.get("lora_rank", 0) or 0) > 0:
            # One vLLM LoRA slot per role; set under SP too (one adapter, as in the fork).
            if arr.rollout.get("engine_kwargs") is None:
                arr.rollout.engine_kwargs = {}
            if arr.rollout.engine_kwargs.get("vllm") is None:
                arr.rollout.engine_kwargs.vllm = {}
            arr.rollout.engine_kwargs.vllm.max_loras = len(names)

        init = resolve_initial_lora_weights(config, recipe)
        if init:
            if sp:
                arr.model.lora_adapter_path = init.get("generator") or next(iter(init.values()))
            else:
                unknown = sorted(set(init) - set(names))
                if unknown:
                    raise ValueError(f"initial_lora_weights roles {unknown} are not in trainer.agent_names {names}")
                arr.initial_lora_adapters = init

        if config.get("marl") is None:
            config.marl = {}
        _resolve_dataset_names(config, recipe)
        if recipe.task == "deepcoder":
            _deepcoder_defaults(config)
