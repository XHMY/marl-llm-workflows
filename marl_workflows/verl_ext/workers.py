"""``MultiAgentAsyncActorRolloutRefWorker``: verl's hybrid actor/rollout worker with one LoRA adapter per role.

Under IP (``share_policy=False``) every role in ``actor_rollout_ref.agent_names`` gets its own PEFT adapter,
created before FSDP wrapping (patch P1): the first role is PEFT's initial adapter and the others are added to
it, so the model carries exactly the role adapters and no adapter named ``default``. All adapters live in one
model, one AdamW optimizer and one LR scheduler; ``set_active_lora`` selects the adapter that the next
forward/backward uses and freezes the others, so inactive adapters get ``None`` gradients and AdamW leaves
them untouched. Every weight sync registers each adapter in vLLM under ``agent_names.index(role) + 123``.
The FSDP checkpoint holds the role adapters only; upstream's ``actor/lora_adapter/`` export (which reads the
``default`` adapter) is skipped, and the trainer writes ``lora_adapter_{role}/`` per role instead. Under SP
nothing differs from upstream: the single ``default`` adapter is registered under 123 on every sync, the
first one included, and upstream's save writes ``actor/lora_adapter/``.

The guards on ``len(agent_names) >= 1`` (contract C9) keep a single-role IP run (the SA-RL baseline) on the
multi-adapter path. Importing this module patches nothing (``patches.applied_patches()`` stays empty).
"""

from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import asdict
from typing import Any

import torch.distributed as dist
from omegaconf import DictConfig
from peft import PeftModel, TaskType
from safetensors.torch import save_file
from torch.distributed.tensor import DTensor
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_device_id, get_device_name, get_torch_device, set_expandable_segments
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import fsdp_version, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu, replace_lora_wrapper
from verl.utils.logger import log_with_rank
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.model import convert_weight_keys
from verl.utils.profiler import log_gpu_memory_usage
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
from verl.workers.rollout.vllm_rollout.utils import VLLM_LORA_INT_ID

from marl_workflows import contracts
from marl_workflows.verl_ext import patches
from marl_workflows.verl_ext.actor import MultiAgentDPActor
from marl_workflows.verl_ext.lora_utils import collect_lora_params, layered_summon_lora_params

__all__ = ["MultiAgentAsyncActorRolloutRefWorker", "param_belongs_to_adapter"]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def param_belongs_to_adapter(param_name: str, adapter_name: str) -> bool:
    """True if ``adapter_name`` is a path component of ``param_name``.

    PEFT stores adapter weights under ``...lora_A.<adapter>.weight`` (FSDP1 flat params under
    ``...lora_A.<adapter>._flat_param``), so matching the component is exact. The research fork used a
    substring test (``role in name``); ``init_model`` checks that both select the same parameters.
    """
    return adapter_name in param_name.split(".")


def _materialize(params: dict[str, Any]) -> Any:
    device = get_device_id()  # used when fsdp2 set cpu_offload_policy
    return (
        (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
        for name, param in params.items()
    )


class MultiAgentAsyncActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    """Subclass of ``verl@v0.6.1:verl/workers/fsdp_workers.py:1928`` (base ``ActorRolloutRefWorker`` ``:134``)."""

    def __init__(self, config: DictConfig, role: str, **kwargs: Any) -> None:
        super().__init__(config, role, **kwargs)
        self.current_active_lora: str | None = None
        self.rollout_lora_registered: bool = False
        self.rollout_lora_int_ids: list[int] = []
        self._marl_adapters_added = False

    # ------------------------------------------------------------------ config accessors

    def _marl_agent_names(self) -> list[str]:
        return list(self.config.get("agent_names", []) or [])

    def _marl_share_policy(self) -> bool:
        return bool(self.config.get("share_policy", False))

    def _marl_multi_adapter(self) -> bool:
        """GUARD 1 (C9): one adapter per role."""
        return len(self._marl_agent_names()) >= 1 and not self._marl_share_policy()

    # ------------------------------------------------------------------ adapter creation

    def _build_model_optimizer(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("role", "actor") != "actor" or not self._is_lora or not self._marl_multi_adapter():
            return super()._build_model_optimizer(*args, **kwargs)
        contracts.validate_agent_names(self._marl_agent_names(), share_policy=False)
        if self.config.model.get("lora_adapter_path") is not None:
            raise ValueError(
                "actor_rollout_ref.model.lora_adapter_path loads a single 'default' adapter (SP warm start); "
                "under IP use actor_rollout_ref.initial_lora_adapters.<role> instead"
            )
        self._marl_adapters_added = False
        with patches.scoped_multi_adapter_peft(self._build_role_adapters):
            built = super()._build_model_optimizer(*args, **kwargs)
        if not self._marl_adapters_added:
            raise RuntimeError("P1 did not run: upstream _build_model_optimizer no longer calls get_peft_model")
        return built

    def _build_role_adapters(self, get_peft_model: Any, model: Any, lora_config: Any) -> Any:
        """Wrap ``model`` with one adapter per role and return the ``PeftModel`` (P1 builder; IP only).

        The first role of ``agent_names`` is PEFT's initial adapter (``get_peft_model(..., adapter_name=role)``,
        or ``PeftModel.from_pretrained`` for a warm start) and the others are added in ``agent_names`` order
        (``add_adapter``, or ``load_adapter`` for a warm start), so no adapter named ``default`` exists. Each
        fresh role gets its own deep copy of the config upstream built, as the fork's
        ``LoraConfig(**lora_config)`` per adapter (fork ``fsdp_workers.py:470-505``).
        ``initial_lora_adapters[role]`` warm-starts a role from a saved PEFT adapter directory.

        Unlike the fork, no frozen ``default`` adapter is created first, so the random LoRA-A initialization
        of role ``k`` draws what the fork's adapter ``k - 1`` (``default`` for ``k = 0``) drew, and the FSDP
        shards hold the role adapters only (the fork's IP shards do not load; see ``load_checkpoint``).
        """
        names = self._marl_agent_names()
        initial = self.config.get("initial_lora_adapters", None) or {}
        use_shm = self.config.model.get("use_shm", False)
        if self.rank == 0:
            print(f"[Multi-Agent LoRA] Initializing {len(names)} adapters: {names}")
        peft_model: Any = None
        for name in names:
            path = initial.get(name) if initial else None
            if path:
                local_path = copy_to_local(path, use_shm=use_shm)
                if peft_model is None:
                    peft_model = PeftModel.from_pretrained(model, local_path, adapter_name=name, is_trainable=True)
                else:
                    peft_model.load_adapter(local_path, adapter_name=name, is_trainable=True)
                if isinstance(peft_model.peft_config[name].task_type, str):
                    peft_model.peft_config[name].task_type = TaskType.CAUSAL_LM
                if self.rank == 0:
                    print(f"[Multi-Agent LoRA] Loaded pre-trained adapter '{name}' from: {path}")
            else:
                if peft_model is None:
                    peft_model = get_peft_model(model, copy.deepcopy(lora_config), adapter_name=name)
                else:
                    peft_model.add_adapter(adapter_name=name, peft_config=copy.deepcopy(lora_config))
                if self.rank == 0:
                    print(f"[Multi-Agent LoRA] Added fresh adapter: {name}")
        if list(peft_model.peft_config) != names:
            raise RuntimeError(
                f"[Multi-Agent LoRA] adapters {list(peft_model.peft_config)} differ from agent_names {names}"
            )
        # The FSDP wrap policy and the optimizer's parameter list are built from requires_grad, so every
        # LoRA parameter (each belongs to a role; there is no dead adapter) is trainable here;
        # set_active_lora narrows it to one role per update.
        for pname, param in peft_model.named_parameters():
            param.requires_grad = "lora" in pname
        self.current_active_lora = names[0]
        self._marl_adapters_added = True
        if self.rank == 0:
            print(f"[Multi-Agent LoRA] All adapters initialized. Active adapter: {names[0]}")
        return peft_model

    def _marl_check_role_params(self) -> None:
        """Each role selects at least one parameter, only LoRA ones, the same set as the fork's rule, and every
        LoRA parameter belongs to some role (no adapter outside ``agent_names``, such as ``default``)."""
        param_names = [name for name, _ in self.actor_module_fsdp.named_parameters()]
        roles = self._marl_agent_names()
        for role in roles:
            selected = [n for n in param_names if param_belongs_to_adapter(n, role)]
            fork_selected = [n for n in param_names if role in n]
            if not selected or any("lora" not in n for n in selected) or selected != fork_selected:
                bad = sorted(set(fork_selected) ^ set(selected)) or [n for n in selected if "lora" not in n]
                raise ValueError(
                    f"[Multi-Agent LoRA] role {role!r} does not select exactly its LoRA parameters "
                    f"({len(selected)} selected; e.g. {bad[:3]})"
                )
        stray = [n for n in param_names if "lora" in n and not any(param_belongs_to_adapter(n, r) for r in roles)]
        if stray:
            raise ValueError(f"[Multi-Agent LoRA] LoRA parameters outside agent_names {roles}: e.g. {stray[:3]}")

    # ------------------------------------------------------------------ RPCs

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self) -> None:
        patches.apply_worker_patches()
        super().init_model()
        if self._is_actor:
            # Same optimizer object, so the checkpoint manager built by super() stays valid.
            self.actor = MultiAgentDPActor(
                config=self.actor.config, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer
            )
            if self._is_lora and self._marl_multi_adapter():
                self._marl_check_role_params()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_active_lora(self, agent_role: str, lora_config: dict | None = None) -> None:
        """Make ``agent_role``'s adapter the active and only trainable one (GUARD 3; fork semantics).

        ``lora_config`` is unused (kept for the fork's call signature). Parameters are matched to the role by
        PEFT adapter name (a path component of the parameter name) rather than by the fork's substring test;
        ``init_model`` verifies that the two select the same parameters.
        """
        names = self._marl_agent_names()
        if not self._is_lora or not names or len(names) < 1:
            return
        if not self._is_actor:
            return
        if agent_role not in names:
            raise ValueError(f"Adapter '{agent_role}' not found. Available adapters: {names}")
        if self.rank == 0:
            print(f"[Multi-Agent LoRA] Switching active adapter to: {agent_role}")
        peft_model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        peft_model.set_adapter(agent_role)
        for name, param in self.actor_module_fsdp.named_parameters():
            param.requires_grad = param_belongs_to_adapter(name, agent_role)
        self.current_active_lora = agent_role

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_single_lora_adapter(self, agent_name: str, save_path: str, global_step: int = 0) -> None:
        """Write ``agent_name``'s adapter as a standard PEFT adapter directory (fork ``:1064-1124``).

        ``adapter_model.safetensors`` keys have the adapter name stripped
        (``base_model.model.model.layers.L.self_attn.q_proj.lora_A.weight``), so the directory loads with
        ``PeftModel.from_pretrained`` or as a vLLM LoRA module. Unlike the fork, the active adapter is not
        switched (that had no effect on what is written).
        """
        if not self._is_actor or not self._is_lora:
            return
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        peft_model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        if agent_name not in peft_model.peft_config:
            raise ValueError(f"Adapter '{agent_name}' not found. Available: {list(peft_model.peft_config)}")
        peft_config: dict[str, Any] = {}
        if dist.get_rank() == 0:
            os.makedirs(save_path, exist_ok=True)
            peft_config = asdict(peft_model.peft_config[agent_name])
            peft_config["task_type"] = peft_config["task_type"].value
            peft_config["peft_type"] = peft_config["peft_type"].value
            peft_config["target_modules"] = list(peft_config["target_modules"])
        try:
            if fsdp_version(self.actor_module_fsdp) > 0:
                self.actor_module_fsdp = self.actor_module_fsdp.to(get_device_name())
                lora_params = layered_summon_lora_params(self.actor_module_fsdp, adapter_name=agent_name)
                if dist.get_rank() == 0:
                    save_file(lora_params, os.path.join(save_path, "adapter_model.safetensors"))
                    with open(os.path.join(save_path, "adapter_config.json"), "w", encoding="utf-8") as f:
                        json.dump(peft_config, f, ensure_ascii=False, indent=4)
        except Exception as e:  # same policy as upstream save_checkpoint (fsdp_workers.py:1048-1072)
            log_with_rank(
                f"Save LoRA Adapter Error for {agent_name} ({e})",
                rank=dist.get_rank(),
                logger=logger,
                log_only_rank_0=True,
            )
        dist.barrier()
        log_with_rank(
            f"[rank-{self.rank}]: Saved LoRA adapter '{agent_name}' (step {global_step}) to: {save_path}",
            rank=dist.get_rank(),
            logger=logger,
            log_only_rank_0=True,
        )
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None) -> None:
        """FSDP checkpoint; under IP without upstream's ``lora_adapter/`` export (``fsdp_workers.py:1048-1072``).

        Upstream exports ``peft_config["default"]`` to ``{local_path}/lora_adapter/``; under IP that adapter does
        not exist (upstream would fail on ``asdict({})``) and the per-role directories are written by
        ``save_single_lora_adapter``. The FSDP part is upstream's ``:1037-1046`` and ``:1074-1075``. SP and
        non-LoRA runs call upstream unchanged.
        """
        if not (self._is_lora and self._marl_multi_adapter()):
            return super().save_checkpoint(
                local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
            )
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )
        dist.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False) -> None:
        """Upstream load; under IP a failure names the unsupported source (the research fork's IP shards).

        The fork's IP shards also hold its frozen ``default`` adapter, so their keys and FSDP flat-parameter
        layout differ from this worker's; initialise from the fork's ``lora_adapter_{role}`` directories with
        ``initial_lora_adapters`` instead.
        """
        try:
            return super().load_checkpoint(local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load)
        except (RuntimeError, KeyError, ValueError) as e:
            if local_path is None or not (self._is_lora and self._marl_multi_adapter()):
                raise
            raise RuntimeError(
                f"[Multi-Agent LoRA] could not load the IP checkpoint {local_path}. IP runs resume only from FSDP "
                "shards written by marl-llm-workflows; shards of the research fork carry a 'default' adapter and "
                "are not supported (initialise from its lora_adapter_{role} directories via "
                "actor_rollout_ref.initial_lora_adapters instead)."
            ) from e

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def probe_rollout_lora_state(self) -> dict[str, Any]:
        """Rollout LoRA registration state of this rank (the worker group returns one dict per rank)."""
        return {
            "rank": self.rank,
            "base_sync_done": bool(getattr(self, "base_sync_done", False)),
            "rollout_lora_registered": bool(getattr(self, "rollout_lora_registered", False)),
            "lora_int_ids": list(getattr(self, "rollout_lora_int_ids", [])),
        }

    # ------------------------------------------------------------------ weight sync

    async def rollout_mode(self) -> None:
        """Context switch to rollout mode (replaces ``fsdp_workers.py:651-732``; fork ``:720-749``)."""
        aggressive_empty_cache(force_sync=True)
        log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=logger)

        peft_model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        names = self._marl_agent_names()
        # GUARD 2 (C9)
        if hasattr(peft_model, "peft_config") and len(names) >= 1 and not self._marl_share_policy():
            await self._sync_multi_agent_lora_weights(peft_model, names)
        else:
            await self._sync_single_agent_weights(peft_model)

        self.base_sync_done = True
        # important: need to manually set the random states of each tp to be identical.
        self.torch_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.gen_random_states)

    async def _sync_single_agent_weights(self, peft_model: Any) -> None:
        """Push the weights without LoRA, or base + the ``default`` adapter (SP); port of fork ``:751-833``.

        With LoRA the adapter is collected and registered under 123 on every sync, the first included, so the
        first rollout after a ``lora_adapter_path`` warm start is served by base + adapter (upstream's first
        sync pushes only the base weights and registers no adapter). Base weights are pushed on the first sync
        and, with ``sleep_level == 2``, on every sync because vLLM discards them on sleep.
        """
        peft_config = None
        if hasattr(peft_model, "peft_config"):  # LoRA
            peft_config = peft_model.peft_config.get(contracts.SHARED_ADAPTER, None)
        need_base_sync = False
        if peft_config is None:
            params = convert_weight_keys(self.actor_module_fsdp.state_dict(), peft_model)
        else:
            layered_summon = self.config.rollout.get("layered_summon", False)
            # Collect while the model is still on GPU (before offload).
            adapter_params = collect_lora_params(
                module=self.actor_module_fsdp,
                layered_summon=layered_summon,
                base_sync_done=True,
                adapter_name=contracts.SHARED_ADAPTER,
            )
            adapter_params = convert_weight_keys(adapter_params, peft_model)
            need_base_sync = not self.base_sync_done or getattr(self.rollout, "sleep_level", None) == 2
            if need_base_sync:
                base_params = collect_lora_params(
                    module=self.actor_module_fsdp, layered_summon=layered_summon, base_sync_done=False
                )
                base_params = {replace_lora_wrapper(k, peft_config): v for k, v in base_params.items()}
                base_params = convert_weight_keys(base_params, peft_model)

        log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)
        set_expandable_segments(False)
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["weights"])
        log_gpu_memory_usage("After resume weights", logger=logger)

        if peft_config is None:
            await self.rollout.update_weights(
                _materialize(params), peft_config=None, base_sync_done=self.base_sync_done
            )
            del params
        else:
            if need_base_sync:
                await self.rollout.update_weights(_materialize(base_params), base_sync_done=False)
                del base_params
                aggressive_empty_cache(force_sync=False)
            # The rollout calls dict()/len() on a LoRA payload: pass the sized items() view.
            await self.rollout.update_weights(
                adapter_params.items(), peft_config=peft_config, base_sync_done=True, lora_int_id=VLLM_LORA_INT_ID
            )
            self.rollout_lora_int_ids = [VLLM_LORA_INT_ID]
            self.rollout_lora_registered = True
            del adapter_params

        log_gpu_memory_usage("After update_weights", logger=logger)
        aggressive_empty_cache(force_sync=True)
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["kv_cache"])
        log_gpu_memory_usage("After resume kv_cache", logger=logger)

    async def _sync_multi_agent_lora_weights(self, peft_model: Any, agent_names: list[str]) -> None:
        """Push base weights when needed and every role adapter under ``index + 123`` (fork ``:870-1003``)."""
        available = set(peft_model.peft_config.keys())
        missing = [n for n in agent_names if n not in available]
        if missing:
            raise ValueError(f"[Multi-Agent LoRA] Adapters {missing} not found in peft_config: {sorted(available)}")
        if self.rank == 0:
            logger.info(f"[Multi-Agent LoRA] Syncing {len(agent_names)} adapters, base_sync_done={self.base_sync_done}")

        # 1. Collect every adapter while the model is on GPU.
        layered_summon = self.config.rollout.get("layered_summon", False)
        all_adapter_params = {}
        for name in agent_names:
            params = collect_lora_params(
                module=self.actor_module_fsdp, layered_summon=layered_summon, base_sync_done=True, adapter_name=name
            )
            all_adapter_params[name] = convert_weight_keys(params, peft_model)

        # 2. Base weights on the first sync, and on every sync with sleep_level == 2 (vLLM discards them).
        first_config = peft_model.peft_config[agent_names[0]]
        need_base_sync = not self.base_sync_done or getattr(self.rollout, "sleep_level", None) == 2
        if need_base_sync:
            base_params = collect_lora_params(
                module=self.actor_module_fsdp,
                layered_summon=layered_summon,
                base_sync_done=False,
                adapter_name=agent_names[0],
            )
            base_params = {replace_lora_wrapper(k, first_config): v for k, v in base_params.items()}
            base_params = convert_weight_keys(base_params, peft_model)

        # 3. Offload and wake the engine's weights.
        log_gpu_memory_usage("Before offload_fsdp_model_to_cpu (multi-agent)", logger=logger)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After offload_fsdp_model_to_cpu (multi-agent)", logger=logger)
        set_expandable_segments(False)
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["weights"])
        log_gpu_memory_usage("After resume weights (multi-agent)", logger=logger)

        # 4. Push.
        if need_base_sync:
            await self.rollout.update_weights(_materialize(base_params), base_sync_done=False)
            del base_params
            aggressive_empty_cache(force_sync=False)
        for idx, name in enumerate(agent_names):
            lora_int_id = idx + VLLM_LORA_INT_ID  # == contracts.lora_int_id_for(name, agent_names, False)
            await self.rollout.update_weights(
                all_adapter_params[name].items(),
                peft_config=peft_model.peft_config[name],
                base_sync_done=True,
                lora_int_id=lora_int_id,
            )
            if self.rank == 0:
                logger.info(f"[Multi-Agent LoRA] Synced adapter '{name}' with lora_int_id={lora_int_id}")
            aggressive_empty_cache(force_sync=False)
        self.rollout_lora_int_ids = [idx + VLLM_LORA_INT_ID for idx in range(len(agent_names))]
        self.rollout_lora_registered = True
        del all_adapter_params

        log_gpu_memory_usage("After update_weights (multi-agent)", logger=logger)
        aggressive_empty_cache(force_sync=True)
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["kv_cache"])
        log_gpu_memory_usage("After resume kv_cache (multi-agent)", logger=logger)
