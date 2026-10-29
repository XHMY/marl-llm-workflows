"""Route log-prob and update calls to one LoRA adapter per role (IP).

:class:`RoleRoutedWorkerGroup` wraps the actor/rollout ``RayWorkerGroup``. The rLLM training
loop keeps calling ``compute_log_prob``, ``compute_ref_log_prob`` and ``update_actor`` on the
whole batch; under IP the proxy splits the batch by role (parsed from ``trajectory_ids``,
contract C2), activates the role's adapter, calls the worker group once per role and scatters
the results back. Under SP the calls pass through, except for the optional per-role loss
weight (T1). This recreates the research fork's trainer regions
``rllm/trainer/verl/agent_workflow_trainer.py:138-183`` (split), ``463-499`` (old log-probs),
``517-551`` (ref log-probs) and ``653-715`` (update and metric renaming) without copying the
training loop.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.utils.metric import reduce_metrics

from marl_workflows.contracts import (
    META_KL_OVERRIDE,
    META_TEMPERATURE,
    role_metric_key,
    roles_from_trajectory_ids,
)
from marl_workflows.trainer.loss_weight import apply_per_role_loss_weight

__all__ = ["RoleRoutedWorkerGroup", "split_batch_by_role"]


def split_batch_by_role(batch: DataProto, world_size: int) -> OrderedDict[str, tuple[DataProto, np.ndarray]]:
    """Split ``batch`` into one sub-batch per role, in order of first appearance.

    Each sub-batch is padded to a multiple of ``world_size`` by duplicating its first rows
    (the duplicates enter the loss, as in the fork). The returned indices are the original
    row positions without padding, for scattering results back. The order determines the
    per-role update order and therefore the LR-scheduler order within a step (C12).
    """
    roles = roles_from_trajectory_ids(batch.non_tensor_batch["trajectory_ids"])
    groups: dict[str, list[int]] = {}
    for i, role in enumerate(roles):
        groups.setdefault(role, []).append(i)
    out: OrderedDict[str, tuple[DataProto, np.ndarray]] = OrderedDict()
    for role, rows in groups.items():
        idx = np.asarray(rows, dtype=np.int64)
        sub = batch.select_idxs(idx)
        if world_size > 0 and len(sub) % world_size != 0:
            sub, _ = pad_dataproto_to_divisor(sub, world_size)
        # select_idxs shares the parent's meta_info dict; per-role overrides must not leak back.
        sub.meta_info = dict(sub.meta_info)
        out[role] = (sub, idx)
    return out


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    value = cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)
    return default if value is None else value


class RoleRoutedWorkerGroup:
    """Proxy around the actor/rollout worker group that adds per-role routing under IP.

    Every attribute not defined here (``save_checkpoint``, ``load_checkpoint``,
    ``set_active_lora``, ``save_single_lora_adapter``, ...) is delegated to the wrapped group.
    """

    def __init__(
        self,
        wg: Any,
        *,
        agent_names: Sequence[str],
        share_policy: bool,
        trainer_config: Any,
        rollout_temperature: float,
    ) -> None:
        self._wg = wg
        self.agent_names = [str(n) for n in agent_names]
        self.share_policy = bool(share_policy)
        self.trainer_config = trainer_config
        self.rollout_temperature = float(rollout_temperature)

    def __getattr__(self, name: str) -> Any:
        # Only called for attributes missing on the proxy; read _wg via __dict__ so that a
        # half-constructed or unpickled proxy does not recurse.
        wg = self.__dict__.get("_wg")
        if wg is None:
            raise AttributeError(name)
        return getattr(wg, name)

    @property
    def world_size(self) -> int:
        return self._wg.world_size

    # ------------------------------------------------------------------ helpers

    def _split(self, batch: DataProto) -> OrderedDict[str, tuple[DataProto, np.ndarray]]:
        parts = split_batch_by_role(batch, self.world_size)
        unknown = [r for r in parts if r not in self.agent_names]
        if unknown:
            raise ValueError(f"Roles {unknown} from trajectory_ids are not in trainer.agent_names {self.agent_names}")
        return parts

    def _activate(self, role: str) -> None:
        self._wg.set_active_lora(agent_role=role, lora_config={})

    @staticmethod
    def _response_buffer(batch: DataProto) -> torch.Tensor:
        n = batch.batch["attention_mask"].shape[0]
        resp_len = batch.batch["responses"].shape[1]
        return torch.zeros(n, resp_len, dtype=torch.float32, device=batch.batch["attention_mask"].device)

    # ------------------------------------------------------------------ routed RPCs

    def compute_log_prob(self, batch: DataProto) -> DataProto:
        """Old log-probs and entropies, each role computed under its own adapter (IP)."""
        if self.share_policy:
            return self._wg.compute_log_prob(batch)
        old_log_probs = self._response_buffer(batch)
        entropys = self._response_buffer(batch)
        for role, (sub, idx) in self._split(batch).items():
            self._activate(role)
            out = self._wg.compute_log_prob(sub)
            old_log_probs[idx] = out.batch["old_log_probs"][: len(idx)].to(old_log_probs)
            entropys[idx] = out.batch["entropys"][: len(idx)].to(entropys)
        return DataProto.from_dict(
            tensors={"old_log_probs": old_log_probs, "entropys": entropys},
            meta_info={META_TEMPERATURE: self.rollout_temperature},
        )

    def compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        """Reference log-probs per role sub-batch, without switching adapters.

        The worker computes the reference pass with every adapter disabled, so the active
        adapter does not matter; the split only reproduces the fork's per-role batching.
        """
        if self.share_policy:
            return self._wg.compute_ref_log_prob(batch)
        ref = self._response_buffer(batch)
        for _role, (sub, idx) in self._split(batch).items():
            out = self._wg.compute_ref_log_prob(sub)
            ref[idx] = out.batch["ref_log_prob"][: len(idx)].to(ref)
        return DataProto.from_dict(tensors={"ref_log_prob": ref})

    def update_actor(self, batch: DataProto) -> DataProto:
        """One ``update_actor`` call per role (IP) or one call with optional T1 weights (SP)."""
        if self.share_policy:
            lw_cfg = _cfg_get(self.trainer_config, "loss_weight_per_role", {})
            lw_metrics: dict[str, float] = {}
            if str(_cfg_get(lw_cfg, "mode", "none")) != "none":
                batch, lw_metrics = apply_per_role_loss_weight(batch, lw_cfg)
            out = self._wg.update_actor(batch)
            if lw_metrics:
                out.meta_info.setdefault("metrics", {}).update(lw_metrics)
            return out

        kl_per_role = _cfg_get(self.trainer_config, "kl_loss_coef_per_role", {}) or {}
        renamed: dict[str, Any] = {}
        for role, (sub, _idx) in self._split(batch).items():
            self._activate(role)
            if role in kl_per_role:
                sub.meta_info[META_KL_OVERRIDE] = float(kl_per_role[role])
            sub.meta_info.setdefault(META_TEMPERATURE, self.rollout_temperature)
            out = self._wg.update_actor(sub)
            # Same insertion order as the fork, so leaf collisions resolve identically (C13).
            for key, value in reduce_metrics(out.meta_info.get("metrics", {})).items():
                renamed[role_metric_key(role, key)] = value
        return DataProto(meta_info={"metrics": renamed})
