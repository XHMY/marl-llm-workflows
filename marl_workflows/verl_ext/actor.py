"""``MultiAgentDPActor``: verl's FSDP PPO actor plus the per-role KL coefficient and optimizer-step counters."""

from __future__ import annotations

import math
from typing import Any

import torch
from verl import DataProto
from verl.utils.py_functional import append_to_dict
from verl.workers.actor.dp_actor import DataParallelPPOActor

from marl_workflows import contracts

__all__ = ["MultiAgentDPActor"]


class MultiAgentDPActor(DataParallelPPOActor):
    """Subclass of ``verl@v0.6.1:verl/workers/actor/dp_actor.py:49``.

    ``update_policy`` adds three scalars per call (``contracts.OPTIM_COUNTER_KEYS``): the number of
    mini-batches, the number of optimizer steps it took, and the AdamW ``state["step"]`` of the active
    adapter. Under IP the trainer calls ``update_actor`` once per role, so these show how many optimizer
    steps each role's adapter receives per training step.

    It also honours ``data.meta_info["kl_loss_coef_override"]`` (contract C11): upstream reads
    ``self.config.kl_loss_coef`` in the loss and in the ``actor/kl_coef`` metric, so the frozen config
    value is swapped for the duration of the call and restored afterwards. Without the key nothing changes.
    """

    def update_policy(self, data: DataProto) -> dict[str, Any]:
        self._marl_optim_calls = 0
        override = data.meta_info.get(contracts.META_KL_OVERRIDE)
        previous_coef = self.config.kl_loss_coef
        if override is not None:
            # ActorConfig is frozen (verl/base_config.py:33-38); bypass it for this call only.
            object.__setattr__(self.config, "kl_loss_coef", float(override))
        try:
            metrics = super().update_policy(data)
        finally:
            if override is not None:
                object.__setattr__(self.config, "kl_loss_coef", previous_coef)
        # Same count as len(data.split(ppo_mini_batch_size)) in upstream update_policy.
        num_mini_batches = math.ceil(len(data) / self.config.ppo_mini_batch_size)
        num_key, calls_key, count_key = contracts.OPTIM_COUNTER_KEYS
        append_to_dict(
            metrics,
            {
                num_key: num_mini_batches,
                calls_key: self._marl_optim_calls,
                count_key: self._optim_state_step(),
            },
        )
        return metrics

    def _optimizer_step(self) -> torch.Tensor:
        self._marl_optim_calls = getattr(self, "_marl_optim_calls", 0) + 1
        return super()._optimizer_step()

    def _optim_state_step(self) -> float:
        """AdamW ``state["step"]`` of the first trainable parameter with optimizer state (0 if none).

        Inactive adapters keep their own (unchanged) state, so a trainable parameter is preferred; a frozen
        one is the fallback. Copied from the research fork's ``dp_actor.py:717-731``.
        """
        fallback = None
        for group in self.actor_optimizer.param_groups:
            for p in group["params"]:
                state = self.actor_optimizer.state.get(p, None)
                if not state or "step" not in state:
                    continue
                step = state["step"]
                step = float(step.item()) if torch.is_tensor(step) else float(step)
                if p.requires_grad:
                    return step
                if fallback is None:
                    fallback = step
        return fallback if fallback is not None else 0.0
