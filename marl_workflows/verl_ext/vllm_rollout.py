"""``MultiLoRAvLLMAsyncRollout``: the hybrid worker's vLLM engine with one LoRA slot per role.

Selected through patch P2 (``patches.apply_worker_patches``).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from verl.utils.vllm import TensorLoRARequest
from verl.workers.rollout.vllm_rollout.utils import VLLM_LORA_INT_ID, VLLM_LORA_PATH
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMAsyncRollout

__all__ = ["MultiLoRAvLLMAsyncRollout", "max_loras_from"]

logger = logging.getLogger(__name__)


def max_loras_from(rollout_config: Any) -> int:
    """``rollout.engine_kwargs.vllm.max_loras`` (default 1).

    ``RolloutConfig`` is a strict dataclass, so the value travels in the free ``engine_kwargs`` dict
    (contract C10); the config injection sets it to ``len(agent_names)``.
    """
    engine_kwargs = rollout_config.get("engine_kwargs", None) or {}
    vllm_kwargs = engine_kwargs.get("vllm", None) if isinstance(engine_kwargs, Mapping) else None
    value = (vllm_kwargs or {}).get("max_loras", None)
    return int(value) if value is not None else 1


class MultiLoRAvLLMAsyncRollout(vLLMAsyncRollout):
    """Subclass of ``verl@v0.6.1:verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:515``.

    Two changes: ``max_loras`` comes from the config (upstream fixes it to 1 at ``:528-532`` and copies it
    into ``vllm_config.lora_config`` at ``:593-595``), and ``update_weights`` registers the adapter under the
    ``lora_int_id`` passed by the worker (default 123, so SP is unchanged).
    """

    def __init__(self, config: Any, model_config: Any, device_mesh: Any) -> None:
        super().__init__(config, model_config, device_mesh)
        if self.lora_config:
            self.lora_config["max_loras"] = max_loras_from(config)

    async def update_weights(self, weights: Any, **kwargs: Any) -> None:
        """Register a LoRA adapter under ``kwargs["lora_int_id"]``, or load base weights (upstream path).

        A LoRA payload must be sized (a dict or ``dict.items()``), as upstream calls ``dict()`` and ``len()``
        on it. Re-registering an id is idempotent because the old adapter is removed first.
        """
        lora_int_id = kwargs.pop("lora_int_id", None)
        peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)
        if not (peft_config and base_sync_done):
            return await super().update_weights(weights, **kwargs)
        lid = VLLM_LORA_INT_ID if lora_int_id is None else int(lora_int_id)
        self.inference_engine.worker.remove_lora(lid)
        # lora_name str(123) == "123" == VLLM_LORA_NAME, so the SP adapter is registered exactly as upstream.
        lora_request = TensorLoRARequest(
            lora_name=str(lid),
            lora_int_id=lid,
            lora_path=VLLM_LORA_PATH,
            peft_config=asdict(peft_config),
            lora_tensors=dict(weights),
        )
        self.inference_engine.worker.add_lora(lora_request)
        logger.info(f"vLLM load weights for lora_int_id={lid}, loaded_params: {len(weights)}")
