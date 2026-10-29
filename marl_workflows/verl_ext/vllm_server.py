"""vLLM async server and replica that serve each request with the LoRA adapter of its role.

The stock server (``verl@v0.6.1:verl/workers/rollout/vllm_rollout/vllm_async_server.py``) starts vLLM with
``max_loras=1`` (``:270-277``) and serves every request with adapter 123 when it is loaded (``:399-406``).
Here ``run_server`` / ``run_headless`` fix ``max_loras`` on the parsed CLI namespace before the engine is
built, and the engine is wrapped in :class:`_LoRARoutingEngine`, which replaces the ``lora_request`` of the
requests that carry an explicit id. The upstream ``generate`` and ``launch_server`` run unchanged. The replica
is selected through hook H1 (``patches.apply_driver_patches``).
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import ray
from verl.workers.rollout.replica import RolloutReplica, TokenOutput
from verl.workers.rollout.vllm_rollout.utils import VLLM_LORA_PATH
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServerBase, vLLMReplica
from vllm.lora.request import LoRARequest

from marl_workflows import contracts
from marl_workflows.verl_ext.vllm_rollout import max_loras_from

__all__ = [
    "MultiLoRAvLLMHttpServer",
    "MultiLoRAvLLMHttpServerBase",
    "MultiLoRAvLLMReplica",
    "load_multi_lora_replica",
]

logger = logging.getLogger(__name__)

# Value stored for a request whose id is not registered under MARL_LORA_STRICT=0: serve the base model,
# which is what the research fork did for an unregistered id.
_BASE_MODEL = None


class _LoRARoutingEngine:
    """Thin proxy around vLLM's ``AsyncLLM`` that picks the adapter per request id.

    ``lora_by_request`` maps a vLLM request id to a LoRA id (or ``None`` for the base model). Requests not in
    the map keep the ``lora_request`` the upstream server built (adapter 123 under SP). vLLM request ids are
    unique per call (``MultiLoRAServerManager``), so entries cannot collide.
    """

    def __init__(self, inner: Any, lora_by_request: dict[str, int | None]) -> None:
        self._inner = inner
        self._lora_by_request = lora_by_request

    def generate(self, *args: Any, request_id: str, lora_request: Any = None, **kwargs: Any) -> Any:
        if request_id in self._lora_by_request:
            lid = self._lora_by_request[request_id]
            lora_request = (
                None
                if lid is _BASE_MODEL
                else LoRARequest(lora_name=str(lid), lora_int_id=lid, lora_path=VLLM_LORA_PATH)
            )
        return self._inner.generate(*args, request_id=request_id, lora_request=lora_request, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # list_loras, sleep, wake_up, reset_prefix_cache, ... go to the real engine.
        return getattr(self._inner, name)


class MultiLoRAvLLMHttpServerBase(vLLMHttpServerBase):
    """Subclass of ``vllm_async_server.py:123`` with per-request LoRA ids and a configurable ``max_loras``."""

    def __init__(
        self,
        config: Any,
        model_config: Any,
        rollout_mode: Any,
        workers: list[Any],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
    ) -> None:
        super().__init__(config, model_config, rollout_mode, workers, replica_rank, node_rank, gpus_per_node, nnodes)
        self._max_loras = max_loras_from(self.config)
        self._lora_by_request: dict[str, int | None] = {}
        self._lora_strict = contracts.lora_strict_from_env()

    def _fix_lora_args(self, args: argparse.Namespace) -> None:
        if getattr(args, "enable_lora", False):
            args.max_loras = self._max_loras
            if self.replica_rank == 0:
                logger.info(f"[Multi-Agent LoRA] vLLM server max_loras={self._max_loras}")

    async def run_server(self, args: argparse.Namespace) -> None:
        self._fix_lora_args(args)
        await super().run_server(args)
        self.engine = _LoRARoutingEngine(self.engine, self._lora_by_request)

    async def run_headless(self, args: argparse.Namespace) -> None:
        self._fix_lora_args(args)
        await super().run_headless(args)

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: list[Any] | None = None,
        lora_int_id: int | None = None,
    ) -> TokenOutput:
        """Upstream ``generate`` (``:373-415``), served with adapter ``lora_int_id`` when it is given.

        ``None`` keeps the upstream behaviour (adapter 123 if loaded; the SP path). An id that is not
        registered in vLLM raises, unless ``MARL_LORA_STRICT=0``, in which case the base model serves it.
        """
        if lora_int_id is not None and self.model_config.lora_rank > 0:
            if lora_int_id in await self.engine.list_loras():
                self._lora_by_request[request_id] = int(lora_int_id)
            elif self._lora_strict:
                raise RuntimeError(
                    f"LoRA id {lora_int_id} is not registered in vLLM (request {request_id}); "
                    f"set {contracts.ENV_STRICT_LORA}=0 to serve the base model instead"
                )
            else:
                self._lora_by_request[request_id] = _BASE_MODEL
        try:
            return await super().generate(prompt_ids, sampling_params, request_id, image_data=image_data)
        finally:
            self._lora_by_request.pop(request_id, None)


@ray.remote(num_cpus=1)
class MultiLoRAvLLMHttpServer(MultiLoRAvLLMHttpServerBase):
    """Ray actor form, mirroring ``@ray.remote(num_cpus=1) class vLLMHttpServer`` (``:444``)."""


class MultiLoRAvLLMReplica(vLLMReplica):
    """Subclass of ``vllm_async_server.py:469`` that launches :class:`MultiLoRAvLLMHttpServer`."""

    def __init__(
        self,
        replica_rank: int,
        config: Any,
        model_config: Any,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ) -> None:
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.server_class = MultiLoRAvLLMHttpServer


def load_multi_lora_replica() -> type[RolloutReplica]:
    """Loader registered under ``"vllm"`` in ``RolloutReplicaRegistry`` by hook H1."""
    return MultiLoRAvLLMReplica
