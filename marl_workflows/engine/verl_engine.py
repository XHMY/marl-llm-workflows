"""Training-time rollout engine: each model call is served by the LoRA adapter of its role.

``MultiLoRAVerlEngine.get_model_response`` maps the workflow's ``agent_name`` to a vLLM LoRA id (contract C6)
and puts it in ``sampling_params`` under ``contracts.SAMPLING_LORA_KEY``: upstream copies the call's kwargs
into ``sampling_params`` (``rllm@186d0bbb:rllm/engine/rollout/verl_engine.py:58``) and hands that dict only to
``server_manager.generate`` (``:80``), so the id reaches :class:`MultiLoRAServerManager` without copying the
upstream method. The manager pops it before anything reaches vLLM (C20) and forwards it to the server.
"""

from __future__ import annotations

import uuid
from typing import Any

from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.engine.rollout.verl_engine import VerlEngine
from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

from marl_workflows import contracts

__all__ = ["MultiLoRAServerManager", "MultiLoRAVerlEngine"]


class MultiLoRAServerManager(AsyncLLMServerManager):
    """Subclass of ``verl@v0.6.1:verl/experimental/agent_loop/agent_loop.py:52`` (``generate`` ``:89-110``).

    Two changes: the LoRA id is popped from ``sampling_params`` and passed to the server, and every vLLM
    request gets a unique id. Upstream passes ``request_id`` straight to vLLM, but workflows reuse one
    ``application_id`` for concurrent calls (Math Voting's three voters), which vLLM rejects as a duplicate.
    The sticky server choice still uses the caller's ``request_id`` (prefix caching across turns).
    """

    @classmethod
    def from_manager(cls, manager: AsyncLLMServerManager) -> MultiLoRAServerManager:
        """Take over an already-built upstream manager's state (server order, load heap, sticky cache).

        Calling the constructor again would ``random.shuffle`` ``server_handles`` a second time; the research
        fork built one manager, so the engine adopts the one upstream ``VerlEngine.__init__`` built.
        """
        new = cls.__new__(cls)
        new.__dict__.update(vars(manager))
        return new

    @rollout_trace_op
    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: list[Any] | None = None,
    ) -> TokenOutput:
        sampling_params = dict(sampling_params)
        lora_int_id = sampling_params.pop(contracts.SAMPLING_LORA_KEY, None)
        server = self._choose_server(request_id)
        vllm_request_id = f"{request_id}_{uuid.uuid4().hex[:8]}"
        return await server.generate.remote(
            request_id=vllm_request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            lora_int_id=lora_int_id,
        )


class MultiLoRAVerlEngine(VerlEngine):
    """Subclass of ``rllm@186d0bbb:rllm/engine/rollout/verl_engine.py:12`` (``get_model_response`` ``:48-106``).

    Reads ``trainer.agent_names`` and ``trainer.share_policy``. Under IP a call without ``agent_name`` raises
    unless ``MARL_LORA_STRICT=0`` (then the server's default adapter 123 serves it, as in the research fork).
    """

    def __init__(self, config: Any, rollout_manager: Any, tokenizer: Any, processor: Any = None, **kwargs: Any) -> None:
        super().__init__(config, rollout_manager, tokenizer, processor=processor, **kwargs)
        self.server_manager = MultiLoRAServerManager.from_manager(self.server_manager)
        self.agent_names: list[str] = list(config.trainer.get("agent_names", None) or [])
        self.share_policy: bool = bool(config.trainer.get("share_policy", False))
        self.strict: bool = contracts.lora_strict_from_env()

    def lora_int_id_for(self, agent_name: str | None) -> int | None:
        """vLLM LoRA id serving ``agent_name`` (``None``: the server's default, adapter 123)."""
        return contracts.lora_int_id_for(agent_name, self.agent_names, self.share_policy, strict=self.strict)

    async def get_model_response(self, messages: list[dict], **kwargs: Any) -> ModelOutput:
        agent_name = kwargs.pop("agent_name", None)
        kwargs[contracts.SAMPLING_LORA_KEY] = self.lora_int_id_for(agent_name)
        return await super().get_model_response(messages, **kwargs)
