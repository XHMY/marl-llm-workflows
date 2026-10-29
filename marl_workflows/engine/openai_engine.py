"""``LoRAOpenAIEngine``: rLLM's ``OpenAIEngine`` routing each call to a role's LoRA adapter.

The evaluator serves every role adapter from one vLLM OpenAI server under its own model
name and runs the workflows through this engine. Workflows pass ``agent_name`` (the role) on
every call; the engine maps it to a served model name.

Why not set ``self.model``: the research fork's version (``dashboard/evaluate_checkpoints.py:
756-800``) assigned ``self.model`` before each call. Many workflow coroutines share one
engine, and upstream ``OpenAIEngine.completion`` re-reads ``self.model`` on every retry after
an ``await asyncio.sleep`` (rllm@186d0bbb ``rllm/engine/rollout/openai_engine.py:139-217``), so
a retry could be sent to another role's adapter. Here the model name travels with the call
inside ``sampling_params`` under :data:`contracts.OPENAI_MODEL_KEY`; a thin client proxy pops it
and passes it as ``model``. ``self.model`` is never mutated.

The proxy also applies the fork's ``timeout=3600`` (``openai_engine.py:163``) to every request
that does not set one, without editing the submodule.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rllm.engine.rollout.openai_engine import OpenAIEngine
from rllm.engine.rollout.rollout_engine import ModelOutput

from marl_workflows.contracts import OPENAI_MODEL_KEY, SHARED_ADAPTER

__all__ = ["DEFAULT_REQUEST_TIMEOUT", "LoRAOpenAIEngine"]

# The fork's per-request timeout (seconds): long Math generations at 8k tokens under heavy
# server load exceeded the openai client's default.
DEFAULT_REQUEST_TIMEOUT: float = 3600.0


class _RoutedEndpoint:
    """Wraps ``client.completions`` / ``client.chat.completions``; ``create`` honours the carrier key."""

    def __init__(self, inner: Any, default_timeout: float) -> None:
        self._inner = inner
        self._default_timeout = default_timeout

    async def create(self, *, model: str, **kwargs: Any) -> Any:
        # ``kwargs`` is a fresh dict per call (built from ``**sampling_params``), so popping
        # here never touches the engine's per-call sampling_params, which retries reuse.
        model = kwargs.pop(OPENAI_MODEL_KEY, model)
        kwargs.setdefault("timeout", self._default_timeout)
        return await self._inner.create(model=model, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _RoutedChat:
    def __init__(self, inner: Any, default_timeout: float) -> None:
        self._inner = inner
        self.completions = _RoutedEndpoint(inner.completions, default_timeout)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _ModelRoutingClient:
    """Proxy around ``openai.AsyncOpenAI`` used by :class:`LoRAOpenAIEngine`."""

    def __init__(self, inner: Any, default_timeout: float = DEFAULT_REQUEST_TIMEOUT) -> None:
        self._inner = inner
        self.completions = _RoutedEndpoint(inner.completions, default_timeout)
        self.chat = _RoutedChat(inner.chat, default_timeout)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class LoRAOpenAIEngine(OpenAIEngine):
    """``OpenAIEngine`` that selects the served model (base or LoRA adapter) per call from ``agent_name``.

    ``lora_names`` maps a role to the name its adapter is served under. Resolution order (the
    fork's): the role's own entry, then a ``"default"`` entry, then the base model. A role with
    no entry therefore runs on the base model, which the base-model and generator-transfer
    evaluation modes rely on.
    """

    def __init__(
        self,
        base_model: str,
        tokenizer: Any,
        base_url: str = "http://localhost:8000/v1",
        lora_names: Mapping[str, str] | None = None,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("validate", None)  # accepted by VerlEngine call sites, not by OpenAIEngine
        super().__init__(model=base_model, tokenizer=tokenizer, base_url=base_url, api_key="EMPTY", **kwargs)
        self.base_model_name = base_model
        self.lora_names: dict[str, str] = dict(lora_names or {})
        self.client = _ModelRoutingClient(self.client, request_timeout)

    def set_lora_names(self, lora_names: Mapping[str, str]) -> None:
        """Replace the role -> served-name map (one assignment, so concurrent readers see old or new)."""
        self.lora_names = dict(lora_names)

    def model_for(self, agent_name: str | None) -> str:
        """Served model name for a role (role entry, else ``"default"``, else the base model)."""
        names = self.lora_names
        if agent_name and agent_name in names:
            return names[agent_name]
        if SHARED_ADAPTER in names:
            return names[SHARED_ADAPTER]
        return self.base_model_name

    async def get_model_response(self, messages: list[dict], **kwargs: Any) -> ModelOutput:
        """Route by ``agent_name`` (popped so it never reaches the API) and delegate."""
        kwargs[OPENAI_MODEL_KEY] = self.model_for(kwargs.pop("agent_name", None))
        return await super().get_model_response(messages, **kwargs)
