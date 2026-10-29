"""LoRAOpenAIEngine routes each call to its role's served model without shared mutable state."""

from __future__ import annotations

import asyncio
import random
from types import SimpleNamespace
from typing import Any

import pytest

from marl_workflows.contracts import OPENAI_MODEL_KEY

pytestmark = pytest.mark.stack

LORA_NAMES = {"generator": "exp_step10_generator", "aggregator": "exp_step10_aggregator"}


class FakeEndpoint:
    """Records every request; optionally fails the first attempt of selected prompts."""

    def __init__(self, fail_first: set[str] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_first = set(fail_first or ())
        self._failed: set[str] = set()

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        key = str(kwargs.get("prompt", kwargs.get("messages")))
        # Yield so that other coroutines interleave between the request and its result.
        await asyncio.sleep(random.random() * 0.01)
        if key in self.fail_first and key not in self._failed:
            self._failed.add(key)
            raise RuntimeError("transient server error")
        text = f"served-by={kwargs['model']}"
        usage = SimpleNamespace(prompt_tokens=3, completion_tokens=2)
        if "messages" in kwargs:
            msg = SimpleNamespace(content=text, reasoning=None, tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=usage)
        choice = SimpleNamespace(text=text, token_ids=[1, 2], logprobs=None, prompt_logprobs=None, finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=usage)


class FakeClient:
    def __init__(self, endpoint: FakeEndpoint) -> None:
        self.completions = endpoint
        self.chat = SimpleNamespace(completions=endpoint)
        self.base_url = "http://fake/v1"


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) % 50 for c in text][:16]


class FakeParser:
    """Stands in for rLLM's ChatTemplateParser (no HF tokenizer needed)."""

    def parse(self, messages: list[dict], **_: Any) -> str:
        return messages[-1]["content"]

    def parse_completion(self, completion_ids: list[int]) -> dict:
        return {"content": "c", "reasoning": "", "tool_calls": []}


def _engine(endpoint: FakeEndpoint, *, completions_path: bool = True, **kw: Any):
    from marl_workflows.engine.openai_engine import LoRAOpenAIEngine, _ModelRoutingClient

    engine = LoRAOpenAIEngine(base_model="Qwen/Qwen3-0.6B", tokenizer=None, lora_names=LORA_NAMES, **kw)
    engine.client = _ModelRoutingClient(FakeClient(endpoint), kw.get("request_timeout", 3600.0))
    if completions_path:
        # The evaluator uses the completions endpoint with a tokenizer; fake the two pieces it needs.
        engine._use_chat_completions = False
        engine.tokenizer = FakeTokenizer()
        engine.chat_parser = FakeParser()
    return engine


@pytest.mark.parametrize("completions_path", [True, False])
async def test_concurrent_routing_with_retries(completions_path: bool) -> None:
    endpoint = FakeEndpoint(fail_first={f"p{i}" for i in range(0, 50, 3)} | {str([{"role": "user", "content": "p1"}])})
    engine = _engine(endpoint, completions_path=completions_path, api_retries=2)
    agents = ["generator", "aggregator", None, "unknown"] * 13
    expected = {
        "generator": LORA_NAMES["generator"],
        "aggregator": LORA_NAMES["aggregator"],
        None: "Qwen/Qwen3-0.6B",
        "unknown": "Qwen/Qwen3-0.6B",
    }

    async def call(i: int, agent: str | None) -> tuple[int, str | None, str]:
        kwargs: dict[str, Any] = {"application_id": f"app{i}"}
        if agent is not None:
            kwargs["agent_name"] = agent
        out = await engine.get_model_response([{"role": "user", "content": f"p{i}"}], **kwargs)
        return i, agent, out.text

    results = await asyncio.gather(*(call(i, agents[i]) for i in range(50)))
    for _i, agent, text in results:
        assert text == f"served-by={expected[agent]}"
    # Every request, including retries, went to the right model with the default timeout, and
    # neither the carrier key nor agent_name nor application_id reached the client.
    assert len(endpoint.calls) > 50  # retries happened
    for call_kwargs in endpoint.calls:
        assert OPENAI_MODEL_KEY not in call_kwargs
        assert "agent_name" not in call_kwargs and "application_id" not in call_kwargs
        assert call_kwargs["timeout"] == 3600
    prompts = [c.get("prompt", str(c.get("messages"))) for c in endpoint.calls]
    for i in range(50):
        models = {
            c["model"]
            for c, p in zip(endpoint.calls, prompts, strict=True)
            if p in (f"p{i}", str([{"role": "user", "content": f"p{i}"}]))
        }
        assert models == {expected[agents[i]]}, i
    # The shared attribute is never switched per call.
    assert engine.model == "Qwen/Qwen3-0.6B"


async def test_default_entry_and_base_fallback() -> None:
    endpoint = FakeEndpoint()
    engine = _engine(endpoint)
    assert engine.model_for("generator") == LORA_NAMES["generator"]
    engine.set_lora_names({"default": "shared"})
    assert engine.model_for("generator") == "shared"
    assert engine.model_for(None) == "shared"
    engine.set_lora_names({})
    assert engine.model_for("generator") == "Qwen/Qwen3-0.6B"
    out = await engine.get_model_response([{"role": "user", "content": "x"}], agent_name="generator")
    assert out.text == "served-by=Qwen/Qwen3-0.6B"


async def test_custom_timeout_and_explicit_timeout_kept() -> None:
    from marl_workflows.engine.openai_engine import _RoutedEndpoint

    endpoint = FakeEndpoint()
    routed = _RoutedEndpoint(endpoint, default_timeout=12.0)
    await routed.create(model="base", prompt="a", **{OPENAI_MODEL_KEY: "lora"})
    await routed.create(model="base", prompt="b", timeout=5)
    assert [(c["model"], c["timeout"]) for c in endpoint.calls] == [("lora", 12.0), ("base", 5)]


def test_real_client_is_wrapped() -> None:
    from marl_workflows.engine.openai_engine import LoRAOpenAIEngine, _ModelRoutingClient

    engine = LoRAOpenAIEngine(base_model="m", tokenizer=None, base_url="http://localhost:1/v1", validate=True)
    assert isinstance(engine.client, _ModelRoutingClient)
    assert str(engine.client.base_url).startswith("http://localhost:1")  # delegated attribute
