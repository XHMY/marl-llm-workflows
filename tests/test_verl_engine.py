"""MultiLoRAVerlEngine / MultiLoRAServerManager: agent_name -> LoRA id carrier, unique vLLM request ids,
sticky server choice. The engine test runs the real upstream ``get_model_response`` with fake tokenizer,
parser and server handles."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

pytestmark = pytest.mark.stack

pytest.importorskip("rllm.engine.rollout.verl_engine")

from rllm.engine.rollout.verl_engine import VerlEngine  # noqa: E402
from verl.workers.rollout.replica import TokenOutput  # noqa: E402

from marl_workflows import contracts  # noqa: E402
from marl_workflows.engine.verl_engine import MultiLoRAServerManager, MultiLoRAVerlEngine  # noqa: E402

ROLES = ["generator", "aggregator"]


class FakeServer:
    """Ray actor handle stand-in: ``await server.generate.remote(**kw)``."""

    def __init__(self, name: str):
        self.name = name
        self.calls: list[dict] = []
        self.generate = SimpleNamespace(remote=self._remote)

    async def _remote(self, **kwargs):
        self.calls.append(kwargs)
        return TokenOutput(token_ids=[5, 6, 7], log_probs=[-0.1, -0.2, -0.3])


def make_manager(n_servers: int = 2):
    servers = [FakeServer(f"s{i}") for i in range(n_servers)]
    return MultiLoRAServerManager(OmegaConf.create({}), server_handles=list(servers)), servers


async def test_manager_pops_carrier_and_forwards_id():
    manager, servers = make_manager(1)
    params = {"temperature": 0.7, contracts.SAMPLING_LORA_KEY: 124}
    out = await manager.generate("app", prompt_ids=[1, 2], sampling_params=params)
    assert out.token_ids == [5, 6, 7]
    call = servers[0].calls[-1]
    assert call["lora_int_id"] == 124
    assert call["sampling_params"] == {"temperature": 0.7}
    assert contracts.SAMPLING_LORA_KEY in params  # the caller's dict is not mutated
    await manager.generate("app2", prompt_ids=[1], sampling_params={"temperature": 0.7})
    assert servers[0].calls[-1]["lora_int_id"] is None


async def test_manager_unique_request_ids_and_sticky_server():
    manager, servers = make_manager(2)
    await asyncio.gather(*[manager.generate("app", prompt_ids=[1], sampling_params={}) for _ in range(3)])
    sticky = [s for s in servers if s.calls]
    assert len(sticky) == 1, "same request_id must stay on one server"
    ids = [c["request_id"] for c in sticky[0].calls]
    assert len(set(ids)) == 3 and all(i.startswith("app_") for i in ids)
    await manager.generate("other", prompt_ids=[1], sampling_params={})
    assert all(s.calls for s in servers), "a new request_id goes to the least-loaded server"


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]

    def decode(self, ids, skip_special_tokens=True):
        return "x" * len(ids)


class FakeParser:
    def parse(self, messages, **kwargs):
        return "prompt"

    def parse_completion(self, ids):
        return {"content": "x", "reasoning": "", "tool_calls": []}


def make_engine(share_policy: bool = False, strict: bool = True):
    engine = object.__new__(MultiLoRAVerlEngine)
    engine.config = OmegaConf.create({"trainer": {"agent_names": ROLES, "share_policy": share_policy}})
    engine.server_manager, servers = make_manager(1)
    engine.tokenizer = FakeTokenizer()
    engine.processor = None
    engine.chat_parser = FakeParser()
    engine.max_prompt_length = 100
    engine.max_response_length = 16
    engine.accumulate_reasoning = False
    engine.train_sampling_params = {"temperature": 1.0, "top_k": -1, "top_p": 1.0, "logprobs": 1}
    engine.val_sampling_params = {"temperature": 0.6, "top_k": -1, "top_p": 0.95, "logprobs": 1}
    engine.validate = False
    engine.agent_names = list(ROLES)
    engine.share_policy = share_policy
    engine.strict = strict
    return engine, servers[0]


MSGS = [{"role": "user", "content": "hi"}]


async def test_engine_ip_routes_agent_name_to_id_through_upstream():
    engine, server = make_engine()
    out = await engine.get_model_response(MSGS, agent_name="aggregator", application_id="task1")
    assert out.completion_ids == [5, 6, 7]
    call = server.calls[-1]
    assert call["lora_int_id"] == 124
    assert contracts.SAMPLING_LORA_KEY not in call["sampling_params"]
    assert "agent_name" not in call["sampling_params"]
    assert call["request_id"].startswith("task1_")
    await engine.get_model_response(MSGS, agent_name="generator", application_id="task1")
    assert server.calls[-1]["lora_int_id"] == 123


async def test_engine_ip_missing_or_unknown_agent_name():
    engine, server = make_engine(strict=True)
    with pytest.raises(ValueError, match="required under IP"):
        await engine.get_model_response(MSGS)
    with pytest.raises(ValueError, match="Unknown agent_name"):
        await engine.get_model_response(MSGS, agent_name="critic")
    lenient, server = make_engine(strict=False)
    await lenient.get_model_response(MSGS)
    assert server.calls[-1]["lora_int_id"] is None


async def test_engine_sp_sends_no_id():
    engine, server = make_engine(share_policy=True)
    await engine.get_model_response(MSGS, agent_name="aggregator")
    assert server.calls[-1]["lora_int_id"] is None


def test_engine_init_reads_config_and_env(monkeypatch):
    def fake_init(self, config, rollout_manager, tokenizer, processor=None, **kwargs):
        self.config = config

    config = OmegaConf.create({"trainer": {"agent_names": ROLES, "share_policy": False}})
    rollout_manager = SimpleNamespace(server_handles=[FakeServer("s0")])
    with patch.object(VerlEngine, "__init__", fake_init):
        monkeypatch.setenv(contracts.ENV_STRICT_LORA, "0")
        engine = MultiLoRAVerlEngine(config, rollout_manager, tokenizer=None)
        assert engine.strict is False
        monkeypatch.delenv(contracts.ENV_STRICT_LORA)
        engine = MultiLoRAVerlEngine(config, rollout_manager, tokenizer=None)
    assert isinstance(engine.server_manager, MultiLoRAServerManager)
    assert engine.agent_names == ROLES and engine.share_policy is False and engine.strict is True
    assert engine.lora_int_id_for("aggregator") == 124
