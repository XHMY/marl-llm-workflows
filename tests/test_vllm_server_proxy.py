"""vLLM server/rollout subclasses: per-request LoRA routing through the real upstream ``generate``,
strict handling of unregistered ids, ``max_loras`` on both sides, and per-id adapter registration."""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

pytestmark = pytest.mark.stack

server_mod = pytest.importorskip("verl.workers.rollout.vllm_rollout.vllm_async_server")

from peft import LoraConfig  # noqa: E402
from verl.workers.config import RolloutConfig  # noqa: E402
from verl.workers.rollout.vllm_rollout import vllm_rollout_spmd  # noqa: E402

from marl_workflows import contracts  # noqa: E402
from marl_workflows.verl_ext import vllm_rollout, vllm_server  # noqa: E402

UPSTREAM_SERVER = server_mod.vLLMHttpServerBase


class FakeAsyncLLM:
    """Records the lora_request of each generate call; ids in ``loaded`` are registered adapters."""

    def __init__(self, loaded=(123, 124, 125)):
        self.loaded = set(loaded)
        self.calls: list[tuple[str, object]] = []

    async def list_loras(self):
        return set(self.loaded)

    def generate(self, *, prompt, sampling_params, request_id, lora_request=None):
        self.calls.append((request_id, lora_request))

        async def gen():
            out = SimpleNamespace(token_ids=[7, 8], logprobs=None)
            yield SimpleNamespace(outputs=[out])

        return gen()

    async def reset_prefix_cache(self):
        return "reset"


class _Cfg(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def make_server(inner: FakeAsyncLLM, strict: bool = True, lora_rank: int = 8):
    s = object.__new__(vllm_server.MultiLoRAvLLMHttpServerBase)
    s.config = _Cfg(max_model_len=64)
    s.model_config = SimpleNamespace(lora_rank=lora_rank, processor=None)
    s._lora_by_request = {}
    s._lora_strict = strict
    s._max_loras = 3
    s.replica_rank = 0
    s.engine = vllm_server._LoRARoutingEngine(inner, s._lora_by_request)
    return s


def lora_id_of(call):
    lr = call[1]
    return None if lr is None else (lr.lora_int_id, lr.lora_name, lr.lora_path)


async def test_ip_request_served_by_its_adapter():
    inner = FakeAsyncLLM()
    s = make_server(inner)
    out = await s.generate([1, 2, 3], {}, "app_0001", lora_int_id=124)
    assert out.token_ids == [7, 8]
    assert lora_id_of(inner.calls[-1]) == (124, "124", contracts.LORA_PATH_PLACEHOLDER)
    assert s._lora_by_request == {}


async def test_sp_request_keeps_upstream_adapter_123():
    inner = FakeAsyncLLM()
    s = make_server(inner)
    await s.generate([1, 2, 3], {}, "app_0002", lora_int_id=None)
    assert lora_id_of(inner.calls[-1]) == (123, "123", contracts.LORA_PATH_PLACEHOLDER)
    # nothing loaded yet (first rollout before any sync): upstream serves the base model
    inner.loaded.clear()
    await s.generate([1, 2, 3], {}, "app_0003")
    assert lora_id_of(inner.calls[-1]) is None


async def test_unregistered_id_strict_raises_and_lenient_serves_base():
    inner = FakeAsyncLLM(loaded=(123, 124))
    with pytest.raises(RuntimeError, match="not registered"):
        await make_server(inner, strict=True).generate([1], {}, "app_x", lora_int_id=130)
    assert inner.calls == []
    s = make_server(inner, strict=False)
    await s.generate([1], {}, "app_y", lora_int_id=130)
    assert lora_id_of(inner.calls[-1]) is None
    assert s._lora_by_request == {}


async def test_no_lora_model_ignores_id():
    inner = FakeAsyncLLM()
    await make_server(inner, lora_rank=0).generate([1], {}, "app_z", lora_int_id=124)
    assert lora_id_of(inner.calls[-1]) is None


async def test_concurrent_requests_do_not_mix_adapters():
    import asyncio

    inner = FakeAsyncLLM()
    s = make_server(inner)
    await asyncio.gather(*[s.generate([1], {}, f"app_{i}", lora_int_id=123 + i % 3) for i in range(9)])
    for rid, lr in inner.calls:
        assert lr.lora_int_id == 123 + int(rid.split("_")[1]) % 3


async def test_proxy_delegates_other_attributes():
    inner = FakeAsyncLLM()
    proxy = vllm_server._LoRARoutingEngine(inner, {})
    assert await proxy.reset_prefix_cache() == "reset"
    assert await proxy.list_loras() == {123, 124, 125}


async def test_run_server_sets_max_loras_and_wraps_engine():
    inner = FakeAsyncLLM()
    seen = {}

    async def fake_run_server(self, args):
        seen["max_loras"] = args.max_loras
        self.engine = inner

    s = make_server(inner)
    with patch.object(UPSTREAM_SERVER, "run_server", fake_run_server):
        await s.run_server(argparse.Namespace(enable_lora=True, max_loras=1))
    assert seen["max_loras"] == 3
    assert isinstance(s.engine, vllm_server._LoRARoutingEngine) and s.engine._inner is inner

    async def fake_headless(self, args):
        seen["headless"] = args.max_loras

    with patch.object(UPSTREAM_SERVER, "run_headless", fake_headless):
        await s.run_headless(argparse.Namespace(enable_lora=True, max_loras=1))
        no_lora = argparse.Namespace(enable_lora=False, max_loras=1)
        await s.run_headless(no_lora)
    assert seen["headless"] == 1 and no_lora.max_loras == 1


def test_max_loras_from_all_config_forms():
    assert vllm_rollout.max_loras_from(RolloutConfig(name="vllm", engine_kwargs={"vllm": {"max_loras": 3}})) == 3
    assert vllm_rollout.max_loras_from(RolloutConfig(name="vllm")) == 1
    assert vllm_rollout.max_loras_from(OmegaConf.create({"engine_kwargs": {"vllm": {"max_loras": 4}}})) == 4
    assert vllm_rollout.max_loras_from(OmegaConf.create({"engine_kwargs": {"vllm": None}})) == 1
    assert vllm_rollout.max_loras_from({"engine_kwargs": {"sglang": {}}}) == 1


def test_rollout_init_sets_worker_side_max_loras():
    def fake_init(self, config, model_config, device_mesh):
        self.config = config
        self.lora_config = {"max_loras": 1, "max_lora_rank": 8} if model_config.lora_rank > 0 else {}

    cfg = RolloutConfig(name="vllm", engine_kwargs={"vllm": {"max_loras": 4}})
    with patch.object(vllm_rollout_spmd.vLLMAsyncRollout, "__init__", fake_init):
        r = vllm_rollout.MultiLoRAvLLMAsyncRollout(cfg, SimpleNamespace(lora_rank=8), None)
        assert r.lora_config == {"max_loras": 4, "max_lora_rank": 8}
        r0 = vllm_rollout.MultiLoRAvLLMAsyncRollout(cfg, SimpleNamespace(lora_rank=0), None)
        assert r0.lora_config == {}


class FakeVllmWorker:
    def __init__(self):
        self.events: list = []

    def remove_lora(self, lid):
        self.events.append(("remove", lid))

    def add_lora(self, req):
        self.events.append(("add", req.lora_int_id, req.lora_name, req.lora_path, sorted(req.lora_tensors)))


def make_rollout():
    r = object.__new__(vllm_rollout.MultiLoRAvLLMAsyncRollout)
    r.inference_engine = SimpleNamespace(worker=FakeVllmWorker())
    return r


async def test_update_weights_registers_given_id():
    r = make_rollout()
    cfg = LoraConfig(r=4, target_modules=["q_proj"])
    weights = {"a.lora_A.weight": 1, "a.lora_B.weight": 2}
    await r.update_weights(weights.items(), peft_config=cfg, base_sync_done=True, lora_int_id=124)
    await r.update_weights(weights.items(), peft_config=cfg, base_sync_done=True)
    assert r.inference_engine.worker.events == [
        ("remove", 124),
        ("add", 124, "124", contracts.LORA_PATH_PLACEHOLDER, sorted(weights)),
        ("remove", 123),
        ("add", 123, "123", contracts.LORA_PATH_PLACEHOLDER, sorted(weights)),
    ]


async def test_update_weights_base_path_goes_upstream():
    r = make_rollout()
    seen = {}

    async def fake_upstream(self, weights, **kwargs):
        seen.update(kwargs)

    with patch.object(vllm_rollout_spmd.vLLMAsyncRollout, "update_weights", fake_upstream):
        await r.update_weights(iter([]), base_sync_done=False)
        await r.update_weights(iter([]), peft_config=None, base_sync_done=True, lora_int_id=124)
    assert "lora_int_id" not in seen
    assert r.inference_engine.worker.events == []


def test_replica_uses_multi_lora_server():
    def fake_init(self, *args, **kwargs):
        self.server_class = "upstream"

    with patch.object(server_mod.vLLMReplica, "__init__", fake_init):
        replica = vllm_server.MultiLoRAvLLMReplica(0, None, None)
    assert replica.server_class is vllm_server.MultiLoRAvLLMHttpServer
    assert vllm_server.load_multi_lora_replica() is vllm_server.MultiLoRAvLLMReplica
