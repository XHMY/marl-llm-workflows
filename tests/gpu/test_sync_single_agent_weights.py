"""Weight-sync contract of ``MultiAgentAsyncActorRolloutRefWorker`` (C7).

Port of the research fork's ``tests/trainer/verl/test_sync_single_agent_weights.py`` (five tests for the SP
path: the ``default`` adapter is registered under 123 on every sync, the first included) with the patch
targets moved to ``marl_workflows.verl_ext.workers``, plus two tests for the IP path (ids 123..123+N-1 on the
first and on later syncs; collection precedes offload).

The collaborators are patched out and a recording stub stands in for the rollout engine, so the tests also run
on CPU once the stack imports; they carry the ``gpu`` marker so that they run with the GPU suite.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

pytestmark = [pytest.mark.gpu, pytest.mark.stack]

workers = pytest.importorskip("marl_workflows.verl_ext.workers")
Worker = workers.MultiAgentAsyncActorRolloutRefWorker

ADAPTER_KEYS = ("layer0.lora_A.weight", "layer0.lora_B.weight")
BASE_KEYS = ("layer0.weight", "layer1.weight")
RLW = "|rlw"  # marker appended by the fake replace_lora_wrapper
ROLES = ["generator", "aggregator", "worker"]


class RecordingRollout:
    """Stand-in for ``MultiLoRAvLLMAsyncRollout`` that records what the worker pushes."""

    def __init__(self, events, sleep_level=2, free_cache_engine=True):
        self.events = events
        self.sleep_level = sleep_level
        self.free_cache_engine = free_cache_engine

    async def resume(self, tags=None):
        for tag in tags or []:
            self.events.append(("resume", tag))

    async def update_weights(self, weights, lora_int_id=None, **kwargs):
        peft_config = kwargs.get("peft_config", None)
        base_sync_done = kwargs.get("base_sync_done", False)
        is_lora_registration = bool(peft_config) and bool(base_sync_done)
        if is_lora_registration:
            # The rollout does dict(weights) and len(weights) on this payload.
            assert hasattr(weights, "__len__"), "LoRA payload must be a sized view (dict/items), not a generator"
            names = tuple(sorted(dict(weights).keys()))
        else:
            names = tuple(sorted(name for name, _ in weights))
        self.events.append(
            (
                "push",
                "lora" if is_lora_registration else "full",
                names,
                lora_int_id,
                base_sync_done,
                peft_config is not None,
            )
        )


def _make_worker(events, *, base_sync_done, sleep_level, is_lora=True, offload=True, adapters=("default",)):
    peft_cfgs = {name: SimpleNamespace(r=8, lora_alpha=16, name=name) for name in adapters}
    if is_lora:
        inner = SimpleNamespace(
            peft_config=peft_cfgs,
            state_dict=lambda: OrderedDict((k, object()) for k in BASE_KEYS),
        )
    else:
        inner = SimpleNamespace(state_dict=lambda: OrderedDict((k, object()) for k in BASE_KEYS))
    actor_module_fsdp = SimpleNamespace(
        _fsdp_wrapped_module=inner,
        state_dict=lambda: OrderedDict((k, object()) for k in BASE_KEYS),
    )
    worker = SimpleNamespace(
        actor_module_fsdp=actor_module_fsdp,
        config=OmegaConf.create({"rollout": {"layered_summon": False, "free_cache_engine": True}}),
        _is_offload_param=offload,
        layered_summon=False,
        base_sync_done=base_sync_done,
        rollout_lora_registered=False,
        rollout_lora_int_ids=[],
        rank=0,
        rollout=RecordingRollout(events, sleep_level=sleep_level),
    )
    return worker, inner


def _run(coro_fn, events):
    def fake_collect(module=None, layered_summon=False, base_sync_done=False, adapter_name="default", **kwargs):
        events.append(("collect", bool(base_sync_done), adapter_name))
        keys = ADAPTER_KEYS if base_sync_done else BASE_KEYS
        return OrderedDict((k, object()) for k in keys)

    def fake_replace_lora_wrapper(k, peft_config):
        return k + RLW

    def fake_offload(module):
        events.append(("offload",))

    patches = [
        patch.object(workers, "collect_lora_params", fake_collect),
        patch.object(workers, "replace_lora_wrapper", fake_replace_lora_wrapper),
        patch.object(workers, "convert_weight_keys", lambda params, model=None: params),
        patch.object(workers, "log_gpu_memory_usage", lambda *a, **k: None),
        patch.object(workers, "set_expandable_segments", lambda *a, **k: None),
        patch.object(workers, "aggressive_empty_cache", lambda *a, **k: None),
        patch.object(workers, "get_device_id", lambda: "cpu"),
        patch.object(workers, "offload_fsdp_model_to_cpu", fake_offload),
        patch.object(workers, "load_fsdp_model_to_gpu", lambda module: None),
    ]
    for p in patches:
        p.start()
    try:
        asyncio.run(coro_fn())
    finally:
        for p in reversed(patches):
            p.stop()
    return events


def _run_single(worker, peft_model, events):
    return _run(lambda: Worker._sync_single_agent_weights(worker, peft_model), events)


def _run_multi(worker, peft_model, events, names):
    return _run(lambda: Worker._sync_multi_agent_lora_weights(worker, peft_model, names), events)


def _kinds(events):
    return [e[0] for e in events]


def _pushes(events):
    return [e for e in events if e[0] == "push"]


def _collects(events):
    return [(e[1], e[2]) for e in events if e[0] == "collect"]


def _assert_base_push(push):
    assert push[1] == "full", f"base push should not be a LoRA registration: {push}"
    assert push[2] == tuple(sorted(k + RLW for k in BASE_KEYS)), push
    assert push[3] is None, f"base push must not carry a lora_int_id: {push}"
    assert push[4] is False, f"base push must set base_sync_done=False: {push}"


def _assert_adapter_push(push, lora_int_id=workers.VLLM_LORA_INT_ID):
    assert push[1] == "lora", f"adapter push must register a LoRA adapter: {push}"
    assert push[2] == tuple(sorted(ADAPTER_KEYS)), push
    assert push[3] == lora_int_id, f"adapter must be registered under {lora_int_id}: {push}"
    assert push[4] is True, push
    assert push[5] is True, push


# ----------------------------------------------------------------------------- SP / single adapter (ported)


@pytest.mark.parametrize("sleep_level", [1, 2])
def test_first_sync_registers_adapter(sleep_level):
    """First sync (base_sync_done False) pushes base weights AND registers the adapter."""
    events = []
    worker, peft_model = _make_worker(events, base_sync_done=False, sleep_level=sleep_level)
    _run_single(worker, peft_model, events)

    assert _kinds(events) == ["collect", "collect", "offload", "resume", "push", "push", "resume"], events
    # adapter collected first (model still on GPU), then base
    assert _collects(events) == [(True, "default"), (False, "default")], events
    assert events[3] == ("resume", "weights"), events
    base_push, adapter_push = _pushes(events)
    _assert_base_push(base_push)
    _assert_adapter_push(adapter_push)
    assert events[-1] == ("resume", "kv_cache"), events

    assert worker.rollout_lora_registered is True
    assert worker.rollout_lora_int_ids == [workers.VLLM_LORA_INT_ID]


def test_later_sync_sleep_level_2_pushes_base_and_adapter():
    """sleep_level=2 destroys base weights on sleep, so both are pushed every sync."""
    events = []
    worker, peft_model = _make_worker(events, base_sync_done=True, sleep_level=2)
    _run_single(worker, peft_model, events)

    assert _collects(events) == [(True, "default"), (False, "default")], events
    pushes = _pushes(events)
    assert len(pushes) == 2, events
    _assert_base_push(pushes[0])
    _assert_adapter_push(pushes[1])
    assert worker.rollout_lora_registered is True
    assert worker.rollout_lora_int_ids == [workers.VLLM_LORA_INT_ID]


def test_later_sync_sleep_level_1_pushes_adapter_only():
    """sleep_level=1 keeps base weights resident: only the adapter is re-pushed."""
    events = []
    worker, peft_model = _make_worker(events, base_sync_done=True, sleep_level=1)
    _run_single(worker, peft_model, events)

    assert _collects(events) == [(True, "default")], events
    pushes = _pushes(events)
    assert len(pushes) == 1, events
    _assert_adapter_push(pushes[0])
    assert _kinds(events) == ["collect", "offload", "resume", "push", "resume"], events
    assert worker.rollout_lora_registered is True
    assert worker.rollout_lora_int_ids == [workers.VLLM_LORA_INT_ID]


def test_collection_precedes_offload():
    """Weights must be collected while the model is still on GPU."""
    events = []
    worker, peft_model = _make_worker(events, base_sync_done=False, sleep_level=2, offload=True)
    _run_single(worker, peft_model, events)

    kinds = _kinds(events)
    assert "offload" in kinds, events
    offload_at = kinds.index("offload")
    collect_at = [i for i, k in enumerate(kinds) if k == "collect"]
    assert collect_at, events
    assert max(collect_at) < offload_at, events
    assert all(i > offload_at for i, k in enumerate(kinds) if k == "push"), events


def test_non_lora_single_full_push():
    """Without LoRA the whole state dict is pushed once and no adapter is registered."""
    events = []
    worker, peft_model = _make_worker(events, base_sync_done=False, sleep_level=2, is_lora=False)
    _run_single(worker, peft_model, events)

    assert not any(e[0] == "collect" for e in events), events
    pushes = _pushes(events)
    assert len(pushes) == 1, events
    push = pushes[0]
    assert push[1] == "full", push
    assert push[2] == tuple(sorted(BASE_KEYS)), push
    assert push[5] is False, f"peft_config must be None in the non-LoRA path: {push}"
    assert push[4] is False, push
    assert worker.rollout_lora_registered is False
    assert worker.rollout_lora_int_ids == []


# ----------------------------------------------------------------------------- IP / one adapter per role (new)


@pytest.mark.parametrize(("base_sync_done", "sleep_level"), [(False, 1), (False, 2), (True, 1), (True, 2)])
def test_multi_agent_registers_every_role_on_every_sync(base_sync_done, sleep_level):
    """Every role adapter is registered under index + 123, on the first and on later syncs."""
    events = []
    worker, peft_model = _make_worker(
        events, base_sync_done=base_sync_done, sleep_level=sleep_level, adapters=("default", *ROLES)
    )
    _run_multi(worker, peft_model, events, ROLES)

    need_base = not base_sync_done or sleep_level == 2
    expected_collects = [(True, r) for r in ROLES] + ([(False, ROLES[0])] if need_base else [])
    assert _collects(events) == expected_collects, events
    pushes = _pushes(events)
    if need_base:
        _assert_base_push(pushes.pop(0))
    assert len(pushes) == len(ROLES), events
    for idx, push in enumerate(pushes):
        _assert_adapter_push(push, lora_int_id=workers.VLLM_LORA_INT_ID + idx)
    assert events[-1] == ("resume", "kv_cache"), events
    assert worker.rollout_lora_registered is True
    assert worker.rollout_lora_int_ids == [123, 124, 125]


def test_multi_agent_collection_precedes_offload_and_validates_names():
    events = []
    worker, peft_model = _make_worker(events, base_sync_done=False, sleep_level=2, adapters=("default", *ROLES))
    _run_multi(worker, peft_model, events, ROLES)
    kinds = _kinds(events)
    offload_at = kinds.index("offload")
    assert max(i for i, k in enumerate(kinds) if k == "collect") < offload_at, events
    assert all(i > offload_at for i, k in enumerate(kinds) if k == "push"), events

    events = []
    worker, peft_model = _make_worker(events, base_sync_done=True, sleep_level=1, adapters=("default", "generator"))
    with pytest.raises(ValueError, match="not found in peft_config"):
        _run_multi(worker, peft_model, events, ["generator", "aggregator"])
