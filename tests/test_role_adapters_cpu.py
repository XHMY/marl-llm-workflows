"""Per-role adapters on a tiny random Qwen2 model (CPU, no download, no FSDP).

Covers adapter creation through P1 (order, RNG equivalence with the research fork, warm start from a saved
adapter), the requires_grad rule, ``set_active_lora`` isolation (inactive adapters get no gradient and no
AdamW update), the adapter-aware ``collect_lora_params`` and the PEFT directory written by
``save_single_lora_adapter``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytestmark = pytest.mark.stack

fsdp_workers = pytest.importorskip("verl.workers.fsdp_workers")

from omegaconf import OmegaConf  # noqa: E402
from peft import LoraConfig, PeftModel, TaskType, get_peft_model  # noqa: E402
from peft.utils.save_and_load import get_peft_model_state_dict  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from marl_workflows.verl_ext import lora_utils, patches  # noqa: E402
from marl_workflows.verl_ext import workers as W  # noqa: E402

ROLES = ["generator", "aggregator"]


def tiny_model(seed: int = 0) -> Qwen2ForCausalLM:
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return Qwen2ForCausalLM(cfg)


def lora_kwargs() -> dict:
    return {
        "task_type": TaskType.CAUSAL_LM,
        "r": 4,
        "lora_alpha": 8,
        "target_modules": ["q_proj", "v_proj"],
        "exclude_modules": None,
        "bias": "none",
    }


def make_worker(
    agent_names=ROLES, share_policy=False, initial=None, module=None
) -> W.MultiAgentAsyncActorRolloutRefWorker:
    w = object.__new__(W.MultiAgentAsyncActorRolloutRefWorker)
    w.config = OmegaConf.create(
        {
            "agent_names": list(agent_names),
            "share_policy": share_policy,
            "initial_lora_adapters": initial or {},
            "model": {"use_shm": False, "lora_adapter_path": None},
        }
    )
    w._rank = 0
    w._is_lora = True
    w._is_actor = True
    w._is_offload_param = False
    w.current_active_lora = None
    w._marl_adapters_added = False
    w.actor_module_fsdp = module
    return w


def build_with_p1(worker, seed: int = 0):
    """What upstream _build_model_optimizer does at fsdp_workers.py:439, with P1 in force."""
    model = tiny_model(seed)
    torch.manual_seed(1234)
    with patches.scoped_multi_adapter_peft(worker._add_role_adapters):
        peft_model = fsdp_workers.get_peft_model(model, LoraConfig(**lora_kwargs()))
    worker.actor_module_fsdp = peft_model
    return peft_model


def build_like_fork(seed: int = 0):
    """The research fork's code path (fsdp_workers.py:470-505): fresh LoraConfig per extra adapter."""
    model = tiny_model(seed)
    torch.manual_seed(1234)
    peft_model = get_peft_model(model, LoraConfig(**lora_kwargs()))
    for name in ROLES:
        peft_model.add_adapter(adapter_name=name, peft_config=LoraConfig(**lora_kwargs()))
    return peft_model


def test_adapters_created_in_order_and_match_fork_rng():
    worker = make_worker()
    ours = build_with_p1(worker)
    assert list(ours.peft_config) == ["default", *ROLES]
    assert worker.current_active_lora == "generator"
    assert patches.applied_patches() == []
    fork = build_like_fork()
    ours_sd, fork_sd = ours.state_dict(), fork.state_dict()
    assert ours_sd.keys() == fork_sd.keys()
    for k in ours_sd:
        assert torch.equal(ours_sd[k], fork_sd[k]), k
    # each role adapter gets its own config object
    assert ours.peft_config["generator"] is not ours.peft_config["aggregator"]


def test_requires_grad_rule_after_creation():
    peft_model = build_with_p1(make_worker())
    for name, p in peft_model.named_parameters():
        assert p.requires_grad == ("lora" in name and "default" not in name), name
    assert any(p.requires_grad for n, p in peft_model.named_parameters() if ".generator." in n)
    assert any(p.requires_grad for n, p in peft_model.named_parameters() if ".aggregator." in n)


def test_initial_adapter_is_loaded(tmp_path):
    donor = get_peft_model(tiny_model(), LoraConfig(**lora_kwargs()))
    with torch.no_grad():
        for n, p in donor.named_parameters():
            if "lora_B" in n:
                p.fill_(0.5)
    donor.save_pretrained(str(tmp_path / "agg"))

    worker = make_worker(initial={"aggregator": str(tmp_path / "agg")})
    with patch.object(W, "copy_to_local", side_effect=lambda path, use_shm=False: path) as ctl:
        peft_model = build_with_p1(worker)
    ctl.assert_called_once()
    assert peft_model.peft_config["aggregator"].task_type == TaskType.CAUSAL_LM
    loaded = get_peft_model_state_dict(peft_model, adapter_name="aggregator")
    expected = get_peft_model_state_dict(donor, adapter_name="default")
    assert loaded.keys() == expected.keys()
    for k in loaded:
        assert torch.equal(loaded[k], expected[k]), k
    for n, p in peft_model.named_parameters():
        if ".aggregator." in n:
            assert p.requires_grad, n


def test_set_active_lora_isolates_inactive_adapters():
    worker = make_worker()
    peft_model = build_with_p1(worker)
    opt = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=1e-2)
    # nonzero lora_B so that every adapter would receive a gradient if it were trainable
    with torch.no_grad():
        for n, p in peft_model.named_parameters():
            if "lora_B" in n:
                p.normal_()

    worker.set_active_lora("aggregator")
    assert worker.current_active_lora == "aggregator"
    assert peft_model.active_adapter == "aggregator"
    for n, p in peft_model.named_parameters():
        assert p.requires_grad == (".aggregator." in n), n

    before = {n: p.detach().clone() for n, p in peft_model.named_parameters()}
    out = peft_model(input_ids=torch.randint(0, 64, (2, 8)))
    out.logits.float().pow(2).mean().backward()
    for n, p in peft_model.named_parameters():
        if ".aggregator." in n:
            assert p.grad is not None, n
        else:
            assert p.grad is None, n
    opt.step()
    for n, p in peft_model.named_parameters():
        changed = not torch.equal(before[n], p.detach())
        assert changed == (".aggregator." in n and "lora" in n), n


def test_set_active_lora_guards_and_unknown_role():
    worker = make_worker()
    build_with_p1(worker)
    with pytest.raises(ValueError, match="not found"):
        worker.set_active_lora("critic")
    # GUARD 3: non-LoRA or non-actor workers return without touching the model
    sentinel = SimpleNamespace(named_parameters=lambda: pytest.fail("touched"))
    for attrs in ({"_is_lora": False}, {"_is_actor": False}):
        w = make_worker(module=sentinel)
        for k, v in attrs.items():
            setattr(w, k, v)
        assert w.set_active_lora("generator") is None
    w = make_worker(agent_names=[], module=sentinel)
    assert w.set_active_lora("generator") is None


def test_role_parameter_check():
    worker = make_worker()
    build_with_p1(worker)
    worker._marl_check_role_params()  # passes
    # A role that is a substring of a base parameter name (q_proj) would train base weights under the
    # fork's substring rule; the check refuses it.
    bad = make_worker(agent_names=["proj"])
    build_with_p1(bad)
    with pytest.raises(ValueError, match="does not select exactly"):
        bad._marl_check_role_params()


def test_param_belongs_to_adapter():
    assert W.param_belongs_to_adapter("m.layers.0.q_proj.lora_A.generator.weight", "generator")
    assert W.param_belongs_to_adapter("m.layers.0.q_proj.lora_A.generator._flat_param", "generator")
    assert not W.param_belongs_to_adapter("m.layers.0.q_proj.lora_A.generators.weight", "generator")
    assert not W.param_belongs_to_adapter("m.layers.0.q_proj.base_layer.weight", "proj")


def test_collect_lora_params_per_adapter():
    peft_model = build_with_p1(make_worker())
    with torch.no_grad():
        for n, p in peft_model.named_parameters():
            if "lora_B.aggregator" in n:
                p.fill_(1.0)
    gen = lora_utils.collect_lora_params(
        peft_model, layered_summon=False, base_sync_done=True, adapter_name="generator"
    )
    agg = lora_utils.collect_lora_params(
        peft_model, layered_summon=False, base_sync_done=True, adapter_name="aggregator"
    )
    assert gen.keys() == agg.keys()
    assert all("generator" not in k and "aggregator" not in k for k in gen)
    b_keys = [k for k in gen if "lora_B" in k]
    assert b_keys and all(torch.count_nonzero(gen[k]) == 0 for k in b_keys)
    assert all(torch.all(agg[k] == 1.0) for k in b_keys)
    default = lora_utils.collect_lora_params(peft_model, layered_summon=False, base_sync_done=True)
    assert default.keys() == gen.keys()


def test_save_single_lora_adapter_writes_peft_dir(tmp_path):
    worker = make_worker()
    peft_model = build_with_p1(worker)
    with torch.no_grad():
        for n, p in peft_model.named_parameters():
            if "lora_B.aggregator" in n:
                p.fill_(0.25)

    def fake_layered_summon(module, adapter_name="default"):
        return dict(get_peft_model_state_dict(module, adapter_name=adapter_name))

    fake_dist = SimpleNamespace(get_rank=lambda: 0, barrier=lambda: None)
    with (
        patch.object(W, "dist", fake_dist),
        patch.object(W, "fsdp_version", lambda m: 1),
        patch.object(W, "get_device_name", lambda: "cpu"),
        patch.object(W, "layered_summon_lora_params", fake_layered_summon),
    ):
        worker.save_single_lora_adapter("aggregator", str(tmp_path / "lora_adapter_aggregator"), global_step=3)
        with pytest.raises(ValueError, match="not found"):
            worker.save_single_lora_adapter("critic", str(tmp_path / "x"))

    out = tmp_path / "lora_adapter_aggregator"
    tensors = load_file(str(out / "adapter_model.safetensors"))
    assert tensors and all("aggregator" not in k for k in tensors)
    assert any(k.endswith("lora_A.weight") for k in tensors)
    cfg = json.loads((out / "adapter_config.json").read_text())
    assert cfg["task_type"] == "CAUSAL_LM" and cfg["peft_type"] == "LORA" and cfg["r"] == 4
    assert sorted(cfg["target_modules"]) == ["q_proj", "v_proj"]
    assert worker.current_active_lora == "generator"  # saving does not switch adapters

    reloaded = PeftModel.from_pretrained(tiny_model(), str(out))
    sd = get_peft_model_state_dict(reloaded)
    for k, v in tensors.items():
        assert torch.equal(sd[k], v), k


def test_build_model_optimizer_guard_1():
    calls = []

    def fake_super_build(self, *args, **kwargs):
        # stands in for upstream: wraps the model at fsdp_workers.py:439
        peft_model = fsdp_workers.get_peft_model(tiny_model(), LoraConfig(**lora_kwargs()))
        calls.append(list(peft_model.peft_config))
        return peft_model

    with patch.object(fsdp_workers.ActorRolloutRefWorker, "_build_model_optimizer", fake_super_build):
        W.MultiAgentAsyncActorRolloutRefWorker._build_model_optimizer(make_worker(), role="actor")
        W.MultiAgentAsyncActorRolloutRefWorker._build_model_optimizer(make_worker(share_policy=True), role="actor")
        W.MultiAgentAsyncActorRolloutRefWorker._build_model_optimizer(make_worker(), role="ref")
        single = make_worker(agent_names=["generator"])
        W.MultiAgentAsyncActorRolloutRefWorker._build_model_optimizer(single, role="actor")
        warm = make_worker()
        warm.config.model.lora_adapter_path = "/some/adapter"
        with pytest.raises(ValueError, match="initial_lora_adapters"):
            W.MultiAgentAsyncActorRolloutRefWorker._build_model_optimizer(warm, role="actor")
        reserved = make_worker(agent_names=["default", "aggregator"])
        with pytest.raises(ValueError, match="reserved"):
            W.MultiAgentAsyncActorRolloutRefWorker._build_model_optimizer(reserved, role="actor")
    assert calls == [["default", *ROLES], ["default"], ["default"], ["default", "generator"]]
    assert fsdp_workers.get_peft_model is get_peft_model
    assert patches.applied_patches() == []
