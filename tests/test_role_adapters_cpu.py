"""Per-role adapters on a tiny random Qwen2 model (CPU, no download, no FSDP).

Covers adapter creation through P1 under IP (exactly the role adapters, no ``default``; order and RNG use of a
plain PEFT construction; warm start of the first and of a later role from a saved adapter), SP creation
(bit-identical to the research fork's path), the requires_grad rule, ``set_active_lora`` isolation (inactive
adapters get no gradient and no AdamW update), the adapter-aware ``collect_lora_params``, the PEFT directory
written by ``save_single_lora_adapter`` for every role, the state-dict round trip of an IP model, and the IP
``save_checkpoint`` / ``load_checkpoint`` overrides.
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
    with patches.scoped_multi_adapter_peft(worker._build_role_adapters):
        peft_model = fsdp_workers.get_peft_model(model, LoraConfig(**lora_kwargs()))
    worker.actor_module_fsdp = peft_model
    return peft_model


def build_plain_ip(seed: int = 0, roles=ROLES):
    """Plain PEFT: the first role is the initial adapter, the others are added with a fresh LoraConfig."""
    model = tiny_model(seed)
    torch.manual_seed(1234)
    peft_model = get_peft_model(model, LoraConfig(**lora_kwargs()), adapter_name=roles[0])
    for name in roles[1:]:
        peft_model.add_adapter(adapter_name=name, peft_config=LoraConfig(**lora_kwargs()))
    return peft_model


def build_like_fork_sp(seed: int = 0):
    """The research fork's SP path: upstream get_peft_model only (guard 1 is off under SP)."""
    model = tiny_model(seed)
    torch.manual_seed(1234)
    return get_peft_model(model, LoraConfig(**lora_kwargs()))


def adapter_names_in(keys) -> set[str]:
    """Adapter-name components of PEFT LoRA keys (``...lora_A.<adapter>.weight``)."""
    out = set()
    for k in keys:
        parts = k.split(".")
        for i, part in enumerate(parts[:-1]):
            if part.startswith("lora_"):
                out.add(parts[i + 1])
    return out


def test_ip_adapters_are_exactly_the_roles():
    worker = make_worker()
    ours = build_with_p1(worker)
    assert list(ours.peft_config) == ROLES
    assert "default" not in ours.peft_config
    assert not any("default" in n.split(".") for n, _ in ours.named_parameters())
    assert adapter_names_in(ours.state_dict()) == set(ROLES)
    assert ours.active_adapter == "generator" and worker.current_active_lora == "generator"
    assert patches.applied_patches() == []
    # same order and RNG use as a plain PEFT construction with the first role as the initial adapter
    plain = build_plain_ip()
    ours_sd, plain_sd = ours.state_dict(), plain.state_dict()
    assert ours_sd.keys() == plain_sd.keys()
    for k in ours_sd:
        assert torch.equal(ours_sd[k], plain_sd[k]), k
    # each role adapter gets its own config object
    assert ours.peft_config["generator"] is not ours.peft_config["aggregator"]
    assert ours.peft_config["generator"].r == ours.peft_config["aggregator"].r == 4


def test_sp_creation_matches_fork_bit_identical():
    """SP never enters P1: the single adapter is upstream's ``default``, bit-identical to the fork's path."""
    built = {}

    def fake_super_build(self, *args, **kwargs):
        model = tiny_model(0)
        torch.manual_seed(1234)
        built["m"] = fsdp_workers.get_peft_model(model, LoraConfig(**lora_kwargs()))
        return built["m"]

    with patch.object(fsdp_workers.ActorRolloutRefWorker, "_build_model_optimizer", fake_super_build):
        W.MultiAgentAsyncActorRolloutRefWorker._build_model_optimizer(
            make_worker(agent_names=["generator"], share_policy=True), role="actor"
        )
    ours, fork = built["m"], build_like_fork_sp()
    assert list(ours.peft_config) == ["default"]
    ours_sd, fork_sd = ours.state_dict(), fork.state_dict()
    assert ours_sd.keys() == fork_sd.keys()
    for k in ours_sd:
        assert torch.equal(ours_sd[k], fork_sd[k]), k


def test_requires_grad_rule_after_creation():
    peft_model = build_with_p1(make_worker())
    for name, p in peft_model.named_parameters():
        assert p.requires_grad == ("lora" in name), name
    assert any(p.requires_grad for n, p in peft_model.named_parameters() if ".generator." in n)
    assert any(p.requires_grad for n, p in peft_model.named_parameters() if ".aggregator." in n)


@pytest.mark.parametrize("role", ROLES)  # the first role takes the from_pretrained path, the second load_adapter
def test_initial_adapter_is_loaded(tmp_path, role):
    donor = get_peft_model(tiny_model(), LoraConfig(**lora_kwargs()))
    with torch.no_grad():
        for n, p in donor.named_parameters():
            if "lora_B" in n:
                p.fill_(0.5)
    donor.save_pretrained(str(tmp_path / "donor"))

    worker = make_worker(initial={role: str(tmp_path / "donor")})
    with patch.object(W, "copy_to_local", side_effect=lambda path, use_shm=False: path) as ctl:
        peft_model = build_with_p1(worker)
    ctl.assert_called_once()
    assert list(peft_model.peft_config) == ROLES
    assert peft_model.peft_config[role].task_type == TaskType.CAUSAL_LM
    loaded = get_peft_model_state_dict(peft_model, adapter_name=role)
    expected = get_peft_model_state_dict(donor, adapter_name="default")
    assert loaded.keys() == expected.keys()
    for k in loaded:
        assert torch.equal(loaded[k], expected[k]), k
    (other,) = [r for r in ROLES if r != role]
    fresh = get_peft_model_state_dict(peft_model, adapter_name=other)
    assert all(torch.count_nonzero(v) == 0 for k, v in fresh.items() if "lora_B" in k)
    for n, p in peft_model.named_parameters():
        assert p.requires_grad == ("lora" in n), n
    assert peft_model.active_adapter == "generator"


def test_set_active_lora_isolates_inactive_adapters():
    worker = make_worker()
    peft_model = build_with_p1(worker)
    opt = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=1e-2)
    # nonzero lora_B so that every adapter would receive a gradient if it were trainable
    with torch.no_grad():
        for n, p in peft_model.named_parameters():
            if "lora_B" in n:
                p.normal_()

    # every parameter AdamW holds is a role adapter parameter (no dead adapter in the parameter walk)
    assert adapter_names_in(n for n, p in peft_model.named_parameters() if p.requires_grad) == set(ROLES)

    for role in ("aggregator", "generator"):
        worker.set_active_lora(role)
        assert worker.current_active_lora == role
        assert peft_model.active_adapter == role
        for n, p in peft_model.named_parameters():
            assert p.requires_grad == (f".{role}." in n), n

        opt.zero_grad(set_to_none=True)
        before = {n: p.detach().clone() for n, p in peft_model.named_parameters()}
        out = peft_model(input_ids=torch.randint(0, 64, (2, 8)))
        out.logits.float().pow(2).mean().backward()
        for n, p in peft_model.named_parameters():
            if f".{role}." in n:
                assert p.grad is not None, n
            else:
                assert p.grad is None, n
        opt.step()
        for n, p in peft_model.named_parameters():
            changed = not torch.equal(before[n], p.detach())
            assert changed == (f".{role}." in n and "lora" in n), n


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
    # an adapter outside agent_names (e.g. a dead ``default``) is refused
    stray = make_worker()
    peft_model = build_with_p1(stray)
    peft_model.add_adapter(adapter_name="default", peft_config=LoraConfig(**lora_kwargs()))
    with pytest.raises(ValueError, match="outside agent_names"):
        stray._marl_check_role_params()


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
    # no silent fallback: without adapter_name upstream's "default" is asked for, and it does not exist
    with pytest.raises(KeyError, match="default"):
        lora_utils.collect_lora_params(peft_model, layered_summon=False, base_sync_done=True)


def _save_patches():
    def fake_layered_summon(module, adapter_name="default"):
        return dict(get_peft_model_state_dict(module, adapter_name=adapter_name))

    fake_dist = SimpleNamespace(get_rank=lambda: 0, barrier=lambda: None)
    return (
        patch.object(W, "dist", fake_dist),
        patch.object(W, "fsdp_version", lambda m: 1),
        patch.object(W, "get_device_name", lambda: "cpu"),
        patch.object(W, "layered_summon_lora_params", fake_layered_summon),
    )


def test_save_single_lora_adapter_writes_peft_dir_per_role(tmp_path):
    worker = make_worker()
    peft_model = build_with_p1(worker)
    fill = {"generator": 0.75, "aggregator": 0.25}
    with torch.no_grad():
        for n, p in peft_model.named_parameters():
            for role, v in fill.items():
                if f"lora_B.{role}." in n:
                    p.fill_(v)

    p1, p2, p3, p4 = _save_patches()
    with p1, p2, p3, p4:
        for role in ROLES:
            worker.save_single_lora_adapter(role, str(tmp_path / f"lora_adapter_{role}"), global_step=3)
        with pytest.raises(ValueError, match="not found"):
            worker.save_single_lora_adapter("default", str(tmp_path / "x"))
    assert worker.current_active_lora == "generator"  # saving does not switch adapters

    for role in ROLES:
        out = tmp_path / f"lora_adapter_{role}"
        tensors = load_file(str(out / "adapter_model.safetensors"))
        assert tensors and all(r not in k for k in tensors for r in ROLES)
        assert any(k.endswith("lora_A.weight") for k in tensors)
        assert all(torch.all(v == fill[role]) for k, v in tensors.items() if "lora_B" in k)
        cfg = json.loads((out / "adapter_config.json").read_text())
        assert cfg["task_type"] == "CAUSAL_LM" and cfg["peft_type"] == "LORA" and cfg["r"] == 4
        assert sorted(cfg["target_modules"]) == ["q_proj", "v_proj"]

        reloaded = PeftModel.from_pretrained(tiny_model(), str(out))
        sd = get_peft_model_state_dict(reloaded)
        assert sd.keys() == tensors.keys()
        for k, v in tensors.items():
            assert torch.equal(sd[k], v), k
        # and it warm-starts the same role of a new IP model through initial_lora_adapters
        warm = make_worker(initial={role: str(out)})
        with patch.object(W, "copy_to_local", side_effect=lambda path, use_shm=False: path):
            again = build_with_p1(warm, seed=1)
        got = get_peft_model_state_dict(again, adapter_name=role)
        for k, v in tensors.items():
            assert torch.equal(got[k], v), k


def test_ip_state_dict_round_trip_has_only_role_adapters(tmp_path):
    """FSDP-free stand-in for the sharded checkpoint: the key set is the base model plus the role adapters."""
    src = build_with_p1(make_worker(), seed=0)
    with torch.no_grad():
        for n, p in src.named_parameters():
            if "lora_B" in n:
                p.normal_()
    sd = src.state_dict()
    lora_keys = [k for k in sd if "lora_" in k]
    assert lora_keys and adapter_names_in(lora_keys) == set(ROLES)
    base_keys = {k for k in sd if "lora_" not in k}
    assert base_keys == {k for k in build_like_fork_sp().state_dict() if "lora_" not in k}
    torch.save(sd, tmp_path / "model.pt")

    dst = build_with_p1(make_worker(), seed=7)
    result = dst.load_state_dict(torch.load(tmp_path / "model.pt"), strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    for k, v in dst.state_dict().items():
        assert torch.equal(v, sd[k]), k
    assert list(dst.peft_config) == ROLES

    # a state dict that still carries a ``default`` adapter (the research fork's IP layout) does not load
    fork_like = build_plain_ip(roles=["default", *ROLES])
    with pytest.raises(RuntimeError, match="default"):
        build_with_p1(make_worker(), seed=7).load_state_dict(fork_like.state_dict(), strict=True)


def test_ip_save_checkpoint_skips_shared_adapter_export(tmp_path):
    calls = []
    worker = make_worker()
    build_with_p1(worker)
    worker.checkpoint_manager = SimpleNamespace(save_checkpoint=lambda **kw: calls.append(("fsdp", kw)))

    def upstream_save(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        calls.append(("upstream", local_path))

    fake_dist = SimpleNamespace(get_rank=lambda: 0, barrier=lambda: None)
    with (
        patch.object(W, "dist", fake_dist),
        patch.object(fsdp_workers.ActorRolloutRefWorker, "save_checkpoint", upstream_save),
    ):
        worker.save_checkpoint(str(tmp_path / "actor"), global_step=5, max_ckpt_to_keep=2)
        assert calls == [
            (
                "fsdp",
                {"local_path": str(tmp_path / "actor"), "hdfs_path": None, "global_step": 5, "max_ckpt_to_keep": 2},
            )
        ]
        assert not (tmp_path / "actor" / "lora_adapter").exists()
        calls.clear()
        sp = make_worker(agent_names=["generator"], share_policy=True)
        sp.save_checkpoint(str(tmp_path / "sp"), global_step=5)
        assert calls == [("upstream", str(tmp_path / "sp"))]


def test_ip_load_checkpoint_names_fork_shards():
    def failing_load(self, local_path, hdfs_path=None, del_local_after_load=False):
        raise RuntimeError("Unexpected key(s) in state_dict: lora_A.default.weight")

    with patch.object(fsdp_workers.ActorRolloutRefWorker, "load_checkpoint", failing_load):
        with pytest.raises(RuntimeError, match="research fork") as info:
            make_worker().load_checkpoint("/ckpt/global_step_10/actor")
        assert "lora_A.default" in str(info.value.__cause__)
        with pytest.raises(RuntimeError, match="Unexpected key"):  # SP: upstream's error unchanged
            make_worker(share_policy=True).load_checkpoint("/ckpt/global_step_10/actor")


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
    assert calls == [ROLES, ["default"], ["default"], ["generator"]]
    assert fsdp_workers.get_peft_model is get_peft_model
    assert patches.applied_patches() == []
