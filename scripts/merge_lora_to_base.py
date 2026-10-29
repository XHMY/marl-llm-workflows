#!/usr/bin/env python3
# COPIED FROM fork scripts/merge_lora_to_base.py:1-235 and scripts/verify_merge.py:23-226 (verify_weights and
# its loaders) (research fork @f0240732); diff regions marked MARL(...).
"""Merge a LoRA adapter into its base model and save a standalone model (PEFT ``merge_and_unload``).

Used to build the ``*_s300`` init weights: the SA-RL generator adapter at step 300 merged into
the base model, saved as ``checkpoints/init_weight/<task>/qwen3_<size>_s300`` (``$MARL_INIT_WEIGHT_DIR``),
which ``training_metadata.json`` ``base_model`` tags refer to.

Usage::

    python scripts/merge_lora_to_base.py --base_model Qwen/Qwen3-1.7B \\
        --adapter_path ckpts/single_agent-qwen3_1.7b-multi_lora-math/global_step_300/actor/lora_adapter_generator \\
        --output_dir checkpoints/init_weight/math/qwen3_1.7b_s300 --verify

    # shorthand for the SA-RL step-300 generator of an experiment directory:
    python scripts/merge_lora_to_base.py --base_model Qwen/Qwen3-1.7B --sa_experiment_dir ckpts/single_agent-... \\
        --task math

A key check (merged keys equal the base model's; every LoRA target found) always runs;
``--verify`` adds the numeric check merged == base + scale * B @ A for every target and
identity for every other tensor. The adapter directory is never modified (the fork patched
a null ``base_model_name_or_path`` in place; the base model is passed explicitly here).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


# ---------------------------------------------------------------------------- verification
def load_adapter_weights(adapter_path: Path) -> dict[str, torch.Tensor]:
    """Adapter tensors from ``adapter_model.safetensors`` (or ``.bin``)."""
    st_path = adapter_path / "adapter_model.safetensors"
    bin_path = adapter_path / "adapter_model.bin"
    if st_path.exists():
        return load_safetensors(str(st_path))
    if bin_path.exists():
        return torch.load(str(bin_path), map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No adapter weights in {adapter_path} (adapter_model.safetensors or .bin)")


def load_merged_weights(merged_path: Path) -> dict[str, torch.Tensor]:
    """Tensors of a saved model (sharded or single safetensors / bin)."""
    for index_name, loader in (
        ("model.safetensors.index.json", lambda p: load_safetensors(str(p))),
        ("pytorch_model.bin.index.json", lambda p: torch.load(str(p), map_location="cpu", weights_only=True)),
    ):
        index_path = merged_path / index_name
        if index_path.exists():
            shards = set(json.loads(index_path.read_text())["weight_map"].values())
            state: dict[str, torch.Tensor] = {}
            for shard in sorted(shards):
                state.update(loader(merged_path / shard))
            return state
    if (merged_path / "model.safetensors").exists():
        return load_safetensors(str(merged_path / "model.safetensors"))
    if (merged_path / "pytorch_model.bin").exists():
        return torch.load(str(merged_path / "pytorch_model.bin"), map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No model weights found in {merged_path}")


def build_lora_key_mapping(adapter_weights: dict[str, torch.Tensor]) -> dict[str, tuple[str, str]]:
    """Base weight key -> (lora_A key, lora_B key).

    Adapter keys look like ``base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight``
    (no adapter name, the checkpoint layout of C14).
    """
    mapping: dict[str, tuple[str, str]] = {}
    for a_key in (k for k in adapter_weights if ".lora_A." in k):
        base_key = a_key.replace(".lora_A.weight", ".weight").replace("base_model.model.", "", 1)
        b_key = a_key.replace(".lora_A.", ".lora_B.")
        if b_key in adapter_weights:
            mapping[base_key] = (a_key, b_key)
    return mapping


def check_keys(
    base_keys: set[str],
    merged_keys: set[str],
    adapter_weights: dict[str, torch.Tensor],
    tied_keys: Iterable[str] = (),
) -> tuple[bool, list[str]]:
    """Cheap structural check: identical key sets, and every LoRA target present in the base model.

    ``tied_keys`` (e.g. ``lm_head.weight`` with tied embeddings) are in the in-memory state
    dict but not in the saved files, so they may be missing from ``merged_keys``.
    """
    msgs: list[str] = []
    ok = True
    extra, missing = merged_keys - base_keys, base_keys - merged_keys - set(tied_keys)
    if extra or missing:
        ok = False
        msgs.append(f"FAIL: merged has {len(extra)} extra / {len(missing)} missing keys vs base")
    else:
        msgs.append(f"PASS: key sets match ({len(base_keys)} keys)")
    n_a = sum(1 for k in adapter_weights if ".lora_A." in k)
    mapping = build_lora_key_mapping(adapter_weights)
    absent = [k for k in mapping if k not in base_keys]
    if n_a == 0 or len(mapping) != n_a or absent:
        ok = False
        msgs.append(f"FAIL: {n_a} lora_A tensors, {len(mapping)} A/B pairs, {len(absent)} targets missing in base")
    else:
        msgs.append(f"PASS: {len(mapping)} LoRA targets, all present in the base model")
    return ok, msgs


def verify_weights(
    base_weights: dict[str, torch.Tensor],
    adapter_weights: dict[str, torch.Tensor],
    merged_weights: dict[str, torch.Tensor],
    adapter_config: dict[str, Any],
    tol: float = 1e-3,
) -> tuple[bool, list[str]]:
    """Numeric check: LoRA targets equal ``base + (B @ A) * alpha / r`` (fp32, cast), others identical."""
    scaling = adapter_config["lora_alpha"] / adapter_config["r"]
    mapping = build_lora_key_mapping(adapter_weights)
    msgs = [f"LoRA config: r={adapter_config['r']}, alpha={adapter_config['lora_alpha']}, scaling={scaling}"]
    ok = True
    max_diff = 0.0
    n_bad = 0
    for base_key, (a_key, b_key) in sorted(mapping.items()):
        if base_key not in base_weights or base_key not in merged_weights:
            continue
        merged_w = merged_weights[base_key]
        delta = (adapter_weights[b_key].float() @ adapter_weights[a_key].float()) * scaling
        expected = (base_weights[base_key].float() + delta).to(merged_w.dtype)
        d = (merged_w.float() - expected.float()).abs().max().item()
        max_diff = max(max_diff, d)
        n_bad += d > tol
    if n_bad:
        ok = False
        msgs.append(f"FAIL: {n_bad}/{len(mapping)} LoRA targets differ by more than {tol} (max {max_diff:.3e})")
    else:
        msgs.append(f"PASS: all {len(mapping)} LoRA targets match (max diff {max_diff:.3e})")
    mismatched = [
        k
        for k in sorted(set(base_weights) & set(merged_weights))
        if k not in mapping and not torch.equal(base_weights[k].to(merged_weights[k].dtype), merged_weights[k])
    ]
    if mismatched:
        ok = False
        msgs.append(f"FAIL: {len(mismatched)} non-LoRA tensors differ, e.g. {mismatched[:3]}")
    else:
        msgs.append("PASS: all non-LoRA tensors identical")
    return ok, msgs


# ---------------------------------------------------------------------------- merge
def sa_adapter_from_experiment(experiment_dir: Path, step: int) -> Path:
    """``{experiment}/global_step_{step}/actor/lora_adapter_generator`` (SA-RL layout)."""
    # >>> MARL(layout): path from contracts instead of a hand-written string
    from marl_workflows.contracts import actor_ckpt_dir, adapter_dirname

    return Path(actor_ckpt_dir(str(experiment_dir), step)) / adapter_dirname("generator")
    # <<< MARL(layout)


def default_output_dir(task: str, base_model: str, step: int) -> Path:
    """``$MARL_INIT_WEIGHT_DIR/<task>/qwen3_<size>_s<step>``."""
    from marl_workflows.utils import init_weight_dir

    m = re.search(r"(\d+\.?\d*)[Bb]", base_model)
    if m is None:
        raise ValueError(f"cannot infer the model size from {base_model!r}; pass --output_dir")
    return Path(init_weight_dir()) / task / f"qwen3_{m.group(1).lower()}b_s{step}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge a LoRA adapter into the base model and save")
    p.add_argument("--base_model", required=True, help="base model name or path (e.g. Qwen/Qwen3-1.7B)")
    p.add_argument("--adapter_path", default=None, help="adapter dir (adapter_config.json + adapter_model.safetensors)")
    p.add_argument("--sa_experiment_dir", default=None, help="SA-RL experiment dir; uses global_step_<step> generator")
    p.add_argument("--step", type=int, default=300, help="step for --sa_experiment_dir (default 300)")
    p.add_argument("--task", choices=["math", "deepcoder"], default=None, help="task folder for the default output")
    p.add_argument("--output_dir", default=None, help="default: $MARL_INIT_WEIGHT_DIR/<task>/qwen3_<size>_s<step>")
    p.add_argument("--tokenizer_path", default=None, help="tokenizer source (default: base_model)")
    p.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    p.add_argument("--no_safe_serialization", action="store_false", dest="safe_serialization")
    p.add_argument("--verify", action="store_true", help="numeric merge verification after saving")
    args = p.parse_args(argv)
    if (args.adapter_path is None) == (args.sa_experiment_dir is None):
        p.error("pass exactly one of --adapter_path / --sa_experiment_dir")
    if args.output_dir is None and args.task is None:
        p.error("pass --output_dir or --task")
    return args


def main(argv: list[str] | None = None) -> int:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args(argv)
    adapter_path = (
        Path(args.adapter_path)
        if args.adapter_path
        else sa_adapter_from_experiment(Path(args.sa_experiment_dir), args.step)
    )
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.task, args.base_model, args.step)
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")
    adapter_config = json.loads(config_path.read_text())
    torch_dtype = DTYPES[args.dtype]

    # Loading the base in the target dtype is memory efficient; PEFT computes the CPU
    # LoRA delta in fp32 internally and casts back.
    print(f"Loading base model: {args.base_model} (dtype={args.dtype})")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch_dtype, device_map="cpu", trust_remote_code=True
    )
    base_state = {k: v.detach().clone() for k, v in base.state_dict().items()} if args.verify else None
    base_keys = set(base.state_dict().keys())
    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base, str(adapter_path), torch_dtype=torch_dtype, torch_device="cpu")
    print("Merging LoRA weights into base model...")
    merged = model.merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to: {output_dir}")
    merged.save_pretrained(str(output_dir), safe_serialization=args.safe_serialization)
    tokenizer_source = args.tokenizer_path or args.base_model
    AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True).save_pretrained(str(output_dir))

    adapter_w = load_adapter_weights(adapter_path)
    merged_w = load_merged_weights(output_dir)
    tied = set(getattr(merged, "_tied_weights_keys", None) or []) if merged.config.tie_word_embeddings else set()
    ok, msgs = check_keys(base_keys, set(merged_w), adapter_w, tied)
    if args.verify and base_state is not None:
        ok_w, msgs_w = verify_weights(base_state, adapter_w, merged_w, adapter_config)
        ok, msgs = ok and ok_w, msgs + msgs_w
    print("\n".join(msgs))
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name} ({f.stat().st_size / 2**20:.1f} MB)")
    print("Verification PASSED" if ok else "Verification FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
