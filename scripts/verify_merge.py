#!/usr/bin/env python3
"""
Verify that a LoRA merge is numerically correct.

Compares weights and logits between:
  (a) base model + LoRA adapter (via PEFT)
  (b) merged model (standalone)

Usage:
    python scripts/verify_merge.py \
        --base_model Qwen/Qwen3-1.7B \
        --adapter_path checkpoints/.../lora_adapter_generator \
        --merged_model_path checkpoints/init_weight/qwen3_1.7b_s430
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_adapter_weights(adapter_path: Path) -> dict[str, torch.Tensor]:
    """Load LoRA adapter weights from safetensors or pytorch bin."""
    st_path = adapter_path / "adapter_model.safetensors"
    bin_path = adapter_path / "adapter_model.bin"
    if st_path.exists():
        return load_safetensors(str(st_path))
    elif bin_path.exists():
        return torch.load(str(bin_path), map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(
            f"No adapter weights found in {adapter_path}. "
            "Expected adapter_model.safetensors or adapter_model.bin"
        )


def load_merged_weights(merged_path: Path) -> dict[str, torch.Tensor]:
    """Load merged model weights from safetensors or pytorch bin."""
    # Check for safetensors index
    st_index = merged_path / "model.safetensors.index.json"
    if st_index.exists():
        with open(st_index) as f:
            index = json.load(f)
        shard_files = set(index["weight_map"].values())
        state_dict = {}
        for shard in shard_files:
            state_dict.update(load_safetensors(str(merged_path / shard)))
        return state_dict

    # Single safetensors file
    st_path = merged_path / "model.safetensors"
    if st_path.exists():
        return load_safetensors(str(st_path))

    # Pytorch bin
    bin_path = merged_path / "pytorch_model.bin"
    if bin_path.exists():
        return torch.load(str(bin_path), map_location="cpu", weights_only=True)

    # Pytorch bin index (sharded)
    bin_index = merged_path / "pytorch_model.bin.index.json"
    if bin_index.exists():
        with open(bin_index) as f:
            index = json.load(f)
        shard_files = set(index["weight_map"].values())
        state_dict = {}
        for shard in shard_files:
            state_dict.update(
                torch.load(str(merged_path / shard), map_location="cpu", weights_only=True)
            )
        return state_dict

    raise FileNotFoundError(f"No model weights found in {merged_path}")


def build_lora_key_mapping(
    adapter_weights: dict[str, torch.Tensor],
    target_modules: list[str],
) -> dict[str, tuple[str, str]]:
    """Map base model weight keys to their (lora_A, lora_B) adapter weight keys.

    Returns:
        {base_key: (lora_A_key, lora_B_key)}
    """
    # Adapter keys look like: base_model.model.{base_key}.lora_A.weight
    mapping = {}
    lora_a_keys = [k for k in adapter_weights if ".lora_A." in k]

    for a_key in lora_a_keys:
        # Extract the base parameter name
        # e.g. "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        # -> "model.layers.0.self_attn.q_proj.weight"
        parts = a_key.replace(".lora_A.weight", ".weight")
        base_key = parts.replace("base_model.model.", "", 1)
        b_key = a_key.replace(".lora_A.", ".lora_B.")
        if b_key in adapter_weights:
            mapping[base_key] = (a_key, b_key)

    return mapping


def verify_weights(
    base_weights: dict[str, torch.Tensor],
    adapter_weights: dict[str, torch.Tensor],
    merged_weights: dict[str, torch.Tensor],
    adapter_config: dict,
) -> tuple[bool, list[str]]:
    """Verify merged weights match base + LoRA delta.

    Returns:
        (passed, messages)
    """
    messages = []
    passed = True

    lora_alpha = adapter_config["lora_alpha"]
    lora_r = adapter_config["r"]
    scaling = lora_alpha / lora_r
    target_modules = adapter_config["target_modules"]

    messages.append(f"LoRA config: r={lora_r}, alpha={lora_alpha}, scaling={scaling}")
    messages.append(f"Target modules: {target_modules}")

    # Build mapping from base keys to adapter keys
    lora_mapping = build_lora_key_mapping(adapter_weights, target_modules)
    messages.append(f"Found {len(lora_mapping)} LoRA target weights in adapter")

    # Stage 1: Key structure check
    messages.append("\n=== Stage 1: Key Structure Check ===")
    base_keys = set(base_weights.keys())
    merged_keys = set(merged_weights.keys())
    extra_keys = merged_keys - base_keys
    missing_keys = base_keys - merged_keys
    if extra_keys:
        messages.append(f"FAIL: Merged model has {len(extra_keys)} extra keys: {sorted(extra_keys)[:5]}...")
        passed = False
    if missing_keys:
        messages.append(f"FAIL: Merged model is missing {len(missing_keys)} keys: {sorted(missing_keys)[:5]}...")
        passed = False
    if not extra_keys and not missing_keys:
        messages.append(f"PASS: Key sets match ({len(base_keys)} keys)")

    # Stage 2: LoRA weight verification
    messages.append("\n=== Stage 2: LoRA Weight Verification ===")
    lora_max_diffs = []
    lora_mean_diffs = []
    lora_fail_count = 0

    for base_key, (a_key, b_key) in sorted(lora_mapping.items()):
        if base_key not in base_weights:
            messages.append(f"  SKIP: {base_key} not in base model")
            continue
        if base_key not in merged_weights:
            messages.append(f"  SKIP: {base_key} not in merged model")
            continue

        base_w = base_weights[base_key]
        merged_w = merged_weights[base_key]
        lora_a = adapter_weights[a_key]
        lora_b = adapter_weights[b_key]

        # Compute expected: base + (B @ A) * scaling, in the merged model's dtype
        # Do the LoRA matmul in float32 for precision, then cast
        delta = (lora_b.float() @ lora_a.float()) * scaling
        expected = base_w.float() + delta
        expected = expected.to(merged_w.dtype)

        diff = (merged_w.float() - expected.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        lora_max_diffs.append(max_diff)
        lora_mean_diffs.append(mean_diff)

        # For bf16, 1 ULP ≈ 2^-8 for values near 1.0, so allow small tolerance
        # But for a correct merge, we expect exact match (0 diff) since both
        # operations should produce the same bf16 result
        if max_diff > 1e-3:
            short_key = base_key.split("model.layers.")[-1] if "model.layers." in base_key else base_key
            messages.append(
                f"  WARN: {short_key}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}"
            )
            lora_fail_count += 1

    if lora_max_diffs:
        overall_max = max(lora_max_diffs)
        overall_mean = sum(lora_mean_diffs) / len(lora_mean_diffs)
        messages.append(f"  Overall: max_diff={overall_max:.6e}, mean_diff={overall_mean:.6e}")
        messages.append(f"  Weights with diff > 1e-3: {lora_fail_count}/{len(lora_max_diffs)}")
        if overall_max > 1e-3:
            messages.append(f"  FAIL: LoRA weights have significant differences")
            passed = False
        else:
            messages.append(f"  PASS: All LoRA weights match within tolerance")
    else:
        messages.append("  WARN: No LoRA weights found to verify")

    # Stage 3: Non-LoRA weight verification
    messages.append("\n=== Stage 3: Non-LoRA Weight Verification ===")
    non_lora_mismatch = 0
    non_lora_checked = 0
    non_lora_max_diff = 0.0

    for key in sorted(base_keys & merged_keys):
        if key in lora_mapping:
            continue
        non_lora_checked += 1
        base_w = base_weights[key]
        merged_w = merged_weights[key]

        if base_w.dtype != merged_w.dtype:
            # dtype mismatch — compare after casting to same type
            diff = (merged_w.float() - base_w.float()).abs().max().item()
            if diff > 0:
                non_lora_mismatch += 1
                non_lora_max_diff = max(non_lora_max_diff, diff)
                if non_lora_mismatch <= 3:
                    messages.append(
                        f"  WARN: {key}: dtype mismatch ({base_w.dtype} vs {merged_w.dtype}), max_diff={diff:.6e}"
                    )
        elif not torch.equal(base_w, merged_w):
            diff = (merged_w.float() - base_w.float()).abs().max().item()
            non_lora_mismatch += 1
            non_lora_max_diff = max(non_lora_max_diff, diff)
            if non_lora_mismatch <= 3:
                messages.append(f"  WARN: {key}: max_diff={diff:.6e}")

    if non_lora_mismatch > 0:
        messages.append(
            f"  FAIL: {non_lora_mismatch}/{non_lora_checked} non-LoRA weights differ (max_diff={non_lora_max_diff:.6e})"
        )
        passed = False
    else:
        messages.append(f"  PASS: All {non_lora_checked} non-LoRA weights are identical")

    return passed, messages


def verify_logits(
    base_model_path: str,
    adapter_path: str,
    merged_model_path: str,
    torch_dtype: torch.dtype,
) -> tuple[bool, list[str]]:
    """Validate merged model logits against an in-memory merged reference.

    Notes:
      - In low precision (bf16/fp16), PEFT runtime logits and merged-runtime
        logits can differ due to different floating-point operation order.
      - The correctness check here is disk-merged vs in-memory merged reference.
    """
    from peft import PeftModel

    messages = []
    messages.append("\n=== Stage 4: Logit Comparison ===")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    test_prompt = "The square root of 144 is"
    inputs = tokenizer(test_prompt, return_tensors="pt")

    # Load PEFT model and get runtime logits (informational only)
    messages.append("  Loading base + adapter (PEFT)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch_dtype, device_map="cpu", trust_remote_code=True
    )
    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch_dtype,
        torch_device="cpu",
    )
    peft_model.eval()

    with torch.no_grad():
        peft_out = peft_model(**inputs)
        peft_logits = peft_out.logits

    # Merge the same PEFT model in-memory and get merged-reference logits.
    messages.append("  Creating in-memory merged reference...")
    merged_ref_model = peft_model.merge_and_unload()
    merged_ref_model.eval()
    with torch.no_grad():
        merged_ref_logits = merged_ref_model(**inputs).logits

    # Free PEFT/ref model memory before loading merged-from-disk
    del peft_model, merged_ref_model, base_model

    # Load merged model from disk
    messages.append("  Loading merged model from disk...")
    merged_model = AutoModelForCausalLM.from_pretrained(
        merged_model_path, torch_dtype=torch_dtype, device_map="cpu", trust_remote_code=True
    )
    merged_model.eval()

    with torch.no_grad():
        merged_out = merged_model(**inputs)
        merged_logits = merged_out.logits

    del merged_model

    def compare_logits(
        lhs_name: str, lhs_logits: torch.Tensor, rhs_name: str, rhs_logits: torch.Tensor
    ) -> tuple[float, float, float, bool]:
        diff = (lhs_logits.float() - rhs_logits.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        lhs_last = lhs_logits[0, -1].float()
        rhs_last = rhs_logits[0, -1].float()
        cos_sim = torch.nn.functional.cosine_similarity(
            lhs_last.unsqueeze(0), rhs_last.unsqueeze(0)
        ).item()

        lhs_tokens = lhs_logits[0, -1].topk(5).indices.tolist()
        rhs_tokens = rhs_logits[0, -1].topk(5).indices.tolist()
        top5_match = lhs_tokens == rhs_tokens

        messages.append(
            f"  {lhs_name} vs {rhs_name}: max={max_diff:.6e}, mean={mean_diff:.6e}, cos={cos_sim:.8f}, top5_match={top5_match}"
        )
        messages.append(
            f"    {lhs_name} top-5: {lhs_tokens} (decoded: {[tokenizer.decode(t) for t in lhs_tokens]})"
        )
        messages.append(
            f"    {rhs_name} top-5: {rhs_tokens} (decoded: {[tokenizer.decode(t) for t in rhs_tokens]})"
        )
        return max_diff, mean_diff, cos_sim, top5_match

    messages.append("  Informational: PEFT runtime vs merged runtime")
    compare_logits("PEFT", peft_logits, "MergedDisk", merged_logits)

    messages.append("  Verification target: in-memory merged reference vs merged disk")
    ref_max, _, _, ref_top5_match = compare_logits(
        "MergedRef", merged_ref_logits, "MergedDisk", merged_logits
    )

    # For the same merge path/config, these should be effectively identical.
    passed = ref_max < 1e-5 and ref_top5_match
    if passed:
        messages.append("  PASS: Disk merged logits match in-memory merged reference")
    else:
        messages.append(
            f"  FAIL: Disk merged logits differ from merged reference (max_diff={ref_max:.6e}, top5_match={ref_top5_match})"
        )
    messages.append(
        "  Note: PEFT runtime vs merged-runtime diff can be non-zero in bf16/fp16 due to different floating-point paths."
    )

    return passed, messages


def main():
    parser = argparse.ArgumentParser(description="Verify LoRA merge correctness")
    parser.add_argument("--base_model", type=str, required=True, help="Base model name or path")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--merged_model_path", type=str, required=True, help="Path to merged model directory")
    parser.add_argument(
        "--skip_logits", action="store_true",
        help="Skip logit comparison (faster, weight-only check)"
    )
    args = parser.parse_args()

    adapter_path = Path(args.adapter_path)
    merged_path = Path(args.merged_model_path)

    # Load adapter config
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        print(f"ERROR: adapter_config.json not found in {adapter_path}")
        sys.exit(1)
    with open(config_path) as f:
        adapter_config = json.load(f)

    # Determine dtype from merged model config
    model_config_path = merged_path / "config.json"
    torch_dtype = torch.bfloat16  # default
    if model_config_path.exists():
        with open(model_config_path) as f:
            model_config = json.load(f)
        dtype_str = model_config.get("torch_dtype", "bfloat16")
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        torch_dtype = dtype_map.get(dtype_str, torch.bfloat16)
    print(f"Using dtype: {torch_dtype}")

    # Load weights
    print("Loading base model weights...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch_dtype, device_map="cpu", trust_remote_code=True
    )
    base_weights = dict(base_model.state_dict())
    del base_model

    print("Loading adapter weights...")
    adapter_weights = load_adapter_weights(adapter_path)

    print("Loading merged model weights...")
    merged_weights = load_merged_weights(merged_path)

    # Run weight verification
    weight_passed, weight_msgs = verify_weights(base_weights, adapter_weights, merged_weights, adapter_config)
    for msg in weight_msgs:
        print(msg)

    # Free weight dicts before logit check
    del base_weights, adapter_weights, merged_weights

    # Run logit verification
    logit_passed = True
    if not args.skip_logits:
        logit_passed, logit_msgs = verify_logits(
            args.base_model, str(adapter_path), str(merged_path), torch_dtype
        )
        for msg in logit_msgs:
            print(msg)
    else:
        print("\n=== Stage 4: Logit Comparison (SKIPPED) ===")

    # Summary
    print("\n" + "=" * 50)
    overall = weight_passed and logit_passed
    if overall:
        print("VERDICT: PASS — Merge is numerically correct")
    else:
        print("VERDICT: FAIL — Merge has issues")
        if not weight_passed:
            print("  - Weight verification failed")
        if not logit_passed:
            print("  - Logit verification failed")
    print("=" * 50)

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
