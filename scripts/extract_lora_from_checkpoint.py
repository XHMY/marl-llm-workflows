#!/usr/bin/env python3
"""
Extract LoRA adapters from existing multi-agent checkpoints.

This script reads the full model checkpoint (model_world_size_N_rank_*.pt) and
extracts individual LoRA adapters for each agent.

Usage:
    # For multi-agent (share_policy=False):
    python scripts/extract_lora_from_checkpoint.py \
        --checkpoint_dir checkpoints/my_experiment/global_step_20/actor \
        --agent_names generator evaluator

    # For single-agent (share_policy=True):
    python scripts/extract_lora_from_checkpoint.py \
        --checkpoint_dir checkpoints/my_experiment/global_step_20/actor \
        --share_policy

    # Auto-detect adapters from checkpoint:
    python scripts/extract_lora_from_checkpoint.py \
        --checkpoint_dir checkpoints/my_experiment/global_step_20/actor \
        --auto_detect
"""

import argparse
import json
import os
import re
import shutil
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import save_file


def find_checkpoint_files(checkpoint_dir: str) -> list[str]:
    """Find all model checkpoint files in the directory."""
    checkpoint_dir = Path(checkpoint_dir)
    pattern = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt")

    files = []
    for f in checkpoint_dir.iterdir():
        if pattern.match(f.name):
            files.append(f)

    # Sort by rank
    files.sort(key=lambda x: int(pattern.match(x.name).group(2)))
    return files


def get_local_tensor(tensor):
    """Extract local tensor from DTensor or regular tensor."""
    if hasattr(tensor, 'to_local'):
        return tensor.to_local().cpu()
    elif hasattr(tensor, '_local_tensor'):
        return tensor._local_tensor.cpu()
    return tensor.cpu()


def detect_adapters(state_dict: dict) -> list[str]:
    """Detect adapter names from state dict keys."""
    adapters = set()
    # Pattern: ...lora_A.{adapter_name}.weight or ...lora_B.{adapter_name}.weight
    pattern = re.compile(r"\.lora_[AB]\.([^.]+)\.weight$")

    for key in state_dict.keys():
        match = pattern.search(key)
        if match:
            adapters.add(match.group(1))

    return sorted(adapters)


def extract_adapter_weights(
    checkpoint_files: list[Path],
    adapter_name: str,
) -> OrderedDict:
    """
    Extract LoRA weights for a specific adapter from sharded checkpoints.

    Args:
        checkpoint_files: List of checkpoint file paths (one per rank)
        adapter_name: Name of the adapter to extract

    Returns:
        OrderedDict with adapter weights in PEFT-compatible format
    """
    # Load all shards
    shards = []
    for f in checkpoint_files:
        state_dict = torch.load(f, map_location='cpu', weights_only=False)
        shards.append(state_dict)

    # Pattern to match LoRA keys for this adapter
    # e.g., base_model.model.model.layers.0.self_attn.q_proj.lora_A.generator.weight
    pattern = re.compile(
        rf"(.*\.lora_[AB])\.{re.escape(adapter_name)}\.weight$"
    )

    # Find all keys for this adapter
    adapter_keys = {}
    for key in shards[0].keys():
        match = pattern.match(key)
        if match:
            # Convert to PEFT format: ...lora_A.weight (without adapter name)
            peft_key = f"{match.group(1)}.weight"
            adapter_keys[key] = peft_key

    if not adapter_keys:
        raise ValueError(f"No LoRA weights found for adapter '{adapter_name}'")

    # Combine shards and extract weights
    lora_params = OrderedDict()
    for orig_key, peft_key in sorted(adapter_keys.items()):
        # Collect tensors from all ranks
        tensors = []
        for shard in shards:
            tensor = get_local_tensor(shard[orig_key])
            tensors.append(tensor)

        # Concatenate along dim 0 (FSDP shards along this dimension)
        combined = torch.cat(tensors, dim=0)
        lora_params[peft_key] = combined

    return lora_params


def create_adapter_config(
    checkpoint_dir: Path,
    adapter_name: str,
) -> dict:
    """
    Create adapter config from existing config or infer from weights.

    Args:
        checkpoint_dir: Path to checkpoint directory
        adapter_name: Name of the adapter

    Returns:
        PEFT adapter config dict
    """
    # Try to read existing config
    existing_config_path = checkpoint_dir / "lora_adapter" / "adapter_config.json"
    if existing_config_path.exists():
        with open(existing_config_path) as f:
            config = json.load(f)
        return config

    # Try huggingface directory
    hf_config_path = checkpoint_dir / "huggingface" / "adapter_config.json"
    if hf_config_path.exists():
        with open(hf_config_path) as f:
            config = json.load(f)
        return config

    # Return a minimal default config (user should verify)
    print(f"Warning: Could not find existing adapter_config.json, using minimal config")
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "inference_mode": False,
    }


def save_adapter(
    lora_params: OrderedDict,
    config: dict,
    save_path: Path,
):
    """Save adapter weights and config to directory."""
    save_path.mkdir(parents=True, exist_ok=True)

    # Save weights
    weights_path = save_path / "adapter_model.safetensors"
    save_file(lora_params, str(weights_path))
    print(f"  Saved weights to: {weights_path}")

    # Save config
    config_path = save_path / "adapter_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  Saved config to: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract LoRA adapters from multi-agent checkpoints"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to actor checkpoint directory (containing model_world_size_*.pt files)",
    )
    parser.add_argument(
        "--agent_names",
        type=str,
        nargs="+",
        default=None,
        help="Names of agents to extract (e.g., generator evaluator)",
    )
    parser.add_argument(
        "--share_policy",
        action="store_true",
        help="If set, only extract the 'default' adapter (for share_policy=True training)",
    )
    parser.add_argument(
        "--auto_detect",
        action="store_true",
        help="Auto-detect adapter names from checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: same as checkpoint_dir)",
    )
    parser.add_argument(
        "--remove_default_dir",
        action="store_true",
        default=True,
        help="Remove the default lora_adapter directory after extraction",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only detect adapters without saving",
    )

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir

    # Find checkpoint files
    checkpoint_files = find_checkpoint_files(checkpoint_dir)
    if not checkpoint_files:
        print(f"Error: No checkpoint files found in {checkpoint_dir}")
        return 1

    print(f"Found {len(checkpoint_files)} checkpoint shards")

    # Load first shard to detect adapters
    print("Loading checkpoint to detect adapters...")
    first_shard = torch.load(checkpoint_files[0], map_location='cpu', weights_only=False)
    detected_adapters = detect_adapters(first_shard)
    print(f"Detected adapters: {detected_adapters}")

    if args.dry_run:
        print("\nDry run - not saving any files")
        return 0

    # Determine which adapters to extract
    if args.share_policy:
        # Single-agent mode: only extract default
        adapters_to_extract = ["default"]
        print("\nShare policy mode: extracting only 'default' adapter")
    elif args.auto_detect:
        # Extract all detected adapters except 'default'
        adapters_to_extract = [a for a in detected_adapters if a != "default"]
        if not adapters_to_extract:
            print("Warning: No non-default adapters found, extracting 'default'")
            adapters_to_extract = ["default"]
        print(f"\nAuto-detect mode: extracting adapters {adapters_to_extract}")
    elif args.agent_names:
        adapters_to_extract = args.agent_names
        print(f"\nExtracting specified adapters: {adapters_to_extract}")
    else:
        print("Error: Must specify --agent_names, --share_policy, or --auto_detect")
        return 1

    # Verify all requested adapters exist
    for adapter in adapters_to_extract:
        if adapter not in detected_adapters:
            print(f"Error: Adapter '{adapter}' not found in checkpoint")
            print(f"Available adapters: {detected_adapters}")
            return 1

    # Get base config
    base_config = create_adapter_config(checkpoint_dir, adapters_to_extract[0])

    # Extract and save each adapter
    print(f"\nExtracting {len(adapters_to_extract)} adapter(s)...")
    for adapter_name in adapters_to_extract:
        print(f"\nProcessing adapter: {adapter_name}")

        # Extract weights
        lora_params = extract_adapter_weights(checkpoint_files, adapter_name)
        print(f"  Extracted {len(lora_params)} parameter tensors")

        # Determine save path
        if args.share_policy or adapter_name == "default":
            save_path = output_dir / "lora_adapter"
        else:
            save_path = output_dir / f"lora_adapter_{adapter_name}"

        # Save adapter
        save_adapter(lora_params, base_config, save_path)

    # Remove default lora_adapter directory if multi-agent mode
    if not args.share_policy and args.remove_default_dir:
        default_lora_path = output_dir / "lora_adapter"
        if default_lora_path.exists():
            shutil.rmtree(default_lora_path)
            print(f"\nRemoved default lora_adapter directory: {default_lora_path}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
