"""Convert share_policy checkpoints to multi-lora format.

For each global_step_* directory:
1. Rename actor/lora_adapter/ -> actor/lora_adapter_generator/
2. Add 'generator' adapter keys (cloned from 'default') to model .pt files
"""

import argparse
import os
import re
import shutil

import torch
import torch.distributed.tensor


def convert_step(step_dir: str, agent_name: str = "generator", dry_run: bool = False):
    actor_dir = os.path.join(step_dir, "actor")
    if not os.path.isdir(actor_dir):
        print(f"  Skipping {step_dir}: no actor/ directory")
        return

    # Step 1: Rename lora_adapter -> lora_adapter_{agent_name}
    src_lora = os.path.join(actor_dir, "lora_adapter")
    dst_lora = os.path.join(actor_dir, f"lora_adapter_{agent_name}")

    if os.path.isdir(src_lora) and not os.path.isdir(dst_lora):
        print(f"  Renaming lora_adapter -> lora_adapter_{agent_name}")
        if not dry_run:
            os.rename(src_lora, dst_lora)
    elif os.path.isdir(dst_lora):
        print(f"  lora_adapter_{agent_name} already exists, skipping rename")
    else:
        print(f"  WARNING: no lora_adapter/ found in {actor_dir}")

    # Step 2: Add agent keys to model .pt files
    pt_files = sorted(
        f for f in os.listdir(actor_dir)
        if f.startswith("model_world_size_") and f.endswith(".pt")
    )

    for pt_file in pt_files:
        pt_path = os.path.join(actor_dir, pt_file)
        print(f"  Processing {pt_file}...")

        ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)

        # Find all default LoRA keys
        default_keys = [
            k for k in ckpt.keys()
            if re.search(r"lora_[AB]\.default\.weight$", k)
        ]

        if not default_keys:
            print(f"    No default LoRA keys found, skipping")
            continue

        # Check if agent keys already exist
        agent_key_sample = default_keys[0].replace(".default.", f".{agent_name}.")
        if agent_key_sample in ckpt:
            print(f"    Agent keys already present, skipping")
            continue

        # Add agent keys as clones of default keys
        added = 0
        for key in default_keys:
            new_key = key.replace(".default.", f".{agent_name}.")
            ckpt[new_key] = ckpt[key].clone()
            added += 1

        print(f"    Added {added} {agent_name} keys (total LoRA keys: {len(default_keys) + added})")

        if not dry_run:
            torch.save(ckpt, pt_path)


def main():
    parser = argparse.ArgumentParser(description="Convert share_policy checkpoints to multi-lora format")
    parser.add_argument("checkpoint_dir", help="Root checkpoint directory containing global_step_* subdirs")
    parser.add_argument("--agent-name", default="generator", help="Agent name for the new LoRA adapter (default: generator)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without making changes")
    parser.add_argument("--steps", nargs="*", type=int, help="Only convert specific steps (e.g., --steps 10 20 30)")
    args = parser.parse_args()

    root = args.checkpoint_dir
    if not os.path.isdir(root):
        print(f"Error: {root} is not a directory")
        return

    # Find all global_step_* directories
    step_dirs = sorted(
        [d for d in os.listdir(root) if d.startswith("global_step_")],
        key=lambda x: int(x.split("_")[-1]),
    )

    if args.steps:
        step_dirs = [d for d in step_dirs if int(d.split("_")[-1]) in args.steps]

    print(f"Found {len(step_dirs)} steps to convert")
    if args.dry_run:
        print("DRY RUN - no changes will be made\n")

    for step_name in step_dirs:
        step_path = os.path.join(root, step_name)
        print(f"\n[{step_name}]")
        convert_step(step_path, agent_name=args.agent_name, dry_run=args.dry_run)

    print("\nDone!")


if __name__ == "__main__":
    main()
