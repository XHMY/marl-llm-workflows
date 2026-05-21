"""Clean trajectory JSON files by removing token ID and logprob fields."""
import argparse
import json
import os


STEP_FIELDS = ("prompt_ids", "response_ids", "logprobs")
MODEL_OUTPUT_FIELDS = ("prompt_ids", "response_ids", "completion_ids")


def clean_episode(data: dict) -> bool:
    """Remove bloat fields from an episode dict. Returns True if anything changed."""
    changed = False
    for traj in data.get("trajectories", []):
        for step in traj.get("steps", []):
            for field in STEP_FIELDS:
                if field in step:
                    del step[field]
                    changed = True
            model_output = step.get("model_output")
            if isinstance(model_output, dict):
                for field in MODEL_OUTPUT_FIELDS:
                    if field in model_output:
                        del model_output[field]
                        changed = True
    return changed


def main():
    parser = argparse.ArgumentParser(description="Strip prompt_ids/response_ids/logprobs from trajectory JSONs")
    parser.add_argument("directory", nargs="?", default="evaluation_trajectories",
                        help="Directory containing trajectory JSON files (default: evaluation_trajectories/)")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without modifying files")
    args = parser.parse_args()

    # Gather all .json files recursively
    json_files = []
    for root, _, files in os.walk(args.directory):
        for f in files:
            if f.endswith(".json"):
                json_files.append(os.path.join(root, f))

    cleaned = 0
    total_saved = 0
    for path in sorted(json_files):
        orig_size = os.path.getsize(path)
        with open(path) as f:
            data = json.load(f)
        if clean_episode(data):
            if not args.dry_run:
                with open(path, "w") as f:
                    json.dump(data, f, indent=2)
                new_size = os.path.getsize(path)
            else:
                new_size = orig_size  # estimate not available in dry-run
            saved = orig_size - new_size
            total_saved += saved
            cleaned += 1
            label = "[dry-run] " if args.dry_run else ""
            print(f"{label}Cleaned {path} (saved {saved:,} bytes)")

    print(f"\nDone. {cleaned}/{len(json_files)} files cleaned. Total saved: {total_saved:,} bytes.")


if __name__ == "__main__":
    main()
