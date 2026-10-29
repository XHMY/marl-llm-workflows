"""Prepare and register DAPO-Math-17k (``dapo_math`` train / test) under ``$MARL_DATA_DIR``.

Port of the research fork's ``examples/math_reasoning/prepare_dapo_math_data.py``: the
``en`` config of ``open-r1/DAPO-Math-17k-Processed`` with ``prompt -> question`` and
``solution -> final_answer``, split with ``train_test_split(test_size=0.1, seed=42)``. The
test split (1412 problems) is the paper's Math validation set; the train split has 12704.

This module also holds the helpers every ``prepare_*`` module shares (checksums,
registration, the ``--verify`` command line), so that the data package needs no extra
module.

Usage::

    python -m marl_workflows.data.prepare_dapo_math            # download + register
    python -m marl_workflows.data.prepare_dapo_math --verify   # ... and check checksums.json
    python -m marl_workflows.data.prepare_dapo_math --verify-only

Checksum definition (``checksums.json``): per row, ``sha256(json.dumps([question, answer]))``
where ``answer`` is ``ground_truth`` when that column exists and is not null, else
``final_answer`` (non-strings via ``str``); ``sha256_sorted`` hashes the sorted row digests
(order-free content identity), ``sha256_ordered`` the digests in file order (exact
reproduction, which matters for data-loader order).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "CHECKSUMS_PATH",
    "CONFIG",
    "DATASET_NAME",
    "GOLDEN_SOURCES",
    "REVISION",
    "answer_of",
    "checksum_parquet",
    "checksum_rows",
    "load_checksums",
    "main",
    "prepare_dapo_math_data",
    "register_split",
    "run_prepare_cli",
    "verify_registered",
    "write_golden_checksums",
]

DATASET_NAME = "open-r1/DAPO-Math-17k-Processed"
CONFIG = "en"
# HfApi().dataset_info(DATASET_NAME).sha on 2026-10-29; equals the snapshot the fork's files came from.
REVISION = "31dd309567e3da778038cc87d868b6097a3ccf68"
TEST_SIZE = 0.1
SEED = 42

CHECKSUMS_PATH = Path(__file__).resolve().parent / "checksums.json"

# Registry name -> sub-directory of the research fork's rllm/data/datasets/ holding the golden
# parquet files. MATH-500 was registered there as "math500_answer" (answer-only ground truth).
GOLDEN_SOURCES: dict[str, str] = {
    "dapo_math": "dapo_math",
    "aime2025": "aime2025",
    "aime2024": "aime2024",
    "math500": "math500_answer",
    "deepcoder_primeintellect": "deepcoder_primeintellect",
}

# ======================================================================================
# Shared helpers (used by every prepare_* module)
# ======================================================================================


def answer_of(row: Mapping[str, Any]) -> str:
    """The graded answer of a row: ``ground_truth`` if present and not null, else ``final_answer``."""
    value = row.get("ground_truth")
    if value is None:
        value = row.get("final_answer")
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def checksum_rows(rows: Iterable[tuple[str, str]]) -> dict[str, Any]:
    """Row count plus order-free and ordered sha256 over ``(question, answer)`` pairs."""
    digests: list[str] = []
    for question, answer in rows:
        payload = json.dumps([question, answer], ensure_ascii=False, separators=(",", ":"))
        digests.append(hashlib.sha256(payload.encode("utf-8")).hexdigest())
    ordered = hashlib.sha256("\n".join(digests).encode("ascii")).hexdigest()
    sorted_ = hashlib.sha256("\n".join(sorted(digests)).encode("ascii")).hexdigest()
    return {"rows": len(digests), "sha256_sorted": sorted_, "sha256_ordered": ordered}


def checksum_parquet(path: str | Path) -> dict[str, Any]:
    """:func:`checksum_rows` of a registered (non-verl) parquet file, reading only the needed columns."""
    import polars as pl

    schema = pl.read_parquet_schema(str(path))
    cols = [c for c in ("question", "ground_truth", "final_answer") if c in schema]
    if "question" not in cols:
        raise ValueError(f"{path} has no 'question' column")
    df = pl.read_parquet(str(path), columns=cols)
    return checksum_rows((str(r["question"]), answer_of(r)) for r in df.iter_rows(named=True))


def load_checksums(path: str | Path = CHECKSUMS_PATH) -> dict[str, Any]:
    """Read ``checksums.json``."""
    with open(path) as f:
        return json.load(f)


def register_split(name: str, data: Any, split: str) -> Any:
    """Register one split with :class:`MarlDatasetRegistry` (parquet + ``_verl`` parquet under ``$MARL_DATA_DIR``)."""
    from marl_workflows.data.registry import MarlDatasetRegistry

    ds = MarlDatasetRegistry.register_dataset(name, data, split)
    print(f"registered {name}/{split}: {len(ds)} rows -> {MarlDatasetRegistry.data_path(name, split)}")
    return ds


def verify_registered(
    name: str,
    splits: Sequence[str],
    checksums: Mapping[str, Any] | None = None,
) -> list[str]:
    """Compare registered splits with ``checksums.json``; return a list of problems (empty = OK)."""
    from marl_workflows.data.registry import MarlDatasetRegistry

    golden = (checksums or load_checksums())["datasets"].get(name, {})
    problems: list[str] = []
    for split in splits:
        expected = golden.get(split)
        path = MarlDatasetRegistry.data_path(name, split)
        if expected is None:
            problems.append(f"{name}/{split}: no entry in checksums.json")
            continue
        if path is None:
            problems.append(f"{name}/{split}: not registered under {MarlDatasetRegistry._REGISTRY_FILE}")
            continue
        got = checksum_parquet(path)
        for key in ("rows", "sha256_sorted", "sha256_ordered"):
            if got[key] != expected[key]:
                problems.append(f"{name}/{split}: {key} {got[key]} != expected {expected[key]}")
        status = "OK" if not any(p.startswith(f"{name}/{split}:") for p in problems) else "MISMATCH"
        print(f"verify {name}/{split}: {status} rows={got['rows']} sha256_sorted={got['sha256_sorted'][:16]}...")
    return problems


def run_prepare_cli(
    description: str,
    prepare: Callable[[argparse.Namespace], None],
    targets: Sequence[tuple[str, Sequence[str]]],
    argv: list[str] | None = None,
    add_arguments: Callable[[argparse.ArgumentParser], None] | None = None,
) -> int:
    """Shared ``main``: prepare (unless ``--verify-only``), then optionally verify ``targets``."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--verify", action="store_true", help="after preparing, check checksums.json")
    parser.add_argument("--verify-only", action="store_true", help="only check the registered files")
    if add_arguments is not None:
        add_arguments(parser)
    args = parser.parse_args(argv)
    if not args.verify_only:
        prepare(args)
    if args.verify or args.verify_only:
        problems: list[str] = []
        for name, splits in targets:
            problems.extend(verify_registered(name, splits))
        if problems:
            print("VERIFY FAILED:\n  " + "\n  ".join(problems), file=sys.stderr)
            return 1
        print("VERIFY OK")
    return 0


def write_golden_checksums(fork_datasets_dir: str | Path, out: str | Path = CHECKSUMS_PATH) -> dict[str, Any]:
    """Recompute the ``datasets`` section of ``checksums.json`` from the fork's registered parquet files.

    Other top-level sections (``files``, ``_meta``) and datasets without a fork source
    (``smoke_math``) are preserved.
    """
    root = Path(fork_datasets_dir)
    existing = load_checksums(out) if Path(out).exists() else {}
    datasets: dict[str, Any] = {k: v for k, v in existing.get("datasets", {}).items() if k not in GOLDEN_SOURCES}
    for name, subdir in GOLDEN_SOURCES.items():
        for parquet in sorted((root / subdir).glob("*.parquet")):
            if parquet.stem.endswith("_verl"):
                continue
            datasets.setdefault(name, {})[parquet.stem] = checksum_parquet(parquet)
            print(f"{name}/{parquet.stem}: {datasets[name][parquet.stem]}")
    existing["datasets"] = datasets
    Path(out).write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
    return existing


# ======================================================================================
# DAPO-Math
# ======================================================================================


def prepare_dapo_math_data(
    dataset_name: str = DATASET_NAME,
    config: str = CONFIG,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
    revision: str | None = REVISION,
) -> tuple[Any, Any]:
    """Download, map, split and register ``dapo_math`` (fork semantics, pinned revision)."""
    from datasets import load_dataset

    print(f"Loading dataset: {dataset_name} (config={config}, revision={revision})")
    ds = load_dataset(dataset_name, config, revision=revision)["train"]

    # Field map verbatim from the fork (prepare_dapo_math_data.py:39-47).
    def map_fields(example: dict) -> dict:
        return {
            "question": example["prompt"],
            "final_answer": example["solution"],
            "data_source": example.get("data_source", "dapo_math"),
            "ability": example.get("ability", ""),
            "reward_model": example.get("reward_model", {}),
            "extra_info": example.get("extra_info", {}),
        }

    ds = ds.map(map_fields)
    split = ds.train_test_split(test_size=test_size, seed=seed)
    train = register_split("dapo_math", split["train"], "train")
    test = register_split("dapo_math", split["test"], "test")
    return train, test


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--write-checksums"]:
        # Maintenance: python -m marl_workflows.data.prepare_dapo_math --write-checksums <fork>/rllm/data/datasets
        if len(argv) != 2:
            raise SystemExit("usage: --write-checksums FORK_DATASETS_DIR")
        write_golden_checksums(argv[1])
        return 0
    return run_prepare_cli(
        "Prepare and register dapo_math (train/test).",
        lambda _args: prepare_dapo_math_data(),
        [("dapo_math", ("train", "test"))],
        argv,
    )


if __name__ == "__main__":
    sys.exit(main())
