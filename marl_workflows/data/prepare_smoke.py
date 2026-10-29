"""Register the tiny ``smoke_math`` set (train 32 / test 8) from the packaged fixture; no download.

``fixtures/smoke_math.jsonl`` holds 40 short arithmetic problems in the Math schema
(``question``, ``final_answer``, ``ground_truth``, ``data_source``). The first 32 rows are the
train split and the last 8 the test split. Used by the 1-GPU smoke run (``scripts/smoke_gpu.sh``).

Usage::

    python -m marl_workflows.data.prepare_smoke [--verify | --verify-only]

``--verify`` checks the fixture's sha256 and the registered row counts against
``checksums.json`` (``files`` and ``datasets`` sections).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from marl_workflows.data.build_deepcoder_passing_index import sha256_file
from marl_workflows.data.prepare_dapo_math import load_checksums, register_split, run_prepare_cli

__all__ = ["FIXTURE_PATH", "N_TRAIN", "load_fixture", "main", "prepare_smoke"]

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "smoke_math.jsonl"
N_TRAIN = 32


def load_fixture(path: str | Path = FIXTURE_PATH) -> list[dict[str, Any]]:
    """Read the JSONL fixture."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def prepare_smoke(path: str | Path = FIXTURE_PATH) -> tuple[Any, Any]:
    """Register ``smoke_math`` train / test."""
    rows = load_fixture(path)
    if len(rows) <= N_TRAIN:
        raise ValueError(f"{path} must hold more than {N_TRAIN} rows, got {len(rows)}")
    train = register_split("smoke_math", rows[:N_TRAIN], "train")
    test = register_split("smoke_math", rows[N_TRAIN:], "test")
    return train, test


def _check_fixture() -> None:
    expected = load_checksums().get("files", {}).get("fixtures/smoke_math.jsonl", {}).get("sha256")
    got = sha256_file(FIXTURE_PATH)
    if expected is not None and got != expected:
        raise SystemExit(f"fixture sha256 {got} != checksums.json {expected}")


def main(argv: list[str] | None = None) -> int:
    def prepare(_args: object) -> None:
        _check_fixture()
        prepare_smoke()

    return run_prepare_cli(
        "Register smoke_math (train 32 / test 8) from the packaged fixture.",
        prepare,
        [("smoke_math", ("train", "test"))],
        argv,
    )


if __name__ == "__main__":
    sys.exit(main())
