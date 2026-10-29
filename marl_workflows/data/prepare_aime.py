"""Prepare and register the AIME test sets ``aime2025`` and ``aime2024`` (``test`` split).

- ``aime2025``: port of the fork's ``examples/math_reasoning/prepare_aime_test_data.py``
  (``MathArena/aime_2025``; all source columns kept, plus ``question``, ``final_answer`` (int)
  and ``data_source="math"``). It is the in-training validation set of every Math recipe.
- ``aime2024``: the form the fork actually registered (``analysis/revision_testsets/
  prepare_test_sets.py``, which superseded ``prepare_aime2024_data.py``): ``MathArena/
  aime_2024_I`` followed by ``aime_2024_II``, rows ``{question, final_answer, ground_truth,
  data_source="math"}`` with the answer as a string. Independent test set; not used for
  checkpoint selection.

Usage::

    python -m marl_workflows.data.prepare_aime [--only aime2025|aime2024] [--verify | --verify-only]
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from marl_workflows.data.prepare_dapo_math import register_split, run_prepare_cli

__all__ = ["AIME2024_SOURCES", "AIME2025_SOURCE", "main", "prepare_aime2024", "prepare_aime2025"]

# (repo, revision). Revisions are the snapshots the fork's files were built from; the content
# check in checksums.json guards against silent upstream edits.
AIME2025_SOURCE = ("MathArena/aime_2025", "beca2d7875cf92cdac07acefbccad3c4d16e2916")
AIME2024_SOURCES = (
    ("MathArena/aime_2024_I", "ea5b061c3e8039dc9858defaafc407d04b995e9f"),
    ("MathArena/aime_2024_II", "29d5d31e9b46e215fc24d9b2a3047506823dd101"),
)


def prepare_aime2025() -> Any:
    """Register ``aime2025``/``test`` (30 problems)."""
    from datasets import load_dataset

    repo, revision = AIME2025_SOURCE
    test_dataset = load_dataset(repo, split="train", revision=revision)

    def preprocess_fn(example: dict, idx: int) -> dict:
        return {"question": example["problem"], "final_answer": example["answer"], "data_source": "math"}

    test_dataset = test_dataset.map(preprocess_fn, with_indices=True)
    return register_split("aime2025", test_dataset, "test")


def prepare_aime2024() -> Any:
    """Register ``aime2024``/``test`` (30 problems, answers as strings)."""
    from datasets import concatenate_datasets, load_dataset

    parts = [load_dataset(repo, split="train", revision=rev) for repo, rev in AIME2024_SOURCES]
    aime = concatenate_datasets(parts)
    rows = [
        {
            "question": aime[i]["problem"],
            "final_answer": str(aime[i]["answer"]),
            "ground_truth": str(aime[i]["answer"]),
            "data_source": "math",
        }
        for i in range(len(aime))
    ]
    return register_split("aime2024", rows, "test")


_PREPARERS = {"aime2025": prepare_aime2025, "aime2024": prepare_aime2024}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    # --only must be known before the targets list is built.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--only", choices=sorted(_PREPARERS), default=None)
    known, _ = pre.parse_known_args(argv)
    names = [known.only] if known.only else list(_PREPARERS)

    def prepare(_args: argparse.Namespace) -> None:
        for name in names:
            _PREPARERS[name]()

    return run_prepare_cli(
        "Prepare and register the AIME 2025 / 2024 test sets.",
        prepare,
        [(name, ("test",)) for name in names],
        argv,
        add_arguments=lambda p: p.add_argument("--only", choices=sorted(_PREPARERS), default=None),
    )


if __name__ == "__main__":
    sys.exit(main())
