# COPIED FROM fork analysis/revision_testsets/prepare_test_sets.py:37-108 (research fork @f0240732);
# diff regions marked MARL(...).
"""Prepare and register MATH-500 as ``math500``/``test`` (answer-only ground truth, DAPO duplicates dropped).

``HuggingFaceH4/MATH-500`` with ``final_answer = ground_truth = answer`` (the final answer,
not the worked solution that upstream rLLM's ``examples/simple_math`` registers, which the
math workflows cannot grade against a boxed answer). Problems that duplicate a DAPO-Math
training problem are dropped, as in the fork's registration (``math500_answer``, 474 of 500):

Texts are normalized (lowercase, whitespace, ``$``, ``\\left``/``\\right`` and spacing macros
removed). For each MATH-500 problem, training problems with word-token Jaccard >= 0.45 are
candidates; their similarity is difflib's ``SequenceMatcher`` ratio between the normalized
MATH-500 text and the normalized training text truncated to 1.15x its length (DAPO appends an
answer-format instruction). A problem is dropped when its best similarity is >= 0.9, or when
it is in ``MANUAL_SAME_GIVENS`` (similarity 0.7-0.9 candidates that a reading showed pose the
same givens). It needs the ``dapo_math`` train split registered first
(``python -m marl_workflows.data.prepare_dapo_math``). Independent test set; not used by the
paper's main numbers.

Usage::

    python -m marl_workflows.data.prepare_math500 [--keep-duplicates] [--verify | --verify-only]
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from collections.abc import Collection, Sequence
from typing import Any

from marl_workflows.data.prepare_dapo_math import register_split, run_prepare_cli

__all__ = ["MANUAL_SAME_GIVENS", "REVISION", "find_duplicates", "main", "norm", "prepare_math500", "toks"]

DATASET_NAME = "HuggingFaceH4/MATH-500"
REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"

# MATH-500 indices (HuggingFaceH4/MATH-500 test order) with the same givens as a DAPO training
# problem, found by reading every candidate with similarity 0.7-0.9 (fork, 2026-09-26).
MANUAL_SAME_GIVENS = frozenset({25, 126, 153, 189, 192, 295, 339})


def norm(s: str) -> str:
    """Normalize a problem text for the similarity ratio."""
    s = s.lower()
    s = re.sub(
        r"\\left|\\right|\\!|\\,|\;|\\quad|\$|\\displaystyle|\\dfrac",
        lambda m: "\\frac" if m.group(0) == "\\dfrac" else "",
        s,
    )
    return re.sub(r"\s+", "", s)


def toks(s: str) -> set[str]:
    """Word / number / symbol tokens for the Jaccard prefilter."""
    return set(re.findall(r"[a-z]+|\d+|[^\sa-z\d]", s.lower()))


def find_duplicates(
    questions: Sequence[str],
    train_questions: Sequence[str],
    manual: Collection[int] = frozenset(),
) -> set[int]:
    """Indices of ``questions`` that duplicate a training problem (fork ``overlap`` without the report)."""
    train_n = [norm(q) for q in train_questions]
    train_t = [toks(q) for q in train_questions]
    dup: set[int] = set()
    for i, q in enumerate(questions):
        nq, tq = norm(q), toks(q)
        best_r = 0.0
        for k, tt in enumerate(train_t):
            inter = len(tq & tt)
            if not inter:
                continue
            j = inter / len(tq | tt)
            if j < 0.45:
                continue
            r = difflib.SequenceMatcher(None, nq, train_n[k][: int(len(nq) * 1.15) + 5], autojunk=False).ratio()
            best_r = max(best_r, r)
        if best_r >= 0.9 or i in manual:
            dup.add(i)
    return dup


def prepare_math500(keep_duplicates: bool = False) -> Any:
    """Register ``math500``/``test``."""
    from datasets import load_dataset

    m5 = load_dataset(DATASET_NAME, split="test", revision=REVISION)
    dup: set[int] = set()
    if not keep_duplicates:
        # >>> MARL(train-source): the DAPO train split comes from our registry, not a fork path
        from marl_workflows.data.registry import MarlDatasetRegistry

        train = MarlDatasetRegistry.load_dataset("dapo_math", "train")
        if train is None:
            raise FileNotFoundError(
                "dapo_math/train is not registered; run python -m marl_workflows.data.prepare_dapo_math first "
                "(or pass --keep-duplicates)"
            )
        train_q = [row["question"] for row in train.get_data()]
        # <<< MARL(train-source)
        dup = find_duplicates(m5["problem"], train_q, MANUAL_SAME_GIVENS)
    keep = [i for i in range(len(m5)) if i not in dup]
    rows = [
        {
            "question": m5[i]["problem"],
            "final_answer": m5[i]["answer"],
            "ground_truth": m5[i]["answer"],
            "data_source": "math500",
            "unique_id": m5[i]["unique_id"],
            "subject": m5[i]["subject"],
            "level": int(m5[i]["level"]),
        }
        for i in keep
    ]
    print(f"math500: {len(rows)} problems kept, {len(dup)} dropped as DAPO-train duplicates")
    return register_split("math500", rows, "test")


def main(argv: list[str] | None = None) -> int:
    def add(p: argparse.ArgumentParser) -> None:
        p.add_argument("--keep-duplicates", action="store_true", help="register all 500 problems (no DAPO dedup)")

    return run_prepare_cli(
        "Prepare and register MATH-500 (math500/test).",
        lambda args: prepare_math500(keep_duplicates=args.keep_duplicates),
        [("math500", ("test",))],
        argv,
        add_arguments=add,
    )


if __name__ == "__main__":
    sys.exit(main())
