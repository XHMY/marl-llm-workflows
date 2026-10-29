# COPIED FROM fork examples/deepcoder/prepare_deepcoder_data.py:1-100 (research fork @f0240732);
# diff regions marked MARL(...).
"""Prepare and register DeepCoder (``deepcoder_primeintellect`` train / test).

The ``primeintellect`` subset of ``agentica-org/DeepCoder-Preview-Dataset`` (16 252
problems). Tests whose reference solution failed the runtime measurement are dropped, and
problems with no passing test are removed (3 295), using ``deepcoder_passing_tests.json.gz``
(built from the fork's ``runtime_results.json`` by ``build_deepcoder_passing_index.py``).
The rest is shuffled with seed 42; the first 1000 problems are the test split (the paper's
Code validation set) and the remaining 11 957 the train split. Questions get the
LiveCodeBench system prompt; ``ground_truth`` is the JSON list of tests.

Filter semantics (identical to the fork's ``filter_tests_by_runtime``): a problem absent from
the index keeps all tests; a present problem keeps only the listed tests, in original order;
an empty list drops the problem.

Usage::

    python -m marl_workflows.data.prepare_deepcoder [--verify | --verify-only] [--num-proc 16]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from marl_workflows.data.build_deepcoder_passing_index import DEFAULT_INDEX_PATH, load_passing_index
from marl_workflows.data.prepare_dapo_math import register_split, run_prepare_cli

__all__ = [
    "DATASET_NAME",
    "REVISION",
    "TEST_SIZE",
    "filter_tests_by_passing_index",
    "main",
    "prepare_deepcoder_data",
    "preprocess_fn",
]

DATASET_NAME = "agentica-org/DeepCoder-Preview-Dataset"
SUBSET = "primeintellect"
REVISION = "177913a7bd43791646ef6a43645caa3c871ab3db"
TEST_SIZE = 1000
SEED = 42


def filter_tests_by_passing_index(example: dict, idx: int, passing_index: Mapping[str, Sequence[int]]) -> dict:
    """Keep only the passing tests of problem ``idx``; marks ``_num_passing_tests`` (-1 = not measured)."""
    problem_key = str(idx)
    if problem_key not in passing_index:
        example["_num_passing_tests"] = -1
        return example

    # >>> MARL(passing-index): the listed indices replace the fork's per-test status lookup
    passing = set(passing_index[problem_key])
    # <<< MARL(passing-index)
    tests_raw = example["tests"]
    tests = json.loads(tests_raw) if isinstance(tests_raw, str) else tests_raw

    filtered_tests = [test for test_idx, test in enumerate(tests) if test_idx in passing]

    example["tests"] = json.dumps(filtered_tests)
    example["_num_passing_tests"] = len(filtered_tests)
    return example


def preprocess_fn(example: dict, idx: int) -> dict:
    """Fork ``preprocess_fn`` verbatim: LCB prompt, normalized test list, metadata as JSON."""
    from rllm.data.utils import fetch_live_code_bench_system_prompt

    starter_code = example.get("starter_code", "")
    question = fetch_live_code_bench_system_prompt(example["problem"], starter_code if starter_code else None)

    tests_raw = example["tests"]
    # Handle different test formats
    if isinstance(tests_raw, str):
        tests = json.loads(tests_raw)
    else:
        tests = tests_raw
    metadata = example.get("metadata", {})

    # Convert TACO format to standard format
    if isinstance(tests, dict) and "inputs" in tests and "outputs" in tests:
        normalized_tests = []
        for input_val, output_val in zip(tests["inputs"], tests["outputs"], strict=False):
            normalized_tests.append({"input": input_val, "output": output_val, "testtype": "stdin_stdout"})
        tests = normalized_tests

    # Ensure tests is always a list
    if not isinstance(tests, list):
        tests = [tests] if tests else []

    for test in tests:
        if test.get("testtype") == "functional" and metadata.get("func_name") is not None:
            test["metadata"] = {"func_name": str(metadata["func_name"])}
        else:
            test["metadata"] = {"func_name": None}

    return {
        "question": question,
        "ground_truth": json.dumps(tests),
        "data_source": "livecodebench",
        "uid": f"deepcoder_{idx}",
        "index": idx,
        "starter_code": starter_code,
        "metadata": json.dumps(metadata),
    }


def prepare_deepcoder_data(
    train_size: int | None = None,
    test_size: int | None = None,
    dataset_name: str = "deepcoder_primeintellect",
    passing_index_path: str | Path = DEFAULT_INDEX_PATH,
    num_proc: int = 16,
    revision: str | None = REVISION,
) -> tuple[Any, Any]:
    """Download, filter, split, preprocess and register the DeepCoder splits (fork semantics)."""
    from datasets import load_dataset

    passing_index = load_passing_index(passing_index_path)
    dataset = load_dataset(DATASET_NAME, name=SUBSET, split="train", revision=revision)

    dataset = dataset.map(
        filter_tests_by_passing_index,
        with_indices=True,
        num_proc=1,
        fn_kwargs={"passing_index": passing_index},
    )
    dataset = dataset.filter(lambda example: example["_num_passing_tests"] != 0)
    dataset = dataset.remove_columns(["_num_passing_tests"])

    # Train/test split: 1000 for test, rest for train
    dataset = dataset.shuffle(seed=SEED)
    test_dataset = dataset.select(range(TEST_SIZE))
    train_dataset = dataset.select(range(TEST_SIZE, len(dataset)))

    if train_size:
        train_dataset = train_dataset.select(range(min(train_size, len(train_dataset))))
    if test_size:
        test_dataset = test_dataset.select(range(min(test_size, len(test_dataset))))

    map_kwargs = {"with_indices": True, "writer_batch_size": 10, "num_proc": num_proc}
    train_dataset = train_dataset.map(preprocess_fn, remove_columns=train_dataset.column_names, **map_kwargs)
    test_dataset = test_dataset.map(preprocess_fn, remove_columns=test_dataset.column_names, **map_kwargs)
    train = register_split(dataset_name, train_dataset, "train")
    test = register_split(dataset_name, test_dataset, "test")
    return train, test


def main(argv: list[str] | None = None) -> int:
    def add(p: argparse.ArgumentParser) -> None:
        p.add_argument("--num-proc", type=int, default=16, help="worker processes for the preprocessing map")
        p.add_argument("--passing-index", default=str(DEFAULT_INDEX_PATH), help="deepcoder_passing_tests.json.gz")

    return run_prepare_cli(
        "Prepare and register deepcoder_primeintellect (train/test).",
        lambda args: prepare_deepcoder_data(passing_index_path=args.passing_index, num_proc=args.num_proc),
        [("deepcoder_primeintellect", ("train", "test"))],
        argv,
        add_arguments=add,
    )


if __name__ == "__main__":
    sys.exit(main())
