"""The passing-test index reproduces the fork's runtime-based DeepCoder test filter."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from marl_workflows.data.build_deepcoder_passing_index import (
    DEFAULT_INDEX_PATH,
    build_passing_index,
    load_passing_index,
    sha256_file,
    write_passing_index,
)
from marl_workflows.data.prepare_dapo_math import CHECKSUMS_PATH, load_checksums
from marl_workflows.data.prepare_deepcoder import filter_tests_by_passing_index


def fork_filter_tests_by_runtime(example: dict, idx: int, runtime_results: dict) -> dict:
    """Verbatim copy of fork examples/deepcoder/prepare_deepcoder_data.py:24-46 (the reference)."""
    problem_key = str(idx)
    if problem_key not in runtime_results:
        example["_num_passing_tests"] = -1
        return example

    problem_runtime = runtime_results[problem_key]
    tests_raw = example["tests"]
    if isinstance(tests_raw, str):
        tests = json.loads(tests_raw)
    else:
        tests = tests_raw

    filtered_tests = [
        test
        for test_idx, test in enumerate(tests)
        if str(test_idx) in problem_runtime and problem_runtime[str(test_idx)].get("status") == "passed"
    ]

    example["tests"] = json.dumps(filtered_tests)
    example["_num_passing_tests"] = len(filtered_tests)
    return example


def _tests(n: int) -> list[dict]:
    return [{"input": f"{i}\n", "output": f"{i * i}\n", "testtype": "stdin_stdout"} for i in range(n)]


# Problems 0..5 cover: partial pass (string tests), all pass, none pass (empty), problem-level
# error record, absent from the measurement, partial pass with list-typed tests and statuses
# beyond the test count.
DATASET = [
    {"tests": json.dumps(_tests(4))},
    {"tests": json.dumps(_tests(2))},
    {"tests": json.dumps(_tests(3))},
    {"tests": json.dumps(_tests(2))},
    {"tests": json.dumps(_tests(5))},
    {"tests": _tests(3)},
]
RUNTIME = {
    "0": {
        "0": {"time": 0.1, "status": "passed"},
        "1": {"time": 10, "status": "timeout"},
        "2": {"time": 0.2, "status": "passed"},
        "3": {"time": 0, "status": "wrong_answer"},
    },
    "1": {"0": {"time": 0.1, "status": "passed"}, "1": {"time": 0.1, "status": "passed"}},
    "2": {str(i): {"time": 0.0, "status": "error", "error": "X"} for i in range(3)},
    "3": {"error": "invalid_tests_json"},
    # "4" absent: keep every test
    "5": {
        "1": {"time": 0.1, "status": "passed"},
        "7": {"time": 0.1, "status": "passed"},
        "0": {"time": 0.0, "status": "subprocess_error", "error": "boom"},
    },
}


def test_filter_equals_fork_filter() -> None:
    index = build_passing_index(RUNTIME)
    assert index == {"0": [0, 2], "1": [0, 1], "2": [], "3": [], "5": [1, 7]}
    for idx, row in enumerate(DATASET):
        ours = filter_tests_by_passing_index(json.loads(json.dumps(row)), idx, index)
        ref = fork_filter_tests_by_runtime(json.loads(json.dumps(row)), idx, RUNTIME)
        assert ours == ref, idx
    # The semantic cases, spelled out.
    kept = [filter_tests_by_passing_index(json.loads(json.dumps(r)), i, index) for i, r in enumerate(DATASET)]
    assert [k["_num_passing_tests"] for k in kept] == [2, 2, 0, 0, -1, 1]
    assert [t["input"] for t in json.loads(kept[0]["tests"])] == ["0\n", "2\n"]  # original order
    assert kept[4]["tests"] == DATASET[4]["tests"]  # absent: untouched


def test_written_index_round_trips_and_is_deterministic(tmp_path: Path) -> None:
    index = build_passing_index(RUNTIME)
    a, b = tmp_path / "a.json.gz", tmp_path / "b.json.gz"
    write_passing_index(index, a)
    write_passing_index(dict(reversed(list(index.items()))), b)
    assert a.read_bytes() == b.read_bytes()
    assert load_passing_index(a) == index
    with gzip.open(a, "rt") as f:
        assert list(json.load(f)) == ["0", "1", "2", "3", "5"]


def test_shipped_index_counts_and_checksum() -> None:
    index = load_passing_index(DEFAULT_INDEX_PATH)
    assert len(index) == 16252
    assert sum(1 for v in index.values() if not v) == 3295
    assert all(v == sorted(set(v)) for v in index.values())
    meta = load_checksums(CHECKSUMS_PATH)["files"]["deepcoder_passing_tests.json.gz"]
    assert sha256_file(DEFAULT_INDEX_PATH) == meta["sha256"]
    assert meta["problems"] == 16252 and meta["problems_without_passing_test"] == 3295
