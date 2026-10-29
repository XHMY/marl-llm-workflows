"""Compress the DeepCoder runtime measurement into a per-problem index of passing tests.

``measure_deepcoder_runtime.py`` runs the reference solution of every
``agentica-org/DeepCoder-Preview-Dataset`` (``primeintellect``) problem against each of its
tests and writes ``runtime_results.json`` (129 MB): ``{problem_idx: {test_idx: {"time",
"status", ...}}}``. The data preparation only needs which tests passed, so this script keeps
``{problem_idx: [passing test indices, ascending]}`` for every problem, including problems
with no passing test (empty list), so that "absent" (keep all tests) and "empty" (drop the
problem) stay distinct. The gzip output is byte-deterministic (``mtime=0``, sorted keys).

Usage::

    python -m marl_workflows.data.build_deepcoder_passing_index \\
        --runtime-results /path/to/runtime_results.json \\
        --out marl_workflows/data/deepcoder_passing_tests.json.gz
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_INDEX_PATH",
    "build_passing_index",
    "load_passing_index",
    "main",
    "sha256_file",
    "write_passing_index",
]

DEFAULT_INDEX_PATH = Path(__file__).resolve().parent / "deepcoder_passing_tests.json.gz"


def build_passing_index(runtime_results: Mapping[str, Any]) -> dict[str, list[int]]:
    """``{problem: {test: {"status": ...}}}`` -> ``{problem: sorted passing test indices}``.

    A test passes iff its record is a dict with ``status == "passed"``, which is the test the
    fork applied (``examples/deepcoder/prepare_deepcoder_data.py`` ``filter_tests_by_runtime``).
    Problem-level error records (``{"error": ...}``) have no test entries and map to ``[]``.
    """
    index: dict[str, list[int]] = {}
    for problem_key, tests in runtime_results.items():
        passing: list[int] = []
        if isinstance(tests, Mapping):
            for test_key, record in tests.items():
                if str(test_key).isdigit() and isinstance(record, Mapping) and record.get("status") == "passed":
                    passing.append(int(test_key))
        index[str(problem_key)] = sorted(passing)
    return index


def write_passing_index(index: Mapping[str, list[int]], out_path: str | Path) -> None:
    """Write the index as deterministic gzip-compressed compact JSON (keys in numeric order)."""
    ordered = {k: list(index[k]) for k in sorted(index, key=lambda s: (len(s), s))}
    payload = json.dumps(ordered, separators=(",", ":")).encode("utf-8")
    buf = io.BytesIO()
    # mtime=0 and no file name in the header keep the bytes (and the checksum) reproducible.
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, compresslevel=9, mtime=0) as gz:
        gz.write(payload)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(buf.getvalue())


def load_passing_index(path: str | Path = DEFAULT_INDEX_PATH) -> dict[str, list[int]]:
    """Read the gzip JSON index written by :func:`write_passing_index`."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: str | Path) -> str:
    """Hex sha256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runtime-results", required=True, help="runtime_results.json from measure_deepcoder_runtime")
    parser.add_argument("--out", default=str(DEFAULT_INDEX_PATH), help="output .json.gz path")
    args = parser.parse_args(argv)

    with open(args.runtime_results) as f:
        runtime_results = json.load(f)
    index = build_passing_index(runtime_results)
    write_passing_index(index, args.out)
    n_empty = sum(1 for v in index.values() if not v)
    print(f"problems: {len(index)}  with no passing test: {n_empty}")
    print(f"runtime_results sha256: {sha256_file(args.runtime_results)}")
    print(f"wrote {args.out} sha256: {sha256_file(args.out)}")


if __name__ == "__main__":
    main()
