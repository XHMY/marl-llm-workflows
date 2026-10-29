# COPIED FROM fork coding_dataset_filtering/measure_runtime.py:1-172 (research fork @f0240732);
# diff regions marked MARL(...).
"""Measure the reference solution of every DeepCoder (primeintellect) problem on each of its tests.

Writes ``runtime_results.json``: ``{problem_idx: {test_idx: {"time", "status"[, "error"]}}}``
with status ``passed`` / ``wrong_answer`` / ``timeout`` / ``error`` / ``subprocess_error``.
``build_deepcoder_passing_index.py`` turns it into ``deepcoder_passing_tests.json.gz``, which
``prepare_deepcoder.py`` uses to drop failing tests and problems with no passing test.

The paper's index was measured with a 10 s per-test timeout (the default here). Timing
depends on the machine, so a re-run need not reproduce the shipped index exactly; the
shipped index is the reference (its sha256 is in ``checksums.json``).

Usage::

    python -m marl_workflows.data.measure_deepcoder_runtime --n-jobs 32 --output runtime_results.json

The reference solutions are executed with ``exec`` in a child Python process without a
sandbox; run this only on a machine where that is acceptable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

__all__ = ["DATASET_NAME", "DATASET_SUBSET", "REVISION", "WRAPPER_SCRIPT", "extract_code", "main", "run_single_problem"]

DATASET_NAME = "agentica-org/DeepCoder-Preview-Dataset"
DATASET_SUBSET = "primeintellect"
# >>> MARL(pin): pinned HF revision (the snapshot the paper's data was prepared from)
REVISION = "177913a7bd43791646ef6a43645caa3c871ab3db"
# <<< MARL(pin)

WRAPPER_SCRIPT = textwrap.dedent(r"""
import json
import sys
import io
import time
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Test case timed out")

signal.signal(signal.SIGALRM, timeout_handler)

payload = json.loads(sys.stdin.read())
code = payload["code"]
tests = payload["tests"]
timeout = payload["timeout"]

results = []
for i, test in enumerate(tests):
    inp = test["input"]
    expected = test["output"]
    result = {"test_idx": i}
    old_stdin = sys.stdin
    old_stdout = sys.stdout
    try:
        sys.stdin = io.StringIO(inp)
        sys.stdout = io.StringIO()
        globals_dict = {"__name__": "__main__", "__builtins__": __builtins__}
        signal.setitimer(signal.ITIMER_REAL, timeout)
        t0 = time.perf_counter()
        try:
            exec(code, globals_dict)
        except SystemExit:
            pass
        t1 = time.perf_counter()
        signal.setitimer(signal.ITIMER_REAL, 0)
        actual = sys.stdout.getvalue()
        result["time"] = t1 - t0
        if actual.rstrip() == expected.rstrip():
            result["status"] = "passed"
        else:
            result["status"] = "wrong_answer"
    except TimeoutError:
        signal.setitimer(signal.ITIMER_REAL, 0)
        result["time"] = timeout
        result["status"] = "timeout"
    except Exception as e:
        signal.setitimer(signal.ITIMER_REAL, 0)
        result["time"] = 0.0
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        sys.stdin = old_stdin
        sys.stdout = old_stdout
    results.append(result)

sys.stdout.write(json.dumps(results) + "\n")
""").strip()


def extract_code(solution: str) -> str:
    """Strip the markdown code block wrapper from a reference solution."""
    if solution.startswith("```python\n"):
        solution = solution[len("```python\n") :]
    if solution.endswith("\n```"):
        solution = solution[: -len("\n```")]
    return solution


def run_single_problem(idx: int, code: str, tests_json: str, per_test_timeout: float) -> dict:
    """Run all tests of one problem in a fresh interpreter and return per-test records."""
    try:
        tests = json.loads(tests_json)
    except json.JSONDecodeError:
        return {"error": "invalid_tests_json"}

    num_tests = len(tests)
    subprocess_timeout = min(num_tests * per_test_timeout + 30, 600)

    payload = json.dumps({"code": code, "tests": tests, "timeout": per_test_timeout})

    try:
        proc = subprocess.run(
            # >>> MARL(interpreter): the current interpreter instead of whatever "python3" is on PATH
            [sys.executable, "-c", WRAPPER_SCRIPT],
            # <<< MARL(interpreter)
            input=payload,
            capture_output=True,
            text=True,
            timeout=subprocess_timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            str(j): {"time": 0.0, "status": "subprocess_error", "error": "subprocess timed out"}
            for j in range(num_tests)
        }

    if proc.returncode != 0:
        return {
            str(j): {"time": 0.0, "status": "subprocess_error", "error": proc.stderr[:500]} for j in range(num_tests)
        }

    try:
        results_list = json.loads(proc.stdout.strip())
    except json.JSONDecodeError:
        return {
            str(j): {"time": 0.0, "status": "subprocess_error", "error": "invalid JSON output"}
            for j in range(num_tests)
        }

    result = {}
    for entry in results_list:
        j = entry["test_idx"]
        r = {"time": entry["time"], "status": entry["status"]}
        if "error" in entry:
            r["error"] = entry["error"]
        result[str(j)] = r

    return result


def _run_star(args: tuple[int, str, str, float]) -> dict:
    return run_single_problem(*args)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Measure test case running times for DeepCoder-Preview-Dataset")
    parser.add_argument("--n-jobs", type=int, default=32, help="Number of parallel workers (default: 32)")
    parser.add_argument("--timeout", type=float, default=10, help="Per-test-case timeout in seconds (default: 10)")
    parser.add_argument("--output", type=str, default="runtime_results.json", help="Output JSON file path")
    parser.add_argument("--start-idx", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--end-idx", type=int, default=None, help="End index (exclusive)")
    parser.add_argument("--revision", type=str, default=REVISION, help="HF dataset revision")
    args = parser.parse_args(argv)

    from datasets import load_dataset

    print("Loading dataset...", flush=True)
    dataset = load_dataset(DATASET_NAME, DATASET_SUBSET, split="train", revision=args.revision)

    end_idx = args.end_idx if args.end_idx is not None else len(dataset)
    start_idx = args.start_idx
    subset = dataset.select(range(start_idx, end_idx))
    print(f"Processing problems {start_idx} to {end_idx} ({len(subset)} problems)", flush=True)

    codes = [extract_code(row["solutions"][0]) for row in subset]
    tests = [row["tests"] for row in subset]

    print(f"Running with {args.n_jobs} workers, {args.timeout}s per-test timeout...", flush=True)
    # >>> MARL(executor): a stdlib process pool instead of joblib (not a dependency); same per-problem calls
    jobs = [(i, codes[i], tests[i], args.timeout) for i in range(len(subset))]
    with ProcessPoolExecutor(max_workers=args.n_jobs) as pool:
        results_list = list(tqdm(pool.map(_run_star, jobs, chunksize=4), total=len(jobs), desc="Processing"))
    # <<< MARL(executor)

    results = {}
    for i, r in enumerate(results_list):
        results[str(start_idx + i)] = r

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    total_tests = 0
    status_counts: dict[str, int] = {}
    for problem_results in results.values():
        for test_result in problem_results.values():
            if isinstance(test_result, dict) and "status" in test_result:
                total_tests += 1
                s = test_result["status"]
                status_counts[s] = status_counts.get(s, 0) + 1

    print(f"\nDone. Results saved to {args.output}")
    print(f"Total test cases: {total_tests}")
    for s, c in sorted(status_counts.items()):
        print(f"  {s}: {c} ({100 * c / max(total_tests, 1):.1f}%)")


if __name__ == "__main__":
    main()
