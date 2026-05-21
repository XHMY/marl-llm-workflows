"""Benchmark: BatchTestScheduler (cross-problem) vs lcb_check_correctness_v2 (per-problem).

Measures:
  1. Correctness parity: both paths produce identical pass/fail results
  2. Total subprocess forks (via monkey-patching multiprocessing.Process)
  3. Wall-clock time for batches of problems at various failure rates

Usage:
    conda activate rllm
    python scripts/benchmark_batch_executor.py [--problems 32] [--tests-per-problem 10] [--failure-rate 0.8] [--repeats 3]

The benchmark constructs synthetic problems with known correct/incorrect code
and runs them through both execution paths for comparison.
"""

import argparse
import asyncio
import json
import multiprocessing
import os
import statistics
import sys
import time
from unittest.mock import patch

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rllm.rewards.code_reward import lcb_check_correctness_v2, _temp_run
from rllm.rewards.batch_code_executor import BatchTestScheduler
from rllm.rewards.reward_types import RewardConfig, RewardOutput


# ---------------------------------------------------------------------------
# Synthetic test problems
# ---------------------------------------------------------------------------

def make_correct_stdin_problem(n_tests: int) -> tuple[str, list[dict]]:
    """Create a correct stdin problem (add two numbers) with n_tests test cases."""
    code = "def main():\n    a, b = map(int, input().split())\n    print(a + b)\nif __name__ == '__main__':\n    main()\n"
    tests = []
    for i in range(n_tests):
        a, b = i + 1, (i + 1) * 2
        tests.append({"input": f"{a} {b}\n", "output": f"{a + b}\n", "testtype": "stdin"})
    return code, tests


def make_incorrect_stdin_problem(n_tests: int) -> tuple[str, list[dict]]:
    """Create an incorrect stdin problem (always prints 0) with n_tests test cases."""
    code = "def main():\n    a, b = map(int, input().split())\n    print(0)\nif __name__ == '__main__':\n    main()\n"
    tests = []
    for i in range(n_tests):
        a, b = i + 1, (i + 1) * 2
        tests.append({"input": f"{a} {b}\n", "output": f"{a + b}\n", "testtype": "stdin"})
    return code, tests


def make_correct_functional_problem(n_tests: int) -> tuple[str, list[dict]]:
    """Create a correct functional problem (two-sum) with n_tests test cases."""
    code = (
        "def twoSum(nums, target):\n"
        "    seen = {}\n"
        "    for i, n in enumerate(nums):\n"
        "        if target - n in seen:\n"
        "            return [seen[target - n], i]\n"
        "        seen[n] = i\n"
        "    return []\n"
    )
    # Generate n_tests with known answers
    tests = []
    for i in range(n_tests):
        nums = list(range(i + 2))
        target = nums[-1] + nums[-2]
        expected = [len(nums) - 2, len(nums) - 1]
        tests.append({
            "input": f"{json.dumps(nums)}\n{target}",
            "output": json.dumps(expected),
            "testtype": "functional",
            "metadata": {"func_name": "twoSum"},
        })
    return code, tests


def make_incorrect_functional_problem(n_tests: int) -> tuple[str, list[dict]]:
    """Create an incorrect functional problem (returns empty) with n_tests test cases."""
    code = "def twoSum(nums, target):\n    return []\n"
    tests = []
    for i in range(n_tests):
        nums = list(range(i + 2))
        target = nums[-1] + nums[-2]
        expected = [len(nums) - 2, len(nums) - 1]
        tests.append({
            "input": f"{json.dumps(nums)}\n{target}",
            "output": json.dumps(expected),
            "testtype": "functional",
            "metadata": {"func_name": "twoSum"},
        })
    return code, tests


def make_slow_correct_problem(n_tests: int) -> tuple[str, list[dict]]:
    """Correct code that takes ~0.5s per test (simulates near-TLE correct code)."""
    code = (
        "import time\n"
        "def main():\n"
        "    a, b = map(int, input().split())\n"
        "    time.sleep(0.5)\n"
        "    print(a + b)\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    tests = []
    for i in range(n_tests):
        a, b = i + 1, (i + 1) * 2
        tests.append({"input": f"{a} {b}\n", "output": f"{a + b}\n", "testtype": "stdin"})
    return code, tests


# ---------------------------------------------------------------------------
# Subprocess counter
# ---------------------------------------------------------------------------

class ProcessCounter:
    """Counts multiprocessing.Process instantiations via monkey-patching."""

    def __init__(self):
        self.count = 0
        self._original_init = multiprocessing.Process.__init__

    def __enter__(self):
        self.count = 0
        original = self._original_init
        counter = self

        def counting_init(self_proc, *args, **kwargs):
            counter.count += 1
            return original(self_proc, *args, **kwargs)

        multiprocessing.Process.__init__ = counting_init
        return self

    def __exit__(self, *args):
        multiprocessing.Process.__init__ = self._original_init


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def run_v2(problems: list[tuple[str, list[dict]]]) -> tuple[list[bool], float, int]:
    """Run problems through lcb_check_correctness_v2 sequentially.

    Returns (results, wall_time, subprocess_count).
    """
    results = []
    counter = ProcessCounter()

    start = time.monotonic()
    with counter:
        for code, tests in problems:
            passed, _details = lcb_check_correctness_v2(tests, code, timeout=3, debug=False)
            results.append(passed)
    wall_time = time.monotonic() - start

    return results, wall_time, counter.count


def run_v2_concurrent(problems: list[tuple[str, list[dict]]], concurrency: int) -> tuple[list[bool], float, int]:
    """Run problems through lcb_check_correctness_v2 with asyncio concurrency.

    Simulates the real evaluation path where multiple workflows submit concurrently.
    """
    counter = ProcessCounter()
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(code, tests):
        async with semaphore:
            loop = asyncio.get_event_loop()
            passed, _details = await loop.run_in_executor(
                None, lcb_check_correctness_v2, tests, code, 3, False,
            )
            return passed

    async def run_all():
        tasks = [run_one(code, tests) for code, tests in problems]
        return await asyncio.gather(*tasks)

    start = time.monotonic()
    with counter:
        results = asyncio.run(run_all())
    wall_time = time.monotonic() - start

    return list(results), wall_time, counter.count


def run_batch(problems: list[tuple[str, list[dict]]], pool_size: int) -> tuple[list[bool], float, int]:
    """Run problems through BatchTestScheduler.

    Returns (results, wall_time, subprocess_count).
    """
    counter = ProcessCounter()

    async def run_all():
        scheduler = BatchTestScheduler(
            pool_size=pool_size,
            batch_timeout_ms=100,  # short collection window for benchmark
            per_test_timeout=3,
        )
        try:
            tasks = []
            for code, tests in problems:
                task_info = {
                    "data_source": "livecodebench",
                    "ground_truth": tests,
                }
                model_response = f"```python\n{code}\n```"
                tasks.append(scheduler.submit(task_info, model_response))
            results = await asyncio.gather(*tasks)
            return results
        finally:
            await scheduler.shutdown()

    start = time.monotonic()
    with counter:
        results = asyncio.run(run_all())
    wall_time = time.monotonic() - start

    passed_list = [r.is_correct for r in results]
    return passed_list, wall_time, counter.count


# ---------------------------------------------------------------------------
# Benchmark scenarios
# ---------------------------------------------------------------------------

def build_problem_set(
    n_problems: int,
    n_tests: int,
    failure_rate: float,
    use_functional: bool = False,
) -> tuple[list[tuple[str, list[dict]]], list[bool]]:
    """Build a mix of correct/incorrect problems.

    Returns (problems, expected_results).
    """
    n_incorrect = int(n_problems * failure_rate)
    n_correct = n_problems - n_incorrect

    problems = []
    expected = []

    if use_functional:
        make_correct = make_correct_functional_problem
        make_incorrect = make_incorrect_functional_problem
    else:
        make_correct = make_correct_stdin_problem
        make_incorrect = make_incorrect_stdin_problem

    for _ in range(n_incorrect):
        code, tests = make_incorrect(n_tests)
        problems.append((code, tests))
        expected.append(False)

    for _ in range(n_correct):
        code, tests = make_correct(n_tests)
        problems.append((code, tests))
        expected.append(True)

    return problems, expected


def run_scenario(
    name: str,
    problems: list[tuple[str, list[dict]]],
    expected: list[bool],
    pool_size: int,
    repeats: int,
):
    """Run a benchmark scenario and print results."""
    n_problems = len(problems)
    n_tests = len(problems[0][1]) if problems else 0
    n_correct = sum(expected)
    n_incorrect = n_problems - n_correct

    print(f"\n{'='*70}")
    print(f"Scenario: {name}")
    print(f"  {n_problems} problems x {n_tests} tests | {n_correct} correct, {n_incorrect} incorrect | pool_size={pool_size}")
    print(f"{'='*70}")

    # --- v2 sequential ---
    print(f"\n  v2 (sequential):")
    v2_times, v2_forks_list = [], []
    for _ in range(repeats):
        results_v2, wt, forks = run_v2(problems)
        v2_times.append(wt)
        v2_forks_list.append(forks)
    # Verify correctness on last run
    for i, (got, exp) in enumerate(zip(results_v2, expected)):
        if got != exp:
            print(f"    WARNING: v2 problem {i} expected={exp} got={got}")
    print(f"    Time:  {statistics.median(v2_times):6.2f}s (min={min(v2_times):.2f}, max={max(v2_times):.2f})")
    print(f"    Forks: {v2_forks_list[-1]}")

    # --- v2 concurrent ---
    print(f"\n  v2 (concurrent, semaphore={pool_size}):")
    v2c_times, v2c_forks_list = [], []
    for _ in range(repeats):
        results_v2c, wt, forks = run_v2_concurrent(problems, concurrency=pool_size)
        v2c_times.append(wt)
        v2c_forks_list.append(forks)
    for i, (got, exp) in enumerate(zip(results_v2c, expected)):
        if got != exp:
            print(f"    WARNING: v2-concurrent problem {i} expected={exp} got={got}")
    print(f"    Time:  {statistics.median(v2c_times):6.2f}s (min={min(v2c_times):.2f}, max={max(v2c_times):.2f})")
    print(f"    Forks: {v2c_forks_list[-1]}")

    # --- Batch scheduler ---
    print(f"\n  Batch scheduler (pool_size={pool_size}):")
    batch_times, batch_forks_list = [], []
    for _ in range(repeats):
        results_batch, wt, forks = run_batch(problems, pool_size=pool_size)
        batch_times.append(wt)
        batch_forks_list.append(forks)
    for i, (got, exp) in enumerate(zip(results_batch, expected)):
        if got != exp:
            print(f"    WARNING: batch problem {i} expected={exp} got={got}")
    print(f"    Time:  {statistics.median(batch_times):6.2f}s (min={min(batch_times):.2f}, max={max(batch_times):.2f})")
    print(f"    Forks: {batch_forks_list[-1]}")

    # --- Correctness parity check ---
    parity = all(a == b for a, b in zip(results_v2, results_batch))
    print(f"\n  Correctness parity (v2 == batch): {'PASS' if parity else 'FAIL'}")
    if not parity:
        for i, (a, b) in enumerate(zip(results_v2, results_batch)):
            if a != b:
                print(f"    Mismatch at problem {i}: v2={a}, batch={b}")

    # --- Summary ---
    v2_median = statistics.median(v2_times)
    v2c_median = statistics.median(v2c_times)
    batch_median = statistics.median(batch_times)
    print(f"\n  Summary:")
    print(f"    v2 sequential:  {v2_median:.2f}s, {v2_forks_list[-1]} forks")
    print(f"    v2 concurrent:  {v2c_median:.2f}s, {v2c_forks_list[-1]} forks")
    print(f"    batch:          {batch_median:.2f}s, {batch_forks_list[-1]} forks")
    if v2c_median > 0:
        print(f"    Speedup (batch vs v2-concurrent): {v2c_median / batch_median:.2f}x")
    print(f"    Fork reduction: {v2c_forks_list[-1]} -> {batch_forks_list[-1]} ({(1 - batch_forks_list[-1] / max(v2c_forks_list[-1], 1)) * 100:.0f}% fewer)")

    return {
        "name": name,
        "v2_seq_time": v2_median,
        "v2_conc_time": v2c_median,
        "batch_time": batch_median,
        "v2_forks": v2_forks_list[-1],
        "batch_forks": batch_forks_list[-1],
        "parity": parity,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark BatchTestScheduler vs lcb_check_correctness_v2")
    parser.add_argument("--problems", type=int, default=16, help="Number of problems per scenario")
    parser.add_argument("--tests-per-problem", type=int, default=10, help="Tests per problem")
    parser.add_argument("--failure-rate", type=float, default=0.8, help="Fraction of problems that are incorrect")
    parser.add_argument("--pool-size", type=int, default=0, help="Pool size (0 = CPU count)")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per measurement")
    parser.add_argument("--quick", action="store_true", help="Run a minimal quick benchmark")
    args = parser.parse_args()

    pool_size = args.pool_size or os.cpu_count() or 4
    repeats = args.repeats

    if args.quick:
        args.problems = 8
        args.tests_per_problem = 5
        repeats = 1

    print(f"Benchmark: BatchTestScheduler vs lcb_check_correctness_v2")
    print(f"CPUs: {os.cpu_count()}, pool_size: {pool_size}, repeats: {repeats}")
    print(f"Problems: {args.problems}, tests/problem: {args.tests_per_problem}, failure_rate: {args.failure_rate}")

    all_results = []

    # Scenario 1: Standard benchmark (stdin problems, configurable failure rate)
    problems, expected = build_problem_set(args.problems, args.tests_per_problem, args.failure_rate)
    all_results.append(run_scenario(
        f"stdin {int(args.failure_rate * 100)}% failure",
        problems, expected, pool_size, repeats,
    ))

    # Scenario 2: Functional problems
    problems, expected = build_problem_set(args.problems, args.tests_per_problem, args.failure_rate, use_functional=True)
    all_results.append(run_scenario(
        f"functional {int(args.failure_rate * 100)}% failure",
        problems, expected, pool_size, repeats,
    ))

    # Scenario 3: All correct (worst case for batch - must run all rounds)
    problems, expected = build_problem_set(args.problems, args.tests_per_problem, 0.0)
    all_results.append(run_scenario(
        "all correct (worst case for batch)",
        problems, expected, pool_size, repeats,
    ))

    # Scenario 4: All incorrect (best case for batch - eliminated in round 1)
    problems, expected = build_problem_set(args.problems, args.tests_per_problem, 1.0)
    all_results.append(run_scenario(
        "all incorrect (best case for batch)",
        problems, expected, pool_size, repeats,
    ))

    # Scenario 5: Slow correct code (tests batch's round-based latency)
    if not args.quick:
        slow_problems = []
        slow_expected = []
        # Mix: 2 slow correct + several fast incorrect
        for _ in range(2):
            code, tests = make_slow_correct_problem(5)
            slow_problems.append((code, tests))
            slow_expected.append(True)
        for _ in range(6):
            code, tests = make_incorrect_stdin_problem(5)
            slow_problems.append((code, tests))
            slow_expected.append(False)
        all_results.append(run_scenario(
            "mixed: 2 slow-correct + 6 fast-incorrect",
            slow_problems, slow_expected, pool_size, repeats,
        ))

    # --- Final summary ---
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Scenario':<45} {'v2-conc':>8} {'batch':>8} {'speedup':>8} {'fork-reduce':>12} {'parity':>7}")
    print(f"{'-'*45} {'-'*8} {'-'*8} {'-'*8} {'-'*12} {'-'*7}")
    for r in all_results:
        speedup = r["v2_conc_time"] / r["batch_time"] if r["batch_time"] > 0 else float("inf")
        fork_pct = (1 - r["batch_forks"] / max(r["v2_forks"], 1)) * 100
        print(f"{r['name']:<45} {r['v2_conc_time']:>7.2f}s {r['batch_time']:>7.2f}s {speedup:>7.2f}x {fork_pct:>10.0f}% {'PASS' if r['parity'] else 'FAIL':>7}")


if __name__ == "__main__":
    main()
