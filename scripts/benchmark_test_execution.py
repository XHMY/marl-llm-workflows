"""Benchmark: multiprocessing.Process (fork) vs Singularity container for test execution.

Simulates the code reward evaluation workload: compile a Python solution + run test cases.
Compares fork-based subprocess isolation vs Singularity container isolation.

Usage:
    python scripts/benchmark_test_execution.py [--sif scripts/python311.sif] [--tests 20] [--repeats 5]
"""

import argparse
import multiprocessing
import os
import subprocess
import statistics
import textwrap
import time

# Realistic test workload: compile code + run one function-call test
TEST_CODE = textwrap.dedent("""\
    def solution(nums, target):
        seen = {}
        for i, n in enumerate(nums):
            if target - n in seen:
                return [seen[target - n], i]
            seen[n] = i
        return []
""")

TEST_INPUT = "[2, 7, 11, 15], 9"
TEST_EXPECTED = "[0, 1]"

# Slow test: simulates a test that takes 2s (near TLE)
SLOW_CODE = textwrap.dedent("""\
    import time
    def solution(n):
        time.sleep(2)
        return n * 2
""")


# --- Fork approach (current production code) ---

def _fork_worker(code, test_input, expected, conn):
    """Simulate _temp_run: compile code, run one test, send result."""
    try:
        compiled = compile(code, "<test>", "exec")
        ns = {}
        exec(compiled, ns)
        inp = eval(test_input)
        result = ns["solution"](*inp) if isinstance(inp, list | tuple) else ns["solution"](inp)
        conn.send(str(result) == expected)
    except Exception as e:
        conn.send(False)
    finally:
        conn.close()


def bench_fork_single(code=TEST_CODE, test_input=TEST_INPUT, expected=TEST_EXPECTED):
    parent, child = multiprocessing.Pipe(duplex=False)
    p = multiprocessing.Process(target=_fork_worker, args=(code, test_input, expected, child))
    p.start()
    child.close()
    p.join(timeout=10)
    if p.is_alive():
        p.kill()
        p.join(timeout=5)
    try:
        result = parent.recv() if parent.poll() else None
    except (EOFError, OSError):
        result = None
    finally:
        parent.close()
        p.close()
    return result


def bench_fork_parallel(n, code=TEST_CODE, test_input=TEST_INPUT, expected=TEST_EXPECTED):
    pipes, procs = [], []
    for _ in range(n):
        parent, child = multiprocessing.Pipe(duplex=False)
        p = multiprocessing.Process(target=_fork_worker, args=(code, test_input, expected, child))
        p.start()
        child.close()
        pipes.append(parent)
        procs.append(p)

    deadline = time.monotonic() + 15
    for p in procs:
        remaining = max(0, deadline - time.monotonic())
        p.join(timeout=remaining)
    for p in procs:
        if p.is_alive():
            p.kill()
            p.join(timeout=5)

    results = []
    for conn in pipes:
        try:
            results.append(conn.recv() if conn.poll() else None)
        except (EOFError, OSError):
            results.append(None)
        finally:
            conn.close()
    for p in procs:
        try:
            p.close()
        except ValueError:
            pass
    return results


# --- Singularity approach ---

def _singularity_cmd(sif_path, code, test_input):
    """Build apptainer exec command that compiles + runs one test."""
    python_script = code + f"\nresult = solution(*({test_input},))\nprint(result)"
    # Use a temp file to avoid shell quoting issues
    return ["apptainer", "exec", sif_path, "python3", "-c", python_script]


def bench_singularity_single(sif_path, code=TEST_CODE, test_input=TEST_INPUT, expected=TEST_EXPECTED):
    cmd = _singularity_cmd(sif_path, code, test_input)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() == expected
    except subprocess.TimeoutExpired:
        return None


def bench_singularity_parallel(n, sif_path, code=TEST_CODE, test_input=TEST_INPUT, expected=TEST_EXPECTED):
    procs = []
    for _ in range(n):
        cmd = _singularity_cmd(sif_path, code, test_input)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        procs.append(p)

    results = []
    for p in procs:
        try:
            stdout, _ = p.communicate(timeout=15)
            results.append(stdout.decode().strip() == expected)
        except subprocess.TimeoutExpired:
            p.kill()
            p.communicate()
            results.append(None)
    return results


# --- Mixed: 19 fast tests + 1 slow test (simulates the real bottleneck) ---

def bench_fork_mixed(n_fast, sif_path=None):
    """Fork: 1 slow test (2s sleep) + n_fast fast tests, all in parallel."""
    pipes, procs = [], []
    # 1 slow test
    parent, child = multiprocessing.Pipe(duplex=False)
    p = multiprocessing.Process(target=_fork_worker, args=(SLOW_CODE, "5", "10", child))
    p.start()
    child.close()
    pipes.append(parent)
    procs.append(p)
    # n_fast fast tests
    for _ in range(n_fast):
        parent, child = multiprocessing.Pipe(duplex=False)
        p = multiprocessing.Process(target=_fork_worker, args=(TEST_CODE, TEST_INPUT, TEST_EXPECTED, child))
        p.start()
        child.close()
        pipes.append(parent)
        procs.append(p)

    deadline = time.monotonic() + 15
    for p in procs:
        remaining = max(0, deadline - time.monotonic())
        p.join(timeout=remaining)
    for p in procs:
        if p.is_alive():
            p.kill()
            p.join(timeout=5)
    results = []
    for conn in pipes:
        try:
            results.append(conn.recv() if conn.poll() else None)
        except (EOFError, OSError):
            results.append(None)
        finally:
            conn.close()
    for p in procs:
        try:
            p.close()
        except ValueError:
            pass
    return results


# --- Timing utility ---

def time_fn(fn, *args, repeats=5):
    times = []
    for _ in range(repeats):
        start = time.monotonic()
        fn(*args)
        times.append(time.monotonic() - start)
    return {
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "all": times,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark fork vs Singularity for test execution")
    parser.add_argument("--sif", default="scripts/python311.sif", help="Path to SIF image")
    parser.add_argument("--tests", type=int, default=20, help="Number of parallel tests")
    parser.add_argument("--repeats", type=int, default=5, help="Number of repeats per benchmark")
    args = parser.parse_args()

    n_tests = args.tests
    repeats = args.repeats
    sif_path = args.sif

    has_sif = os.path.exists(sif_path)
    if not has_sif:
        print(f"WARNING: SIF image not found at {sif_path}")
        print(f"  Run: apptainer pull {sif_path} docker://python:3.11-slim")
        print(f"  Skipping Singularity benchmarks.\n")

    print(f"Configuration: {n_tests} tests, {repeats} repeats")
    print(f"{'='*60}")

    # --- Single test ---
    print("\n1. Single test (compile + run one test case)")
    print(f"{'-'*60}")

    t = time_fn(bench_fork_single, repeats=repeats)
    print(f"  Fork:          {t['median']*1000:6.0f}ms (min={t['min']*1000:.0f}, max={t['max']*1000:.0f})")

    if has_sif:
        t = time_fn(bench_singularity_single, sif_path, repeats=repeats)
        print(f"  Singularity:   {t['median']*1000:6.0f}ms (min={t['min']*1000:.0f}, max={t['max']*1000:.0f})")

    # --- N tests in parallel (all fast) ---
    print(f"\n2. {n_tests} fast tests in parallel")
    print(f"{'-'*60}")

    t = time_fn(bench_fork_parallel, n_tests, repeats=repeats)
    print(f"  Fork:          {t['median']*1000:6.0f}ms (min={t['min']*1000:.0f}, max={t['max']*1000:.0f})")

    if has_sif:
        t = time_fn(bench_singularity_parallel, n_tests, sif_path, repeats=repeats)
        print(f"  Singularity:   {t['median']*1000:6.0f}ms (min={t['min']*1000:.0f}, max={t['max']*1000:.0f})")

    # --- Mixed: 1 slow + (N-1) fast tests in parallel ---
    print(f"\n3. Mixed: 1 slow test (2s sleep) + {n_tests-1} fast tests in parallel")
    print(f"   (This is the scenario where per-test parallelism matters)")
    print(f"{'-'*60}")

    t = time_fn(bench_fork_mixed, n_tests - 1, repeats=repeats)
    print(f"  Fork:          {t['median']*1000:6.0f}ms (min={t['min']*1000:.0f}, max={t['max']*1000:.0f})")

    print(f"\n{'='*60}")
    print("Done. Compare median times to decide fork vs Singularity.")


if __name__ == "__main__":
    main()
