"""Code reward: sandboxed execution, ported cases of fork ``tests/rewards/test_code_reward.py``.

Every program here is tiny and every timeout short. Cases run both in-process (the per-test
``lcb_check_correctness_v2`` path) and inside the production executor: a spawn
``ProcessPoolExecutor`` whose workers run ``code_executor_initializer`` (direct mode, one
subprocess per problem with RLIMIT_AS / RLIMIT_CPU).
"""

from __future__ import annotations

import inspect
import json
import multiprocessing
import signal
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest
from rllm.agents.agent import Action
from rllm.rewards.reward_types import RewardConfig, RewardType

from marl_workflows.rewards import code_reward as cr

pytestmark = pytest.mark.stack

# --------------------------------------------------------------------------------------
# Programs and test cases (from fork tests/rewards/test_code_reward.py, trimmed)
# --------------------------------------------------------------------------------------

SHORTEST_ROUTE = """
```python
import sys
from itertools import permutations
def main():
    N, M, R = map(int, sys.stdin.readline().split())
    r = list(map(int, sys.stdin.readline().split()))
    A, B, C = [], [], []
    for _ in range(M):
        a, b, c = map(int, sys.stdin.readline().split())
        A.append(a)
        B.append(b)
        C.append(c)
    INF = float('inf')
    dist = [[INF for _ in range(N+1)] for _ in range(N+1)]
    for i in range(1, N+1):
        dist[i][i] = 0
    for i in range(M):
        a, b, c = A[i], B[i], C[i]
        dist[a][b] = c
        dist[b][a] = c
    for k in range(1, N+1):
        for i in range(1, N+1):
            for j in range(1, N+1):
                if dist[i][k] != INF and dist[k][j] != INF:
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    min_dist = INF
    for perm in permutations(r):
        total = 0
        for i in range(R-1):
            total += dist[perm[i]][perm[i+1]]
        if total < min_dist:
            min_dist = total
    print(min_dist)
if __name__ == "__main__":
    main()
    ```
    """
SHORTEST_ROUTE_TESTS = {
    "inputs": [
        "4 3 3\n1 2 3\n1 2 3\n2 3 2\n3 4 4\n",
        "5 10 4\n1 2 3 4\n1 2 5\n1 3 5\n1 4 5\n1 5 5\n2 3 5\n2 4 5\n2 5 5\n3 4 5\n3 5 5\n4 5 5\n",
    ],
    "outputs": ["5\n", "15\n"],
}

CODEFORCES = """
```python
def main():
    n,m=map(int, input().split())
    a=sum(list(map(int, input().split())))
    if a+(n-1)*10<=m:
        print((m-a)//5)
    else:
        print(-1)
if __name__ == "__main__":
    main()
    ```
    """
CODEFORCES_TESTS = [
    {"input": "3 30\n2 2 1", "output": "5"},
    {"input": "3 20\n2 1 1", "output": "-1"},
    {"input": "1 20\n5", "output": "3"},
    {"input": "2 100\n1 1", "output": "19"},
]

ANAGRAM = """
```python
def is_anagram(test, original):
    return sorted(original.lower()) == sorted(test.lower())
```
"""
ANAGRAM_TESTS = {
    "fn_name": "is_anagram",
    "inputs": [["foefet", "toffee"], ["Buckethead", "DeathCubeK"], ["dumble", "bumble"], ["apple", "pale"]],
    "outputs": [[True], [True], [False], [False]],
}

TOTAL = """
```python
def total(arr):
    while len(arr) > 1:
        arr = [x + y for (x, y) in zip(arr, arr[1:])]
    return arr[0]
```
"""
TOTAL_TESTS = {
    "fn_name": "total",
    "inputs": [[[1, 2, 3, 4, 5]], [[1, 2, 3]], [[-1, 1, -1, 1]], [[42]]],
    "outputs": [[48], [8], [0], [42]],
}

LCB_STDIN = """
Yes of course!
```python
def main():
    n = input()
    phone_numbers = input().strip().split()
    seen = set()
    duplicates = set()
    for number in phone_numbers:
        if number in seen:
            duplicates.add(number)
        else:
            seen.add(number)
    print(len(duplicates)+1)
if __name__ == "__main__":
    main()
```
"""
LCB_STDIN_TESTS = [{"input": "3\n12345 530391 12345\n", "output": "2\n", "testtype": "stdin"}]

LCB_FUNCTIONAL = """
Yes of course!
```python
class Solution:
    def numberOfEmployeesWhoMetTarget(self, hours: List[int], target: int) -> int:
        count = 0
        for hour in hours:
            if hour >= target:
                count += 1
        return count
```
"""
LCB_FUNCTIONAL_TESTS = [
    {
        "input": "[5, 3, 10, 8, 2]\n5",
        "output": "3",
        "testtype": "functional",
        "metadata": {"func_name": "numberOfEmployeesWhoMetTarget"},
    }
]

DOUBLE = "```python\nn = int(input())\nprint(n * 2)\n```"
DOUBLE_TESTS = [
    {"input": "3\n", "output": "6\n", "testtype": "stdin"},
    {"input": "5\n", "output": "10\n", "testtype": "stdin"},
    {"input": "7\n", "output": "14\n", "testtype": "stdin"},
]
DOUBLE_TESTS_SECOND_WRONG = [DOUBLE_TESTS[0], {"input": "5\n", "output": "11\n", "testtype": "stdin"}, DOUBLE_TESTS[2]]

INFINITE_LOOP = '```python\ndef main():\n    while True:\n        pass\nif __name__ == "__main__":\n    main()\n```'
# sum() over a range loops in C without returning to the eval loop, so signal.alarm cannot
# interrupt it. (The fork test used catastrophic regex backtracking, which Python 3.11's re
# module interrupts on signals, so it no longer exercises RLIMIT_CPU.)
CPU_BURN = '```python\ndef main():\n    print(sum(range(10**12)) and 1)\nif __name__ == "__main__":\n    main()\n```'
ONE_TEST = [{"input": "1\n", "output": "1\n", "testtype": "stdin"}]
BIG_ALLOC = "```python\nx = bytearray(5 * 1024**3)\nprint(input())\n```"


def _task(data_source: str, tests) -> dict:
    return {"problem": "", "problem_type": RewardType.CODE, "data_source": data_source, "ground_truth": tests}


# name: (data_source, ground_truth, response, expected is_correct)
CASES = {
    "code_contests": ("code_contests", SHORTEST_ROUTE_TESTS, SHORTEST_ROUTE, True),
    "taco": ("taco", SHORTEST_ROUTE_TESTS, SHORTEST_ROUTE, True),
    "taco_call_based": ("taco", ANAGRAM_TESTS, ANAGRAM, True),
    "taco_call_based2": ("taco", TOTAL_TESTS, TOTAL, True),
    "codeforces": ("codeforces", CODEFORCES_TESTS, CODEFORCES, True),
    "livecodebench_stdin": ("livecodebench", LCB_STDIN_TESTS, LCB_STDIN, True),
    "livecodebench_functional": ("livecodebench", LCB_FUNCTIONAL_TESTS, LCB_FUNCTIONAL, True),
    "primeintellect": ("primeintellect", DOUBLE_TESTS, DOUBLE, True),
    "primeintellect_json_string": ("primeintellect", json.dumps(DOUBLE_TESTS), DOUBLE, True),
    "wrong_answer": ("primeintellect", DOUBLE_TESTS_SECOND_WRONG, DOUBLE, False),
    "no_code_block": ("primeintellect", DOUBLE_TESTS, "n = int(input())\nprint(n * 2)", False),
    "invalid_json_tests": ("primeintellect", "{not json", DOUBLE, False),
    "empty_tests": ("primeintellect", "  ", DOUBLE, False),
}


@pytest.fixture(scope="module")
def code_pool():
    """The production code-reward executor: spawn workers in direct mode."""
    pool = ProcessPoolExecutor(
        max_workers=2,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=cr.code_executor_initializer,
    )
    yield pool
    pool.shutdown(wait=True, cancel_futures=True)


@pytest.fixture
def v2_mode():
    """In-process calls use the per-test (v2) path; restore the module flag afterwards."""
    saved = cr._USE_DIRECT_EXECUTION
    cr.set_direct_execution(False)
    yield
    cr.set_direct_execution(saved)


# --------------------------------------------------------------------------------------
# Source and configuration
# --------------------------------------------------------------------------------------


def test_no_multiprocessing_manager_in_source() -> None:
    """The fork replaced every manager-based result channel with a Pipe (finding 0.9)."""
    source = Path(inspect.getsourcefile(cr)).read_text()
    assert "Manager(" not in source


def test_executor_initializer_sets_direct_mode() -> None:
    saved = cr._USE_DIRECT_EXECUTION
    try:
        cr.set_direct_execution(False)
        cr.code_executor_initializer()
        assert cr._USE_DIRECT_EXECUTION is True
    finally:
        cr.set_direct_execution(saved)


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, 4 * 1024**3),
        ({"RLLM_CODE_REWARD_MAX_MEMORY_GB": "6"}, 6 * 1024**3),
        ({"RLLM_CODE_REWARD_MAX_MEMORY_GB": "0.5"}, 512 * 1024**2),
        ({"RLLM_CODE_REWARD_MAX_MEMORY_BYTES": "123456"}, 123456),
        # The byte value wins over the GiB value.
        ({"RLLM_CODE_REWARD_MAX_MEMORY_BYTES": "1000", "RLLM_CODE_REWARD_MAX_MEMORY_GB": "6"}, 1000),
        ({"RLLM_CODE_REWARD_MAX_MEMORY_GB": "none"}, None),
        ({"RLLM_CODE_REWARD_MAX_MEMORY_GB": " Unlimited "}, None),
        ({"RLLM_CODE_REWARD_MAX_MEMORY_GB": "0"}, None),
        ({"RLLM_CODE_REWARD_MAX_MEMORY_GB": "-2"}, None),
        # Invalid values fall back (to the next variable, then the default).
        ({"RLLM_CODE_REWARD_MAX_MEMORY_GB": "lots"}, 4 * 1024**3),
        ({"RLLM_CODE_REWARD_MAX_MEMORY_BYTES": "1.5", "RLLM_CODE_REWARD_MAX_MEMORY_GB": "2"}, 2 * 1024**3),
    ],
)
def test_memory_env_parsing(monkeypatch: pytest.MonkeyPatch, env: dict, expected) -> None:
    monkeypatch.delenv("RLLM_CODE_REWARD_MAX_MEMORY_GB", raising=False)
    monkeypatch.delenv("RLLM_CODE_REWARD_MAX_MEMORY_BYTES", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert cr._configured_max_memory_bytes() == expected


def test_unsupported_sources_raise() -> None:
    for source in ("leetcode", "kodcode", "humanevalplus", "nope"):
        with pytest.raises(NotImplementedError):
            cr.code_reward_fn(_task(source, DOUBLE_TESTS), DOUBLE)
    config = RewardConfig()
    config.use_together_code_interpreter = True
    with pytest.raises(NotImplementedError):
        cr.RewardCodeFn(config)(_task("taco", SHORTEST_ROUTE_TESTS), SHORTEST_ROUTE)


def test_extract_code_takes_last_block() -> None:
    assert cr.extract_code_from_model("```python\na=1\n```\ntext\n```python\nb=2\n```") == "b=2"
    assert cr.extract_code_from_model("no code") is None


# --------------------------------------------------------------------------------------
# Correctness: in-process v2 path and the spawn pool in direct mode
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(CASES))
def test_reward_v2_in_process(case: str, v2_mode) -> None:
    source, tests, response, expected = CASES[case]
    out = cr.code_reward_fn(_task(source, tests), response)
    assert bool(out.is_correct) is expected
    assert out.reward == (1.0 if expected else 0.0)


def test_reward_in_code_executor_pool(code_pool: ProcessPoolExecutor) -> None:
    names = sorted(CASES)
    futures = [code_pool.submit(cr.code_reward_fn, _task(CASES[n][0], CASES[n][1]), CASES[n][2]) for n in names]
    for name, fut in zip(names, futures, strict=True):
        out = fut.result(timeout=120)
        assert bool(out.is_correct) is CASES[name][3], name
    # Direct mode tests a problem sequentially in one subprocess and stops at the first
    # failure; the v2 path kills the remaining per-test subprocesses instead.
    out = code_pool.submit(cr.code_reward_fn, _task("primeintellect", DOUBLE_TESTS_SECOND_WRONG), DOUBLE).result(120)
    assert [t["passed"] for t in out.metadata["test_results"]] == [True, False, False]
    assert out.metadata["test_results"][2]["error"] == "skipped (prior test failed)"
    assert out.metadata["passed_tests"] == 1 and out.metadata["total_tests"] == 3


@pytest.mark.parametrize("case", ["livecodebench_stdin", "taco_call_based", "codeforces", "wrong_answer"])
def test_direct_and_v2_paths_agree(case: str) -> None:
    source, tests, response, _ = CASES[case]
    tests = cr.taco_to_lcb_format(tests) if source == "taco" else tests
    code = cr.extract_code_from_model(response)
    ok_direct, details_direct = cr.lcb_check_correctness_direct(tests, code)
    ok_v2, details_v2 = cr.lcb_check_correctness_v2(tests, code)
    assert ok_direct == ok_v2
    assert details_direct["total_tests"] == details_v2["total_tests"]
    if ok_direct:
        assert details_direct["passed_tests"] == details_v2["passed_tests"] == details_v2["total_tests"]


def test_action_object_is_unwrapped(v2_mode) -> None:
    assert cr.code_reward_fn(_task("primeintellect", DOUBLE_TESTS), Action(action=DOUBLE)).is_correct


# --------------------------------------------------------------------------------------
# Sandbox limits
# --------------------------------------------------------------------------------------


@pytest.mark.slow
def test_timeout_infinite_loop() -> None:
    """A Python-level loop is stopped by signal.alarm inside the per-test timeout."""
    code = cr.extract_code_from_model(INFINITE_LOOP)
    for check in (cr.lcb_check_correctness_direct, cr.lcb_check_correctness_v2):
        start = time.monotonic()
        ok, _ = check(ONE_TEST, code, timeout=1)
        elapsed = time.monotonic() - start
        assert not ok
        # direct: global cap (1 + 1) * 1 + 5 = 7 s; v2: deadline 1 + 10 = 11 s.
        assert elapsed < 11.5, (check.__name__, elapsed)


@pytest.mark.slow
def test_cpu_burn_killed_by_rlimit_cpu() -> None:
    """C-level work that holds the GIL is ended by RLIMIT_CPU (SIGXCPU), not by our kill."""
    sample = cr.postprocess_lcb_sample(ONE_TEST)
    code = cr.extract_code_from_model(CPU_BURN)
    parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
    # timeout=1: RLIMIT_CPU soft limit (1 + 1) * 1 + 5 = 7 s of CPU time.
    proc = multiprocessing.Process(target=cr._temp_run, args=(sample, code, False, child_conn, 1))
    start = time.monotonic()
    proc.start()
    child_conn.close()
    proc.join(timeout=60)
    elapsed = time.monotonic() - start
    if proc.is_alive():
        proc.kill()
        proc.join()
    try:
        assert proc.exitcode == -signal.SIGXCPU, proc.exitcode
        with pytest.raises(EOFError):  # killed before it could report
            parent_conn.recv()
        assert elapsed < 60
    finally:
        parent_conn.close()
        proc.close()


def test_memory_cap_turns_large_allocation_into_failure(v2_mode) -> None:
    """RLIMIT_AS (4 GiB default) makes a 5 GiB allocation fail with MemoryError."""
    out = cr.code_reward_fn(_task("primeintellect", ONE_TEST), BIG_ALLOC)
    assert not out.is_correct
    assert out.metadata["test_results"][0]["error"] == "MemoryError()"
