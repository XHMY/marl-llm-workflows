"""Fresh-interpreter code sandbox (``rewards/code_sandbox.py``): limits, process model, results.

The programs are tiny; the slowest cases wait for a 7 s global timeout. Only the regression
test for the forked-sandbox memory bug imports torch (skipped without it).
"""

from __future__ import annotations

import json
import os
import random
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from marl_workflows.rewards import code_sandbox as cs

ONE_TEST = [{"input": "3\n", "output": "6\n", "testtype": "stdin"}]
DOUBLE = "print(int(input()) * 2)"
THREE_TESTS = [ONE_TEST[0], {"input": "5\n", "output": "11\n", "testtype": "stdin"}, ONE_TEST[0]]
DETAIL_KEYS = {"input", "expected", "passed", "error", "error_message", "output"}


def _alloc(n_bytes: int) -> str:
    return f"x = bytearray({n_bytes})\nprint(int(input()) * 2)"


def _vsz_bytes() -> int:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[0]) * os.sysconf("SC_PAGE_SIZE")


def _pids_with_cmdline(marker: str) -> list[int]:
    pids = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                if marker.encode() in (entry / "cmdline").read_bytes():
                    pids.append(int(entry.name))
            except OSError:
                pass
    return pids


def _no_process_left(marker: str, wait_s: float = 3.0) -> bool:
    deadline = time.monotonic() + wait_s
    while _pids_with_cmdline(marker):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.1)
    return True


# --------------------------------------------------------------------------------------
# Process model
# --------------------------------------------------------------------------------------


def test_runner_is_fresh_small_and_torch_free() -> None:
    run = cs.run_fresh(cs.postprocess_lcb_sample(ONE_TEST), DOUBLE, timeout=3, global_timeout=60)
    assert run.result == [True] and run.returncode == 0 and not run.timed_out
    assert run.metadata["sandbox_torch_loaded"] is False
    # Interpreter + harness + numpy/pandas (preloaded for the harness's BASE_IMPORTS): ~300 MiB,
    # against ~5.7 GiB for a child forked from an rllm/torch-importing process.
    assert run.metadata["sandbox_vsz_mib"] < 1024


def test_sandbox_env_is_minimal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/somewhere")
    monkeypatch.setenv("MARL_REPO_ROOT", "/somewhere")
    env = cs._sandbox_env()
    assert set(env) == {"PATH", "HOME", "LANG", "PYTHONHASHSEED"} | {
        f"{lib}_NUM_THREADS" for lib in ("OMP", "OPENBLAS", "MKL")
    }


def test_runner_finds_an_installed_rllm(tmp_path: Path, repo_root: Path) -> None:
    """Without third_party next to the runner, it locates rllm on the parent's sys.path."""
    runner = tmp_path / "a" / "b" / "runner.py"
    runner.parent.mkdir(parents=True)
    shutil.copy(cs.RUNNER, runner)
    request = {
        "sample": cs.postprocess_lcb_sample(ONE_TEST),
        "generation": DOUBLE,
        "timeout": 3,
        "max_memory_bytes": 4 * 1024**3,
        "repo_root": None,
        "sys_path": [str(repo_root / "third_party" / "rllm")],
    }
    proc = subprocess.run(
        [sys.executable, "-I", str(runner)], input=json.dumps(request).encode(), capture_output=True, timeout=60
    )
    assert json.loads(proc.stdout)["res"] == [True], proc.stderr


# --------------------------------------------------------------------------------------
# Memory limit
# --------------------------------------------------------------------------------------


def test_memory_cap_admits_256mib_and_stops_5gib() -> None:
    ok, details = cs.grade_sample(ONE_TEST, _alloc(256 * 1024**2))
    assert ok, details
    ok, details = cs.grade_sample(ONE_TEST, _alloc(5 * 1024**3))
    assert not ok
    assert details["test_results"][0]["error"] == "MemoryError()"
    assert details["test_results"][0]["error_message"] == "Runtime Error"


def test_forked_child_of_torch_process_regression() -> None:
    """The bug the fresh interpreter fixes: a 4 GiB RLIMIT_AS set in a child forked from a
    torch-importing process is already used up by the inherited address space, so a 256 MiB
    allocation fails there, while the same program passes in the sandbox."""
    pytest.importorskip("torch")
    if _vsz_bytes() < 4 * 1024**3 + 256 * 1024**2:
        pytest.skip("this process's address space is below the cap; the fork scenario does not apply")
    pid = os.fork()
    if pid == 0:  # child: the fork's _temp_run setting
        code = 2
        try:
            resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
            bytearray(256 * 1024**2)
            code = 0
        except MemoryError:
            code = 1
        finally:
            os._exit(code)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 1  # MemoryError in the forked child
    assert cs.grade_sample(ONE_TEST, _alloc(256 * 1024**2))[0]


# --------------------------------------------------------------------------------------
# Timeouts and children of the solution
# --------------------------------------------------------------------------------------


@pytest.mark.slow
def test_global_timeout_kills_the_whole_session() -> None:
    """A solution that ignores SIGALRM, starts a child and sleeps is killed at the global
    timeout ((1 + 1) * 1 + 5 = 7 s), and the child dies with it."""
    marker = f"{100 + random.random():.9f}"
    code = (
        "import os, signal, time\n"
        "signal.signal(signal.SIGALRM, signal.SIG_IGN)\n"
        f"os.posix_spawnp('sleep', ['sleep', '{marker}'], dict(os.environ))\n"
        "while True:\n"
        "    time.sleep(0.05)\n"
    )
    start = time.monotonic()
    run = cs.run_fresh(cs.postprocess_lcb_sample(ONE_TEST), code, timeout=1, global_timeout=7)
    elapsed = time.monotonic() - start
    assert run.timed_out and run.result is None
    assert 6.5 < elapsed < 11, elapsed
    assert _no_process_left(marker)
    ok, details = cs.build_direct_results(run.result, run.metadata, ["3\n"], ["6\n"])
    assert not ok and details["test_results"][0]["error"] == "global timeout or exception"


def test_children_left_by_a_finished_solution_are_killed() -> None:
    marker = f"{100 + random.random():.9f}"
    code = f"import os\nos.posix_spawnp('sleep', ['sleep', '{marker}'], dict(os.environ))\nprint(int(input()) * 2)"
    ok, _ = cs.grade_sample(ONE_TEST, code)
    assert ok
    assert _no_process_left(marker)


# --------------------------------------------------------------------------------------
# Result classification (shape of the fork's direct mode)
# --------------------------------------------------------------------------------------

OUTCOMES = {
    # name: (code, tests, expected passed flags, check on the first failing detail)
    "passed": (DOUBLE, ONE_TEST, [True], None),
    "wrong_answer": (DOUBLE, THREE_TESTS, [True, False, False], ("error_message", "Wrong answer")),
    "runtime_error": ("raise ValueError('boom')", ONE_TEST, [False], ("error", "ValueError('boom')")),
    "memory_error": (_alloc(5 * 1024**3), ONE_TEST, [False], ("error", "MemoryError()")),
    "timeout": ("while True:\n    pass", ONE_TEST, [False], ("error_message", "Time Limit Exceeded")),
}


@pytest.mark.parametrize("name", sorted(OUTCOMES))
def test_result_classification(name: str) -> None:
    code, tests, passed, check = OUTCOMES[name]
    ok, details = cs.grade_sample(tests, code, timeout=1)
    assert ok is all(passed)
    assert set(details) == {"all_passed", "test_results", "total_tests", "passed_tests"}
    assert [t["passed"] for t in details["test_results"]] == passed
    assert details["total_tests"] == len(tests) and details["passed_tests"] == sum(passed)
    if check is not None:
        first_fail = details["test_results"][passed.index(False)]
        assert set(first_fail) == DETAIL_KEYS
        key, prefix = check
        assert first_fail[key].startswith(prefix), first_fail
    if name == "wrong_answer":  # direct mode stops at the first failure
        assert details["test_results"][2] == {
            "input": "3\n",
            "expected": "6\n",
            "passed": False,
            "error": "skipped (prior test failed)",
        }
