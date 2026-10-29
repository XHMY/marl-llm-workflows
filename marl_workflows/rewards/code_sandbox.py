"""Fresh-interpreter sandbox for the code reward (stdlib only; never imports rllm or torch).

Each sample is graded by ``code_sandbox_runner.py`` in a new ``python -I`` process started in
its own session with a minimal environment. The runner's address space starts small (no torch),
so ``RLIMIT_AS`` (4 GiB by default) bounds what the solution itself can allocate. The global
wall-clock timeout kills the whole session with ``SIGKILL``, which also ends processes the
solution started; the session is killed after every sample, so none outlive their grading.

Result semantics are those of the research fork's direct mode: one process runs all tests of a
problem with 3 s per test, stops at the first failure, and is killed after
``min((3 + 1) * n_tests + 5, 120)`` s. ``grade_sample`` is the target of the code-reward process
pool; it returns ``(all_passed, detailed_results)``.
"""

import json
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

__all__ = ["SandboxRun", "build_direct_results", "grade_sample", "postprocess_lcb_sample", "run_fresh", "worker_stats"]

logger = logging.getLogger(__name__)

RUNNER = Path(__file__).with_name("code_sandbox_runner.py")
MAX_FILE_BYTES = 64 * 1024**2  # RLIMIT_FSIZE of the runner
_DEFAULT_MAX_MEMORY_BYTES = 4 * 1024**3
_MAX_MEMORY_GB_ENV = "RLLM_CODE_REWARD_MAX_MEMORY_GB"
_MAX_MEMORY_BYTES_ENV = "RLLM_CODE_REWARD_MAX_MEMORY_BYTES"
_DISABLE_SENTINELS = {"0", "none", "unlimited", "off", "false"}
_UNSET = object()


def _parse_memory_env(name: str, parser, scale: int, default_bytes: int):
    """Parse a memory-cap env var. Returns int bytes, None (disabled), or _UNSET (not set / invalid)."""
    raw = os.environ.get(name)
    if raw is None:
        return _UNSET
    s = raw.strip().lower()
    if s in _DISABLE_SENTINELS:
        return None
    try:
        value = parser(s)
    except ValueError:
        logger.warning("Invalid %s=%r; falling back to %.1f GiB", name, raw, default_bytes / 1024**3)
        return _UNSET
    return int(value * scale) if value > 0 else None


def _configured_max_memory_bytes(default_bytes: int = _DEFAULT_MAX_MEMORY_BYTES) -> int | None:
    """Return the configured RLIMIT_AS / RLIMIT_DATA cap of the sandbox runner.

    The default is 4 GiB. ``RLLM_CODE_REWARD_MAX_MEMORY_BYTES`` (exact bytes) takes precedence
    over ``RLLM_CODE_REWARD_MAX_MEMORY_GB``. Values of 0, "none", "unlimited", "off" or "false"
    disable the cap; invalid values fall back to the next variable, then the default.
    """
    for name, parser, scale in ((_MAX_MEMORY_BYTES_ENV, int, 1), (_MAX_MEMORY_GB_ENV, float, 1024**3)):
        result = _parse_memory_env(name, parser, scale, default_bytes)
        if result is not _UNSET:
            return result
    return default_bytes


# Resolve once at import, so a misconfigured env var warns once per process.
_RESOLVED_MAX_MEMORY_BYTES = _configured_max_memory_bytes()


def postprocess_lcb_sample(sample):
    sample_inputs = [sample["input"] for sample in sample]
    sample_outputs = [sample["output"] for sample in sample]

    sample_dict = {
        "inputs": sample_inputs,
        "outputs": sample_outputs,
    }

    if sample[0].get("testtype") == "functional":
        metadata = sample[0].get("metadata", {})
        fn_name = metadata.get("func_name", None)
        assert fn_name is not None, (
            f"Function name is not found, check if your LCB data is preprocessed correctly: {metadata}"
            f"\nSample: {sample}"
        )
        # Fill in the blank
        sample_dict["fn_name"] = fn_name

    sample = {
        "input_output": json.dumps(sample_dict),
    }
    return sample


class SandboxRun(NamedTuple):
    result: list | None  # run_test's per-test results; None if the runner sent nothing
    metadata: dict | None
    returncode: int | None
    timed_out: bool


def _sandbox_env() -> dict[str, str]:
    """Minimal environment: no PYTHONPATH or other variable that could make rllm/torch importable."""
    env = {
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": os.environ.get("HOME", "/"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "0"),
    }
    # The harness prepends numpy/pandas imports to every solution; one BLAS thread keeps the
    # runner's address space (and so its headroom under RLIMIT_AS) independent of the core count.
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return env


def _kill_session(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_fresh(sample, generation, timeout=3, debug=False, global_timeout=120, max_memory_bytes=_UNSET) -> SandboxRun:
    """Run upstream ``run_test`` on one processed sample in a fresh interpreter."""
    if max_memory_bytes is _UNSET:
        max_memory_bytes = _RESOLVED_MAX_MEMORY_BYTES
    request = {
        "sample": sample,
        "generation": generation,
        "timeout": timeout,
        "debug": debug,
        "max_memory_bytes": max_memory_bytes,
        "max_file_bytes": MAX_FILE_BYTES,
        "repo_root": os.environ.get("MARL_REPO_ROOT"),
        "sys_path": [p for p in sys.path if p],
    }
    proc = subprocess.Popen(
        [sys.executable, "-I", str(RUNNER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        env=_sandbox_env(),
    )
    timed_out = False
    out = b""
    try:
        out, _ = proc.communicate(json.dumps(request).encode(), timeout=global_timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_session(proc.pid)
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:  # a process outside the session still holds a pipe
            proc.kill()
            proc.wait()
        out = b""
    finally:
        _kill_session(proc.pid)  # processes the solution left behind in the session
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None and not stream.closed:
                stream.close()
        if proc.returncode is None:
            proc.kill()
            proc.wait()
    try:
        response = json.loads(out) if out else None
    except ValueError:
        response = None
    if not isinstance(response, dict):
        return SandboxRun(None, None, proc.returncode, timed_out)
    return SandboxRun(response.get("res"), response.get("metadata"), proc.returncode, timed_out)


def build_direct_results(result, metadata, all_inputs, all_outputs):
    """Per-test details from one direct-mode run (fork ``lcb_check_correctness_direct``)."""
    num_tests = len(all_inputs)
    detailed_results = {"all_passed": False, "test_results": [], "total_tests": num_tests, "passed_tests": 0}

    if result is None:
        for j in range(num_tests):
            detailed_results["test_results"].append(
                {
                    "input": all_inputs[j],
                    "expected": all_outputs[j],
                    "passed": False,
                    "error": "global timeout or exception",
                }
            )
        return False, detailed_results

    for j in range(num_tests):
        if j < len(result):
            passed = result[j] == True  # noqa: E712 (verbatim; result entries are bools or negative ints)
            detail = {
                "input": all_inputs[j],
                "expected": all_outputs[j],
                "passed": passed,
                "error": metadata.get("error", None) if metadata else None,
                "error_message": metadata.get("error_message", None) if metadata else None,
                "output": metadata.get("output", None) if metadata else None,
            }
            detailed_results["test_results"].append(detail)
        else:
            detailed_results["test_results"].append(
                {
                    "input": all_inputs[j],
                    "expected": all_outputs[j],
                    "passed": False,
                    "error": "skipped (prior test failed)",
                }
            )

    detailed_results["passed_tests"] = sum(1 for t in detailed_results["test_results"] if t["passed"])
    detailed_results["all_passed"] = all(t["passed"] for t in detailed_results["test_results"])

    return detailed_results["all_passed"], detailed_results


def grade_sample(sample, generation, timeout=3, debug=False, max_global_timeout=120):
    """Grade one solution against all tests of a problem (LCB-format test list)."""
    assert len(sample) >= 1, "Sample must contain at least one test case"
    processed_sample = postprocess_lcb_sample(sample)
    in_outs = json.loads(processed_sample["input_output"])
    # Incorrect code returns at its first failure and stuck code blocks on the first test, so
    # only correct-but-slow solutions need the full budget.
    global_timeout = min((timeout + 1) * len(in_outs["inputs"]) + 5, max_global_timeout)
    run = run_fresh(processed_sample, generation, timeout=timeout, debug=debug, global_timeout=global_timeout)
    return build_direct_results(run.result, run.metadata, in_outs["inputs"], in_outs["outputs"])


def worker_stats() -> dict:
    """Memory of the calling process and whether rllm/torch are loaded (pool-worker diagnostics)."""
    page = os.sysconf("SC_PAGE_SIZE")
    with open("/proc/self/statm") as f:
        vsz, rss = (int(x) * page / 2**20 for x in f.read().split()[:2])
    return {
        "pid": os.getpid(),
        "vsz_mib": round(vsz, 1),
        "rss_mib": round(rss, 1),
        "torch_loaded": "torch" in sys.modules,
        "rllm_loaded": "rllm" in sys.modules,
    }
