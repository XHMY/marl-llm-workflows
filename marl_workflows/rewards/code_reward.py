"""Code reward: sandboxed unit-test execution of generated programs.

COPIED FROM fork ``rllm/rewards/code_reward.py`` @f0240732 (research fork of rLLM; bounded
copy, only the parts the paper's Code runs use). It replaces these regions of upstream
rllm@186d0bbb ``rllm/rewards/code_reward.py``:

* ``_temp_run`` (:202-206) and ``lcb_check_correctness_v2`` (:208-250), which collect results
  through a multiprocessing manager server process and deadlock or leak under load; the copy
  uses one ``multiprocessing.Pipe`` per subprocess instead,
* ``RewardCodeFn`` (:397-470), which gains the direct-execution branch and a catch-all that
  turns execution errors into an incorrect verdict instead of crashing a pool worker.

Kept verbatim from the fork (line numbers in the fork file): ``set_direct_execution``
(:32-41), memory-cap env parsing (:43-86), ``extract_code_from_model`` (:89-103),
``postprocess_lcb_sample`` (:227-246), ``lcb_check_correctness_direct`` (:249-329),
``_temp_run`` (:360-406), ``lcb_check_correctness_v2`` (:409-543), ``taco_to_lcb_format``
(:625-656) and ``RewardCodeFn`` (:690-783, restricted to the stdin/functional data sources).

Reward semantics (contract C17): in direct mode (set in every code-reward pool worker by
``code_executor_initializer``) one subprocess runs all tests of a problem with 3 s per test
and a global wall-clock cap ``min((3 + 1) * n_tests + 5, 120)`` s. The subprocess closes
inherited file descriptors, sets ``RLIMIT_AS`` (4 GiB by default; env
``RLLM_CODE_REWARD_MAX_MEMORY_GB`` / ``RLLM_CODE_REWARD_MAX_MEMORY_BYTES``) and ``RLIMIT_CPU``
(same formula as the wall-clock cap), and leaves through ``os._exit``. Test execution itself
is upstream's ``rllm.rewards.code_utils.livecodebench.run_test``.

Dropped (not used by any paper run): leetcode, kodcode, humanevalplus, the Together code
interpreter, ``check_correctness`` and ``primeintellect_check_correctness``. Unsupported data
sources raise ``NotImplementedError``.
"""

import json
import logging
import multiprocessing
import os
import re
import resource
import select
import time
from typing import Any

from rllm.agents.agent import Action
from rllm.rewards.code_utils.livecodebench import run_test as lcb_run_test
from rllm.rewards.reward_types import RewardConfig, RewardOutput

__all__ = [
    "RewardCodeFn",
    "SUPPORTED_DATA_SOURCES",
    "code_executor_initializer",
    "code_reward_fn",
    "extract_code_from_model",
    "lcb_check_correctness_direct",
    "lcb_check_correctness_v2",
    "postprocess_lcb_sample",
    "set_direct_execution",
    "taco_to_lcb_format",
]

logger = logging.getLogger(__name__)

_USE_DIRECT_EXECUTION = False
_DEFAULT_MAX_MEMORY_BYTES = 4 * 1024**3
_MAX_MEMORY_GB_ENV = "RLLM_CODE_REWARD_MAX_MEMORY_GB"
_MAX_MEMORY_BYTES_ENV = "RLLM_CODE_REWARD_MAX_MEMORY_BYTES"

_TACO_SOURCES = ("taco", "apps", "code_contests")
_LCB_SOURCES = ("livecodebench", "codeforces", "primeintellect")
SUPPORTED_DATA_SOURCES = _TACO_SOURCES + _LCB_SOURCES


def set_direct_execution(enabled: bool):
    global _USE_DIRECT_EXECUTION
    _USE_DIRECT_EXECUTION = enabled


def code_executor_initializer() -> None:
    """``ProcessPoolExecutor`` initializer for code-reward workers (fork ``agent_workflow_engine.py:26-31``).

    Pool workers run each problem in one subprocess with a kill-based global timeout
    (``lcb_check_correctness_direct``) instead of one subprocess per test.
    """
    set_direct_execution(True)


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
        logger.warning(
            "Invalid %s=%r; falling back to %.1f GiB",
            name,
            raw,
            default_bytes / 1024**3,
        )
        return _UNSET
    return int(value * scale) if value > 0 else None


def _configured_max_memory_bytes(default_bytes: int = _DEFAULT_MAX_MEMORY_BYTES) -> int | None:
    """Return the configured per-test subprocess RLIMIT_AS cap.

    The default remains 4 GiB. Set RLLM_CODE_REWARD_MAX_MEMORY_GB=6 for hw4090,
    or RLLM_CODE_REWARD_MAX_MEMORY_BYTES for an exact byte value. Values of 0,
    "none", "unlimited", "off", or "false" disable the address-space cap.
    """
    for name, parser, scale in (
        (_MAX_MEMORY_BYTES_ENV, int, 1),
        (_MAX_MEMORY_GB_ENV, float, 1024**3),
    ):
        result = _parse_memory_env(name, parser, scale, default_bytes)
        if result is not _UNSET:
            return result
    return default_bytes


# Resolve once at import; subprocesses inherit the cached value and avoid
# repeated warning logs on a misconfigured env var.
_RESOLVED_MAX_MEMORY_BYTES = _configured_max_memory_bytes()


def extract_code_from_model(model_response: str):
    """
    Extracts the code from a Markdown-style code block in an LLM output.

    Parameters:
        model_response (str): The text output from the LLM.

    Returns:
        str: The extracted code, or an empty string if no code block is found.
    """
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", model_response, re.DOTALL)
    if not code_blocks:
        return None
    return code_blocks[-1].strip()


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
            f"Function name is not found, check if your LCB data is preprocessed correctly: {metadata}\nSample: {sample}"
        )
        # Fill in the blank
        sample_dict["fn_name"] = fn_name

    sample = {
        "input_output": json.dumps(sample_dict),
    }
    return sample


def lcb_check_correctness_direct(sample, generation, timeout=3, debug=False, max_global_timeout=120):
    """Run lcb_run_test in a subprocess with kill-based timeout.
    Called from ProcessPoolExecutor workers for problem-level parallelism.
    The subprocess isolates reliability_guard() state corruption and provides
    an uncatchable SIGKILL timeout via p.kill()."""
    assert len(sample) >= 1, "Sample must contain at least one test case"
    processed_sample = postprocess_lcb_sample(sample)

    in_outs = json.loads(processed_sample["input_output"])
    all_inputs = in_outs["inputs"]
    all_outputs = in_outs["outputs"]
    num_tests = len(all_inputs)

    # Cap global_timeout: incorrect code early-returns on first failure,
    # and stuck code (C-level GIL hang) blocks on the first test anyway.
    # Only correct-but-slow solutions need the full time, and those rarely
    # take more than ~1s per test.
    global_timeout = min((timeout + 1) * num_tests + 5, max_global_timeout)

    # Spawn a subprocess to run lcb_run_test with kill-based timeout
    parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
    p = multiprocessing.Process(
        target=_temp_run,
        args=(processed_sample, generation, debug, child_conn, timeout),
    )
    p.start()
    child_conn.close()

    p.join(timeout=global_timeout)
    if p.is_alive():
        p.kill()
        p.join(timeout=5)

    # Read result from pipe
    result, metadata = None, None
    try:
        if parent_conn.poll():
            result, metadata = parent_conn.recv()
    except (EOFError, OSError):
        pass
    finally:
        parent_conn.close()
        p.close()  # Release subprocess resources to prevent zombie/resource leaks

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


def _temp_run(sample, generation, debug, conn, timeout, max_memory_bytes=None):
    try:
        # Close inherited FDs from the parent process (Ray IPC, CUDA, VLLM shared
        # memory, pipes from other threads, etc.) to prevent cross-thread FD
        # accumulation when forking from a multi-threaded process.
        # Only keep stdin/stdout/stderr (0-2) and the pipe FD for sending results.
        pipe_fd = conn.fileno()
        os.closerange(3, pipe_fd)
        os.closerange(pipe_fd + 1, 4096)

        # Set memory limit to prevent LLM-generated code from consuming unbounded memory.
        # RLIMIT_AS caps virtual address space; if exceeded, malloc/mmap fails with MemoryError.
        if max_memory_bytes is None:
            max_memory_bytes = _RESOLVED_MAX_MEMORY_BYTES
        if max_memory_bytes is not None:
            resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))

        # Set CPU time limit as kernel-level backup for signal.alarm.
        # signal.alarm cannot interrupt C-level operations holding the GIL
        # (e.g., "a"*10**9, pathological regex). RLIMIT_CPU is enforced by
        # the kernel regardless of GIL state. Must be set before run_test()
        # calls reliability_guard(), which disables the resource module.
        try:
            in_outs = json.loads(sample["input_output"])
            num_tests_in_batch = len(in_outs.get("inputs", []))
        except (json.JSONDecodeError, KeyError):
            num_tests_in_batch = 1  # fallback: each subprocess runs one test
        cpu_soft_limit = min((timeout + 1) * num_tests_in_batch + 5, 120)
        cpu_hard_limit = cpu_soft_limit + 3
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_soft_limit, cpu_hard_limit))

        res, metadata = lcb_run_test(sample, test=generation, debug=debug, timeout=timeout)
        conn.send((res, metadata))
    except Exception as e:
        # MemoryError is a subclass of Exception, so this catches OOM from RLIMIT_AS.
        # Send a failure result so the parent can distinguish error type from timeout.
        try:
            error_name = type(e).__name__
            conn.send(([-4], {"error_code": -4, "error_message": f"{error_name}: {e}"}))
        except Exception:
            pass  # If even sending fails (e.g. deep OOM), parent gets None → counted as failure
    finally:
        conn.close()
        # Hard-exit to skip atexit handlers (e.g. Ray's shutdown callback).
        # Under RLIMIT_AS, atexit handlers fail with MemoryError when trying to import modules.
        # The pipe is already closed, so results are safely delivered to the parent.
        os._exit(0)


def lcb_check_correctness_v2(sample, generation, timeout=3, debug=False):
    """Check correctness of code generation with per-test parallelism.

    Each test case runs in its own subprocess so a slow or stuck test cannot
    block other tests. The global deadline is bounded by a single test timeout
    (not num_tests * timeout), dramatically reducing worst-case evaluation time
    when tests run at different speeds.

    We use Pipe (not joblib) because reliability_guard() inside run_test
    disables os.fork/os.getcwd/etc., which breaks joblib/loky's pickling.
    Pipe.send() uses low-level file descriptors unaffected by reliability_guard().
    """
    assert len(sample) >= 1, "Sample must contain at least one test case"
    processed_sample = postprocess_lcb_sample(sample)

    in_outs = json.loads(processed_sample["input_output"])
    all_inputs = in_outs["inputs"]
    all_outputs = in_outs["outputs"]
    fn_name = in_outs.get("fn_name", None)
    num_tests = len(all_inputs)

    pipes = []
    processes = []
    try:
        for i in range(num_tests):
            single_dict = {"inputs": [all_inputs[i]], "outputs": [all_outputs[i]]}
            if fn_name is not None:
                single_dict["fn_name"] = fn_name
            single_sample = {"input_output": json.dumps(single_dict)}

            parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
            try:
                p = multiprocessing.Process(
                    target=_temp_run,
                    args=(single_sample, generation, debug, child_conn, timeout),
                )
                p.start()
            except OSError:
                parent_conn.close()
                child_conn.close()
                raise
            child_conn.close()
            pipes.append(parent_conn)
            processes.append(p)
    except OSError:
        # Clean up already-started processes if we hit the process limit mid-loop
        for p in processes:
            if p.is_alive():
                p.kill()
                p.join(timeout=5)
            try:
                p.close()
            except ValueError:
                pass
        for conn in pipes:
            conn.close()
        raise

    # Poll results as they arrive. On first failure, kill all remaining
    # processes — no point running more tests for incorrect code.
    # Use poll() instead of select() because select() has a hard FD_SETSIZE=1024
    # limit — pipe FDs in a Ray worker easily exceed 1024.
    deadline = time.monotonic() + timeout + min(max(10, num_tests), 60)
    test_results = [None] * num_tests
    fd_to_index = {conn.fileno(): i for i, conn in enumerate(pipes)}
    early_fail = False

    poller = select.poll()
    for fd in fd_to_index:
        poller.register(fd, select.POLLIN)

    try:
        while fd_to_index and not early_fail:
            remaining = max(0, deadline - time.monotonic())
            if remaining <= 0:
                break
            # poll() timeout is in milliseconds
            events = poller.poll(remaining * 1000)
            if not events:
                break  # deadline hit with no new results

            for fd, _event in events:
                i = fd_to_index.pop(fd)
                poller.unregister(fd)
                try:
                    result, metadata = pipes[i].recv()
                except (EOFError, OSError):
                    result, metadata = None, None

                if result is None:
                    test_results[i] = {
                        "input": all_inputs[i],
                        "expected": all_outputs[i],
                        "passed": False,
                        "error": "global timeout",
                    }
                    early_fail = True
                else:
                    passed = isinstance(result, list) and len(result) == 1 and result[0] is True
                    test_results[i] = {
                        "input": all_inputs[i],
                        "expected": all_outputs[i],
                        "passed": passed,
                        "error": metadata.get("error", None) if metadata else None,
                        "error_message": metadata.get("error_message", None) if metadata else None,
                        "output": metadata.get("output", None) if metadata else None,
                    }
                    if not passed:
                        early_fail = True
    finally:
        # Always kill remaining processes and close pipes, even on exceptions.
        # Without this finally, an exception (e.g. from recv) would leak pipe
        # FDs and orphan child processes — the root cause of the 17K FD leak.
        for p in processes:
            if p.is_alive():
                p.kill()
                p.join(timeout=5)
        for p in processes:
            try:
                p.close()
            except ValueError:
                pass
        for i, parent_conn in enumerate(pipes):
            if test_results[i] is None:
                test_results[i] = {
                    "input": all_inputs[i],
                    "expected": all_outputs[i],
                    "passed": False,
                    "error": "killed (another test failed)",
                }
            parent_conn.close()

    passed_tests = sum(1 for t in test_results if t["passed"])
    detailed_results = {
        "all_passed": passed_tests == num_tests,
        "test_results": test_results,
        "total_tests": num_tests,
        "passed_tests": passed_tests,
    }
    return detailed_results["all_passed"], detailed_results


def taco_to_lcb_format(tests):
    """
    Given a dictionary with keys "inputs" and "outputs", returns a list of test cases.
    Each test case is a dictionary with keys "input" and "output". If the lists are unequal,
    missing entries are filled by reusing the first element of the shorter list.

    Args:
        data (dict): A dictionary with keys "inputs" and "outputs", each mapped to a list of strings.

    Returns:
        list of dict: A list where each element is a dict with keys "input" and "output".
    """
    inputs = tests.get("inputs", [])
    outputs = tests.get("outputs", [])

    # Determine the number of test cases to create.
    n = max(len(inputs), len(outputs))

    test_cases = []
    for i in range(n):
        # Use the first element as a fallback if the list is shorter than n.
        inp = inputs[i] if i < len(inputs) else (inputs[0] if inputs else "")
        out = outputs[i] if i < len(outputs) else (outputs[0] if outputs else "")
        out = out[0] if isinstance(out, list) else out
        test_case: dict[str, Any] = {"input": inp, "output": out, "metadata": {}}
        if "fn_name" in tests:
            test_case["testtype"] = "functional"
            test_case["metadata"]["func_name"] = tests["fn_name"]
        test_cases.append(test_case)

    return test_cases


class RewardCodeFn:
    """
    Reward function for evaluating code dataset answers.

    This class implements the RewardFunction protocol to process the input and determine
    the reward based on the correctness of the unit tests provided
    """

    def __init__(self, config: RewardConfig):
        self.config = config

    def __call__(self, task_info: dict, action: str) -> RewardOutput:
        """
        Calculate the reward for a code task based on the agent's action.

        Args:
            task_info: Dictionary containing problem, data_source, problem_type, and ground_truth
            action: The agent's response/solution (code)

        Returns:
            RewardOutput: The calculated reward with correctness information
        """
        model_response = action
        dataset_name = task_info.get("data_source", "")
        tests = task_info.get("ground_truth", None)

        if tests is None or (isinstance(tests, str) and not tests.strip()):
            print("No tests found in task_info")
            return RewardOutput(
                reward=self.config.format_error_reward,
                is_correct=False,
                metadata={"error": "No tests found in task_info"},
            )

        model_code = extract_code_from_model(model_response)
        if model_code is None:
            return RewardOutput(
                reward=self.config.format_error_reward,
                is_correct=False,
                metadata={"error": "No code found in model response"},
            )

        # >>> MARL(code-reward-scope): the Together code interpreter branch is not copied.
        if self.config.use_together_code_interpreter:
            raise NotImplementedError("use_together_code_interpreter is not supported by marl_workflows")
        # <<< MARL(code-reward-scope)

        # Tests: List[Dictionary] - Codeforces, LiveCodeBench
        # Tests: Dictionary[Lists] - CodeContests, Taco/Apps
        is_correct = False
        test_details: dict[str, Any] = {}

        try:
            if dataset_name in _TACO_SOURCES:
                tests = taco_to_lcb_format(tests)
                if _USE_DIRECT_EXECUTION:
                    is_correct, test_details = lcb_check_correctness_direct(tests, model_code, debug=False)
                else:
                    is_correct, test_details = lcb_check_correctness_v2(tests, model_code, debug=False)
            elif dataset_name in _LCB_SOURCES:
                # Handle case where tests is a JSON string
                if isinstance(tests, str):
                    try:
                        tests = json.loads(tests)
                    except json.decoder.JSONDecodeError:
                        print("test json invalid: ", tests)
                        return RewardOutput(
                            reward=self.config.format_error_reward,
                            is_correct=False,
                            metadata={"error": "Tests in task_info is invalid json"},
                        )
                if _USE_DIRECT_EXECUTION:
                    is_correct, test_details = lcb_check_correctness_direct(tests, model_code, debug=False)
                else:
                    is_correct, test_details = lcb_check_correctness_v2(tests, model_code, debug=False)
            else:
                # >>> MARL(code-reward-scope): leetcode / kodcode / humanevalplus are not copied.
                raise NotImplementedError(f"Dataset {dataset_name} not implemented")
                # <<< MARL(code-reward-scope)
        except NotImplementedError:
            raise  # Programming error — don't silently swallow
        except Exception as e:
            # Safety net: catch any exception (including MemoryError) to prevent
            # crashing the ProcessPoolExecutor worker, which would cause BrokenProcessPool.
            logger.warning("RewardCodeFn: exception during correctness check: %s: %s", type(e).__name__, e)
            return RewardOutput(
                reward=self.config.incorrect_reward,
                is_correct=False,
                metadata={"error": f"{type(e).__name__}: {e}"},
            )

        if is_correct:
            return RewardOutput(reward=self.config.correct_reward, is_correct=True, metadata=test_details)
        else:
            return RewardOutput(reward=self.config.incorrect_reward, is_correct=False, metadata=test_details)


def code_reward_fn(task_info: dict, action: Any) -> RewardOutput:
    """Code reward function (mirrors fork ``rllm/rewards/reward_fn.py`` ``code_reward_fn``)."""
    reward_fn = RewardCodeFn(RewardConfig())
    if isinstance(action, Action):
        action = action.action
    return reward_fn(task_info, action)
