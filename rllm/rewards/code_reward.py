"""
This module contains the RewardCode class, which evaluates code datasets answers
and assigns rewards based on their correctness on unit tests.
"""

import ast
import json
import logging
import multiprocessing
import os
import re
import resource
import select
import time
from typing import Any

from rllm.rewards.code_utils.firejail_exec import code_exec_firejail as lc_code_exec
from rllm.rewards.code_utils.humanevalplus import get_num_test_cases

# from rllm.rewards.code_utils.swebench import swebench_check_correctness
from rllm.rewards.code_utils.humanevalplus import run_test as humanevalplus_run_test
from rllm.rewards.code_utils.kodcode import code_exec as kod_code_exec

# from rllm.rewards.code_utils.code_contests import run_test as code_contests_run_test
from rllm.rewards.code_utils.livecodebench import run_test as lcb_run_test
from rllm.rewards.code_utils.taco import run_test as taco_run_test
from rllm.rewards.reward_types import RewardConfig, RewardOutput, RewardType
from rllm.tools.code_tools.code_tool import CodeTool
from rllm.tools.code_tools.together_tool import TogetherCodeTool

logger = logging.getLogger(__name__)

_USE_DIRECT_EXECUTION = False
_DEFAULT_MAX_MEMORY_BYTES = 4 * 1024 ** 3
_MAX_MEMORY_GB_ENV = "RLLM_CODE_REWARD_MAX_MEMORY_GB"
_MAX_MEMORY_BYTES_ENV = "RLLM_CODE_REWARD_MAX_MEMORY_BYTES"


def set_direct_execution(enabled: bool):
    global _USE_DIRECT_EXECUTION
    _USE_DIRECT_EXECUTION = enabled


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
            name, raw, default_bytes / 1024 ** 3,
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
        (_MAX_MEMORY_GB_ENV, float, 1024 ** 3),
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


def clean_code_main_block(code: str) -> str:
    """
    Removes `if __name__ == "__main__"` blocks from Python code.

    Args:
        code (str): The input Python code.

    Returns:
        str: Cleaned code without the main execution block.
    """
    code_lines = code.split("\n")
    filtered_lines = []
    skip_block = False

    for line in code_lines:
        if line.strip().startswith('if __name__ == "__main__"') or line.strip().startswith("if __name__ == '__main__'"):
            skip_block = True
            continue
        if skip_block:
            # Check if we're out of the block (less indentation)
            if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                skip_block = False
            else:
                continue
        filtered_lines.append(line)

    return "\n".join(filtered_lines)


def check_correctness(tests: list[dict[str, str]] | dict[str, list[str]], code: str, test_fn, timeout_per_test: int = 12, max_tests: int = 15) -> tuple[bool, dict[str, Any]]:
    """
    Check if generated code passes all test cases within a timeout period.

    Args:
        tests: Test cases in either list of dictionaries or dictionary of lists format
        code: Generated code to test
        test_fn: Function to run tests
        timeout: Maximum execution time in seconds before killing process

    Returns:
        tuple: (bool, dict) where:
            - bool: True if all tests pass, False otherwise
            - dict: Detailed test results with test cases and pass/fail status
    """

    def evaluate_code(tests, generation, debug, conn, test_fn):
        """Helper function to run tests in separate process."""
        try:
            result = test_fn(tests, test=generation, debug=debug, timeout=timeout_per_test)
            conn.send(result)
        except Exception as e:
            print(f"Error in evaluate_code: {e}")
        finally:
            conn.close()

    original_tests = tests
    if isinstance(tests, list):
        list_tests = tests
        total_tests = len(list_tests)
        if total_tests > max_tests:
            # Sort indices by test input length and take the max_tests longest ones
            selected_indices = sorted(range(total_tests), key=lambda i: len(list_tests[i]["input"]), reverse=True)[:max_tests]
            tests = [list_tests[i] for i in selected_indices]
        num_tests = len(tests)
    else:
        dict_tests = tests
        total_tests = len(dict_tests["inputs"])
        if total_tests > max_tests:
            # Select the tests with the longest input length.
            selected_indices = sorted(range(total_tests), key=lambda i: len(dict_tests["inputs"][i]), reverse=True)[:max_tests]
            # Create a new dict with only the selected test cases
            selected_tests: dict[str, list[str]] = {"inputs": [dict_tests["inputs"][i] for i in selected_indices], "outputs": [dict_tests["outputs"][i] for i in selected_indices]}
            tests = selected_tests
        num_tests = len(tests["inputs"])

    parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
    process = multiprocessing.Process(target=evaluate_code, args=(tests, code, False, child_conn, test_fn))
    process.start()
    child_conn.close()  # Close child end in parent so recv() can detect EOF

    global_timeout = (timeout_per_test + 1) * num_tests + 5
    process.join(timeout=global_timeout)

    if process.is_alive():
        process.kill()
        process.join(timeout=5)

    # Read result from pipe
    test_results_data = None
    try:
        if parent_conn.poll():
            test_results_data = parent_conn.recv()
    except (EOFError, OSError):
        pass
    finally:
        parent_conn.close()
        process.close()

    detailed_results: dict[str, Any] = {"all_passed": False, "test_results": [], "total_tests": num_tests, "passed_tests": 0}

    if test_results_data is None:
        return False, detailed_results

    passed_results = [r == True for r in test_results_data]

    # Create detailed test results
    test_results_list_typed: list[dict[str, Any]] = detailed_results["test_results"]
    if isinstance(original_tests, list):
        assert isinstance(tests, list)
        for i, (test, result) in enumerate(zip(tests, passed_results, strict=False)):
            test_results_list_typed.append({"input": test.get("input", ""), "expected": test.get("output", ""), "passed": result})
    else:
        assert isinstance(tests, dict)
        for i, (inp, out, result) in enumerate(zip(tests["inputs"], tests["outputs"], passed_results, strict=False)):
            test_results_list_typed.append({"input": inp, "expected": out, "passed": result})

    detailed_results["passed_tests"] = sum(passed_results)
    detailed_results["all_passed"] = all(passed_results)

    return all(passed_results), detailed_results


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
        assert fn_name is not None, f"Function name is not found, check if your LCB data is preprocessed correctly: {metadata}\nSample: {sample}"
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
            detailed_results["test_results"].append({
                "input": all_inputs[j],
                "expected": all_outputs[j],
                "passed": False,
                "error": "global timeout or exception",
            })
        return False, detailed_results

    for j in range(num_tests):
        if j < len(result):
            passed = result[j] == True
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
            detailed_results["test_results"].append({
                "input": all_inputs[j],
                "expected": all_outputs[j],
                "passed": False,
                "error": "skipped (prior test failed)",
            })

    detailed_results["passed_tests"] = sum(1 for t in detailed_results["test_results"] if t["passed"])
    detailed_results["all_passed"] = all(t["passed"] for t in detailed_results["test_results"])

    return detailed_results["all_passed"], detailed_results


# https://huggingface.co/datasets/PrimeIntellect/verifiable-coding-problems
def primeintellect_check_correctness(tests, code, use_tci=False):
    if isinstance(tests, str):
        try:
            tests = ast.literal_eval(tests)
            assert isinstance(tests, dict)
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing string: {e}")
            return False, {"all_passed": False, "error": str(e)}

    assert len(tests) >= 1, "PrimeIntellect needs at least one test case"
    # Convert the tests to the format expected by the taco_run_test function
    inputs = [t["input"] for t in tests]
    outputs = [t["output"] for t in tests]
    fn_name = tests[0].get("fn_name", None)
    tests_formatted = {
        "inputs": inputs,
        "outputs": outputs,
    }
    if fn_name:
        tests_formatted["fn_name"] = fn_name

    if use_tci:
        codetool = TogetherCodeTool()
        return codetool_check_correctness(tests_formatted, code, codetool, is_taco_format=True)

    return check_correctness(tests_formatted, code, taco_run_test)


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
                        "input": all_inputs[i], "expected": all_outputs[i],
                        "passed": False, "error": "global timeout",
                    }
                    early_fail = True
                else:
                    passed = isinstance(result, list) and len(result) == 1 and result[0] is True
                    test_results[i] = {
                        "input": all_inputs[i], "expected": all_outputs[i],
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
                    "input": all_inputs[i], "expected": all_outputs[i],
                    "passed": False, "error": "killed (another test failed)",
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


def leetcode_check_correctness(tests: dict[str, str], code: str) -> tuple[bool, dict[str, Any]]:
    """
    Check if generated code passes all LeetCode test cases.

    Args:
         tests: Dict of test cases with "functional" key containing test code
         code: Generated code to test
         timeout: Maximum execution time in seconds before killing process
         runtime_debug: Whether to print debug info during test execution

    Returns:
         tuple: (bool, dict) where:
           - bool: True if all tests pass, False otherwise
           - dict: Detailed test results
    """
    succ, output = lc_code_exec(code + "\n" + tests["functional"])
    detailed_results = {"all_passed": succ, "output": output, "test_results": [{"passed": succ, "output": output}]}

    if not succ:
        print(f"Error in code execution: {output}")
    return succ, detailed_results


def kodcode_check_correctness(test: str, code: str, timeout_per_test: int = 5) -> tuple[bool, dict[str, Any]]:
    """
    Check if generated code passes all Kodcode test cases.

    Args:
        test: String of the test file content
        code: Generated code to test
        timeout: Maximum execution time in seconds before killing process
        runtime_debug: Whether to print debug info during test execution

    Returns:
        tuple: (bool, dict) where:
            - bool: True if all tests pass, False otherwise
            - dict: Detailed test results
    """
    # Count the number of test functions in the test file
    num_tests = test.count("def test")

    # Remove 'if __name__ == "__main__":' block if present
    code = clean_code_main_block(code)

    succ, output = kod_code_exec(code, test, timeout_per_test * num_tests)
    detailed_results = {"all_passed": succ, "output": output, "total_tests": num_tests, "test_results": [{"passed": succ, "output": output}]}

    if not succ:
        print(f"Error in code execution: {output}")
    return succ, detailed_results


def humanevalplus_check_correctness(test: str, code: str, timeout_per_test: int = 1) -> tuple[bool, dict[str, Any]]:
    """
    Check if generated code passes all HumanEvalPlus test cases.

    Args:
        test: String of the test file content
        code: Generated code to test
        timeout: Maximum execution time in seconds before killing process
        runtime_debug: Whether to print debug info during test execution

    Returns:
        tuple: (bool, dict) where:
            - bool: True if all tests pass, False otherwise
            - dict: Detailed test results
    """
    code = clean_code_main_block(code)

    num_test_cases = get_num_test_cases(test)
    succ, output = humanevalplus_run_test(code, test, timeout_per_test * num_test_cases)

    detailed_results = {"all_passed": succ, "output": output, "total_tests": num_test_cases, "test_results": [{"passed": succ, "output": output}]}

    if not succ:
        print(f"Error in code execution: {output}")
    return succ, detailed_results


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


def codetool_check_correctness(tests: Any, code: str, codetool: CodeTool, is_taco_format=True, timeout=30) -> tuple[bool, dict[str, Any]]:
    from rllm.tools.utils import call_based_test_code_wrapper, stdin_test_code_wrapper

    fn_name = None
    call_based = False

    if isinstance(tests, dict) and "fn_name" in tests:
        call_based = True
        fn_name = tests.get("fn_name", None)

    new_tests = taco_to_lcb_format(tests) if is_taco_format and not fn_name else tests

    if call_based:
        test_wrapped_code = call_based_test_code_wrapper(code, new_tests)
    else:
        test_wrapped_code = stdin_test_code_wrapper(code, new_tests)

    tool_response = codetool(code=test_wrapped_code, timeout=timeout)

    detailed_results = {"all_passed": not tool_response.error, "output": tool_response.output, "error": tool_response.error, "test_results": []}

    # Try to extract individual test results if possible
    if isinstance(new_tests, list):
        detailed_results["total_tests"] = len(new_tests)
        detailed_results["test_results"] = [{"input": test.get("input", ""), "expected": test.get("output", ""), "passed": not tool_response.error} for test in new_tests]

    if tool_response.error:
        print(f"Error in code execution: {tool_response.error}")
        return False, detailed_results
    return True, detailed_results


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
        # total_start_time = time.time()

        model_response = action
        dataset_name = task_info.get("data_source", "")
        tests = task_info.get("ground_truth", None)

        if tests is None or (isinstance(tests, str) and not tests.strip()):
            print("No tests found in task_info")
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False, metadata={"error": "No tests found in task_info"})

        model_code = extract_code_from_model(model_response)
        if model_code is None:
            # print("No code found in model response")
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False, metadata={"error": "No code found in model response"})

        if self.config.use_together_code_interpreter:
            codetool = TogetherCodeTool()

        # Tests: List[Dictionary] - Codeforces, LiveCodeBench
        # Tests: Dictionary[Lists] - CodeContests, Taco/Apps
        is_correct = False
        test_details: dict[str, Any] = {}

        try:
            if dataset_name in ["taco", "apps", "code_contests"]:
                if self.config.use_together_code_interpreter:
                    is_correct, test_details = codetool_check_correctness(tests, model_code, codetool, is_taco_format=True)
                else:
                    tests = taco_to_lcb_format(tests)
                    if _USE_DIRECT_EXECUTION:
                        is_correct, test_details = lcb_check_correctness_direct(tests, model_code, debug=False)
                    else:
                        is_correct, test_details = lcb_check_correctness_v2(tests, model_code, debug=False)
            elif dataset_name == "leetcode":
                is_correct, test_details = leetcode_check_correctness(tests, model_code)
            elif dataset_name in ["livecodebench", "codeforces", "primeintellect"]:
                # Handle case where tests is a JSON string
                if isinstance(tests, str):
                    try:
                        tests = json.loads(tests)
                    except json.decoder.JSONDecodeError:
                        print("test json invalid: ", tests)
                        return RewardOutput(reward=self.config.format_error_reward, is_correct=False,
                                            metadata={"error": "Tests in task_info is invalid json"})
                if _USE_DIRECT_EXECUTION:
                    is_correct, test_details = lcb_check_correctness_direct(tests, model_code, debug=False)
                else:
                    is_correct, test_details = lcb_check_correctness_v2(tests, model_code, debug=False)
            elif dataset_name == "kodcode":
                is_correct, test_details = kodcode_check_correctness(tests, model_code)
            elif dataset_name == "humanevalplus":
                is_correct, test_details = humanevalplus_check_correctness(tests, model_code)
            else:
                raise NotImplementedError(f"Dataset {dataset_name} not implemented")
        except NotImplementedError:
            raise  # Programming error — don't silently swallow
        except Exception as e:
            # Safety net: catch any exception (including MemoryError) to prevent
            # crashing the ProcessPoolExecutor worker, which would cause BrokenProcessPool.
            logger.warning(
                "RewardCodeFn: exception during correctness check: %s: %s", type(e).__name__, e
            )
            return RewardOutput(reward=self.config.incorrect_reward, is_correct=False,
                                metadata={"error": f"{type(e).__name__}: {e}"})

        # total_time = time.time() - total_start_time
        # print(f"Total reward function execution time: {total_time:.2f} seconds")

        if is_correct:
            return RewardOutput(reward=self.config.correct_reward, is_correct=True, metadata=test_details)
        else:
            return RewardOutput(reward=self.config.incorrect_reward, is_correct=False, metadata=test_details)


def rllm_reward_fn_code(data_source: str, llm_solution: str, ground_truth: dict, **kwargs):
    """Evaluate code solutions against ground truth answers

        This function creates a reward function to evaluate code solutions by pass the test_case from groun_truth. It can optionally use a language model
        for more sophisticated answer validation.

        Args:
            data_source: The source/dataset the problem comes from
            llm_solution: The solution string provided by the language model to evaluate
            ground_truth: some tests for this llm_solution
            enable_llm: Whether to enable language model validation for complex cases (default: False)

        Returns:
            tuple: (bool, dict) where:
                - bool: True if the solution passes all the test_case, False otherwise
                - dict: Detailed test results with test cases and pass/fail status

        Example:
                model_response = '''
    import sys
    from itertools import permutations
    def main():
        n,m=map(int, input().split())
        a=sum(list(map(int, input().split())))
        if a+(n-1)*10<=m:
            print(5)
        else:
            print(5)
    if __name__ == "__main__":
        main()
    '''

        print(f"test the code_forces")
        # tests = [ { "input": "3 30\n2 2 1", "output": "5" }, { "input": "3 10\n3 2 1", "output": "5" } ]
        metadata = {
             "tests": tests,
        }
        True, {"all_passed": True, "test_results": [...]}
    """
    reward_config = RewardConfig()
    reward_fn = RewardCodeFn(reward_config)

    # Convert to new format
    task_info = {"problem": None, "problem_type": RewardType.CODE, "data_source": data_source, "ground_truth": ground_truth}

    reward_response = reward_fn(task_info, llm_solution)
    return reward_response
