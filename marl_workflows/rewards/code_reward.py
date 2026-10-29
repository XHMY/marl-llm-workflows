"""Code reward: unit-test execution of generated programs in a fresh-interpreter sandbox.

COPIED FROM fork ``rllm/rewards/code_reward.py`` @f0240732 (research fork of rLLM; bounded
copy, only the parts the paper's Code runs use), with the sandbox replaced. It replaces these
regions of upstream rllm@186d0bbb ``rllm/rewards/code_reward.py``:

* ``_temp_run`` (:202-206) and ``lcb_check_correctness_v2`` (:208-250): every problem is graded
  by ``code_sandbox.grade_sample`` (one fresh ``python -I`` process per sample, all tests in
  sequence, kill of the whole session at the global timeout),
* ``RewardCodeFn`` (:397-470), which uses that path for every data source and turns execution
  errors into an incorrect verdict instead of crashing a worker.

Kept verbatim from the fork (line numbers in the fork file): ``extract_code_from_model``
(:89-103), ``taco_to_lcb_format`` (:625-656) and ``RewardCodeFn`` (:690-783, restricted to the
stdin/functional data sources, single execution path). ``postprocess_lcb_sample`` (:227-246) and
the result assembly of ``lcb_check_correctness_direct`` (:249-329) live in ``code_sandbox``.

Sandbox change against the paper's runs: the fork ran the harness in a child forked from a
pool worker that had imported rllm and torch, so its 4 GiB ``RLIMIT_AS`` was already almost
used up by the inherited address space and solutions failed on allocations of tens of MB.
``code_sandbox`` starts a fresh interpreter instead (docs/configuration.md, "Code reward
sandbox"). Test semantics (3 s per test, global cap ``min((3 + 1) * n_tests + 5, 120)`` s,
``RLIMIT_CPU`` with the same formula, stop at the first failure) are unchanged.

``CodeRewardExecutor`` is the executor the trainer hands to the code workflows: the reward
function runs on threads of the calling process and each grading goes to a spawn process pool
whose workers import only ``code_sandbox`` (no rllm, no torch).

Dropped (not used by any paper run): leetcode, kodcode, humanevalplus, the Together code
interpreter, ``check_correctness`` and ``primeintellect_check_correctness``. Unsupported data
sources raise ``NotImplementedError``.
"""

import json
import logging
import multiprocessing
import re
import threading
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any

from rllm.agents.agent import Action
from rllm.rewards.reward_types import RewardConfig, RewardOutput

from marl_workflows.rewards.code_sandbox import grade_sample, postprocess_lcb_sample

__all__ = [
    "CodeRewardExecutor",
    "RewardCodeFn",
    "SUPPORTED_DATA_SOURCES",
    "code_reward_fn",
    "extract_code_from_model",
    "lcb_check_correctness_direct",
    "postprocess_lcb_sample",
    "taco_to_lcb_format",
]

logger = logging.getLogger(__name__)

_TACO_SOURCES = ("taco", "apps", "code_contests")
_LCB_SOURCES = ("livecodebench", "codeforces", "primeintellect")
SUPPORTED_DATA_SOURCES = _TACO_SOURCES + _LCB_SOURCES

# The grading pool of the CodeRewardExecutor whose thread is running the reward function.
_executor_context = threading.local()


class CodeRewardExecutor(Executor):
    """Executor for ``code_reward_fn`` calls whose gradings run in an rllm-free process pool.

    ``submit`` runs the function on a thread of this process; inside it,
    ``lcb_check_correctness_direct`` sends ``code_sandbox.grade_sample`` to a spawn
    ``ProcessPoolExecutor``. Pool workers import only ``code_sandbox`` and each grading starts its
    own fresh interpreter, so a worker costs tens of MB instead of a full rllm/torch import.
    """

    def __init__(self, max_workers: int):
        self._pool = ProcessPoolExecutor(max_workers=max_workers, mp_context=multiprocessing.get_context("spawn"))
        self._threads = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="code-reward")

    def submit(self, fn, /, *args, **kwargs) -> Future:
        return self._threads.submit(self._call, fn, args, kwargs)

    def _call(self, fn, args, kwargs):
        _executor_context.pool = self._pool
        try:
            return fn(*args, **kwargs)
        finally:
            _executor_context.pool = None

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self._threads.shutdown(wait=wait, cancel_futures=cancel_futures)
        self._pool.shutdown(wait=wait, cancel_futures=cancel_futures)


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


def lcb_check_correctness_direct(sample, generation, timeout=3, debug=False, max_global_timeout=120):
    """Grade one solution in the sandbox: in the executor's pool when called from a
    ``CodeRewardExecutor`` thread, else from this process. Returns ``(all_passed, details)``."""
    pool = getattr(_executor_context, "pool", None)
    if pool is None:
        return grade_sample(sample, generation, timeout, debug, max_global_timeout)
    return pool.submit(grade_sample, sample, generation, timeout, debug, max_global_timeout).result()


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
                is_correct, test_details = lcb_check_correctness_direct(tests, model_code, debug=False)
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
                is_correct, test_details = lcb_check_correctness_direct(tests, model_code, debug=False)
            else:
                # >>> MARL(code-reward-scope): leetcode / kodcode / humanevalplus are not copied.
                raise NotImplementedError(f"Dataset {dataset_name} not implemented")
                # <<< MARL(code-reward-scope)
        except NotImplementedError:
            raise  # Programming error — don't silently swallow
        except Exception as e:
            # Safety net: catch any exception (including MemoryError) so that a grading failure
            # becomes an incorrect verdict instead of an error in the calling workflow.
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
