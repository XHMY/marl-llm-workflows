"""
Cross-problem batch test scheduler for code evaluation.

Instead of each problem spawning all N test subprocesses at once (lcb_check_correctness_v2),
this scheduler uses a streaming pipeline: tests from different problems share a fixed-size
subprocess pool. When a test passes, the next test for that problem is immediately queued.
Failed problems are eliminated without spawning further tests.

This cuts total subprocess forks dramatically (e.g., 640 → 165 at 80% failure rate) and
avoids the long-tail problem where a few slow survivors under-utilize the pool.

Used for the evaluation path (_USE_DIRECT_EXECUTION=False). The training path (direct) is
unchanged.
"""

import asyncio
import collections
import itertools
import json
import logging
import multiprocessing
import select
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from rllm.rewards.code_reward import (
    _temp_run,
    extract_code_from_model,
    postprocess_lcb_sample,
    taco_to_lcb_format,
)
from rllm.rewards.reward_types import RewardConfig, RewardOutput

logger = logging.getLogger(__name__)

# Datasets that use the LCB subprocess-per-test model and can be batched.
_LCB_DATASETS = {"taco", "apps", "code_contests", "livecodebench", "codeforces", "primeintellect"}


@dataclass
class ProblemState:
    """Tracks one problem's test execution across rounds."""

    problem_id: int
    generation: str  # code to test
    timeout: int  # per-test timeout
    debug: bool

    all_inputs: list[str]
    all_outputs: list[str]
    fn_name: str | None
    num_tests: int

    next_test_idx: int = 0
    test_results: list[dict[str, Any] | None] = field(default_factory=list)
    failed: bool = False
    completed: bool = False
    # Highest test index that has been queued or launched (including speculative).
    # Used to avoid double-queuing when speculatively filling the pool.
    _queued_up_to: int = -1

    future: asyncio.Future | None = field(default=None, repr=False)

    def __post_init__(self):
        if not self.test_results:
            self.test_results = [None] * self.num_tests


@dataclass
class _RunningTest:
    """Tracks a single in-flight test subprocess in the pipeline."""

    problem: ProblemState
    test_idx: int
    process: multiprocessing.Process
    pipe: multiprocessing.connection.Connection  # parent end (read)


class BatchTestScheduler:
    """Cross-problem streaming pipeline test scheduler.

    Collects pending code evaluation requests from concurrent workflows,
    then executes tests using a streaming pipeline: a fixed-size subprocess
    pool is continuously filled with tests from different problems. When a
    test passes, the next test for that problem is immediately queued —
    no waiting for other problems to finish their current test.

    Args:
        pool_size: Max concurrent subprocesses (set to CPU count).
        batch_timeout_ms: Max wait time to collect a batch before starting.
        per_test_timeout: Default per-test timeout in seconds.
    """

    def __init__(self, pool_size: int, batch_timeout_ms: float = 50, per_test_timeout: int = 3):
        self._pool_size = pool_size
        self._batch_timeout_ms = batch_timeout_ms
        self._per_test_timeout = per_test_timeout

        self._pending_queue: asyncio.Queue[ProblemState] = asyncio.Queue()
        self._thread_executor = ThreadPoolExecutor(max_workers=1)
        self._id_counter = itertools.count()

        self._scheduler_task: asyncio.Task | None = None
        self._shutdown = False
        self._loop: asyncio.AbstractEventLoop | None = None

        # Fallback executor for non-LCB datasets
        self._fallback_executor = ThreadPoolExecutor(max_workers=pool_size)

    def _ensure_started(self):
        """Lazily start the scheduler loop on first submit."""
        if self._scheduler_task is None or self._scheduler_task.done():
            self._loop = asyncio.get_event_loop()
            self._scheduler_task = self._loop.create_task(self._scheduler_loop())

    async def submit(self, task_info: dict, action: str) -> RewardOutput:
        """Submit a problem for evaluation. Returns when all tests complete.

        This is the main interface called by Workflow.run_in_code_executor().
        Multiple asyncio tasks call this concurrently.
        """
        dataset_name = task_info.get("data_source", "")
        tests = task_info.get("ground_truth", None)

        # Non-LCB datasets: fall back to running code_reward_fn directly
        if dataset_name not in _LCB_DATASETS:
            from rllm.rewards.reward_fn import code_reward_fn
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._fallback_executor, partial(code_reward_fn, task_info, action),
            )

        # Validate tests
        config = RewardConfig()
        if tests is None or (isinstance(tests, str) and not tests.strip()):
            return RewardOutput(reward=config.format_error_reward, is_correct=False,
                                metadata={"error": "No tests found in task_info"})

        # Extract code
        model_code = extract_code_from_model(action)
        if model_code is None:
            return RewardOutput(reward=config.format_error_reward, is_correct=False,
                                metadata={"error": "No code found in model response"})

        # Prepare tests into LCB format
        try:
            if dataset_name in ("taco", "apps", "code_contests"):
                tests = taco_to_lcb_format(tests)
            elif isinstance(tests, str):
                tests = json.loads(tests)

            assert isinstance(tests, list) and len(tests) >= 1
            processed_sample = postprocess_lcb_sample(tests)

            in_outs = json.loads(processed_sample["input_output"])
            all_inputs = in_outs["inputs"]
            all_outputs = in_outs["outputs"]
            fn_name = in_outs.get("fn_name", None)
        except Exception as e:
            logger.warning("BatchTestScheduler: failed to parse tests: %s: %s", type(e).__name__, e)
            return RewardOutput(reward=config.incorrect_reward, is_correct=False,
                                metadata={"error": f"Test parsing failed: {type(e).__name__}: {e}"})

        # Create problem state and enqueue
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        state = ProblemState(
            problem_id=next(self._id_counter),
            generation=model_code,
            timeout=self._per_test_timeout,
            debug=False,
            all_inputs=all_inputs,
            all_outputs=all_outputs,
            fn_name=fn_name,
            num_tests=len(all_inputs),
            future=future,
        )

        self._ensure_started()
        await self._pending_queue.put(state)
        return await future

    async def _scheduler_loop(self):
        """Background task: collect batches and run the pipeline."""
        while not self._shutdown:
            # Wait for at least one submission
            batch: list[ProblemState] = []
            try:
                first = await asyncio.wait_for(self._pending_queue.get(), timeout=1.0)
                batch.append(first)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                return

            # Drain additional pending items with a short collection window
            deadline = time.monotonic() + self._batch_timeout_ms / 1000
            while time.monotonic() < deadline:
                try:
                    item = self._pending_queue.get_nowait()
                    batch.append(item)
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.005)

            # Run the batch
            try:
                await self._run_batch(batch)
            except Exception as e:
                logger.error("BatchTestScheduler: batch execution failed: %s: %s", type(e).__name__, e)
                # Set error on all unresolved futures
                config = RewardConfig()
                for p in batch:
                    if p.future is not None and not p.future.done():
                        p.future.set_result(
                            RewardOutput(reward=config.incorrect_reward, is_correct=False,
                                         metadata={"error": f"Batch execution failed: {e}"})
                        )

    async def _run_batch(self, problems: list[ProblemState]):
        """Execute all tests for a batch using the streaming pipeline."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._thread_executor, self._run_pipeline_sync, problems)

        # Resolve all problems (set futures)
        for p in problems:
            if not p.completed and not p.failed:
                p.failed = True
                p.completed = True
            self._resolve_problem(p)

    def _run_pipeline_sync(self, problems: list[ProblemState]):
        """Streaming pipeline with speculative parallel execution.

        Continuously fills the subprocess pool with tests. When a test passes,
        the next test is immediately queued. When the pool is underutilized
        (fewer running tests than pool_size), speculatively launches additional
        tests for surviving problems — running them in parallel even before
        earlier tests have returned results. If an earlier test then fails,
        all in-flight speculative tests for that problem are cancelled.

        This ensures that long-tail survivors (few problems, many remaining tests)
        can utilize all available CPU cores instead of running tests one-by-one.

        Runs in a ThreadPoolExecutor worker thread.
        """
        ready_queue: collections.deque[tuple[ProblemState, int]] = collections.deque()
        fd_to_running: dict[int, _RunningTest] = {}
        poller = select.poll()

        # Batch-global deadline
        max_tests = max((p.num_tests for p in problems), default=0)
        batch_deadline = time.monotonic() + (self._per_test_timeout + 1) * max_tests + 15

        # Seed: queue test[0] for each problem
        for p in problems:
            if p.num_tests > 0:
                ready_queue.append((p, 0))
                p._queued_up_to = 0
            else:
                p.completed = True

        try:
            while fd_to_running or ready_queue:
                # --- FILL: launch from ready_queue until pool is full ---
                while ready_queue and len(fd_to_running) < self._pool_size:
                    problem, test_idx = ready_queue.popleft()
                    if problem.failed or problem.completed:
                        continue
                    rt = self._spawn_test(problem, test_idx)
                    if rt is None:
                        continue  # spawn failed, problem marked failed
                    try:
                        fd = rt.pipe.fileno()
                    except OSError:
                        self._cleanup_process(rt)
                        continue
                    fd_to_running[fd] = rt
                    poller.register(fd, select.POLLIN)

                # --- SPECULATIVE FILL: when pool is underutilized, speculatively
                # queue more tests for active problems to fill spare slots ---
                spare = self._pool_size - len(fd_to_running) - len(ready_queue)
                if spare > 0:
                    for p in problems:
                        if spare <= 0:
                            break
                        if p.failed or p.completed:
                            continue
                        # Queue tests beyond what's already been queued
                        next_spec = p._queued_up_to + 1
                        while next_spec < p.num_tests and spare > 0:
                            ready_queue.append((p, next_spec))
                            p._queued_up_to = next_spec
                            next_spec += 1
                            spare -= 1

                    # Launch the speculative tests
                    while ready_queue and len(fd_to_running) < self._pool_size:
                        problem, test_idx = ready_queue.popleft()
                        if problem.failed or problem.completed:
                            continue
                        rt = self._spawn_test(problem, test_idx)
                        if rt is None:
                            continue
                        try:
                            fd = rt.pipe.fileno()
                        except OSError:
                            self._cleanup_process(rt)
                            continue
                        fd_to_running[fd] = rt
                        poller.register(fd, select.POLLIN)

                if not fd_to_running:
                    break

                # --- POLL: wait for at least one result ---
                remaining_ms = max(0, (batch_deadline - time.monotonic()) * 1000)
                if remaining_ms <= 0:
                    break

                events = poller.poll(remaining_ms)
                if not events:
                    break  # deadline hit

                # --- HARVEST: process completed subprocesses ---
                # Note: _cancel_problem may remove FDs from fd_to_running that
                # are also in the events list, so check before accessing.
                for fd, _event in events:
                    if fd not in fd_to_running:
                        continue  # already cancelled by _cancel_problem
                    rt = fd_to_running.pop(fd)
                    poller.unregister(fd)

                    # Read result
                    try:
                        result, metadata = rt.pipe.recv()
                    except (EOFError, OSError):
                        result, metadata = None, None

                    # Clean up this subprocess immediately
                    self._cleanup_process(rt)

                    p = rt.problem
                    test_idx = rt.test_idx

                    # Skip results for already-failed/completed problems
                    # (speculative test that completed after an earlier test failed)
                    if p.failed or p.completed:
                        continue

                    if result is None:
                        p.test_results[test_idx] = {
                            "input": p.all_inputs[test_idx],
                            "expected": p.all_outputs[test_idx],
                            "passed": False,
                            "error": "timeout or crash",
                        }
                        p.failed = True
                        p.completed = True
                        self._cancel_problem(p, fd_to_running, poller)
                    else:
                        passed = isinstance(result, list) and len(result) == 1 and result[0] is True
                        p.test_results[test_idx] = {
                            "input": p.all_inputs[test_idx],
                            "expected": p.all_outputs[test_idx],
                            "passed": passed,
                            "error": metadata.get("error", None) if metadata else None,
                            "error_message": metadata.get("error_message", None) if metadata else None,
                            "output": metadata.get("output", None) if metadata else None,
                        }
                        if not passed:
                            p.failed = True
                            p.completed = True
                            self._cancel_problem(p, fd_to_running, poller)
                        else:
                            # With speculative execution, results arrive out of
                            # order. Check completion by counting filled results,
                            # not by sequential next_test_idx.
                            if all(r is not None for r in p.test_results):
                                p.completed = True
                                p.next_test_idx = p.num_tests
                            elif test_idx + 1 > p._queued_up_to:
                                # Next test not yet queued — queue it now
                                ready_queue.append((p, test_idx + 1))
                                p._queued_up_to = test_idx + 1
                            # else: next test already speculatively queued/launched
        finally:
            # Kill all still-running subprocesses and close pipes
            for rt in fd_to_running.values():
                self._cleanup_process(rt)
            fd_to_running.clear()

    def _cancel_problem(
        self,
        problem: ProblemState,
        fd_to_running: dict[int, _RunningTest],
        poller: select.poll,
    ):
        """Cancel all in-flight speculative tests for a failed problem."""
        to_cancel = [fd for fd, rt in fd_to_running.items() if rt.problem is problem]
        for fd in to_cancel:
            rt = fd_to_running.pop(fd)
            poller.unregister(fd)
            self._cleanup_process(rt)

    def _spawn_test(self, problem: ProblemState, test_idx: int) -> _RunningTest | None:
        """Spawn a subprocess for one test case. Returns None on failure."""
        single_dict: dict[str, Any] = {
            "inputs": [problem.all_inputs[test_idx]],
            "outputs": [problem.all_outputs[test_idx]],
        }
        if problem.fn_name is not None:
            single_dict["fn_name"] = problem.fn_name
        single_sample = {"input_output": json.dumps(single_dict)}

        parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
        try:
            proc = multiprocessing.Process(
                target=_temp_run,
                args=(single_sample, problem.generation, problem.debug, child_conn, problem.timeout),
            )
            proc.start()
        except OSError:
            parent_conn.close()
            child_conn.close()
            problem.test_results[test_idx] = {
                "input": problem.all_inputs[test_idx],
                "expected": problem.all_outputs[test_idx],
                "passed": False,
                "error": "OSError: failed to spawn subprocess",
            }
            problem.failed = True
            problem.completed = True
            return None

        child_conn.close()
        return _RunningTest(
            problem=problem,
            test_idx=test_idx,
            process=proc,
            pipe=parent_conn,
        )

    def _cleanup_process(self, rt: _RunningTest):
        """Kill process if alive, join, close pipe and process handle."""
        if rt.process.is_alive():
            rt.process.kill()
            rt.process.join(timeout=5)
        try:
            rt.process.close()
        except ValueError:
            pass
        try:
            rt.pipe.close()
        except OSError:
            pass

    def _resolve_problem(self, problem: ProblemState):
        """Build RewardOutput and set the caller's future."""
        if problem.future is None or problem.future.done():
            return

        is_correct, detailed_results = self._finalize_problem(problem)
        config = RewardConfig()

        if is_correct:
            reward_output = RewardOutput(reward=config.correct_reward, is_correct=True, metadata=detailed_results)
        else:
            reward_output = RewardOutput(reward=config.incorrect_reward, is_correct=False, metadata=detailed_results)

        # Use call_soon_threadsafe since this may be called from a thread executor
        if self._loop is not None:
            self._loop.call_soon_threadsafe(problem.future.set_result, reward_output)
        else:
            problem.future.set_result(reward_output)

    def _finalize_problem(self, problem: ProblemState) -> tuple[bool, dict[str, Any]]:
        """Build (bool, detailed_results) matching lcb_check_correctness_v2 format."""
        test_results = []
        for i in range(problem.num_tests):
            if problem.test_results[i] is not None:
                test_results.append(problem.test_results[i])
            else:
                test_results.append({
                    "input": problem.all_inputs[i],
                    "expected": problem.all_outputs[i],
                    "passed": False,
                    "error": "skipped (prior test failed or not reached)",
                })

        passed_tests = sum(1 for t in test_results if t["passed"])
        all_passed = passed_tests == problem.num_tests

        detailed_results: dict[str, Any] = {
            "all_passed": all_passed,
            "test_results": test_results,
            "total_tests": problem.num_tests,
            "passed_tests": passed_tests,
        }
        return all_passed, detailed_results

    async def shutdown(self):
        """Stop the scheduler loop and clean up."""
        self._shutdown = True
        if self._scheduler_task is not None:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        self._thread_executor.shutdown(wait=False)
        self._fallback_executor.shutdown(wait=False)
