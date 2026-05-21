"""Code Test Loop Mixin for coding workflows.

This module provides a mixin class that encapsulates test execution
and feedback loop functionality for code generation workflows.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from rllm.rewards.reward_fn import code_reward_fn


@dataclass
class TestRoundResult:
    """Result from a test execution round."""

    all_passed: bool
    test_results: List[Dict[str, Any]]
    passed_tests: int
    total_tests: int
    code: str
    feedback: str


class CodeTestLoopMixin:
    """Mixin providing test execution and feedback loop functionality for code workflows.

    This mixin provides:
    - `run_tests()`: Execute tests on code and return structured result
    - `_format_test_feedback()`: Format test failures for model feedback

    Configuration attributes (should be set in subclass __init__):
    - `enable_test_loop`: Whether to enable the test loop (default: False)
    - `max_test_rounds`: Maximum number of test rounds (default: 2)
    - `max_tests_to_show`: Maximum failed tests to show in feedback (default: 3)
    - `public_test_only`: Only show tests visible in problem statement (default: False)

    Example usage:
        class MyCodeWorkflow(CodeTestLoopMixin, Workflow):
            def __init__(self, ..., enable_test_loop=False, max_test_rounds=2, ...):
                super().__init__(...)
                self.enable_test_loop = enable_test_loop
                self.max_test_rounds = max_test_rounds
                self.max_tests_to_show = 3
                self.public_test_only = False

            async def run(self, task, uid, **kwargs):
                if self.enable_test_loop:
                    for test_round in range(self.max_test_rounds):
                        code = generate(...)
                        test_result = self.run_tests(task, code)
                        if test_result.all_passed:
                            break
                        feedback = test_result.feedback
                else:
                    code = generate(...)
                    reward = code_reward_fn(task, code)
    """

    # Default configuration values (should be overridden in subclass __init__)
    enable_test_loop: bool = False
    max_test_rounds: int = 2
    max_tests_to_show: int = 3
    public_test_only: bool = False

    async def run_tests(self, task: Dict[str, Any], code: str) -> TestRoundResult:
        """Execute tests on code and return structured result.

        Runs code evaluation in a thread pool to avoid blocking the event loop,
        enabling parallel code evaluation across multiple workflows.

        Args:
            task: Task dictionary containing test cases
            code: Code string to test

        Returns:
            TestRoundResult with test execution details and feedback
        """
        # Run blocking code evaluation in thread pool to enable parallelism
        reward_output = await self.run_in_code_executor(code_reward_fn, task, code)

        metadata = reward_output.metadata or {}
        test_results = metadata.get("test_results", [])
        passed_tests = metadata.get("passed_tests", 0)
        total_tests = metadata.get("total_tests", len(test_results))
        all_passed = reward_output.is_correct

        # Format feedback for failed tests
        feedback = self._format_test_feedback(
            test_results,
            task.get("question", ""),
        )

        return TestRoundResult(
            all_passed=all_passed,
            test_results=test_results,
            passed_tests=passed_tests,
            total_tests=total_tests,
            code=code,
            feedback=feedback,
        )

    def _format_test_feedback(
        self,
        test_results: List[Dict[str, Any]],
        question: str,
    ) -> str:
        """Format test results into feedback string for model refinement.

        Based on CompetitionCodingAgent.format_test_results() pattern.

        Args:
            test_results: List of test result dictionaries
            question: Original problem statement (for public test filtering)

        Returns:
            Formatted string describing test failures
        """
        if not test_results:
            return "No test cases were executed."

        def truncate(s, length=300):
            s = str(s) if not isinstance(s, str) else s
            if len(s) <= length:
                return s
            return s[: length // 2] + "...(truncated)..." + s[-length // 2 :]

        # Filter to public tests only if configured
        if self.public_test_only:
            normalized_question = "".join(question.split())
            public_tests = []
            for test in test_results:
                if not isinstance(test, dict) or "input" not in test:
                    continue
                test_input = test["input"]
                if isinstance(test_input, list):
                    strings_to_match = [
                        "".join(str(s).split()) for s in test_input
                    ]
                elif isinstance(test_input, str):
                    strings_to_match = [
                        "".join(s.split()) for s in test_input.split("\n")
                    ]
                else:
                    strings_to_match = []
                if all(s in normalized_question for s in strings_to_match):
                    public_tests.append(test)

            if not public_tests:
                public_tests = test_results[:2]  # Fallback
            test_results = public_tests

        formatted = ""
        n_failed = 0
        for i, test in enumerate(test_results):
            if not test.get("passed", True):
                formatted += f"### Test {i + 1} FAILED\n"
                formatted += f"  Input: {truncate(test.get('input', 'N/A'))}\n"
                formatted += f"  Expected: {truncate(test.get('expected', 'N/A'))}\n"
                if test.get("output") is not None:
                    formatted += f"  Actual: {truncate(test['output'])}\n"
                if test.get("error_message"):
                    formatted += f"  Error: {truncate(test['error_message'])}\n"
                formatted += "\n"
                n_failed += 1
                if n_failed >= self.max_tests_to_show:
                    break

        if n_failed == 0:
            return "All tests passed."

        return formatted
