"""Single-agent workflow for DeepCoder code generation.

This module provides a single-agent workflow for code generation,
comparable to train_deepcoder.py but using the workflow infrastructure.

The workflow supports two modes:
1. Single-pass (default): Generator creates code, reward computed from tests
2. Test loop (enable_test_loop=True): If tests fail, regenerate with feedback
"""

import json
from typing import Any, Dict

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine import ModelOutput, RolloutEngine
from rllm.rewards.reward_fn import code_reward_fn
from rllm.workflows.code_test_loop_mixin import CodeTestLoopMixin, TestRoundResult
from rllm.workflows.workflow import Workflow


class SingleAgentDeepcodeWorkflow(CodeTestLoopMixin, Workflow):
    """
    Single-agent workflow for code generation.

    Supports two modes:
    - Single-pass (enable_test_loop=False, default):
      Generate code, compute reward from test execution
    - Test loop (enable_test_loop=True):
      Generate code, run tests, if fail regenerate with feedback,
      loop until tests pass or max_test_rounds reached

    Example:
        workflow = SingleAgentDeepcodeWorkflow(
            rollout_engine=engine,
            enable_test_loop=False,  # Single-pass mode (default)
        )
        episode = await workflow.run(task, uid)

        # Or with test loop enabled:
        workflow = SingleAgentDeepcodeWorkflow(
            rollout_engine=engine,
            enable_test_loop=True,
            max_test_rounds=2,
        )
    """

    GENERATOR_NAME = "generator"

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        prompts: Dict[str, Any] = None,
        prompt_file: str = "examples/deepcoder/prompt.json",
        enable_test_loop: bool = False,
        max_test_rounds: int = 2,
        max_tests_to_show: int = 3,
        public_test_only: bool = False,
        **kwargs,
    ):
        """Initialize the single-agent DeepCoder workflow.

        Args:
            rollout_engine: Engine for LLM inference
            prompts: Optional pre-loaded prompt templates
            prompt_file: Path to JSON file with prompt templates
            enable_test_loop: Whether to enable test-based refinement loop
            max_test_rounds: Max test execution rounds (when test loop enabled)
            max_tests_to_show: Max failed tests to include in feedback
            public_test_only: Whether to only show public tests in feedback
            **kwargs: Additional arguments passed to parent workflow
        """
        super().__init__(rollout_engine, **kwargs)
        self._prompts = prompts
        self._prompt_file = prompt_file

        # CodeTestLoopMixin configuration
        self.enable_test_loop = enable_test_loop
        self.max_test_rounds = max_test_rounds
        self.max_tests_to_show = max_tests_to_show
        self.public_test_only = public_test_only

    @property
    def prompts(self) -> Dict[str, Any]:
        """Lazy load prompts from file if not provided."""
        if self._prompts is None:
            self._prompts = self._load_prompts(self._prompt_file)
        return self._prompts

    def _load_prompts(self, prompt_file: str) -> Dict[str, Any]:
        """Load prompt templates from JSON file."""
        with open(prompt_file, "r") as f:
            data = json.load(f)
        return data.get("deepcoder_single_agent_prompts", {})

    def _build_initial_prompt(self, problem: str) -> str:
        """Build initial code generation prompt."""
        template = self.prompts.get("generator_initial", {}).get(
            "template",
            "You are an expert competitive programmer. Solve the following "
            "programming problem.\n\nProblem:\n{problem}\n\nRequirements:\n"
            "- Write clean, efficient Python code\n- Handle all edge cases\n"
            "- Output your code in a markdown code block with ```python",
        )
        return template.format(problem=problem)

    def _build_feedback_prompt(self, problem: str, test_feedback: str) -> str:
        """Build prompt with test feedback for refinement."""
        template = self.prompts.get("generator_with_test_feedback", {}).get(
            "template",
            "Your previous solution failed some test cases. Here are the results:\n\n"
            "{test_feedback}\n\nPlease analyze these failures carefully and create "
            "a corrected solution.\n\nOriginal Problem:\n{problem}\n\nRequirements:\n"
            "- Fix the issues identified in the test failures\n"
            "- Ensure your solution handles all edge cases\n"
            "- Output your code in a markdown code block with ```python",
        )
        return template.format(problem=problem, test_feedback=test_feedback)

    def _create_trajectory(
        self,
        messages: list,
        model_output: ModelOutput,
        response: str,
        reward: float,
    ) -> Trajectory:
        """Create a trajectory from generation output."""
        return Trajectory(
            name=self.GENERATOR_NAME,
            steps=[
                Step(
                    chat_completions=messages
                    + [
                        {
                            "role": "assistant",
                            "content": model_output.content,
                            "reasoning": model_output.reasoning,
                        }
                    ],
                    thought=model_output.reasoning,
                    action=response,
                    model_output=model_output,
                    reward=reward,
                )
            ],
            reward=reward,
        )

    async def run(self, task: Dict[str, Any], uid: str, **kwargs) -> Episode:
        """
        Execute single-agent workflow.

        If enable_test_loop=False (default):
            - Generate code solution
            - Compute reward from test execution
            - Return single trajectory

        If enable_test_loop=True:
            - Generate initial code solution
            - Run tests
            - If tests fail and rounds remaining, regenerate with feedback
            - Loop until tests pass or max rounds reached

        Args:
            task: Dictionary with 'question' (problem) and 'ground_truth' (tests)
            uid: Unique identifier for this episode

        Returns:
            Episode with trajectories and metrics
        """
        self.reset(task, uid)

        problem = task["question"]

        if self.enable_test_loop:
            # Test loop mode: multiple rounds with test feedback
            return await self._run_with_test_loop(task, uid, problem)
        else:
            # Single-pass mode: generate once and compute reward
            return await self._run_single_pass(task, uid, problem)

    async def _run_single_pass(
        self,
        task: Dict[str, Any],
        uid: str,
        problem: str,
    ) -> Episode:
        """Execute single-pass generation without test loop."""
        prompt_content = self._build_initial_prompt(problem)
        messages = [{"role": "user", "content": prompt_content}]

        # Generate solution
        output: ModelOutput = await self.rollout_engine.get_model_response(
            messages, agent_name=self.GENERATOR_NAME
        )

        code = output.content or output.text or ""

        # Compute reward from test execution (run in thread pool for parallelism)
        reward_output = await self.run_in_code_executor(code_reward_fn, task, code)
        reward = reward_output.reward
        is_correct = reward_output.is_correct

        # Create trajectory
        trajectory = self._create_trajectory(
            messages=messages,
            model_output=output,
            response=code,
            reward=reward,
        )

        # Build metrics
        metadata = reward_output.metadata or {}
        passed_tests = metadata.get("passed_tests", 0)
        total_tests = metadata.get("total_tests", 0)

        metrics = {
            "generator_acc": float(is_correct),
            "success": int(is_correct),
            "test_rounds": 1,
            "generator_attempts": 1,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "pass_rate": (
                passed_tests / total_tests if total_tests > 0 else 0.0
            ),
        }

        # Attach reward metadata (test details) to episode info for debugging
        episode_info = {}
        if metadata and not is_correct:
            test_results = metadata.get("test_results", [])
            if test_results:
                episode_info["reward_test_details"] = test_results[:3]  # first 3 tests

        return Episode(
            id=uid,
            task=task,
            trajectories=[trajectory],
            is_correct=is_correct,
            metrics=metrics,
            info=episode_info,
        )

    async def _run_with_test_loop(
        self,
        task: Dict[str, Any],
        uid: str,
        problem: str,
    ) -> Episode:
        """Execute workflow with test-based refinement loop."""
        all_trajectories = []
        test_feedback = None
        final_test_result = None

        for test_round in range(self.max_test_rounds):
            # Build prompt based on whether we have test feedback
            if test_round == 0:
                prompt_content = self._build_initial_prompt(problem)
            else:
                prompt_content = self._build_feedback_prompt(problem, test_feedback)

            messages = [{"role": "user", "content": prompt_content}]

            # Generate solution
            output: ModelOutput = await self.rollout_engine.get_model_response(
                messages, agent_name=self.GENERATOR_NAME
            )

            code = output.content or output.text or ""

            # Create trajectory (reward will be updated after all rounds)
            trajectory = self._create_trajectory(
                messages=messages,
                model_output=output,
                response=code,
                reward=0.0,  # Placeholder, updated after tests
            )
            all_trajectories.append(trajectory)

            # Run tests (async to enable parallel evaluation)
            test_result = await self.run_tests(task, code)
            final_test_result = test_result

            if test_result.all_passed:
                break

            # Prepare feedback for next round
            test_feedback = test_result.feedback

        # Assign final reward to all trajectories
        test_passed = final_test_result.all_passed if final_test_result else False
        final_reward = 1.0 if test_passed else 0.0

        for traj in all_trajectories:
            traj.reward = final_reward
            for step in traj.steps:
                step.reward = final_reward

        # Build metrics
        metrics = {
            "generator_acc": float(test_passed),
            "success": int(test_passed),
            "test_rounds": test_round + 1,
            "generator_attempts": len(all_trajectories),
        }

        if final_test_result:
            metrics["passed_tests"] = final_test_result.passed_tests
            metrics["total_tests"] = final_test_result.total_tests
            metrics["pass_rate"] = (
                final_test_result.passed_tests / final_test_result.total_tests
                if final_test_result.total_tests > 0
                else 0.0
            )

        return Episode(
            id=uid,
            task=task,
            trajectories=all_trajectories,
            is_correct=test_passed,
            metrics=metrics,
        )
