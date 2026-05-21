"""Deepcoder Voting (Parallelization) Workflow.

This module provides a concrete implementation of VotingWorkflow
for code generation with parallel generation and aggregation.

The workflow supports two modes:
1. Single-pass (default): Generate N solutions, aggregate, return result
2. Test loop (enable_test_loop=True): If tests fail, regenerate with feedback
"""

import asyncio
import json
import re
from typing import Any, Dict, List

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine.rollout.rollout_engine import RolloutEngine
from rllm.rewards.reward_fn import code_reward_fn
from rllm.rewards.reward_types import RewardOutput
from rllm.workflows.code_test_loop_mixin import CodeTestLoopMixin, TestRoundResult
from rllm.workflows.voting_workflow import VotingWorkflow


class DeepcodeVotingWorkflow(CodeTestLoopMixin, VotingWorkflow):
    """Code-specific voting workflow.

    Supports two modes:
    - Single-pass (enable_test_loop=False, default):
      Generate N solutions in parallel, aggregate, compute reward
    - Test loop (enable_test_loop=True):
      Generate N solutions, aggregate, run tests on selected,
      if fail regenerate with feedback, loop until pass or max rounds

    The aggregator outputs \\boxed{N} where N is the solution number to select.

    Example:
        workflow = DeepcodeVotingWorkflow(
            rollout_engine=engine,
            n_votes=3,
            enable_test_loop=False,  # Single-pass mode (default)
        )
        episode = await workflow.run(task, uid)

        # Or with test loop enabled:
        workflow = DeepcodeVotingWorkflow(
            rollout_engine=engine,
            n_votes=3,
            enable_test_loop=True,
            max_test_rounds=2,
        )
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        prompts: Dict[str, Any] = None,
        prompt_file: str = "examples/deepcoder/prompt.json",
        n_votes: int = 3,
        enable_test_loop: bool = False,
        max_test_rounds: int = 2,
        max_tests_to_show: int = 3,
        public_test_only: bool = False,
        use_final_outcome_reward: bool = False,
        **kwargs,
    ):
        """Initialize the code voting workflow.

        Args:
            rollout_engine: Engine for LLM inference
            prompts: Optional pre-loaded prompt templates
            prompt_file: Path to JSON file with prompt templates
            n_votes: Number of parallel generation attempts
            enable_test_loop: Whether to enable test-based refinement loop
            max_test_rounds: Max test execution rounds (when test loop enabled)
            max_tests_to_show: Max failed tests to include in feedback
            public_test_only: Whether to only show public tests in feedback
            **kwargs: Additional arguments passed to parent workflow
        """
        super().__init__(
            rollout_engine=rollout_engine,
            n_votes=n_votes,
            use_final_outcome_reward=use_final_outcome_reward,
            **kwargs,
        )
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
        return data.get("deepcoder_voting_prompts", {})

    # ===== Required abstract method implementations =====

    def build_generator_prompt(self, task: Dict[str, Any]) -> str:
        """Build code generation prompt."""
        problem = task["question"]
        template = self.prompts.get("generator", {}).get(
            "template",
            "You are an expert competitive programmer. Solve the following "
            "programming problem.\n\nProblem:\n{problem}\n\nRequirements:\n"
            "- Write clean, efficient Python code\n- Handle all edge cases\n"
            "- Output your code in a markdown code block with ```python",
        )
        return template.format(problem=problem)

    def _build_generator_feedback_prompt(
        self,
        problem: str,
        test_feedback: str,
    ) -> str:
        """Build generator prompt with test feedback."""
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

    def build_aggregator_prompt(
        self,
        task: Dict[str, Any],
        responses: list[str],
    ) -> str:
        """Build aggregation prompt with all solutions."""
        problem = task["question"]

        # Build solutions section
        solutions_text = ""
        for i, response in enumerate(responses, 1):
            solutions_text += f"\n--- Solution {i} ---\n{response}\n"

        template = self.prompts.get("aggregator", {}).get(
            "template",
            "You are an expert code reviewer. Your task is to review multiple "
            "code solutions to the same problem and select the best one.\n\n"
            "Problem:\n{problem}\n\n"
            "Solutions to review:{solutions}\n\n"
            "Instructions:\n"
            "1. Analyze each solution for logical correctness\n"
            "2. Check for proper edge case handling\n"
            "3. Look for potential bugs or errors\n"
            "4. Select the solution number most likely to pass all tests\n"
            "5. Output your selection as \\boxed{{N}} where N is the solution "
            "number (1, 2, 3, etc.)\n\n"
            "Your selection:",
        )

        return template.format(problem=problem, solutions=solutions_text)

    def parse_aggregator_response(
        self,
        response: str,
        candidates: list[str],
    ) -> str:
        """Parse aggregator response to get selected solution."""
        # Try to parse \boxed{N} format
        selection_match = re.search(
            r"\\boxed\{(\d+)\}",
            response,
            re.IGNORECASE,
        )

        if selection_match:
            try:
                index = int(selection_match.group(1)) - 1  # Convert to 0-indexed
                if 0 <= index < len(candidates):
                    return candidates[index]
            except (ValueError, IndexError):
                pass

        # Fallback: try to find any digit that could be a selection
        digit_match = re.search(r"\b([1-9])\b", response)
        if digit_match:
            try:
                index = int(digit_match.group(1)) - 1
                if 0 <= index < len(candidates):
                    return candidates[index]
            except (ValueError, IndexError):
                pass

        # Final fallback: return the first candidate
        return candidates[0] if candidates else ""

    async def compute_generator_reward(
        self,
        task: Dict[str, Any],
        response: str,
    ) -> RewardOutput:
        """Compute reward using test execution (async for parallelism)."""
        return await self.run_in_code_executor(code_reward_fn, task, response)

    async def compute_aggregator_reward(
        self,
        task: Dict[str, Any],
        selected_response: str,
    ) -> RewardOutput:
        """Compute reward for aggregator based on selected solution (async for parallelism)."""
        return await self.run_in_code_executor(code_reward_fn, task, selected_response)

    # ===== Workflow execution =====

    async def _generate_single_with_feedback(
        self,
        task: Dict[str, Any],
        vote_index: int,
        test_feedback: str = None,
    ) -> Trajectory:
        """Generate a single response, optionally with test feedback."""
        problem = task["question"]

        if test_feedback:
            prompt = self._build_generator_feedback_prompt(problem, test_feedback)
        else:
            prompt = self.build_generator_prompt(task)

        messages = [{"role": "user", "content": prompt}]

        output = await self.rollout_engine.get_model_response(
            messages,
            agent_name=self.GENERATOR_NAME,
        )

        response = self.extract_response(output)

        trajectory = Trajectory(
            name=self.GENERATOR_NAME,
            steps=[
                Step(
                    chat_completions=messages + [{
                        "role": "assistant",
                        "content": output.content,
                        "reasoning": output.reasoning,
                    }],
                    thought=output.reasoning,
                    action=response,
                    model_output=output,
                )
            ],
        )

        return trajectory

    async def run(self, task: Dict[str, Any], uid: str, **kwargs) -> Episode:
        """Execute the voting workflow.

        If enable_test_loop=False (default):
            - Generate N solutions in parallel
            - Aggregate and select best
            - Compute reward from test execution

        If enable_test_loop=True:
            - Generate N solutions, aggregate, select best
            - Run tests on selected solution
            - If fail, regenerate all with test feedback
            - Loop until pass or max rounds reached

        Args:
            task: Task dictionary containing problem information
            uid: Unique identifier for this episode

        Returns:
            Episode with all generator and aggregator trajectories
        """
        self.reset(task, uid)

        if self.enable_test_loop:
            return await self._run_with_test_loop(task, uid)
        else:
            return await self._run_single_pass(task, uid)

    async def _run_single_pass(
        self,
        task: Dict[str, Any],
        uid: str,
    ) -> Episode:
        """Execute single-pass voting without test loop."""
        # Step 1: Generate N responses in parallel
        generation_tasks = [
            self._generate_single_with_feedback(task, i, None)
            for i in range(self.n_votes)
        ]
        generator_trajectories = await asyncio.gather(*generation_tasks)

        # Extract responses
        responses = [traj.steps[0].action for traj in generator_trajectories]

        # Hook for custom processing after generation
        self.on_generation_complete(list(generator_trajectories))

        # Step 2: Aggregator reviews all responses and selects the best
        agg_prompt = self.build_aggregator_prompt(task, responses)
        agg_messages = [{"role": "user", "content": agg_prompt}]

        agg_output = await self.rollout_engine.get_model_response(
            agg_messages,
            agent_name=self.AGGREGATOR_NAME,
        )

        selected_response = self.parse_aggregator_response(
            agg_output.content,
            responses,
        )

        # Only evaluate the final selected response - skip individual generator rewards
        # since use_final_outcome_reward propagates the final reward to all trajectories
        agg_reward = await self.compute_aggregator_reward(task, selected_response)

        aggregator_trajectory = Trajectory(
            name=self.AGGREGATOR_NAME,
            steps=[
                Step(
                    chat_completions=agg_messages + [{
                        "role": "assistant",
                        "content": agg_output.content,
                        "reasoning": agg_output.reasoning,
                    }],
                    thought=agg_output.reasoning,
                    action=selected_response,
                    model_output=agg_output,
                    reward=agg_reward.reward,
                )
            ],
        )
        aggregator_trajectory.reward = agg_reward.reward

        # Compute metrics
        all_trajectories = list(generator_trajectories) + [aggregator_trajectory]
        final_is_correct = agg_reward.is_correct

        # Propagate the final outcome reward to all trajectories
        final_reward_value = agg_reward.reward
        for trajectory in all_trajectories:
            trajectory.reward = final_reward_value
            for step in trajectory.steps:
                step.reward = final_reward_value

        metrics = {
            f"{self.AGGREGATOR_NAME}_acc": float(final_is_correct),
            "n_votes": self.n_votes,
            "test_rounds": 1,
            "success": int(final_is_correct),
            f"{self.GENERATOR_NAME}_attempts": self.n_votes,
        }

        return Episode(
            id=uid,
            task=task,
            trajectories=all_trajectories,
            is_correct=final_is_correct,
            metrics=metrics,
        )

    async def _run_with_test_loop(
        self,
        task: Dict[str, Any],
        uid: str,
    ) -> Episode:
        """Execute voting workflow with test-based refinement loop."""
        all_trajectories = []
        test_feedback = None
        final_test_result = None

        for test_round in range(self.max_test_rounds):
            # Step 1: Generate N responses in parallel
            generation_tasks = [
                self._generate_single_with_feedback(task, i, test_feedback)
                for i in range(self.n_votes)
            ]
            generator_trajectories = await asyncio.gather(*generation_tasks)

            # Extract responses
            responses = []
            for traj in generator_trajectories:
                response = traj.steps[0].action
                responses.append(response)

            # Add to all trajectories (rewards assigned later)
            all_trajectories.extend(generator_trajectories)

            # Hook for custom processing after generation
            self.on_generation_complete(list(generator_trajectories))

            # Step 2: Aggregator reviews all responses and selects the best
            agg_prompt = self.build_aggregator_prompt(task, responses)
            agg_messages = [{"role": "user", "content": agg_prompt}]

            agg_output = await self.rollout_engine.get_model_response(
                agg_messages,
                agent_name=self.AGGREGATOR_NAME,
            )

            selected_response = self.parse_aggregator_response(
                agg_output.content,
                responses,
            )

            aggregator_trajectory = Trajectory(
                name=self.AGGREGATOR_NAME,
                steps=[
                    Step(
                        chat_completions=agg_messages + [{
                            "role": "assistant",
                            "content": agg_output.content,
                            "reasoning": agg_output.reasoning,
                        }],
                        thought=agg_output.reasoning,
                        action=selected_response,
                        model_output=agg_output,
                    )
                ],
            )
            all_trajectories.append(aggregator_trajectory)

            # Step 3: Run tests on selected solution (async for parallelism)
            test_result = await self.run_tests(task, selected_response)
            final_test_result = test_result

            if test_result.all_passed:
                break

            # Prepare feedback for next round
            test_feedback = test_result.feedback

        # ===== Assign rewards based on final test result =====

        test_passed = final_test_result.all_passed if final_test_result else False
        final_reward = 1.0 if test_passed else 0.0

        # Compute individual generator rewards for metrics (in parallel for speed)
        generator_correct_count = 0
        total_generator_trajs = 0

        # Collect generator trajectories for parallel reward computation
        generator_trajs = [t for t in all_trajectories if t.name == self.GENERATOR_NAME]
        aggregator_trajs = [t for t in all_trajectories if t.name == self.AGGREGATOR_NAME]

        # Compute generator rewards in parallel
        if generator_trajs:
            gen_reward_tasks = [
                self.compute_generator_reward(task, traj.steps[0].action)
                for traj in generator_trajs
            ]
            gen_rewards = await asyncio.gather(*gen_reward_tasks)

            for traj, gen_reward in zip(generator_trajs, gen_rewards):
                traj.steps[0].reward = gen_reward.reward
                traj.reward = gen_reward.reward
                if gen_reward.is_correct:
                    generator_correct_count += 1
                total_generator_trajs += 1

        # Assign aggregator rewards
        for traj in aggregator_trajs:
            traj.steps[0].reward = final_reward
            traj.reward = final_reward

        # If use_final_outcome_reward is enabled, propagate the final reward
        # to all trajectories (all generators + aggregator)
        if self.use_final_outcome_reward:
            for trajectory in all_trajectories:
                trajectory.reward = final_reward
                for step in trajectory.steps:
                    step.reward = final_reward

        # Compute metrics
        n_aggregator_trajs = sum(
            1 for t in all_trajectories if t.name == self.AGGREGATOR_NAME
        )

        metrics = {
            f"{self.GENERATOR_NAME}_acc": (
                generator_correct_count / total_generator_trajs
                if total_generator_trajs > 0 else 0.0
            ),
            f"{self.AGGREGATOR_NAME}_acc": float(test_passed),
            "n_votes": self.n_votes,
            "test_rounds": test_round + 1,
            "any_correct": int(generator_correct_count > 0),
            "success": int(test_passed),
            f"{self.GENERATOR_NAME}_attempts": total_generator_trajs,
            f"{self.AGGREGATOR_NAME}_attempts": n_aggregator_trajs,
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

    def extract_response(self, model_output) -> str:
        """Extract content from model output."""
        return model_output.content or model_output.text or ""
