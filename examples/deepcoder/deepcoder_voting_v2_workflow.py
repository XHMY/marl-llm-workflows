"""Deepcoder Voting V2 (Per-Voter Policy) Workflow.

This module provides a concrete implementation of VotingV2Workflow
for code generation where each voter uses a distinct policy
(LoRA adapter in multi-LoRA mode).

The workflow supports two modes:
1. Single-pass (default): Generate N solutions (each with different voter), aggregate, return result
2. Test loop (enable_test_loop=True): If tests fail, regenerate with feedback
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine.rollout.rollout_engine import RolloutEngine
from rllm.rewards.reward_fn import code_reward_fn
from rllm.rewards.reward_types import RewardOutput
from rllm.workflows.code_test_loop_mixin import CodeTestLoopMixin, TestRoundResult
from rllm.workflows.voting_v2_workflow import VotingV2Workflow


class DeepcodeVotingV2Workflow(CodeTestLoopMixin, VotingV2Workflow):
    """Code-specific voting v2 workflow with per-voter policies.

    Each voter (voterA, voterB, voterC, ...) uses a different LoRA adapter
    to solve the same coding problem, then the aggregator selects the best.

    Supports two modes:
    - Single-pass (enable_test_loop=False, default):
      Generate N solutions in parallel (each voter uses its own LoRA), aggregate, compute reward
    - Test loop (enable_test_loop=True):
      Generate N solutions, aggregate, run tests on selected,
      if fail regenerate with feedback, loop until pass or max rounds

    Example:
        workflow = DeepcodeVotingV2Workflow(
            rollout_engine=engine,
            n_votes=3,
            enable_test_loop=False,
        )
        episode = await workflow.run(task, uid)
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        prompts: Dict[str, Any] = None,
        prompt_file: str = "examples/deepcoder/prompt.json",
        n_votes: int = 3,
        voter_names: Optional[List[str]] = None,
        enable_test_loop: bool = False,
        max_test_rounds: int = 2,
        max_tests_to_show: int = 3,
        public_test_only: bool = False,
        use_final_outcome_reward: bool = False,
        use_rubric_reward: bool = False,
        **kwargs,
    ):
        """Initialize the code voting v2 workflow.

        Args:
            rollout_engine: Engine for LLM inference
            prompts: Optional pre-loaded prompt templates
            prompt_file: Path to JSON file with prompt templates
            n_votes: Number of parallel voters
            voter_names: Explicit voter names (auto-generated if None)
            enable_test_loop: Whether to enable test-based refinement loop
            max_test_rounds: Max test execution rounds (when test loop enabled)
            max_tests_to_show: Max failed tests to include in feedback
            public_test_only: Whether to only show public tests in feedback
            **kwargs: Additional arguments passed to parent workflow
        """
        super().__init__(
            rollout_engine=rollout_engine,
            n_votes=n_votes,
            voter_names=voter_names,
            use_final_outcome_reward=use_final_outcome_reward,
            use_rubric_reward=use_rubric_reward,
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
        selection_match = re.search(
            r"\\boxed\{(\d+)\}",
            response,
            re.IGNORECASE,
        )

        if selection_match:
            try:
                index = int(selection_match.group(1)) - 1
                if 0 <= index < len(candidates):
                    return candidates[index]
            except (ValueError, IndexError):
                pass

        digit_match = re.search(r"\b([1-9])\b", response)
        if digit_match:
            try:
                index = int(digit_match.group(1)) - 1
                if 0 <= index < len(candidates):
                    return candidates[index]
            except (ValueError, IndexError):
                pass

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
        """Generate a single response using voter-specific policy, optionally with test feedback."""
        voter_name = self.voter_names[vote_index]
        problem = task["question"]

        if test_feedback:
            prompt = self._build_generator_feedback_prompt(problem, test_feedback)
        else:
            prompt = self.build_generator_prompt(task)

        messages = [{"role": "user", "content": prompt}]

        output = await self.rollout_engine.get_model_response(
            messages,
            agent_name=voter_name,
        )

        response = self.extract_response(output)

        trajectory = Trajectory(
            name=voter_name,
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
        """Execute the voting v2 workflow.

        If enable_test_loop=False (default):
            - Generate N solutions in parallel (each voter uses its own LoRA)
            - Aggregate and select best
            - Compute reward from test execution

        If enable_test_loop=True:
            - Generate N solutions, aggregate, select best
            - Run tests on selected solution
            - If fail, regenerate all with test feedback
            - Loop until pass or max rounds reached
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
        # Step 1: Generate N responses in parallel, each with a different voter policy
        generation_tasks = [
            self._generate_single_with_feedback(task, i, None)
            for i in range(self.n_votes)
        ]
        voter_trajectories = await asyncio.gather(*generation_tasks)

        # Extract responses
        responses = [traj.steps[0].action for traj in voter_trajectories]

        # Hook for custom processing after generation
        self.on_generation_complete(list(voter_trajectories))

        # If rubric reward is enabled, run tests on each voter's response in parallel
        # to determine per-voter correctness before the aggregator selects.
        voter_reward_results = []
        voter_correct_count = 0
        voter_parseable_count = 0
        per_voter_acc = {}

        if self.use_rubric_reward:
            gen_reward_tasks = [
                self.compute_generator_reward(task, resp)
                for resp in responses
            ]
            voter_reward_results = list(await asyncio.gather(*gen_reward_tasks))
            for i, (traj, rr) in enumerate(zip(voter_trajectories, voter_reward_results)):
                traj.steps[0].reward = rr.reward
                traj.reward = rr.reward
                if rr.is_correct:
                    voter_correct_count += 1
                if self.has_parseable_answer(responses[i]):
                    voter_parseable_count += 1
                per_voter_acc[f"{self.voter_names[i]}_acc"] = float(rr.is_correct)

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

        # Evaluate the final selected response
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

        all_trajectories = list(voter_trajectories) + [aggregator_trajectory]
        final_is_correct = agg_reward.is_correct

        if self.use_rubric_reward:
            _final_is_correct = bool(final_is_correct)
            any_voter_correct = voter_correct_count > 0
            rubric_voter_rewards = []

            for i, traj in enumerate(voter_trajectories):
                voter_is_correct = bool(voter_reward_results[i].is_correct)
                rubric_r = self._compute_rubric_voter_reward(
                    voter_is_correct, _final_is_correct, responses[i]
                )
                traj.reward = rubric_r
                traj.steps[0].reward = rubric_r
                rubric_voter_rewards.append(rubric_r)

            agg_rubric_r = self._compute_rubric_aggregator_reward(any_voter_correct, _final_is_correct)
            aggregator_trajectory.reward = agg_rubric_r
            aggregator_trajectory.steps[0].reward = agg_rubric_r
        else:
            # Propagate the final outcome reward to all trajectories
            final_reward_value = agg_reward.reward
            for trajectory in all_trajectories:
                trajectory.reward = final_reward_value
                for step in trajectory.steps:
                    step.reward = final_reward_value

        any_voter_correct = voter_correct_count > 0
        metrics = {
            f"{self.AGGREGATOR_NAME}_acc": float(final_is_correct),
            "n_votes": self.n_votes,
            "test_rounds": 1,
            "success": int(final_is_correct),
            "voter_attempts": self.n_votes,
            "voter_correct_count": voter_correct_count,
            "voter_parseable_count": voter_parseable_count,
            **per_voter_acc,
        }

        if self.use_rubric_reward:
            metrics["voter_acc"] = voter_correct_count / self.n_votes
            metrics["any_correct"] = int(any_voter_correct)
            metrics["voter_rubric_reward_mean"] = sum(rubric_voter_rewards) / len(rubric_voter_rewards)
            metrics["voter_rubric_reward_min"] = min(rubric_voter_rewards)
            metrics["voter_rubric_reward_max"] = max(rubric_voter_rewards)
            metrics["agg_rubric_reward"] = agg_rubric_r

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
        voter_name_set = set(self.voter_names)

        for test_round in range(self.max_test_rounds):
            # Step 1: Generate N responses in parallel, each with a different voter policy
            generation_tasks = [
                self._generate_single_with_feedback(task, i, test_feedback)
                for i in range(self.n_votes)
            ]
            voter_trajectories = await asyncio.gather(*generation_tasks)

            # Extract responses
            responses = []
            for traj in voter_trajectories:
                response = traj.steps[0].action
                responses.append(response)

            # Add to all trajectories (rewards assigned later)
            all_trajectories.extend(voter_trajectories)

            # Hook for custom processing after generation
            self.on_generation_complete(list(voter_trajectories))

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

        # Compute individual voter rewards for metrics (in parallel for speed)
        voter_correct_count = 0
        total_voter_trajs = 0
        per_voter_correct = {name: [] for name in self.voter_names}

        # Collect voter trajectories for parallel reward computation
        voter_trajs = [t for t in all_trajectories if t.name in voter_name_set]
        aggregator_trajs = [t for t in all_trajectories if t.name == self.AGGREGATOR_NAME]

        # Compute voter rewards in parallel (needed for rubric and per-voter metrics)
        gen_rewards = []
        voter_parseable_count = 0
        if voter_trajs:
            gen_reward_tasks = [
                self.compute_generator_reward(task, traj.steps[0].action)
                for traj in voter_trajs
            ]
            gen_rewards = list(await asyncio.gather(*gen_reward_tasks))

            for traj, gen_reward in zip(voter_trajs, gen_rewards):
                traj.steps[0].reward = gen_reward.reward
                traj.reward = gen_reward.reward
                if gen_reward.is_correct:
                    voter_correct_count += 1
                if self.has_parseable_answer(traj.steps[0].action):
                    voter_parseable_count += 1
                per_voter_correct[traj.name].append(gen_reward.is_correct)
                total_voter_trajs += 1

        any_voter_correct = voter_correct_count > 0

        if self.use_rubric_reward:
            # Apply per-voter rubric rewards (overrides the raw rewards assigned above)
            rubric_voter_rewards = []
            for traj, gen_reward in zip(voter_trajs, gen_rewards):
                voter_is_correct = bool(gen_reward.is_correct)
                rubric_r = self._compute_rubric_voter_reward(
                    voter_is_correct, test_passed, traj.steps[0].action
                )
                traj.reward = rubric_r
                traj.steps[0].reward = rubric_r
                rubric_voter_rewards.append(rubric_r)

            # Apply aggregator rubric reward
            agg_rubric_r = self._compute_rubric_aggregator_reward(any_voter_correct, test_passed)
            for traj in aggregator_trajs:
                traj.reward = agg_rubric_r
                traj.steps[0].reward = agg_rubric_r
        else:
            # Assign aggregator rewards
            for traj in aggregator_trajs:
                traj.steps[0].reward = final_reward
                traj.reward = final_reward

            # If use_final_outcome_reward is enabled, propagate the final reward
            # to all trajectories (all voters + aggregator)
            if self.use_final_outcome_reward:
                for trajectory in all_trajectories:
                    trajectory.reward = final_reward
                    for step in trajectory.steps:
                        step.reward = final_reward

        # Compute metrics
        n_aggregator_trajs = len(aggregator_trajs)

        metrics = {
            "voter_acc": (
                voter_correct_count / total_voter_trajs
                if total_voter_trajs > 0 else 0.0
            ),
            f"{self.AGGREGATOR_NAME}_acc": float(test_passed),
            "n_votes": self.n_votes,
            "test_rounds": test_round + 1,
            "any_correct": int(voter_correct_count > 0),
            "success": int(test_passed),
            "voter_attempts": total_voter_trajs,
            "voter_correct_count": voter_correct_count,
            "voter_parseable_count": voter_parseable_count,
            f"{self.AGGREGATOR_NAME}_attempts": n_aggregator_trajs,
        }

        # Add per-voter accuracy
        for name in self.voter_names:
            if per_voter_correct[name]:
                metrics[f"{name}_acc"] = sum(per_voter_correct[name]) / len(per_voter_correct[name])
            else:
                metrics[f"{name}_acc"] = 0.0

        if self.use_rubric_reward:
            metrics["voter_rubric_reward_mean"] = (
                sum(rubric_voter_rewards) / len(rubric_voter_rewards)
                if rubric_voter_rewards else 0.0
            )
            metrics["voter_rubric_reward_min"] = min(rubric_voter_rewards) if rubric_voter_rewards else 0.0
            metrics["voter_rubric_reward_max"] = max(rubric_voter_rewards) if rubric_voter_rewards else 0.0
            metrics["agg_rubric_reward"] = agg_rubric_r

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

    def has_parseable_answer(self, response: str) -> bool:
        """Return True if the response contains a Python code block.

        A voter without a code block (e.g., truncated response) is not penalized
        for being wrong — only voters with an explicit wrong code answer are.
        """
        return bool(re.search(r"```python", response, re.IGNORECASE))

    def extract_response(self, model_output) -> str:
        """Extract content from model output."""
        return model_output.content or model_output.text or ""
