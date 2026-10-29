"""Deepcoder Voting V2 (Per-Voter Policy) Workflow.

This module provides a concrete implementation of VotingV2Workflow
for code generation where each voter uses a distinct policy
(LoRA adapter in multi-LoRA mode): generate N solutions (each with a different voter),
aggregate, return result. Without the rubric reward the final outcome reward is broadcast
to every trajectory.

Ported from the single-pass path of fork ``examples/deepcoder/deepcoder_voting_v2_workflow.py``;
only the imports, prompt loading (``prompts`` dict or ``load_prompts``, no ``prompt_file``),
the broadcast helper and the removed test loop differ.
"""

import asyncio
import re
from typing import Any

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine.rollout.rollout_engine import RolloutEngine
from rllm.rewards.reward_types import RewardOutput

from marl_workflows.rewards.code_reward import code_reward_fn
from marl_workflows.workflows.base import broadcast_final_reward, load_prompts
from marl_workflows.workflows.code import reject_test_loop
from marl_workflows.workflows.voting_v2 import VotingV2Workflow


class DeepcodeVotingV2Workflow(VotingV2Workflow):
    """Code-specific voting v2 workflow with per-voter policies.

    Each voter (voterA, voterB, voterC, ...) uses a different LoRA adapter
    to solve the same coding problem, then the aggregator selects the best.

    Example:
        workflow = DeepcodeVotingV2Workflow(
            rollout_engine=engine,
            n_votes=3,
        )
        episode = await workflow.run(task, uid)
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        prompts: dict[str, Any] | None = None,
        n_votes: int = 3,
        voter_names: list[str] | None = None,
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
            prompts: Optional pre-loaded prompt templates (default: the packaged
                ``deepcoder_voting_prompts`` section)
            n_votes: Number of parallel voters
            voter_names: Explicit voter names (auto-generated if None)
            enable_test_loop: Must be False (the test loop is not ported)
            max_test_rounds: Kept for config compatibility; unused
            max_tests_to_show: Kept for config compatibility; unused
            public_test_only: Kept for config compatibility; unused
            **kwargs: Additional arguments passed to parent workflow
        """
        reject_test_loop(enable_test_loop)
        super().__init__(
            rollout_engine=rollout_engine,
            n_votes=n_votes,
            voter_names=voter_names,
            use_final_outcome_reward=use_final_outcome_reward,
            use_rubric_reward=use_rubric_reward,
            **kwargs,
        )
        self._prompts = prompts

        self.enable_test_loop = enable_test_loop
        self.max_test_rounds = max_test_rounds
        self.max_tests_to_show = max_tests_to_show
        self.public_test_only = public_test_only

    @property
    def prompts(self) -> dict[str, Any]:
        """Lazy load prompts if not provided."""
        if self._prompts is None:
            self._prompts = load_prompts("deepcoder", "deepcoder_voting_prompts")
        return self._prompts

    # ===== Required abstract method implementations =====

    def build_generator_prompt(self, task: dict[str, Any]) -> str:
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

    def build_aggregator_prompt(
        self,
        task: dict[str, Any],
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
        task: dict[str, Any],
        response: str,
    ) -> RewardOutput:
        """Compute reward using test execution (async for parallelism)."""
        return await self.run_in_code_executor(code_reward_fn, task, response)

    async def compute_aggregator_reward(
        self,
        task: dict[str, Any],
        selected_response: str,
    ) -> RewardOutput:
        """Compute reward for aggregator based on selected solution (async for parallelism)."""
        return await self.run_in_code_executor(code_reward_fn, task, selected_response)

    # ===== Workflow execution =====

    async def _generate_single_with_feedback(
        self,
        task: dict[str, Any],
        vote_index: int,
        test_feedback: str = None,
    ) -> Trajectory:
        """Generate a single response using voter-specific policy (no test feedback: no test loop)."""
        voter_name = self.voter_names[vote_index]

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
                    chat_completions=messages
                    + [
                        {
                            "role": "assistant",
                            "content": output.content,
                            "reasoning": output.reasoning,
                        }
                    ],
                    thought=output.reasoning,
                    action=response,
                    model_output=output,
                )
            ],
        )

        return trajectory

    async def run(self, task: dict[str, Any], uid: str, **kwargs) -> Episode:
        """Execute the voting v2 workflow.

        - Generate N solutions in parallel (each voter uses its own LoRA)
        - Aggregate and select best
        - Compute reward from test execution
        """
        self.reset(task, uid)

        return await self._run_single_pass(task, uid)

    async def _run_single_pass(
        self,
        task: dict[str, Any],
        uid: str,
    ) -> Episode:
        """Execute single-pass voting without test loop."""
        # Step 1: Generate N responses in parallel, each with a different voter policy
        generation_tasks = [self._generate_single_with_feedback(task, i, None) for i in range(self.n_votes)]
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
            gen_reward_tasks = [self.compute_generator_reward(task, resp) for resp in responses]
            voter_reward_results = list(await asyncio.gather(*gen_reward_tasks))
            for i, (traj, rr) in enumerate(zip(voter_trajectories, voter_reward_results, strict=False)):
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
                    chat_completions=agg_messages
                    + [
                        {
                            "role": "assistant",
                            "content": agg_output.content,
                            "reasoning": agg_output.reasoning,
                        }
                    ],
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
                rubric_r = self._compute_rubric_voter_reward(voter_is_correct, _final_is_correct, responses[i])
                traj.reward = rubric_r
                traj.steps[0].reward = rubric_r
                rubric_voter_rewards.append(rubric_r)

            agg_rubric_r = self._compute_rubric_aggregator_reward(any_voter_correct, _final_is_correct)
            aggregator_trajectory.reward = agg_rubric_r
            aggregator_trajectory.steps[0].reward = agg_rubric_r
        else:
            # Propagate the final outcome reward to all trajectories
            broadcast_final_reward(all_trajectories, agg_reward.reward)

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

    def has_parseable_answer(self, response: str) -> bool:
        """Return True if the response contains a Python code block.

        A voter without a code block (e.g., truncated response) is not penalized
        for being wrong — only voters with an explicit wrong code answer are.
        """
        return bool(re.search(r"```python", response, re.IGNORECASE))

    def extract_response(self, model_output) -> str:
        """Extract content from model output."""
        return model_output.content or model_output.text or ""
