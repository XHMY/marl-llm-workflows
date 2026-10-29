"""Deepcoder Voting (Parallelization) Workflow.

This module provides a concrete implementation of VotingWorkflow
for code generation with parallel generation and aggregation: generate N solutions,
aggregate, return result.

As in Math Voting, voters are named ``generator0/1/2`` (one GRPO group per voter). No
``application_id`` is sent, only the selected solution is tested, and the final outcome reward is
broadcast to every trajectory unconditionally. The paper's Code runs pooled the voters into one
``generator`` group, which under the broadcast reward differs only by a constant advantage scale.

Ported from the single-pass path of fork ``examples/deepcoder/deepcoder_voting_workflow.py``
(:1-357); only the imports, prompt loading (``prompts`` dict or ``load_prompts``, no
``prompt_file``), the broadcast helper and the removed test loop differ. The fork's
``_generate_single_with_feedback`` (always called without feedback) equals the base class's
``_generate_single`` without ``uid`` except for the trajectory name, so the base method is used.
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
from marl_workflows.workflows.voting import VotingWorkflow


class DeepcodeVotingWorkflow(VotingWorkflow):
    """Code-specific voting workflow.

    Generate N solutions in parallel, aggregate, compute reward.

    The aggregator outputs \\boxed{N} where N is the solution number to select.

    Example:
        workflow = DeepcodeVotingWorkflow(
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
            prompts: Optional pre-loaded prompt templates (default: the packaged
                ``deepcoder_voting_prompts`` section)
            n_votes: Number of parallel generation attempts
            enable_test_loop: Must be False (the test loop is not ported)
            max_test_rounds: Kept for config compatibility; unused
            max_tests_to_show: Kept for config compatibility; unused
            public_test_only: Kept for config compatibility; unused
            use_final_outcome_reward: Stored for compatibility; the single pass broadcasts
                the final outcome reward regardless
            **kwargs: Additional arguments passed to parent workflow
        """
        reject_test_loop(enable_test_loop)
        super().__init__(
            rollout_engine=rollout_engine,
            n_votes=n_votes,
            use_final_outcome_reward=use_final_outcome_reward,
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

    async def run(self, task: dict[str, Any], uid: str, **kwargs) -> Episode:
        """Execute the voting workflow.

        - Generate N solutions in parallel
        - Aggregate and select best
        - Compute reward from test execution

        Args:
            task: Task dictionary containing problem information
            uid: Unique identifier for this episode

        Returns:
            Episode with all generator and aggregator trajectories
        """
        self.reset(task, uid)

        return await self._run_single_pass(task, uid)

    async def _run_single_pass(
        self,
        task: dict[str, Any],
        uid: str,
    ) -> Episode:
        """Execute single-pass voting without test loop."""
        # Step 1: Generate N responses in parallel
        # No uid: the fork sent no application_id for Code voters.
        generation_tasks = [self._generate_single(task, i) for i in range(self.n_votes)]
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

        # Compute metrics
        all_trajectories = list(generator_trajectories) + [aggregator_trajectory]
        final_is_correct = agg_reward.is_correct

        # Propagate the final outcome reward to all trajectories (unconditionally, as in the fork)
        broadcast_final_reward(all_trajectories, agg_reward.reward)

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

    def extract_response(self, model_output) -> str:
        """Extract content from model output."""
        return model_output.content or model_output.text or ""
