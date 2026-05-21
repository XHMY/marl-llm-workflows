"""Voting (Parallelization) Workflow.

This module provides an abstract base class for implementing voting/parallelization
workflows. The pattern involves:
1. Generator creates N responses in parallel
2. Aggregator reviews all responses and selects/synthesizes the best answer
"""

import asyncio
from abc import abstractmethod
from typing import Any, Dict

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine.rollout.rollout_engine import RolloutEngine
from rllm.rewards.reward_types import RewardOutput
from rllm.workflows.workflow import Workflow


class VotingWorkflow(Workflow):
    """Abstract base class for voting/parallelization workflows.

    This workflow pattern uses two agents:
    - Generator: Creates N responses in parallel for diverse perspectives
    - Aggregator: Reviews all responses and selects/synthesizes the best answer

    The workflow:
    1. Generator is called N times in parallel using asyncio.gather()
    2. Aggregator reviews all generated responses
    3. Aggregator selects the best response or synthesizes a final answer

    Subclasses must implement:
    - build_generator_prompt(): Create the generation prompt
    - build_aggregator_prompt(): Create the aggregation prompt with all responses
    - parse_aggregator_response(): Parse aggregator output to get final answer
    - compute_generator_reward(): Calculate reward for each generator trajectory
    - compute_aggregator_reward(): Calculate reward for aggregator trajectory

    Optional overrides:
    - extract_response(): Extract relevant content from model output
    - on_generation_complete(): Hook called after all parallel generations complete

    Example:
        class MathVotingWorkflow(VotingWorkflow):
            def build_generator_prompt(self, task):
                return f"Solve: {task['question']}"
            # ... implement other abstract methods ...
    """

    # Agent names (no underscores per CLAUDE.md conventions)
    GENERATOR_NAME = "generator"
    AGGREGATOR_NAME = "aggregator"

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        n_votes: int = 3,
        use_final_outcome_reward: bool = False,
        **kwargs,
    ):
        """Initialize the VotingWorkflow.

        Args:
            rollout_engine: Engine for LLM inference
            n_votes: Number of parallel generation attempts
            use_final_outcome_reward: If True, assign the final outcome reward to
                ALL trajectories in the episode.
            **kwargs: Additional arguments passed to parent Workflow
        """
        super().__init__(rollout_engine, **kwargs)
        self.n_votes = n_votes
        self.use_final_outcome_reward = use_final_outcome_reward

    # ===== Abstract methods that subclasses MUST implement =====

    @abstractmethod
    def build_generator_prompt(self, task: Dict[str, Any]) -> str:
        """Build the prompt for generation.

        Args:
            task: Task dictionary containing problem information

        Returns:
            Formatted prompt string for the generator
        """
        pass

    @abstractmethod
    def build_aggregator_prompt(
        self,
        task: Dict[str, Any],
        responses: list[str],
    ) -> str:
        """Build the aggregation prompt with all generated responses.

        Args:
            task: Task dictionary
            responses: List of all generated responses to review

        Returns:
            Formatted prompt string for the aggregator
        """
        pass

    @abstractmethod
    def parse_aggregator_response(
        self,
        response: str,
        candidates: list[str],
    ) -> str:
        """Parse aggregator output to get the final selected/synthesized response.

        Args:
            response: Raw text response from aggregator
            candidates: List of candidate responses that were reviewed

        Returns:
            The final selected or synthesized response
        """
        pass

    @abstractmethod
    def compute_generator_reward(
        self,
        task: Dict[str, Any],
        response: str,
    ) -> RewardOutput:
        """Compute reward for a generator trajectory.

        Args:
            task: Task dictionary (may contain ground truth)
            response: Generator's response

        Returns:
            RewardOutput with reward value and metadata
        """
        pass

    @abstractmethod
    def compute_aggregator_reward(
        self,
        task: Dict[str, Any],
        selected_response: str,
    ) -> RewardOutput:
        """Compute reward for aggregator trajectory.

        Args:
            task: Task dictionary
            selected_response: The final selected/synthesized response

        Returns:
            RewardOutput with reward based on final answer correctness
        """
        pass

    # ===== Optional hooks for customization =====

    def extract_response(self, model_output) -> str:
        """Extract the relevant response content from model output.

        Default: returns model_output.content

        Override for custom extraction (e.g., parsing specific format).

        Args:
            model_output: ModelOutput from rollout engine

        Returns:
            Extracted response string
        """
        return model_output.content or model_output.text or ""

    def on_generation_complete(self, trajectories: list[Trajectory]):
        """Hook called after all parallel generations complete.

        Override to add custom logging, metrics, or processing.

        Args:
            trajectories: All generator trajectories
        """
        pass

    # ===== Core workflow implementation =====

    async def _generate_single(
        self,
        task: Dict[str, Any],
        vote_index: int,
        uid: str = None,
    ) -> Trajectory:
        """Generate a single response.

        Args:
            task: Task dictionary
            vote_index: Index of this vote (for tracking)
            uid: Optional unique identifier for sticky session routing

        Returns:
            Trajectory containing the generation
        """
        prompt = self.build_generator_prompt(task)
        messages = [{"role": "user", "content": prompt}]

        kwargs = {"agent_name": self.GENERATOR_NAME}
        if uid is not None:
            kwargs["application_id"] = uid
        output = await self.rollout_engine.get_model_response(
            messages,
            **kwargs,
        )

        response = self.extract_response(output)

        trajectory = Trajectory(
            name=f"{self.GENERATOR_NAME}{vote_index}",  # Unique name per vote for separate GRPO groups
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

        Args:
            task: Task dictionary containing problem information
            uid: Unique identifier for this episode

        Returns:
            Episode with all generator and aggregator trajectories
        """
        self.reset(task, uid)

        # Step 1: Generate N responses in parallel
        generation_tasks = [
            self._generate_single(task, i, uid=uid)
            for i in range(self.n_votes)
        ]
        generator_trajectories = await asyncio.gather(*generation_tasks)

        # Extract responses and compute rewards for each generator
        responses = []
        generator_correct_count = 0

        for traj in generator_trajectories:
            response = traj.steps[0].action
            responses.append(response)

            reward_result = self.compute_generator_reward(task, response)
            traj.steps[0].reward = reward_result.reward
            traj.reward = reward_result.reward

            if reward_result.is_correct:
                generator_correct_count += 1

            # Commit trajectory immediately to preserve it if later steps fail
            self.commit(trajectory=traj)

        # Hook for custom processing after generation
        self.on_generation_complete(list(generator_trajectories))

        # Step 2: Aggregator reviews all responses and selects the best
        agg_prompt = self.build_aggregator_prompt(task, responses)
        agg_messages = [{"role": "user", "content": agg_prompt}]

        agg_output = await self.rollout_engine.get_model_response(
            agg_messages,
            agent_name=self.AGGREGATOR_NAME,
            application_id=uid,
        )

        selected_response = self.parse_aggregator_response(
            agg_output.content,
            responses,
        )

        # Compute aggregator reward
        agg_reward = self.compute_aggregator_reward(task, selected_response)

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

        # Commit aggregator trajectory immediately
        self.commit(trajectory=aggregator_trajectory)

        # Compute metrics
        all_trajectories = list(generator_trajectories) + [aggregator_trajectory]
        any_correct = generator_correct_count > 0
        final_is_correct = agg_reward.is_correct

        metrics = {
            f"{self.GENERATOR_NAME}_acc": generator_correct_count / self.n_votes,
            f"{self.AGGREGATOR_NAME}_acc": float(final_is_correct),
            "n_votes": self.n_votes,
            "any_correct": int(any_correct),
            "success": int(final_is_correct),
            f"{self.GENERATOR_NAME}_attempts": self.n_votes,
        }

        # If use_final_outcome_reward is enabled, propagate the final reward
        # to all trajectories (all generators + aggregator)
        if self.use_final_outcome_reward:
            final_reward_value = agg_reward.reward
            for trajectory in all_trajectories:
                trajectory.reward = final_reward_value
                for step in trajectory.steps:
                    step.reward = final_reward_value

        return Episode(
            id=uid,
            task=task,
            trajectories=all_trajectories,
            is_correct=final_is_correct,
            metrics=metrics,
        )
