"""Voting V2 (Per-Voter Policy) Workflow.

This module provides an abstract base class for voting workflows where each
voter uses a distinct policy (LoRA adapter in multi-LoRA mode). This creates
structural diversity -- each voter develops its own reasoning strategy.

Differences from VotingWorkflow (V1):
- V1: All N votes share a single "generator" agent/LoRA adapter
- V2: Each vote uses a different voter agent (voterA, voterB, voterC, ...)
      with its own LoRA adapter when share_policy=False

When share_policy=True, all voters use the same base model (identical to V1).
"""

import asyncio
from typing import Any, Dict, List, Optional

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine.rollout.rollout_engine import RolloutEngine
from rllm.workflows.voting_workflow import VotingWorkflow


class VotingV2Workflow(VotingWorkflow):
    """Abstract base class for voting workflows with per-voter policies.

    Each voter uses a distinct agent name (voterA, voterB, voterC, ...),
    which maps to a separate LoRA adapter when share_policy=False.

    Config example:
        trainer:
          agent_names: ['voterA', 'voterB', 'voterC', 'aggregator']
          share_policy: False

    Subclasses must implement the same abstract methods as VotingWorkflow:
    - build_generator_prompt(): Create the generation prompt
    - build_aggregator_prompt(): Create the aggregation prompt with all responses
    - parse_aggregator_response(): Parse aggregator output to get final answer
    - compute_generator_reward(): Calculate reward for each voter trajectory
    - compute_aggregator_reward(): Calculate reward for aggregator trajectory
    """

    VOTER_PREFIX = "voter"

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        n_votes: int = 3,
        voter_names: Optional[List[str]] = None,
        use_rubric_reward: bool = False,
        voter_correct_bonus: float = 0.5,
        final_correct_bonus: float = 0.5,
        voter_wrong_penalty: float = -1.0,
        **kwargs,
    ):
        """Initialize the VotingV2Workflow.

        Args:
            rollout_engine: Engine for LLM inference
            n_votes: Number of parallel voters (must match len(voter_names) if provided)
            voter_names: Explicit voter agent names. If None, auto-generates
                         as ['voterA', 'voterB', 'voterC', ...] from n_votes.
            use_rubric_reward: If True, use per-agent rubric reward instead of
                final outcome reward. Mutually exclusive with use_final_outcome_reward.
            voter_correct_bonus: Reward bonus when a voter's answer is correct (default +0.5).
            final_correct_bonus: Reward bonus added to all voters when the aggregator's
                final selection is correct (default +0.5).
            voter_wrong_penalty: Reward penalty when a voter produces a parseable but
                wrong answer (default -1.0, must be <= 0).
            **kwargs: Additional arguments passed to parent VotingWorkflow
        """
        super().__init__(rollout_engine, n_votes=n_votes, **kwargs)

        if voter_names:
            self.voter_names = list(voter_names)
        else:
            # Auto-generate alphabetic names: voterA, voterB, voterC, ...
            self.voter_names = [
                f"{self.VOTER_PREFIX}{chr(65 + i)}" for i in range(n_votes)
            ]

        assert len(self.voter_names) == self.n_votes, (
            f"voter_names length ({len(self.voter_names)}) must equal "
            f"n_votes ({self.n_votes})"
        )

        # Validate naming conventions
        for name in self.voter_names:
            assert "_" not in name, (
                f"Voter name '{name}' contains underscore. "
                "Agent names must not contain underscores (used as trajectory ID delimiter)."
            )

        self.use_rubric_reward = use_rubric_reward
        self.voter_correct_bonus = voter_correct_bonus
        self.final_correct_bonus = final_correct_bonus
        self.voter_wrong_penalty = voter_wrong_penalty

        if self.use_rubric_reward and self.use_final_outcome_reward:
            raise ValueError(
                "use_rubric_reward and use_final_outcome_reward are mutually exclusive. "
                "Set only one of them to True."
            )

    def has_parseable_answer(self, response: str) -> bool:
        """Return True if the voter response contains a parseable answer.

        Override in subclasses to apply domain-specific parsing checks.
        Default: any non-empty (after stripping) response is considered parseable.

        Args:
            response: The voter's raw response string.

        Returns:
            True if a structured answer can be extracted from the response.
        """
        return len(response.strip()) > 0

    def _compute_rubric_voter_reward(
        self,
        voter_is_correct: bool,
        final_is_correct: bool,
        response: str,
    ) -> float:
        """Compute the additive rubric reward for a single voter.

        Components (all additive, independent of each other):
          +voter_correct_bonus   if voter_is_correct
          +final_correct_bonus   if final_is_correct
          +voter_wrong_penalty   if not voter_is_correct AND has_parseable_answer(response)

        Args:
            voter_is_correct: Whether this voter's answer is correct.
            final_is_correct: Whether the aggregator's final selection is correct.
            response: The voter's raw response (used for parseable check).

        Returns:
            Scalar rubric reward in range [voter_wrong_penalty, voter_correct_bonus + final_correct_bonus].
        """
        reward = 0.0
        if voter_is_correct:
            reward += self.voter_correct_bonus
        if final_is_correct:
            reward += self.final_correct_bonus
        if not voter_is_correct and self.has_parseable_answer(response):
            reward += self.voter_wrong_penalty  # negative float
        return reward

    def _compute_rubric_aggregator_reward(
        self,
        any_voter_correct: bool,
        final_is_correct: bool,
    ) -> float:
        """Compute the rubric reward for the aggregator.

        The aggregator is only held accountable when at least one voter was correct
        (i.e., a correct answer existed in the candidate pool). If all voters were
        wrong, the aggregator cannot pick a right answer and receives 0.0.

        Args:
            any_voter_correct: Whether at least one voter produced a correct answer.
            final_is_correct: Whether the aggregator's final selection is correct.

        Returns:
            +1.0 if aggregator selected correctly (and any voter was correct),
            -1.0 if aggregator failed to select the correct answer (and any voter was correct),
             0.0 if all voters were wrong (aggregator not penalized).
        """
        if not any_voter_correct:
            return 0.0
        return 1.0 if final_is_correct else -1.0

    async def _generate_single(
        self,
        task: Dict[str, Any],
        vote_index: int,
        uid: str = None,
    ) -> Trajectory:
        """Generate a single response using the voter-specific policy.

        Each vote_index maps to a different voter agent name, which routes
        to a different LoRA adapter during inference.

        Args:
            task: Task dictionary
            vote_index: Index of this vote (maps to voter_names[vote_index])
            uid: Optional unique identifier for sticky session routing

        Returns:
            Trajectory with name set to the voter's agent name
        """
        voter_name = self.voter_names[vote_index]
        prompt = self.build_generator_prompt(task)
        messages = [{"role": "user", "content": prompt}]

        kwargs = {"agent_name": voter_name}
        if uid is not None:
            kwargs["application_id"] = uid
        output = await self.rollout_engine.get_model_response(
            messages,
            **kwargs,
        )

        response = self.extract_response(output)

        trajectory = Trajectory(
            name=voter_name,  # Unique per voter for separate LoRA routing and GRPO groups
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

        Args:
            task: Task dictionary containing problem information
            uid: Unique identifier for this episode

        Returns:
            Episode with per-voter trajectories and aggregator trajectory
        """
        self.reset(task, uid)

        # Step 1: Generate N responses in parallel, each with a different voter policy
        generation_tasks = [
            self._generate_single(task, i, uid=uid)
            for i in range(self.n_votes)
        ]
        voter_trajectories = await asyncio.gather(*generation_tasks)

        # Extract responses and compute rewards for each voter
        responses = []
        voter_correct_count = 0
        voter_parseable_count = 0
        voter_reward_results: list = []
        per_voter_acc = {}

        for i, traj in enumerate(voter_trajectories):
            response = traj.steps[0].action
            responses.append(response)

            reward_result = self.compute_generator_reward(task, response)
            traj.steps[0].reward = reward_result.reward
            traj.reward = reward_result.reward

            voter_reward_results.append(reward_result)
            if reward_result.is_correct:
                voter_correct_count += 1
            if self.has_parseable_answer(response):
                voter_parseable_count += 1

            per_voter_acc[f"{self.voter_names[i]}_acc"] = float(reward_result.is_correct)

            # Commit trajectory immediately to preserve it if later steps fail
            self.commit(trajectory=traj)

        # Hook for custom processing after generation
        self.on_generation_complete(list(voter_trajectories))

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
        all_trajectories = list(voter_trajectories) + [aggregator_trajectory]
        any_correct = voter_correct_count > 0
        final_is_correct = agg_reward.is_correct

        metrics = {
            "voter_acc": voter_correct_count / self.n_votes,
            f"{self.AGGREGATOR_NAME}_acc": float(final_is_correct),
            "n_votes": self.n_votes,
            "any_correct": int(any_correct),
            "success": int(final_is_correct),
            "voter_attempts": self.n_votes,
            "voter_correct_count": voter_correct_count,
            "voter_parseable_count": voter_parseable_count,
            **per_voter_acc,
        }

        # If use_final_outcome_reward is enabled, propagate the final reward
        # to all trajectories (all voters + aggregator)
        if self.use_final_outcome_reward:
            final_reward_value = agg_reward.reward
            for trajectory in all_trajectories:
                trajectory.reward = final_reward_value
                for step in trajectory.steps:
                    step.reward = final_reward_value
        elif self.use_rubric_reward:
            # Recompute per-voter and aggregator rewards using the rubric.
            # Safe to update after commit() — live trajectory objects are what
            # the Episode returns; committed copies are for fault tolerance only.
            _final_is_correct = bool(agg_reward.is_correct)
            rubric_voter_rewards = []

            for i, traj in enumerate(voter_trajectories):
                voter_is_correct = bool(voter_reward_results[i].is_correct)
                response = traj.steps[0].action
                rubric_r = self._compute_rubric_voter_reward(
                    voter_is_correct=voter_is_correct,
                    final_is_correct=_final_is_correct,
                    response=response,
                )
                traj.reward = rubric_r
                traj.steps[0].reward = rubric_r
                rubric_voter_rewards.append(rubric_r)

            agg_rubric_r = self._compute_rubric_aggregator_reward(
                any_voter_correct=any_correct,
                final_is_correct=_final_is_correct,
            )
            aggregator_trajectory.reward = agg_rubric_r
            aggregator_trajectory.steps[0].reward = agg_rubric_r

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
