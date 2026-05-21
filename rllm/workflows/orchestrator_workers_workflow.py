"""Orchestrator-Workers Workflow.

This module provides an abstract base class for implementing orchestrator-workers
workflows. The pattern involves three agents:
1. Orchestrator (Proposal): Proposes distinct solution strategies for a task
2. Workers (Execution): Each worker solves the full task using an assigned strategy
3. Synthesizer (Synthesis): Compares worker solutions and picks the best answer
"""

import asyncio
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine.rollout.rollout_engine import RolloutEngine
from rllm.rewards.reward_types import RewardOutput
from rllm.workflows.workflow import Workflow


@dataclass
class SubtaskResult:
    """Result from a worker executing a subtask.

    Attributes:
        subtask_id: Index of the subtask (0-indexed)
        subtask_description: Text description of the subtask
        response: Worker's response to the subtask
        success: Whether the worker completed successfully
        metadata: Additional subtask-specific data
    """

    subtask_id: int
    subtask_description: str
    response: str
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProposalResult:
    """Result from orchestrator's strategy proposal.

    Attributes:
        strategies: List of proposed strategy descriptions
        label: Optional label for the proposal approach
        execution_mode: "parallel" or "sequential" execution of strategies
        metadata: Additional proposal-specific data
    """

    strategies: List[str]
    label: str = ""
    execution_mode: str = "parallel"  # or "sequential"
    metadata: Dict[str, Any] = field(default_factory=dict)


class OrchestratorWorkersWorkflow(Workflow):
    """Abstract base class for orchestrator-workers workflows.

    This workflow pattern uses three agents:
    - Orchestrator: Proposes distinct solution strategies for a task
    - Worker: Solves the full task using an assigned strategy
    - Synthesizer: Compares worker solutions and picks the best answer

    The workflow:
    1. Orchestrator proposes multiple solution strategies
    2. Workers each solve the full task using their assigned strategy
    3. Synthesizer compares complete solutions and picks the best

    Subclasses must implement:
    - build_proposal_prompt(): Create prompt for strategy proposal
    - build_worker_prompt(): Create prompt for worker to solve using a strategy
    - build_synthesis_prompt(): Create prompt to compare worker solutions
    - parse_proposals(): Parse orchestrator response into ProposalResult
    - compute_final_reward(): Calculate reward based on final answer

    Optional overrides:
    - extract_response(): Extract relevant content from model output
    - should_execute_parallel(): Determine execution mode
    - on_proposal_complete(): Hook after proposal
    - on_worker_complete(): Hook after each worker completes
    - on_synthesis_complete(): Hook after synthesis
    - compute_worker_reward(): Calculate per-worker reward (default: 0.0)
    - compute_proposal_reward(): Calculate proposal reward (default: 0.0)

    Example:
        class MathOrchestratorWorkflow(OrchestratorWorkersWorkflow):
            def build_proposal_prompt(self, task, max_subtasks):
                return f"Propose strategies for: {task['question']}"
            # ... implement other abstract methods ...
    """

    # Agent names (no underscores per CLAUDE.md conventions)
    ORCHESTRATOR_NAME = "orchestrator"
    WORKER_NAME = "worker"
    SYNTHESIZER_NAME = "synthesizer"

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        max_subtasks: int = 5,
        default_execution_mode: str = "parallel",
        use_final_outcome_reward: bool = True,
        **kwargs,
    ):
        """Initialize the OrchestratorWorkersWorkflow.

        Args:
            rollout_engine: Engine for LLM inference
            max_subtasks: Maximum number of strategies allowed
            default_execution_mode: Default execution mode ("parallel" or "sequential")
            use_final_outcome_reward: If True, assign the final outcome reward to
                ALL trajectories in the episode
            **kwargs: Additional arguments passed to parent Workflow
        """
        super().__init__(rollout_engine, **kwargs)
        self.max_subtasks = max_subtasks
        self.default_execution_mode = default_execution_mode
        self.use_final_outcome_reward = use_final_outcome_reward

    # ===== Abstract methods that subclasses MUST implement =====

    @abstractmethod
    def build_proposal_prompt(self, task: Dict[str, Any], max_subtasks: int) -> str:
        """Build the prompt for strategy proposal.

        Args:
            task: Task dictionary containing problem information
            max_subtasks: Maximum number of strategies allowed

        Returns:
            Formatted prompt string for the orchestrator to propose strategies
        """
        pass

    @abstractmethod
    def build_worker_prompt(
        self,
        task: Dict[str, Any],
        subtask: str,
        subtask_id: int,
        previous_results: List[SubtaskResult],
    ) -> str:
        """Build the prompt for a worker to solve using a strategy.

        Args:
            task: Original task dictionary
            subtask: Description of the assigned strategy
            subtask_id: Index of this strategy (0-indexed)
            previous_results: Results from previous workers (for sequential mode)

        Returns:
            Formatted prompt string for the worker
        """
        pass

    @abstractmethod
    def build_synthesis_prompt(
        self,
        task: Dict[str, Any],
        proposal: ProposalResult,
        worker_results: List[SubtaskResult],
    ) -> str:
        """Build the prompt for comparing worker solutions.

        Args:
            task: Original task dictionary
            proposal: The proposal result from phase 1
            worker_results: All results from workers

        Returns:
            Formatted prompt string for the synthesizer
        """
        pass

    @abstractmethod
    def parse_proposals(self, orchestrator_response: str) -> ProposalResult:
        """Parse orchestrator response into structured proposal result.

        Args:
            orchestrator_response: Raw text response from orchestrator

        Returns:
            ProposalResult with strategies and execution mode
        """
        pass

    @abstractmethod
    async def compute_final_reward(
        self,
        task: Dict[str, Any],
        final_response: str,
    ) -> RewardOutput:
        """Compute reward based on the final synthesized response.

        Args:
            task: Task dictionary (may contain ground truth)
            final_response: Final synthesized response from orchestrator

        Returns:
            RewardOutput with reward value and metadata
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

    def should_execute_parallel(
        self,
        proposal: ProposalResult,
        task: Dict[str, Any],
    ) -> bool:
        """Determine if strategies should be executed in parallel.

        Default: uses proposal.execution_mode or default_execution_mode

        Override for custom logic based on task properties.

        Args:
            proposal: The proposal result
            task: Task dictionary

        Returns:
            True for parallel execution, False for sequential
        """
        mode = proposal.execution_mode or self.default_execution_mode
        return mode == "parallel"

    def on_proposal_complete(
        self,
        proposal: ProposalResult,
        trajectory: Trajectory,
    ):
        """Hook called after strategy proposal completes.

        Override to add custom logging, metrics, or state updates.

        Args:
            proposal: The parsed proposal result
            trajectory: Trajectory for the proposal step
        """
        pass

    def on_worker_complete(
        self,
        subtask_id: int,
        result: SubtaskResult,
        trajectory: Trajectory,
    ):
        """Hook called after each worker completes.

        Override to add custom logging or intermediate processing.

        Args:
            subtask_id: Index of the completed subtask
            result: The worker's result
            trajectory: Trajectory for this worker step
        """
        pass

    def on_synthesis_complete(
        self,
        final_response: str,
        trajectory: Trajectory,
    ):
        """Hook called after synthesis completes.

        Override to add custom logging or post-processing.

        Args:
            final_response: The synthesized final response
            trajectory: Trajectory for the synthesis step
        """
        pass

    def compute_worker_reward(
        self,
        task: Dict[str, Any],
        subtask: str,
        response: str,
        subtask_id: int,
    ) -> RewardOutput:
        """Compute reward for a worker trajectory.

        Default: returns 0.0 reward (relies on final outcome reward).

        Override to provide intermediate feedback to workers.

        Args:
            task: Task dictionary
            subtask: The strategy description
            response: Worker's response
            subtask_id: Index of the strategy

        Returns:
            RewardOutput with reward value
        """
        return RewardOutput(reward=0.0, is_correct=False)

    def compute_proposal_reward(
        self,
        task: Dict[str, Any],
        proposal: ProposalResult,
    ) -> RewardOutput:
        """Compute reward for the proposal trajectory.

        Default: returns 0.0 reward (relies on final outcome reward).

        Override to provide feedback on proposal quality.

        Args:
            task: Task dictionary
            proposal: The parsed proposal result

        Returns:
            RewardOutput with reward value
        """
        return RewardOutput(reward=0.0, is_correct=False)

    # ===== Core workflow implementation =====

    async def _execute_worker(
        self,
        task: Dict[str, Any],
        subtask: str,
        subtask_id: int,
        previous_results: List[SubtaskResult],
        uid: str = None,
    ) -> tuple[SubtaskResult, Trajectory]:
        """Execute a single worker task.

        Args:
            task: Original task dictionary
            subtask: Description of the strategy
            subtask_id: Index of this strategy
            previous_results: Results from previous workers (for sequential mode)
            uid: Optional unique identifier for sticky session routing

        Returns:
            Tuple of (SubtaskResult, Trajectory)
        """
        prompt = self.build_worker_prompt(task, subtask, subtask_id, previous_results)
        messages = [{"role": "user", "content": prompt}]

        kwargs = {"agent_name": self.WORKER_NAME}
        if uid is not None:
            kwargs["application_id"] = uid
        output = await self.rollout_engine.get_model_response(
            messages,
            **kwargs,
        )

        response = self.extract_response(output)
        reward = self.compute_worker_reward(task, subtask, response, subtask_id)

        result = SubtaskResult(
            subtask_id=subtask_id,
            subtask_description=subtask,
            response=response,
            success=True,
            metadata={"reward": reward.reward},
        )

        trajectory = Trajectory(
            name=self.WORKER_NAME,
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
                    reward=reward.reward,
                )
            ],
        )
        trajectory.reward = reward.reward

        return result, trajectory

    async def run(self, task: Dict[str, Any], uid: str, **kwargs) -> Episode:
        """Execute the orchestrator-workers workflow.

        Args:
            task: Task dictionary containing problem information
            uid: Unique identifier for this episode

        Returns:
            Episode with all orchestrator and worker trajectories
        """
        self.reset(task, uid)

        all_trajectories = []

        # Phase 1: Orchestrator proposes strategies
        proposal_prompt = self.build_proposal_prompt(task, self.max_subtasks)
        proposal_messages = [{"role": "user", "content": proposal_prompt}]

        proposal_output = await self.rollout_engine.get_model_response(
            proposal_messages,
            agent_name=self.ORCHESTRATOR_NAME,
            application_id=uid,
        )

        proposal = self.parse_proposals(proposal_output.content)

        # Build proposal trajectory (needed for both success and failure cases)
        proposal_trajectory = Trajectory(
            name=self.ORCHESTRATOR_NAME,
            steps=[
                Step(
                    chat_completions=proposal_messages + [{
                        "role": "assistant",
                        "content": proposal_output.content,
                        "reasoning": proposal_output.reasoning,
                    }],
                    thought=proposal_output.reasoning,
                    action={
                        "phase": "proposal",
                        "strategies": proposal.strategies,
                        "label": proposal.label,
                        "execution_mode": proposal.execution_mode,
                    },
                    model_output=proposal_output,
                    reward=0.0,  # Will be updated below
                )
            ],
        )

        # Check if orchestrator exceeded max_subtasks limit
        # This is a negative example - orchestrator failed to follow instructions
        if len(proposal.strategies) > self.max_subtasks:
            # Return episode with only proposal trajectory and 0 reward
            proposal_trajectory.reward = 0.0
            for step in proposal_trajectory.steps:
                step.reward = 0.0

            return Episode(
                id=uid,
                task=task,
                trajectories=[proposal_trajectory],
                is_correct=False,
                metrics={
                    "exceeded_max_subtasks": 1,
                    "n_strategies_generated": len(proposal.strategies),
                    "max_subtasks_allowed": self.max_subtasks,
                    "success": 0,
                },
            )

        proposal_reward = self.compute_proposal_reward(task, proposal)

        # Update proposal trajectory with computed reward
        proposal_trajectory.reward = proposal_reward.reward
        for step in proposal_trajectory.steps:
            step.reward = proposal_reward.reward

        all_trajectories.append(proposal_trajectory)

        # Commit proposal trajectory immediately to preserve it if later steps fail
        self.commit(trajectory=proposal_trajectory)

        # Hook for proposal complete
        self.on_proposal_complete(proposal, proposal_trajectory)

        # Phase 2: Workers execute strategies
        worker_results: List[SubtaskResult] = []
        worker_trajectories: List[Trajectory] = []

        if self.should_execute_parallel(proposal, task):
            # Parallel execution using asyncio.gather()
            worker_tasks = [
                self._execute_worker(task, strategy, i, [], uid=uid)
                for i, strategy in enumerate(proposal.strategies)
            ]
            results_and_trajectories = await asyncio.gather(*worker_tasks)

            for i, (result, trajectory) in enumerate(results_and_trajectories):
                worker_results.append(result)
                worker_trajectories.append(trajectory)
                # Commit worker trajectory immediately
                self.commit(trajectory=trajectory)
                self.on_worker_complete(i, result, trajectory)
        else:
            # Sequential execution, passing previous results
            for i, strategy in enumerate(proposal.strategies):
                result, trajectory = await self._execute_worker(
                    task, strategy, i, worker_results, uid=uid
                )
                worker_results.append(result)
                worker_trajectories.append(trajectory)
                # Commit worker trajectory immediately
                self.commit(trajectory=trajectory)
                self.on_worker_complete(i, result, trajectory)

        all_trajectories.extend(worker_trajectories)

        # Phase 3: Synthesizer compares solutions
        synth_prompt = self.build_synthesis_prompt(task, proposal, worker_results)
        synth_messages = [{"role": "user", "content": synth_prompt}]

        synth_output = await self.rollout_engine.get_model_response(
            synth_messages,
            agent_name=self.SYNTHESIZER_NAME,
            application_id=uid,
        )

        final_response = self.extract_response(synth_output)
        final_reward = await self.compute_final_reward(task, final_response)

        synth_trajectory = Trajectory(
            name=self.SYNTHESIZER_NAME,
            steps=[
                Step(
                    chat_completions=synth_messages + [{
                        "role": "assistant",
                        "content": synth_output.content,
                        "reasoning": synth_output.reasoning,
                    }],
                    thought=synth_output.reasoning,
                    action={"phase": "synthesis", "final_response": final_response},
                    model_output=synth_output,
                    reward=final_reward.reward,
                )
            ],
        )
        synth_trajectory.reward = final_reward.reward
        all_trajectories.append(synth_trajectory)

        # Commit synthesis trajectory immediately
        self.commit(trajectory=synth_trajectory)

        # Hook for synthesis complete
        self.on_synthesis_complete(final_response, synth_trajectory)

        # Phase 4: Apply final outcome reward to ALL trajectories if enabled
        if self.use_final_outcome_reward:
            final_reward_value = final_reward.reward
            for trajectory in all_trajectories:
                trajectory.reward = final_reward_value
                for step in trajectory.steps:
                    step.reward = final_reward_value

        # Compute metrics
        metrics = self._compute_workflow_metrics(
            all_trajectories,
            proposal,
            worker_results,
            final_reward.is_correct,
        )

        return Episode(
            id=uid,
            task=task,
            trajectories=all_trajectories,
            is_correct=final_reward.is_correct,
            metrics=metrics,
        )

    def _compute_workflow_metrics(
        self,
        trajectories: List[Trajectory],
        proposal: ProposalResult,
        worker_results: List[SubtaskResult],
        final_correct: bool,
    ) -> Dict[str, Any]:
        """Compute standard workflow metrics.

        Args:
            trajectories: All trajectories in the episode
            proposal: The proposal result
            worker_results: All worker results
            final_correct: Whether final response is correct

        Returns:
            Dictionary of metrics
        """
        n_strategies = len(proposal.strategies)
        n_workers = len(worker_results)
        successful_workers = sum(1 for r in worker_results if r.success)

        # Count agent calls (orchestrator=1 proposal, synthesizer=1 synthesis)
        orchestrator_calls = 1
        synthesizer_calls = 1
        worker_calls = n_workers

        return {
            "n_strategies": n_strategies,
            "n_workers": n_workers,
            "successful_workers": successful_workers,
            "worker_success_rate": successful_workers / n_workers if n_workers > 0 else 0.0,
            f"{self.ORCHESTRATOR_NAME}_calls": orchestrator_calls,
            f"{self.WORKER_NAME}_calls": worker_calls,
            f"{self.SYNTHESIZER_NAME}_calls": synthesizer_calls,
            "success": int(final_correct),
            "total_trajectories": len(trajectories),
        }
