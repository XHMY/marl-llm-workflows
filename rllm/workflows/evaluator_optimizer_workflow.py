"""Evaluator-Optimizer Workflow.

This module provides an abstract base class for implementing evaluator-optimizer
workflows. The pattern involves:
1. Generator creates initial response
2. Evaluator reviews and provides feedback
3. Generator refines based on feedback
4. Loop until satisfied or max iterations reached
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine.rollout.rollout_engine import RolloutEngine
from rllm.rewards.reward_types import RewardOutput
from rllm.workflows.workflow import Workflow


@dataclass
class EvaluationResult:
    """Structured evaluation result from the evaluator agent.

    Attributes:
        is_satisfied: Whether the evaluation criteria are met (controls loop termination)
        feedback: Feedback text for the generator to use in refinement
        verdict: Short verdict string (e.g., "correct", "incorrect", "unknown")
        confidence: Optional confidence score (0.0 to 1.0)
        metadata: Additional evaluation-specific data
    """

    is_satisfied: bool
    feedback: str
    verdict: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvaluatorOptimizerWorkflow(Workflow):
    """Abstract base class for evaluator-optimizer workflows.

    This workflow pattern uses two agents:
    - Generator: Creates initial response AND refines based on feedback
    - Evaluator: Reviews responses against criteria and provides feedback

    The workflow loop:
    1. Generator creates initial response
    2. Evaluator reviews and provides feedback
    3. If not satisfied and under max_iterations: Generator refines
    4. Repeat steps 2-3 until satisfied or max iterations reached

    Subclasses must implement:
    - build_generator_prompt(): Create the initial generation prompt
    - build_evaluator_prompt(): Create the evaluation prompt
    - build_refinement_prompt(): Create the refinement prompt for generator
    - parse_evaluation(): Parse evaluator response into EvaluationResult
    - compute_generator_reward(): Calculate reward for generator trajectory
    - compute_evaluator_reward(): Calculate reward for evaluator trajectory

    Optional overrides:
    - extract_response(): Extract relevant content from model output
    - should_continue(): Additional termination logic
    - on_iteration_complete(): Hook for custom iteration logic

    Example:
        class MathWorkflow(EvaluatorOptimizerWorkflow):
            def build_generator_prompt(self, task):
                return f"Solve: {task['question']}"
            # ... implement other abstract methods ...
    """

    # Agent names (no underscores per CLAUDE.md conventions)
    GENERATOR_NAME = "generator"
    EVALUATOR_NAME = "evaluator"

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        max_iterations: int = 3,
        share_conversation_history: bool = True,
        use_final_outcome_reward: bool = False,
        **kwargs,
    ):
        """Initialize the EvaluatorOptimizerWorkflow.

        Args:
            rollout_engine: Engine for LLM inference
            max_iterations: Maximum number of evaluation-refinement cycles
            share_conversation_history: Whether to pass full conversation history
                to evaluator and generator during refinement
            use_final_outcome_reward: If True, assign the final outcome reward to
                ALL trajectories in the episode. This tests whether multi-agent
                training can converge without fine-grained per-agent reward feedback.
            **kwargs: Additional arguments passed to parent Workflow
        """
        super().__init__(rollout_engine, **kwargs)
        self.max_iterations = max_iterations
        self.share_conversation_history = share_conversation_history
        self.use_final_outcome_reward = use_final_outcome_reward

    # ===== Abstract methods that subclasses MUST implement =====

    @abstractmethod
    def build_generator_prompt(self, task: Dict[str, Any]) -> str:
        """Build the prompt for initial generation.

        Args:
            task: Task dictionary containing problem information

        Returns:
            Formatted prompt string for the generator
        """
        pass

    @abstractmethod
    def build_evaluator_prompt(
        self,
        task: Dict[str, Any],
        current_response: str,
        iteration: int,
        conversation_history: list,
    ) -> str:
        """Build the evaluation prompt.

        Args:
            task: Task dictionary
            current_response: The response to evaluate
            iteration: Current iteration number (0-indexed)
            conversation_history: Full conversation so far (if share_conversation_history=True)

        Returns:
            Formatted prompt string for the evaluator
        """
        pass

    @abstractmethod
    def build_refinement_prompt(
        self,
        task: Dict[str, Any],
        current_response: str,
        evaluation: EvaluationResult,
        iteration: int,
        conversation_history: list,
    ) -> str:
        """Build the refinement prompt for generator.

        Args:
            task: Task dictionary
            current_response: The response that was evaluated
            evaluation: Parsed evaluation result with feedback
            iteration: Current iteration number
            conversation_history: Full conversation so far (if share_conversation_history=True)

        Returns:
            Formatted prompt string for generator refinement
        """
        pass

    @abstractmethod
    def parse_evaluation(self, evaluator_response: str) -> EvaluationResult:
        """Parse evaluator response into structured result.

        Args:
            evaluator_response: Raw text response from evaluator

        Returns:
            EvaluationResult with is_satisfied, feedback, verdict, etc.
        """
        pass

    @abstractmethod
    def compute_generator_reward(
        self,
        task: Dict[str, Any],
        response: str,
    ) -> RewardOutput:
        """Compute reward for generator trajectory.

        Args:
            task: Task dictionary (may contain ground truth)
            response: Generator's response

        Returns:
            RewardOutput with reward value and metadata
        """
        pass

    @abstractmethod
    def compute_evaluator_reward(
        self,
        task: Dict[str, Any],
        evaluated_response: str,
        evaluation: EvaluationResult,
        ground_truth_correct: bool,
    ) -> RewardOutput:
        """Compute reward for evaluator trajectory.

        Args:
            task: Task dictionary
            evaluated_response: The response that was evaluated
            evaluation: Parsed evaluation result
            ground_truth_correct: Whether the evaluated response is actually correct

        Returns:
            RewardOutput with reward based on evaluation accuracy
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

    def should_continue(
        self,
        evaluation: EvaluationResult,
        iteration: int,
        task: Dict[str, Any],
    ) -> bool:
        """Determine if the evaluation-refinement loop should continue.

        Default: continue if not satisfied and under max iterations

        Override for custom termination logic (e.g., early stopping based on metrics).

        Args:
            evaluation: Current evaluation result
            iteration: Current iteration number (0-indexed)
            task: Task dictionary

        Returns:
            True if loop should continue, False to stop
        """
        return not evaluation.is_satisfied and iteration < self.max_iterations - 1

    def on_iteration_complete(
        self,
        iteration: int,
        evaluation: EvaluationResult,
        trajectories: list,
    ):
        """Hook called after each iteration completes.

        Override to add custom logging, metrics collection, or state updates.

        Args:
            iteration: Completed iteration number
            evaluation: Evaluation result from this iteration
            trajectories: All trajectories collected so far
        """
        pass

    # ===== Core workflow implementation =====

    async def run(self, task: Dict[str, Any], uid: str, **kwargs) -> Episode:
        """Execute the evaluator-optimizer workflow.

        Args:
            task: Task dictionary containing problem information
            uid: Unique identifier for this episode

        Returns:
            Episode with all generator and evaluator trajectories
        """
        self.reset(task, uid)

        all_trajectories = []
        conversation_history = []

        # Track per-agent statistics
        generator_attempts = 0
        evaluator_predictions = 0
        generator_correct_count = 0
        evaluator_correct_count = 0

        # Step 1: Generator creates initial response
        gen_prompt = self.build_generator_prompt(task)
        gen_messages = [{"role": "user", "content": gen_prompt}]

        gen_output = await self.rollout_engine.get_model_response(
            gen_messages,
            agent_name=self.GENERATOR_NAME,
            application_id=uid,
        )

        current_response = self.extract_response(gen_output)
        gen_reward = self.compute_generator_reward(task, current_response)

        gen_trajectory = self._create_trajectory(
            name=self.GENERATOR_NAME,
            messages=gen_messages,
            model_output=gen_output,
            response=current_response,
            reward=gen_reward.reward,
        )
        all_trajectories.append(gen_trajectory)
        generator_attempts += 1
        if gen_reward.is_correct:
            generator_correct_count += 1

        # Commit trajectory immediately to preserve it if later steps fail
        self.commit(trajectory=gen_trajectory)

        # Build conversation history
        conversation_history.extend(gen_messages)
        conversation_history.append({
            "role": "assistant",
            "content": gen_output.content,
            "reasoning": gen_output.reasoning,
        })

        # Iterative evaluation-refinement loop
        for iteration in range(self.max_iterations):
            # Step 2: Evaluator reviews current response
            eval_prompt = self.build_evaluator_prompt(
                task,
                current_response,
                iteration,
                conversation_history if self.share_conversation_history else [],
            )

            if self.share_conversation_history:
                eval_messages = conversation_history + [{"role": "user", "content": eval_prompt}]
            else:
                eval_messages = [{"role": "user", "content": eval_prompt}]

            eval_output = await self.rollout_engine.get_model_response(
                eval_messages,
                agent_name=self.EVALUATOR_NAME,
                application_id=uid,
            )

            evaluation = self.parse_evaluation(eval_output.content)

            # Compute evaluator reward
            ground_truth_correct = self.compute_generator_reward(task, current_response).is_correct

            eval_reward = self.compute_evaluator_reward(
                task,
                current_response,
                evaluation,
                ground_truth_correct,
            )

            eval_trajectory = self._create_trajectory(
                name=self.EVALUATOR_NAME,
                messages=eval_messages,
                model_output=eval_output,
                response=eval_output.content,
                reward=eval_reward.reward,
                action={"verdict": evaluation.verdict, "feedback": evaluation.feedback},
            )
            all_trajectories.append(eval_trajectory)
            evaluator_predictions += 1
            if eval_reward.is_correct:
                evaluator_correct_count += 1

            # Commit evaluator trajectory immediately
            self.commit(trajectory=eval_trajectory)

            # Update conversation history with evaluation
            conversation_history.append({
                "role": "assistant",
                "content": eval_output.content,
            })

            # Hook for custom iteration logic
            self.on_iteration_complete(iteration, evaluation, all_trajectories)

            # Check termination
            if not self.should_continue(evaluation, iteration, task):
                break

            # Step 3: Generator refines based on feedback
            refine_prompt = self.build_refinement_prompt(
                task,
                current_response,
                evaluation,
                iteration,
                conversation_history if self.share_conversation_history else [],
            )

            if self.share_conversation_history:
                refine_messages = conversation_history + [{"role": "user", "content": refine_prompt}]
            else:
                refine_messages = [{"role": "user", "content": refine_prompt}]

            refine_output = await self.rollout_engine.get_model_response(
                refine_messages,
                agent_name=self.GENERATOR_NAME,
                application_id=uid,
            )

            current_response = self.extract_response(refine_output)
            refine_reward = self.compute_generator_reward(task, current_response)

            refine_trajectory = self._create_trajectory(
                name=self.GENERATOR_NAME,
                messages=refine_messages,
                model_output=refine_output,
                response=current_response,
                reward=refine_reward.reward,
            )
            all_trajectories.append(refine_trajectory)
            generator_attempts += 1
            if refine_reward.is_correct:
                generator_correct_count += 1

            # Commit refinement trajectory immediately
            self.commit(trajectory=refine_trajectory)

            # Update conversation history with refinement
            conversation_history.append({
                "role": "assistant",
                "content": refine_output.content,
                "reasoning": refine_output.reasoning,
            })

        # Compute final metrics
        final_reward = self.compute_generator_reward(task, current_response)
        final_is_correct = final_reward.is_correct

        # If use_final_outcome_reward is enabled, propagate the final reward
        # to all trajectories (both generator and evaluator, all iterations)
        if self.use_final_outcome_reward:
            final_reward_value = final_reward.reward
            for trajectory in all_trajectories:
                trajectory.reward = final_reward_value
                for step in trajectory.steps:
                    step.reward = final_reward_value

        metrics = self._compute_workflow_metrics(
            all_trajectories,
            generator_attempts,
            evaluator_predictions,
            generator_correct_count,
            evaluator_correct_count,
            final_is_correct,
        )

        return Episode(
            id=uid,
            task=task,
            trajectories=all_trajectories,
            is_correct=final_is_correct,
            metrics=metrics,
        )

    def _create_trajectory(
        self,
        name: str,
        messages: list,
        model_output,
        response: str,
        reward: float,
        action: Any = None,
    ) -> Trajectory:
        """Helper to create trajectory with proper structure.

        Args:
            name: Agent name (e.g., "generator", "evaluator")
            messages: Input messages sent to the model
            model_output: ModelOutput from rollout engine
            response: Extracted response text
            reward: Computed reward value
            action: Optional action data (e.g., parsed verdict for evaluator)

        Returns:
            Trajectory with single step containing all information
        """
        trajectory = Trajectory(
            name=name,
            steps=[
                Step(
                    chat_completions=messages + [{
                        "role": "assistant",
                        "content": model_output.content,
                        "reasoning": model_output.reasoning,
                    }],
                    thought=model_output.reasoning,
                    action=action if action else response,
                    model_output=model_output,
                    reward=reward,
                )
            ],
        )
        trajectory.reward = reward
        return trajectory

    def _compute_workflow_metrics(
        self,
        trajectories: list,
        gen_attempts: int,
        eval_predictions: int,
        gen_correct: int,
        eval_correct: int,
        final_correct: bool,
    ) -> Dict[str, Any]:
        """Compute standard workflow metrics.

        Args:
            trajectories: All trajectories in the episode
            gen_attempts: Number of generation attempts
            eval_predictions: Number of evaluator predictions
            gen_correct: Number of correct generator responses
            eval_correct: Number of correct evaluator predictions
            final_correct: Whether final response is correct

        Returns:
            Dictionary of metrics
        """
        return {
            f"{self.GENERATOR_NAME}_acc": gen_correct / gen_attempts if gen_attempts > 0 else 0.0,
            f"{self.EVALUATOR_NAME}_acc": eval_correct / eval_predictions if eval_predictions > 0 else 0.0,
            "total_iterations": eval_predictions,
            "success": int(final_correct),
            f"{self.GENERATOR_NAME}_attempts": gen_attempts,
            f"{self.EVALUATOR_NAME}_predictions": eval_predictions,
        }
