"""Evaluator-Optimizer V2 Workflow with proper multi-turn conversation structure.

This module provides an abstract base class for evaluator-optimizer workflows
where each agent maintains its own conversation history with proper user/assistant
alternation.

Key differences from V1 (evaluator_optimizer_workflow.py):
- Agent-specific conversation histories (no shared history mixing both agents)
- Proper user/assistant alternation (no consecutive assistant messages)
- Evaluator feedback embedded as user messages in generator's history
- Generator solution embedded in evaluator's user messages
- Prompts must be self-contained (embed solution/feedback directly)
"""

from abc import abstractmethod
from typing import Any, Dict

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine.rollout.rollout_engine import RolloutEngine
from rllm.rewards.reward_types import RewardOutput
from rllm.workflows.evaluator_optimizer_workflow import EvaluationResult
from rllm.workflows.workflow import Workflow


class EvaluatorOptimizerV2Workflow(Workflow):
    """Abstract base class for evaluator-optimizer V2 workflows.

    This workflow uses two agents with agent-specific multi-turn histories:
    - Generator: Sees its own solutions + reviewer feedback as user messages
    - Evaluator: Sees problem+solution embedded in user messages + its own evaluations

    The conversation structure ensures:
    - Strict user/assistant alternation in every trajectory
    - Evaluator feedback naturally becomes a user turn for the generator
    - Generator solutions are embedded in the evaluator's user prompts
    - No consecutive assistant messages

    Subclasses must implement the same abstract methods as V1:
    - build_generator_prompt(): Create the initial generation prompt
    - build_evaluator_prompt(): Create self-contained evaluation prompt (embed problem + solution)
    - build_refinement_prompt(): Create refinement prompt (embed feedback)
    - parse_evaluation(): Parse evaluator response into EvaluationResult
    - compute_generator_reward(): Calculate reward for generator trajectory
    - compute_evaluator_reward(): Calculate reward for evaluator trajectory
    """

    GENERATOR_NAME = "generator"
    EVALUATOR_NAME = "evaluator"

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        max_iterations: int = 3,
        use_final_outcome_reward: bool = False,
        **kwargs,
    ):
        """Initialize the EvaluatorOptimizerV2Workflow.

        Args:
            rollout_engine: Engine for LLM inference
            max_iterations: Maximum number of evaluation-refinement cycles
            use_final_outcome_reward: If True, assign the final outcome reward to
                ALL trajectories in the episode.
            **kwargs: Additional arguments passed to parent Workflow
        """
        super().__init__(rollout_engine, **kwargs)
        self.max_iterations = max_iterations
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
        """Build a self-contained evaluation prompt.

        Must embed the problem and current solution directly in the prompt,
        since the evaluator's conversation history only contains its own
        prior evaluations.

        Args:
            task: Task dictionary
            current_response: The response to evaluate (embed in prompt)
            iteration: Current iteration number (0-indexed)
            conversation_history: Evaluator's own conversation history

        Returns:
            Self-contained prompt string for the evaluator
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
        """Build a refinement prompt embedding the reviewer's feedback.

        The generator's prior solutions are visible as assistant turns in its
        conversation history. This prompt should embed the feedback and instruct
        the model to fix the identified issue incrementally.

        Args:
            task: Task dictionary
            current_response: The response that was evaluated
            evaluation: Parsed evaluation result with feedback (embed in prompt)
            iteration: Current iteration number
            conversation_history: Generator's own conversation history

        Returns:
            Prompt string embedding feedback for the generator
        """
        pass

    @abstractmethod
    def parse_evaluation(self, evaluator_response: str) -> EvaluationResult:
        """Parse evaluator response into structured result."""
        pass

    @abstractmethod
    def compute_generator_reward(
        self,
        task: Dict[str, Any],
        response: str,
    ) -> RewardOutput:
        """Compute reward for generator trajectory."""
        pass

    @abstractmethod
    def compute_evaluator_reward(
        self,
        task: Dict[str, Any],
        evaluated_response: str,
        evaluation: EvaluationResult,
        ground_truth_correct: bool,
    ) -> RewardOutput:
        """Compute reward for evaluator trajectory."""
        pass

    # ===== Optional hooks =====

    def extract_response(self, model_output) -> str:
        """Extract the relevant response content from model output."""
        return model_output.content or model_output.text or ""

    def should_continue(
        self,
        evaluation: EvaluationResult,
        iteration: int,
        task: Dict[str, Any],
    ) -> bool:
        """Determine if the evaluation-refinement loop should continue."""
        return not evaluation.is_satisfied and iteration < self.max_iterations - 1

    def on_iteration_complete(
        self,
        iteration: int,
        evaluation: EvaluationResult,
        trajectories: list,
    ):
        """Hook called after each iteration completes."""
        pass

    # ===== Core workflow implementation =====

    async def run(self, task: Dict[str, Any], uid: str, **kwargs) -> Episode:
        """Execute the evaluator-optimizer V2 workflow.

        Uses agent-specific conversation histories to maintain proper
        user/assistant alternation for each agent.

        Args:
            task: Task dictionary containing problem information
            uid: Unique identifier for this episode

        Returns:
            Episode with all generator and evaluator trajectories
        """
        self.reset(task, uid)

        all_trajectories = []

        # Agent-specific multi-turn conversation histories
        gen_conversation = []    # Generator: [user, asst, user, asst, ...]
        eval_conversation = []   # Evaluator: [user, asst, user, asst, ...]

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

        self.commit(trajectory=gen_trajectory)

        # Build generator's conversation history
        gen_conversation = [
            {"role": "user", "content": gen_prompt},
            {"role": "assistant", "content": gen_output.content, "reasoning": gen_output.reasoning},
        ]

        # Iterative evaluation-refinement loop
        for iteration in range(self.max_iterations):
            # Step 2: Evaluator reviews current response
            # Prompt is self-contained: embeds problem + solution
            eval_prompt = self.build_evaluator_prompt(
                task, current_response, iteration, eval_conversation,
            )

            eval_messages = eval_conversation + [{"role": "user", "content": eval_prompt}]

            eval_output = await self.rollout_engine.get_model_response(
                eval_messages,
                agent_name=self.EVALUATOR_NAME,
                application_id=uid,
            )

            evaluation = self.parse_evaluation(eval_output.content)

            # Compute evaluator reward
            ground_truth_correct = self.compute_generator_reward(task, current_response).is_correct

            eval_reward = self.compute_evaluator_reward(
                task, current_response, evaluation, ground_truth_correct,
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

            self.commit(trajectory=eval_trajectory)

            # Update evaluator's conversation history
            eval_conversation.extend([
                {"role": "user", "content": eval_prompt},
                {"role": "assistant", "content": eval_output.content},
            ])

            self.on_iteration_complete(iteration, evaluation, all_trajectories)

            # Check termination
            if not self.should_continue(evaluation, iteration, task):
                break

            # Step 3: Generator refines based on feedback
            # Prompt embeds the reviewer's feedback; prior solutions visible in gen_conversation
            refine_prompt = self.build_refinement_prompt(
                task, current_response, evaluation, iteration, gen_conversation,
            )

            refine_messages = gen_conversation + [{"role": "user", "content": refine_prompt}]

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

            self.commit(trajectory=refine_trajectory)

            # Update generator's conversation history
            gen_conversation.extend([
                {"role": "user", "content": refine_prompt},
                {"role": "assistant", "content": refine_output.content, "reasoning": refine_output.reasoning},
            ])

        # Compute final metrics
        final_reward = self.compute_generator_reward(task, current_response)
        final_is_correct = final_reward.is_correct

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
        """Helper to create trajectory with proper structure."""
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
        """Compute standard workflow metrics."""
        return {
            f"{self.GENERATOR_NAME}_acc": gen_correct / gen_attempts if gen_attempts > 0 else 0.0,
            f"{self.EVALUATOR_NAME}_acc": eval_correct / eval_predictions if eval_predictions > 0 else 0.0,
            "total_iterations": eval_predictions,
            "success": int(final_correct),
            f"{self.GENERATOR_NAME}_attempts": gen_attempts,
            f"{self.EVALUATOR_NAME}_predictions": eval_predictions,
        }
