"""Math-specific implementation of the Evaluator-Optimizer workflow.

This module provides a concrete implementation of EvaluatorOptimizerWorkflow
for mathematical problem solving with iterative refinement based on feedback.
"""

import json
import re
from typing import Any, Dict

from rllm.engine.rollout.rollout_engine import RolloutEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.rewards.reward_types import RewardOutput
from rllm.workflows.evaluator_optimizer_workflow import (
    EvaluationResult,
    EvaluatorOptimizerWorkflow,
)


class EvaluatorOptimizerMathWorkflow(EvaluatorOptimizerWorkflow):
    """Math-specific evaluator-optimizer workflow.

    This workflow implements the evaluator-optimizer pattern for mathematical
    problem solving:

    1. Generator: Solves math problems, outputs answer in \\boxed{} format
    2. Evaluator: Checks solution correctness, outputs \\boxed{Correct/Incorrect}
    3. Generator: Refines solution based on evaluator feedback

    The loop continues until the evaluator says "Correct" or max iterations reached.

    Example:
        workflow = EvaluatorOptimizerMathWorkflow(
            rollout_engine=engine,
            reward_function=math_reward_fn,
            max_iterations=3,
        )
        episode = await workflow.run(task, uid)
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        reward_function: RewardFunction,
        prompts: Dict[str, Any] = None,
        prompt_file: str = "examples/math_reasoning/prompt.json",
        max_iterations: int = 3,
        use_final_outcome_reward: bool = False,
        **kwargs,
    ):
        """Initialize the math evaluator-optimizer workflow.

        Args:
            rollout_engine: Engine for LLM inference
            reward_function: Function to compute rewards based on ground truth
            prompts: Optional pre-loaded prompt templates
            prompt_file: Path to JSON file with prompt templates
            max_iterations: Maximum evaluation-refinement cycles
            use_final_outcome_reward: If True, assign the final outcome reward to
                ALL trajectories. Useful for testing multi-agent training without
                fine-grained per-agent reward feedback.
            **kwargs: Additional arguments passed to parent workflow
        """
        super().__init__(
            rollout_engine=rollout_engine,
            max_iterations=max_iterations,
            share_conversation_history=True,
            use_final_outcome_reward=use_final_outcome_reward,
            **kwargs,
        )
        self.reward_function = reward_function
        self._prompts = prompts
        self._prompt_file = prompt_file

    @property
    def prompts(self) -> Dict[str, Any]:
        """Lazy load prompts from file if not provided."""
        if self._prompts is None:
            self._prompts = self._load_prompts(self._prompt_file)
        return self._prompts

    def _load_prompts(self, prompt_file: str) -> Dict[str, Any]:
        """Load prompt templates from JSON file.

        Args:
            prompt_file: Path to JSON file

        Returns:
            Dictionary of prompt templates
        """
        with open(prompt_file, "r") as f:
            data = json.load(f)
        # Try evaluator_optimizer_prompts first, fall back to multi_agent_math_prompts
        return data.get("evaluator_optimizer_prompts", data.get("multi_agent_math_prompts", {}))

    # ===== Required abstract method implementations =====

    def build_generator_prompt(self, task: Dict[str, Any]) -> str:
        """Build math problem solving prompt.

        Args:
            task: Task dictionary with 'question' field

        Returns:
            Formatted prompt asking to solve the math problem
        """
        problem = task["question"]
        template = self.prompts.get("generator_initial", {}).get(
            "template",
            "You are a math problem solver. Your task is to solve the given "
            "mathematical problem step by step. You must provide the final answer "
            "in \\boxed{{}} format.\n\nProblem: {problem}",
        )
        return template.format(problem=problem)

    def build_evaluator_prompt(
        self,
        task: Dict[str, Any],
        current_response: str,
        iteration: int,
        conversation_history: list,
    ) -> str:
        """Build math solution evaluation prompt.

        Args:
            task: Task dictionary
            current_response: The solution to evaluate
            iteration: Current iteration number
            conversation_history: Full conversation so far

        Returns:
            Formatted prompt asking for Correct/Incorrect verdict
        """
        template = self.prompts.get("evaluator_critique", {}).get(
            "template",
            "You are reviewing the solution above. Your task is to analyze it "
            "for correctness.\n\nImportant instructions:\n- Provide your verdict "
            "as \\boxed{Correct} or \\boxed{Incorrect}\n- If incorrect, briefly "
            "explain the main issue\n- Do not reveal the correct answer in your response",
        )
        return template

    def build_refinement_prompt(
        self,
        task: Dict[str, Any],
        current_response: str,
        evaluation: EvaluationResult,
        iteration: int,
        conversation_history: list,
    ) -> str:
        """Build math solution refinement prompt.

        Args:
            task: Task dictionary
            current_response: The solution that was evaluated
            evaluation: Parsed evaluation result with feedback
            iteration: Current iteration number
            conversation_history: Full conversation so far

        Returns:
            Formatted prompt asking for a new solution approach
        """
        template = self.prompts.get("generator_refinement", {}).get(
            "template",
            "You are a math problem solver creating a new solution based on the "
            "teacher's feedback above. Your task is to solve the problem using a "
            "completely different approach.\n\nImportant instructions:\n- You must "
            "provide the final answer in \\boxed{{}} format\n- Use a DIFFERENT method "
            "than your previous attempt\n- Address the issues identified in the "
            "teacher's feedback",
        )
        return template

    def parse_evaluation(self, evaluator_response: str) -> EvaluationResult:
        """Parse \\boxed{Correct/Incorrect} verdict from evaluator.

        Args:
            evaluator_response: Raw text response from evaluator

        Returns:
            EvaluationResult with is_satisfied=True if verdict is "correct"
        """
        verdict_match = re.search(
            r"\\boxed\{(Correct|Incorrect)\}",
            evaluator_response,
            re.IGNORECASE,
        )

        if verdict_match:
            verdict = verdict_match.group(1).lower()
            is_satisfied = verdict == "correct"
        else:
            verdict = "unknown"
            is_satisfied = False  # Continue refining if can't parse verdict

        return EvaluationResult(
            is_satisfied=is_satisfied,
            feedback=evaluator_response,
            verdict=verdict,
            metadata={"raw_response": evaluator_response},
        )

    def compute_generator_reward(
        self,
        task: Dict[str, Any],
        response: str,
    ) -> RewardOutput:
        """Compute reward using ground truth math evaluation.

        Args:
            task: Task dictionary with ground truth answer
            response: Generator's response

        Returns:
            RewardOutput from reward function
        """
        # Ensure ground_truth is set for the reward function
        if "ground_truth" not in task:
            task["ground_truth"] = task.get("final_answer", "")
        return self.reward_function(task, response)

    def compute_evaluator_reward(
        self,
        task: Dict[str, Any],
        evaluated_response: str,
        evaluation: EvaluationResult,
        ground_truth_correct: bool,
    ) -> RewardOutput:
        """Reward evaluator for accurate predictions.

        Evaluator receives reward 1.0 if its verdict matches the ground truth
        correctness, 0.0 otherwise.

        Args:
            task: Task dictionary
            evaluated_response: The response that was evaluated
            evaluation: Parsed evaluation result
            ground_truth_correct: Whether the evaluated response is actually correct

        Returns:
            RewardOutput based on verdict accuracy
        """
        # Evaluator is correct if verdict matches ground truth
        evaluator_correct = (
            (evaluation.verdict == "correct" and ground_truth_correct)
            or (evaluation.verdict == "incorrect" and not ground_truth_correct)
        )

        return RewardOutput(
            reward=1.0 if evaluator_correct else 0.0,
            is_correct=evaluator_correct,
            metadata={
                "verdict": evaluation.verdict,
                "ground_truth_correct": ground_truth_correct,
            },
        )

    # ===== Optional customizations =====

    def extract_response(self, model_output) -> str:
        """Extract content from model output.

        Args:
            model_output: ModelOutput from rollout engine

        Returns:
            Full content (not just boxed answer) for context
        """
        return model_output.content or model_output.text or ""
