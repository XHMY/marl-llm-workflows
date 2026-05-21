"""Math-specific implementation of the Evaluator-Optimizer V2 workflow.

This module provides a concrete implementation of EvaluatorOptimizerV2Workflow
for mathematical problem solving with proper multi-turn conversation structure.

Key differences from V1 (evaluator_optimizer_math_workflow.py):
- Evaluator prompt is self-contained: embeds problem + solution directly
- Evaluator focuses on pinpointing reasoning flaws, not solving the problem
- Refinement prompt embeds reviewer feedback and encourages incremental fixes
- Generator reuses correct reasoning from prior turns instead of re-deriving
"""

import json
import re
from typing import Any, Dict

from rllm.engine.rollout.rollout_engine import RolloutEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.rewards.reward_types import RewardOutput
from rllm.workflows.evaluator_optimizer_workflow import EvaluationResult
from rllm.workflows.evaluator_optimizer_v2_workflow import (
    EvaluatorOptimizerV2Workflow,
)


class EvaluatorOptimizerV2MathWorkflow(EvaluatorOptimizerV2Workflow):
    """Math-specific evaluator-optimizer V2 workflow.

    Uses multi-turn conversation structure where:
    - Generator sees its prior solutions as assistant turns and reviewer
      feedback as user turns, enabling incremental refinement
    - Evaluator sees problem+solution embedded in each user turn and
      pinpoints reasoning flaws without solving the problem itself
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
        """Initialize the math evaluator-optimizer V2 workflow.

        Args:
            rollout_engine: Engine for LLM inference
            reward_function: Function to compute rewards based on ground truth
            prompts: Optional pre-loaded prompt templates
            prompt_file: Path to JSON file with prompt templates
            max_iterations: Maximum evaluation-refinement cycles
            use_final_outcome_reward: If True, assign the final outcome reward to
                ALL trajectories.
            **kwargs: Additional arguments passed to parent workflow
        """
        super().__init__(
            rollout_engine=rollout_engine,
            max_iterations=max_iterations,
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
        """Load prompt templates from JSON file."""
        with open(prompt_file, "r") as f:
            data = json.load(f)
        return data.get("evaluator_optimizer_v2_prompts", {})

    # ===== Required abstract method implementations =====

    def build_generator_prompt(self, task: Dict[str, Any]) -> str:
        """Build math problem solving prompt."""
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
        """Build self-contained evaluation prompt with problem and solution embedded.

        The evaluator's conversation history only contains its own prior evaluations,
        so the problem and current solution must be embedded directly in the prompt.
        """
        problem = task["question"]
        template = self.prompts.get("evaluator_critique", {}).get(
            "template",
            "You are a reviewer checking a math solution for correctness.\n\n"
            "Problem: {problem}\n\n"
            "Solution to review:\n{solution}\n\n"
            "Instructions:\n"
            "- Check the reasoning for correctness\n"
            "- Provide your verdict as \\boxed{{Correct}} or \\boxed{{Incorrect}}\n"
            "- If incorrect, point out where the reasoning goes wrong and explain "
            "why it is flawed\n"
            "- Do NOT solve the problem yourself or suggest the correct approach\n"
            "- Do NOT reveal the correct answer",
        )
        return template.format(problem=problem, solution=current_response)

    def build_refinement_prompt(
        self,
        task: Dict[str, Any],
        current_response: str,
        evaluation: EvaluationResult,
        iteration: int,
        conversation_history: list,
    ) -> str:
        """Build refinement prompt embedding reviewer feedback.

        The generator's prior solutions are visible as assistant turns in its
        conversation history. This prompt embeds the feedback and instructs the
        model to fix the identified issue incrementally.
        """
        template = self.prompts.get("generator_refinement", {}).get(
            "template",
            "A reviewer identified an issue in your solution:\n\n"
            "{feedback}\n\n"
            "Revise your solution to fix the identified issue.\n\n"
            "Instructions:\n"
            "- Your previous reasoning is shown above in the conversation\n"
            "- Keep the parts that are correct — do not re-derive them, refer to "
            'them briefly (e.g. "As shown in my previous reasoning, ...")\n'
            "- Focus on correcting the flawed part and adjusting any reasoning "
            "that depends on it\n"
            "- Provide the final answer in \\boxed{{}} format",
        )
        return template.format(feedback=evaluation.feedback)

    def parse_evaluation(self, evaluator_response: str) -> EvaluationResult:
        """Parse \\boxed{Correct/Incorrect} verdict from evaluator."""
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
            is_satisfied = False

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
        """Compute reward using ground truth math evaluation."""
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
        """Reward evaluator for accurate predictions."""
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

    def extract_response(self, model_output) -> str:
        """Extract content from model output."""
        return model_output.content or model_output.text or ""
