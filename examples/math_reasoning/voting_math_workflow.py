"""Math-specific implementation of the Voting (Parallelization) workflow.

This module provides a concrete implementation of VotingWorkflow
for mathematical problem solving with parallel generation and aggregation.
"""

import json
import re
from typing import Any, Dict

from rllm.engine.rollout.rollout_engine import RolloutEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.rewards.reward_types import RewardOutput
from rllm.workflows.voting_workflow import VotingWorkflow


class VotingMathWorkflow(VotingWorkflow):
    """Math-specific voting workflow.

    This workflow implements the voting/parallelization pattern for mathematical
    problem solving:

    1. Generator: Generates N solutions in parallel, each in \\boxed{} format
    2. Aggregator: Reviews all solutions and selects the best one

    The aggregator outputs \\boxed{N} where N is the solution number to select.

    Example:
        workflow = VotingMathWorkflow(
            rollout_engine=engine,
            reward_function=math_reward_fn,
            n_votes=3,
        )
        episode = await workflow.run(task, uid)
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        reward_function: RewardFunction,
        prompts: Dict[str, Any] = None,
        prompt_file: str = "examples/math_reasoning/prompt.json",
        n_votes: int = 3,
        **kwargs,
    ):
        """Initialize the math voting workflow.

        Args:
            rollout_engine: Engine for LLM inference
            reward_function: Function to compute rewards based on ground truth
            prompts: Optional pre-loaded prompt templates
            prompt_file: Path to JSON file with prompt templates
            n_votes: Number of parallel generation attempts
            **kwargs: Additional arguments passed to parent workflow
        """
        super().__init__(
            rollout_engine=rollout_engine,
            n_votes=n_votes,
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
        # Try voting_prompts first, fall back to other prompt sections
        return data.get("voting_prompts", data.get("multi_agent_math_prompts", {}))

    # ===== Required abstract method implementations =====

    def build_generator_prompt(self, task: Dict[str, Any]) -> str:
        """Build math problem solving prompt.

        Args:
            task: Task dictionary with 'question' field

        Returns:
            Formatted prompt asking to solve the math problem
        """
        problem = task["question"]
        template = self.prompts.get("generator", {}).get(
            "template",
            "You are a math problem solver. Your task is to solve the given "
            "mathematical problem step by step. You must provide the final answer "
            "in \\boxed{{}} format.\n\nProblem: {problem}",
        )
        return template.format(problem=problem)

    def build_aggregator_prompt(
        self,
        task: Dict[str, Any],
        responses: list[str],
    ) -> str:
        """Build aggregation prompt with all solutions.

        Args:
            task: Task dictionary
            responses: List of all generated solutions

        Returns:
            Formatted prompt asking to select the best solution
        """
        problem = task["question"]

        # Build solutions section
        solutions_text = ""
        for i, response in enumerate(responses, 1):
            solutions_text += f"\n--- Solution {i} ---\n{response}\n"

        template = self.prompts.get("aggregator", {}).get(
            "template",
            "You are an expert math reviewer. Your task is to review multiple "
            "solutions to the same problem and select the best one.\n\n"
            "Problem: {problem}\n\n"
            "Solutions to review:{solutions}\n\n"
            "Instructions:\n"
            "1. Analyze each solution for correctness and completeness\n"
            "2. Check the mathematical reasoning and final answer\n"
            "3. Select the solution number that is most likely correct\n"
            "4. Output your selection as \\boxed{{N}} where N is the solution number "
            "(1, 2, 3, etc.)\n\n"
            "Your selection:",
        )

        return template.format(problem=problem, solutions=solutions_text)

    def parse_aggregator_response(
        self,
        response: str,
        candidates: list[str],
    ) -> str:
        """Parse aggregator response to get selected solution.

        Args:
            response: Raw text response from aggregator
            candidates: List of candidate solutions

        Returns:
            The selected solution text
        """
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

    def compute_aggregator_reward(
        self,
        task: Dict[str, Any],
        selected_response: str,
    ) -> RewardOutput:
        """Compute reward for aggregator based on selected answer.

        Args:
            task: Task dictionary
            selected_response: The solution selected by the aggregator

        Returns:
            RewardOutput based on whether selected answer is correct
        """
        # Ensure ground_truth is set for the reward function
        if "ground_truth" not in task:
            task["ground_truth"] = task.get("final_answer", "")
        return self.reward_function(task, selected_response)

    # ===== Optional customizations =====

    def extract_response(self, model_output) -> str:
        """Extract content from model output.

        Args:
            model_output: ModelOutput from rollout engine

        Returns:
            Full content for context
        """
        return model_output.content or model_output.text or ""
