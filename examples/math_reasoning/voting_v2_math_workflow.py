"""Math-specific implementation of the Voting V2 (Per-Voter Policy) workflow.

This module provides a concrete implementation of VotingV2Workflow
for mathematical problem solving where each voter uses a distinct policy
(LoRA adapter in multi-LoRA mode).
"""

import json
import re
from typing import Any, Dict, List, Optional

from rllm.engine.rollout.rollout_engine import RolloutEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.rewards.reward_types import RewardOutput
from rllm.workflows.voting_v2_workflow import VotingV2Workflow


class VotingV2MathWorkflow(VotingV2Workflow):
    """Math-specific voting v2 workflow with per-voter policies.

    Each voter (voterA, voterB, voterC, ...) uses a different LoRA adapter
    to solve the same math problem, then the aggregator selects the best.

    Example:
        workflow = VotingV2MathWorkflow(
            rollout_engine=engine,
            reward_function=math_reward_fn,
            n_votes=3,
            # voter_names auto-generated as ['voterA', 'voterB', 'voterC']
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
        voter_names: Optional[List[str]] = None,
        **kwargs,
    ):
        """Initialize the math voting v2 workflow.

        Args:
            rollout_engine: Engine for LLM inference
            reward_function: Function to compute rewards based on ground truth
            prompts: Optional pre-loaded prompt templates
            prompt_file: Path to JSON file with prompt templates
            n_votes: Number of parallel voters
            voter_names: Explicit voter names (auto-generated if None)
            **kwargs: Additional arguments passed to parent workflow
        """
        super().__init__(
            rollout_engine=rollout_engine,
            n_votes=n_votes,
            voter_names=voter_names,
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
        return data.get("voting_prompts", data.get("multi_agent_math_prompts", {}))

    # ===== Required abstract method implementations =====

    def build_generator_prompt(self, task: Dict[str, Any]) -> str:
        """Build math problem solving prompt."""
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
        """Build aggregation prompt with all solutions."""
        problem = task["question"]

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

    def compute_generator_reward(
        self,
        task: Dict[str, Any],
        response: str,
    ) -> RewardOutput:
        """Compute reward using ground truth math evaluation."""
        if "ground_truth" not in task:
            task["ground_truth"] = task.get("final_answer", "")
        return self.reward_function(task, response)

    def compute_aggregator_reward(
        self,
        task: Dict[str, Any],
        selected_response: str,
    ) -> RewardOutput:
        """Compute reward for aggregator based on selected answer."""
        if "ground_truth" not in task:
            task["ground_truth"] = task.get("final_answer", "")
        return self.reward_function(task, selected_response)

    # ===== Optional customizations =====

    def has_parseable_answer(self, response: str) -> bool:
        r"""Return True if the response contains a \boxed{...} expression.

        A voter that produced no boxed answer (e.g., truncated response) is not
        penalized for being wrong — only voters with an explicit wrong answer are.
        """
        return bool(re.search(r"\\boxed\{", response))

    def extract_response(self, model_output) -> str:
        """Extract content from model output."""
        return model_output.content or model_output.text or ""
