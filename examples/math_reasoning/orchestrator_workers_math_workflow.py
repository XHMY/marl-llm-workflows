"""Math-specific implementation of the Orchestrator-Workers workflow.

This module provides a concrete implementation of OrchestratorWorkersWorkflow
for mathematical problem solving. The orchestrator proposes distinct solution
strategies, workers each solve the full problem using their assigned strategy,
and the synthesizer compares complete solutions to pick the best answer.
"""

import json
import re
from typing import Any, Dict, List

from rllm.engine.rollout.rollout_engine import RolloutEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.rewards.reward_types import RewardOutput
from rllm.workflows.orchestrator_workers_workflow import (
    OrchestratorWorkersWorkflow,
    ProposalResult,
    SubtaskResult,
)


class OrchestratorWorkersMathWorkflow(OrchestratorWorkersWorkflow):
    """Math-specific orchestrator-workers workflow.

    1. Orchestrator proposes 2-3 distinct solution strategies
    2. Workers each solve the *full* problem using their assigned strategy
    3. Synthesizer compares complete solutions and picks the best answer

    Example:
        workflow = OrchestratorWorkersMathWorkflow(
            rollout_engine=engine,
            reward_function=math_reward_fn,
            max_subtasks=3,
        )
        episode = await workflow.run(task, uid)
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        reward_function: RewardFunction,
        prompts: Dict[str, Any] = None,
        prompt_file: str = "examples/math_reasoning/prompt.json",
        max_subtasks: int = 4,
        use_final_outcome_reward: bool = True,
        **kwargs,
    ):
        """Initialize the math orchestrator-workers workflow.

        Args:
            rollout_engine: Engine for LLM inference
            reward_function: Function to compute rewards based on ground truth
            prompts: Optional pre-loaded prompt templates
            prompt_file: Path to JSON file with prompt templates
            max_subtasks: Maximum number of strategies allowed (default: 4)
            use_final_outcome_reward: If True, assign final reward to all trajectories
            **kwargs: Additional arguments passed to parent workflow
        """
        super().__init__(
            rollout_engine=rollout_engine,
            max_subtasks=max_subtasks,
            default_execution_mode="parallel",
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
        return data.get("orchestrator_workers_prompts", {})

    # ===== Required abstract method implementations =====

    def build_proposal_prompt(self, task: Dict[str, Any], max_subtasks: int) -> str:
        """Build prompt for the orchestrator to propose solution strategies.

        Args:
            task: Task dictionary with 'question' field
            max_subtasks: Maximum number of strategies allowed

        Returns:
            Formatted prompt for the orchestrator
        """
        problem = task["question"]

        template = self.prompts.get("orchestrator_propose", {}).get(
            "template",
            "You are a math problem-solving strategist. Your task is to propose "
            "at most {max_subtasks} DIFFERENT solution strategies for the following "
            "math problem. Each strategy should represent a distinct mathematical "
            "approach.\n\n"
            "Problem: {problem}\n\n"
            "Instructions:\n"
            "1. Think about different mathematical techniques that could solve "
            "this problem\n"
            "2. Propose at most {max_subtasks} distinct approaches (e.g., "
            "algebraic manipulation, coordinate geometry, trigonometric identities, "
            "combinatorial argument, etc.)\n"
            "3. Each strategy should be a brief description of the approach, "
            "NOT a solution\n"
            "4. Do NOT solve the problem — only describe the approaches\n"
            "5. Format your response as:\n"
            "   STRATEGY 1: [description of first approach]\n"
            "   STRATEGY 2: [description of second approach]\n"
            "   (and so on, up to {max_subtasks} strategies maximum)\n\n"
            "Propose strategies now:",
        )
        return template.format(problem=problem, max_subtasks=max_subtasks)

    def build_worker_prompt(
        self,
        task: Dict[str, Any],
        subtask: str,
        subtask_id: int,
        previous_results: List[SubtaskResult],
    ) -> str:
        """Build prompt for a worker to solve the full problem using a strategy.

        Args:
            task: Original task dictionary
            subtask: Assigned strategy description
            subtask_id: Index of this strategy (0-indexed)
            previous_results: Results from previous workers (for sequential mode)

        Returns:
            Formatted prompt for the worker
        """
        problem = task["question"]

        template = self.prompts.get("worker_full_solve", {}).get(
            "template",
            "You are a math problem solver. Solve the following problem "
            "using the specified strategy.\n\n"
            "Problem: {problem}\n\n"
            "Assigned strategy: {strategy}\n\n"
            "Instructions:\n"
            "1. Follow the assigned strategy as your primary approach\n"
            "2. Solve the COMPLETE problem step by step\n"
            "3. Show your work clearly\n"
            "4. Provide the final answer in \\boxed{{}} format\n\n"
            "Solve the problem:",
        )
        return template.format(problem=problem, strategy=subtask)

    def build_synthesis_prompt(
        self,
        task: Dict[str, Any],
        proposal: ProposalResult,
        worker_results: List[SubtaskResult],
    ) -> str:
        """Build prompt for the synthesizer to compare solutions.

        Args:
            task: Original task dictionary
            proposal: The proposal result from phase 1
            worker_results: All results from workers

        Returns:
            Formatted prompt for the synthesizer
        """
        problem = task["question"]

        solutions_text = ""
        for result in worker_results:
            solutions_text += (
                f"\n--- Solution {result.subtask_id + 1} "
                f"(Strategy: {result.subtask_description}) ---\n"
                f"{result.response}\n"
            )

        template = self.prompts.get("orchestrator_compare", {}).get(
            "template",
            "You are a math problem-solving judge. Multiple solvers have each "
            "attempted to solve the same problem using different strategies. "
            "Compare their complete solutions and determine the correct "
            "answer.\n\n"
            "Original problem: {problem}\n\n"
            "Complete solutions:{solutions}\n\n"
            "Instructions:\n"
            "1. Read each complete solution carefully\n"
            "2. Check the mathematical reasoning in each solution\n"
            "3. Identify which solutions arrive at the same answer "
            "(agreement = higher confidence)\n"
            "4. If solutions disagree, determine which reasoning is correct\n"
            "5. Provide the final answer in \\boxed{{}} format\n\n"
            "Your judgment:",
        )

        return template.format(problem=problem, solutions=solutions_text)

    def parse_proposals(self, orchestrator_response: str) -> ProposalResult:
        """Parse orchestrator response to extract proposed strategies.

        Looks for ``STRATEGY N:`` patterns, falls back to numbered lists
        and line-splitting.

        Args:
            orchestrator_response: Raw text response from orchestrator

        Returns:
            ProposalResult with extracted strategies
        """
        strategies = []

        # Try to parse STRATEGY N: pattern
        primary_pattern = (
            r"STRATEGY\s*(\d+)\s*:\s*(.+?)"
            r"(?=STRATEGY\s*\d+\s*:|$)"
        )
        matches = re.findall(
            primary_pattern, orchestrator_response, re.IGNORECASE | re.DOTALL
        )

        if matches:
            sorted_matches = sorted(matches, key=lambda x: int(x[0]))
            strategies = [match[1].strip() for match in sorted_matches]
        else:
            # Try numbered list pattern (1. description)
            numbered_pattern = r"^\s*(\d+)[.)]\s*(.+?)(?=^\s*\d+[.)]|\Z)"
            matches = re.findall(
                numbered_pattern, orchestrator_response, re.MULTILINE | re.DOTALL
            )

            if matches:
                sorted_matches = sorted(matches, key=lambda x: int(x[0]))
                strategies = [match[1].strip() for match in sorted_matches]
            else:
                # Fallback: split by newlines and filter non-empty lines
                lines = [
                    line.strip()
                    for line in orchestrator_response.split("\n")
                    if line.strip()
                ]
                strategies = [line for line in lines if len(line) > 10][
                    : self.max_subtasks
                ]

        # Ensure we have at least one strategy
        if not strategies:
            strategies = [orchestrator_response.strip()]

        # Limit to max_subtasks
        strategies = strategies[: self.max_subtasks]

        return ProposalResult(
            strategies=strategies,
            label="math_propose_solve",
            execution_mode="parallel",
            metadata={"raw_response": orchestrator_response},
        )

    async def compute_final_reward(
        self,
        task: Dict[str, Any],
        final_response: str,
    ) -> RewardOutput:
        """Compute reward using ground truth math evaluation.

        Args:
            task: Task dictionary with ground truth answer
            final_response: Final synthesized response from orchestrator

        Returns:
            RewardOutput from reward function
        """
        # Ensure ground_truth is set for the reward function
        if "ground_truth" not in task:
            task["ground_truth"] = task.get("final_answer", "")
        return self.reward_function(task, final_response)

    # ===== Optional customizations =====

    def extract_response(self, model_output) -> str:
        """Extract content from model output.

        Args:
            model_output: ModelOutput from rollout engine

        Returns:
            Full content for context
        """
        return model_output.content or model_output.text or ""

    def compute_worker_reward(
        self,
        task: Dict[str, Any],
        subtask: str,
        response: str,
        subtask_id: int,
    ) -> RewardOutput:
        """Compute reward for a worker trajectory.

        Returns 0.0 and relies on final outcome reward.

        Args:
            task: Task dictionary
            subtask: The strategy description
            response: Worker's response
            subtask_id: Index of the strategy

        Returns:
            RewardOutput with 0.0 reward (relies on final outcome)
        """
        return RewardOutput(reward=0.0, is_correct=False)

    def compute_proposal_reward(
        self,
        task: Dict[str, Any],
        proposal: ProposalResult,
    ) -> RewardOutput:
        """Compute reward for the proposal trajectory.

        Returns 0.0 and relies on final outcome reward.

        Args:
            task: Task dictionary
            proposal: The parsed proposal result

        Returns:
            RewardOutput with 0.0 reward
        """
        return RewardOutput(reward=0.0, is_correct=False)
