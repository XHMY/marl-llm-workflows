import asyncio
import json
import re
from typing import Dict, Any

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine import ModelOutput, RolloutEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.workflows.workflow import Workflow


class Generator:
    """Generator agent for initial math problem solving."""

    def __init__(self, rollout_engine: RolloutEngine, prompts: Dict[str, Any], **kwargs):
        self.rollout_engine = rollout_engine
        self.prompts = prompts

    async def generate_solution(self, problem: str) -> Trajectory:
        """Generate initial solution for a math problem."""
        if "generator_initial" not in self.prompts:
            raise KeyError("'generator_initial' prompt template not found")

        template = self.prompts["generator_initial"]["template"]
        prompt_content = template.format(problem=problem)

        messages = [{"role": "user", "content": prompt_content}]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages, agent_name="generator")

        return Trajectory(
            name="generator",
            steps=[
                Step(
                    chat_completions=messages + [{
                        "role": "assistant",
                        "content": output.content,
                        "reasoning": output.reasoning
                    }],
                    thought=output.reasoning,
                    action=output.content,
                    model_output=output,
                )
            ],
        )


class SingleAgentMathWorkflow(Workflow):
    """
    Simplified single-agent baseline workflow for math reasoning.
    Uses only the Generator agent without Evaluator or Refiner.
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        reward_function: RewardFunction = None,
        **kwargs
    ):
        super().__init__(rollout_engine, **kwargs)
        self.reward_function = reward_function
        self.generator = None

    def _initialize_agent(self, prompts: Dict[str, Any]):
        """Initialize generator with prompt templates."""
        if self.generator is None:
            self.generator = Generator(self.rollout_engine, prompts)

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        """
        Execute single-agent workflow: generate solution and compute reward.

        Flow:
        1. Generator creates solution
        2. Reward computed using ground truth

        Args:
            task: Dictionary with 'question', 'final_answer'
            uid: Unique identifier for this episode

        Returns:
            Episode with single trajectory and metrics
        """
        self.reset(task, uid)

        problem = task["question"]
        task["ground_truth"] = task["final_answer"]

        # Load prompts (reuse multi-agent prompts)
        with open("examples/math_reasoning/prompt.json", "r") as f:
            prompts = json.load(f)["multi_agent_math_prompts"]

        # Initialize generator
        self._initialize_agent(prompts)

        # Generate solution
        generator_trajectory = await self.generator.generate_solution(problem)
        answer = generator_trajectory.steps[0].action

        # Compute reward
        reward_result = self.reward_function(task, answer)
        generator_trajectory.steps[0].reward = reward_result.reward

        # Build metrics
        metrics = {
            "generator_acc": float(reward_result.is_correct),
            "success": int(reward_result.is_correct),
        }

        # Return episode
        return Episode(
            id=uid,
            task=task,
            trajectories=[generator_trajectory],
            is_correct=reward_result.is_correct,
            metrics=metrics,
        )
