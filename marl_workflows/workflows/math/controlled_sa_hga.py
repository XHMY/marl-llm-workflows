"""Single-role control for Math (``controlled_sa_hga``).

This workflow keeps the task semantically single-agent: there is no
aggregator, evaluator, or final-outcome reward. It creates multiple virtual
generator trajectories for the same prompt so GRPO groups stay separate by
trajectory name (``generator0..N-1``), while the trainer routes all of them to the
same "generator" LoRA adapter by stripping numeric suffixes during actor update.
Each virtual generator is rewarded on its own answer; ``use_final_outcome_reward`` is
ignored (absorbed by ``**kwargs``).

Ported from fork ``examples/math_reasoning/controlled_sa_hga_math_workflow.py``; only the
imports, the base class (``MultiAgentWorkflow``) and prompt loading (``load_prompts``) differ.
"""

import asyncio
from typing import Any

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.rewards.reward_fn import RewardFunction

from marl_workflows.workflows.base import MultiAgentWorkflow, load_prompts


class ControlledSAHGAMathWorkflow(MultiAgentWorkflow):
    """Single-agent control with Voting-like same-role adapter routing."""

    GENERATOR_NAME = "generator"

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        reward_function: RewardFunction = None,
        n_virtual_generators: int = 3,
        **kwargs,
    ):
        super().__init__(rollout_engine, **kwargs)
        self.reward_function = reward_function
        self.n_virtual_generators = n_virtual_generators
        self.prompts = None

    def _load_prompts(self) -> dict[str, Any]:
        if self.prompts is None:
            self.prompts = load_prompts("math", "multi_agent_math_prompts")
        return self.prompts

    def _build_generator_prompt(self, problem: str) -> str:
        prompts = self._load_prompts()
        if "generator_initial" not in prompts:
            raise KeyError("'generator_initial' prompt template not found")
        template = prompts["generator_initial"]["template"]
        return template.format(problem=problem)

    async def _generate_virtual(
        self,
        problem: str,
        virtual_index: int,
        uid: str = None,
    ) -> Trajectory:
        prompt_content = self._build_generator_prompt(problem)
        messages = [{"role": "user", "content": prompt_content}]

        kwargs = {"agent_name": self.GENERATOR_NAME}
        if uid is not None:
            kwargs["application_id"] = uid
        output: ModelOutput = await self.rollout_engine.get_model_response(
            messages,
            **kwargs,
        )

        response = output.content or output.text or ""
        return Trajectory(
            name=f"{self.GENERATOR_NAME}{virtual_index}",
            steps=[
                Step(
                    chat_completions=messages
                    + [
                        {
                            "role": "assistant",
                            "content": output.content,
                            "reasoning": output.reasoning,
                        }
                    ],
                    thought=output.reasoning,
                    action=response,
                    model_output=output,
                )
            ],
        )

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        self.reset(task, uid)

        problem = task["question"]
        task["ground_truth"] = task["final_answer"]

        generation_tasks = [self._generate_virtual(problem, i, uid=uid) for i in range(self.n_virtual_generators)]
        generator_trajectories = await asyncio.gather(*generation_tasks)

        correct_count = 0
        for trajectory in generator_trajectories:
            response = trajectory.steps[0].action
            reward_result = self.reward_function(task, response)
            trajectory.steps[0].reward = reward_result.reward
            trajectory.reward = reward_result.reward
            correct_count += int(reward_result.is_correct)

        any_correct = correct_count > 0
        metrics = {
            "generator_acc": correct_count / self.n_virtual_generators,
            "generator_attempts": self.n_virtual_generators,
            "success": int(any_correct),
        }

        return Episode(
            id=uid,
            task=task,
            trajectories=generator_trajectories,
            is_correct=any_correct,
            metrics=metrics,
        )
