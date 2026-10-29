"""Prompt-heterogeneity ablation of VotingMathWorkflow (per-voter prompts control).

This workflow ablates same-prompt density by injecting a per-vote-index reasoning-style
suffix into the generator prompt, while keeping one shared generator adapter and the
broadcast outcome reward (`use_final_outcome_reward=true`) intact.

With `n_votes=3` the three distinct styles map one-to-one onto vote indices.
Larger `n_votes` rotate through the same list.

Ported from fork ``examples/math_reasoning/voting_math_prompt_het_workflow.py``; only the
imports differ. ``STYLE_SUFFIXES`` is verbatim.
"""

from typing import Any

from rllm.agents.agent import Step, Trajectory

from marl_workflows.workflows.math.voting import VotingMathWorkflow

STYLE_SUFFIXES = [
    (
        "\n\nApproach: algebraic manipulation. Set up equations, isolate "
        "variables, and simplify expressions step by step."
    ),
    (
        "\n\nApproach: geometric / coordinate reasoning. Visualize the "
        "problem, place coordinates or draw a diagram, and argue from "
        "geometric relationships."
    ),
    (
        "\n\nApproach: case analysis and small-case verification. Try "
        "small instances first, enumerate cases, and generalize the pattern."
    ),
]


class VotingMathPromptHetWorkflow(VotingMathWorkflow):
    """Voting Math workflow with per-vote-index prompt style suffixes."""

    STYLE_SUFFIXES = STYLE_SUFFIXES

    def _style_for_index(self, vote_index: int) -> str:
        return self.STYLE_SUFFIXES[vote_index % len(self.STYLE_SUFFIXES)]

    async def _generate_single(
        self,
        task: dict[str, Any],
        vote_index: int,
        uid: str = None,
    ) -> Trajectory:
        prompt = self.build_generator_prompt(task) + self._style_for_index(vote_index)
        messages = [{"role": "user", "content": prompt}]

        kwargs = {"agent_name": self.GENERATOR_NAME}
        if uid is not None:
            kwargs["application_id"] = uid
        output = await self.rollout_engine.get_model_response(
            messages,
            **kwargs,
        )

        response = self.extract_response(output)

        trajectory = Trajectory(
            name=f"{self.GENERATOR_NAME}{vote_index}",
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

        return trajectory
