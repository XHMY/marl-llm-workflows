"""Deepcoder Evaluator-Optimizer Workflow.

This workflow implements a 2-agent pattern for code generation:
generate -> evaluate logic -> refine, then test the final code once.

The evaluator reviews code LOGIC without running tests. Test execution happens after the
eval-opt loop completes, on the final code only.

Ported from the single-pass path of fork
``examples/deepcoder/deepcoder_evaluator_optimizer_workflow.py``; only the imports, prompt
loading (``prompts`` dict or ``load_prompts``, no ``prompt_file``), the broadcast helper and
the removed test loop differ.
"""

import re
from typing import Any

from rllm.agents.agent import Episode
from rllm.engine.rollout.rollout_engine import RolloutEngine
from rllm.rewards.reward_types import RewardOutput

from marl_workflows.rewards.code_reward import code_reward_fn
from marl_workflows.workflows.base import broadcast_final_reward, load_prompts
from marl_workflows.workflows.code import reject_test_loop
from marl_workflows.workflows.evaluator_optimizer import (
    EvaluationResult,
    EvaluatorOptimizerWorkflow,
)


class DeepcodeEvaluatorOptimizerWorkflow(EvaluatorOptimizerWorkflow):
    """Deepcoder workflow with evaluator-optimizer pattern.

    Run the eval-opt loop once, compute reward from the final code's tests.

    Reward Structure (before the optional final-outcome broadcast):
        - Generator: 1.0 if tests pass, 0.0 otherwise (assigned at end)
        - Evaluator: 1.0 if verdict matches test result, 0.0 otherwise

    Example:
        workflow = DeepcodeEvaluatorOptimizerWorkflow(
            rollout_engine=engine,
            max_iterations=2,
        )
        episode = await workflow.run(task, uid)
    """

    GENERATOR_NAME = "generator"
    EVALUATOR_NAME = "evaluator"

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        prompts: dict[str, Any] | None = None,
        max_iterations: int = 2,
        enable_test_loop: bool = False,
        max_test_rounds: int = 2,
        max_tests_to_show: int = 3,
        public_test_only: bool = False,
        use_final_outcome_reward: bool = False,
        **kwargs,
    ):
        """Initialize the Deepcoder evaluator-optimizer workflow.

        Args:
            rollout_engine: Engine for LLM inference
            prompts: Optional pre-loaded prompt templates (default: the packaged
                ``deepcoder_evaluator_optimizer_prompts`` section)
            max_iterations: Max eval-opt iterations
            enable_test_loop: Must be False (the test loop is not ported)
            max_test_rounds: Kept for config compatibility; unused
            max_tests_to_show: Kept for config compatibility; unused
            public_test_only: Kept for config compatibility; unused
            **kwargs: Additional arguments passed to parent workflow
        """
        reject_test_loop(enable_test_loop)
        super().__init__(
            rollout_engine=rollout_engine,
            max_iterations=max_iterations,
            share_conversation_history=True,
            use_final_outcome_reward=use_final_outcome_reward,
            **kwargs,
        )
        self._prompts = prompts

        self.enable_test_loop = enable_test_loop
        self.max_test_rounds = max_test_rounds
        self.max_tests_to_show = max_tests_to_show
        self.public_test_only = public_test_only

        # Track test results for reward computation
        self._test_passed = False
        self._current_test_feedback = None

    @property
    def prompts(self) -> dict[str, Any]:
        """Lazy load prompts if not provided."""
        if self._prompts is None:
            self._prompts = load_prompts("deepcoder", "deepcoder_evaluator_optimizer_prompts")
        return self._prompts

    # ===== Abstract method implementations =====

    def build_generator_prompt(self, task: dict[str, Any]) -> str:
        """Build initial code generation prompt."""
        problem = task["question"]

        if self._current_test_feedback:
            # We have test feedback from a previous round
            template = self.prompts.get("generator_with_test_feedback", {}).get(
                "template",
                "Your previous solution failed. Fix it based on:\n{test_feedback}",
            )
            return template.format(test_feedback=self._current_test_feedback)
        else:
            # Initial generation
            template = self.prompts.get("generator_initial", {}).get(
                "template",
                "Solve this programming problem:\n{problem}",
            )
            return template.format(problem=problem)

    def build_evaluator_prompt(
        self,
        task: dict[str, Any],
        current_response: str,
        iteration: int,
        conversation_history: list,
    ) -> str:
        """Build code evaluation prompt for logic review."""
        template = self.prompts.get("evaluator_critique", {}).get(
            "template",
            "Review the code above for logical correctness.\nOutput \\boxed{Correct} or \\boxed{Incorrect}.",
        )
        return template

    def build_refinement_prompt(
        self,
        task: dict[str, Any],
        current_response: str,
        evaluation: EvaluationResult,
        iteration: int,
        conversation_history: list,
    ) -> str:
        """Build code refinement prompt."""
        template = self.prompts.get("generator_refinement", {}).get(
            "template",
            "Based on the feedback, create an improved solution.",
        )
        return template

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
            is_satisfied = False  # Continue refining if can't parse

        return EvaluationResult(
            is_satisfied=is_satisfied,
            feedback=evaluator_response,
            verdict=verdict,
            metadata={"raw_response": evaluator_response},
        )

    def compute_generator_reward(
        self,
        task: dict[str, Any],
        response: str,
    ) -> RewardOutput:
        """Compute generator reward based on test execution."""
        return code_reward_fn(task, response)

    def compute_evaluator_reward(
        self,
        task: dict[str, Any],
        evaluated_response: str,
        evaluation: EvaluationResult,
        ground_truth_correct: bool,
    ) -> RewardOutput:
        """Compute evaluator reward based on prediction accuracy.

        Evaluator is rewarded for correctly predicting whether the code
        will pass tests:
        - Says "correct" and tests pass -> 1.0
        - Says "incorrect" and tests fail -> 1.0
        - Otherwise -> 0.0
        """
        evaluator_prediction_correct = (evaluation.verdict == "correct" and ground_truth_correct) or (
            evaluation.verdict == "incorrect" and not ground_truth_correct
        )

        return RewardOutput(
            reward=1.0 if evaluator_prediction_correct else 0.0,
            is_correct=evaluator_prediction_correct,
            metadata={
                "verdict": evaluation.verdict,
                "ground_truth_correct": ground_truth_correct,
            },
        )

    # ===== Override run() =====

    async def run(self, task: dict[str, Any], uid: str, **kwargs) -> Episode:
        """Execute the evaluator-optimizer workflow.

        - Run single eval-opt cycle
        - Compute reward from test execution

        Args:
            task: Task dictionary containing problem information
            uid: Unique identifier for this episode

        Returns:
            Episode with all generator and evaluator trajectories
        """
        self.reset(task, uid)

        return await self._run_single_pass(task, uid)

    async def _run_single_pass(
        self,
        task: dict[str, Any],
        uid: str,
    ) -> Episode:
        """Execute single-pass eval-opt without test loop."""
        self._test_passed = False
        self._current_test_feedback = None

        all_trajectories = []

        # Metrics tracking
        total_generator_attempts = 0
        total_evaluator_predictions = 0

        # Reset conversation history
        conversation_history = []

        # Step 1: Generator creates initial response
        gen_prompt = self.build_generator_prompt(task)
        gen_messages = [{"role": "user", "content": gen_prompt}]

        gen_output = await self.rollout_engine.get_model_response(
            gen_messages,
            agent_name=self.GENERATOR_NAME,
        )

        current_response = self.extract_response(gen_output)

        # Create trajectory
        gen_trajectory = self._create_trajectory(
            name=self.GENERATOR_NAME,
            messages=gen_messages,
            model_output=gen_output,
            response=current_response,
            reward=0.0,  # Placeholder
        )
        all_trajectories.append(gen_trajectory)
        total_generator_attempts += 1

        # Build conversation history
        conversation_history.extend(gen_messages)
        conversation_history.append(
            {
                "role": "assistant",
                "content": gen_output.content,
                "reasoning": gen_output.reasoning,
            }
        )

        # Inner eval-opt loop
        for iteration in range(self.max_iterations):
            # Step 2: Evaluator reviews current code (logic only)
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
            )

            evaluation = self.parse_evaluation(eval_output.content)

            # Evaluator trajectory
            eval_trajectory = self._create_trajectory(
                name=self.EVALUATOR_NAME,
                messages=eval_messages,
                model_output=eval_output,
                response=eval_output.content,
                reward=0.0,  # Placeholder
                action={
                    "verdict": evaluation.verdict,
                    "feedback": evaluation.feedback,
                },
            )
            eval_trajectory.metadata = {
                "iteration": iteration,
                "verdict": evaluation.verdict,
            }
            all_trajectories.append(eval_trajectory)
            total_evaluator_predictions += 1

            # Update conversation history
            conversation_history.append(
                {
                    "role": "assistant",
                    "content": eval_output.content,
                }
            )

            # Check if evaluator is satisfied
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
            )

            current_response = self.extract_response(refine_output)

            refine_trajectory = self._create_trajectory(
                name=self.GENERATOR_NAME,
                messages=refine_messages,
                model_output=refine_output,
                response=current_response,
                reward=0.0,  # Placeholder
            )
            all_trajectories.append(refine_trajectory)
            total_generator_attempts += 1

            # Update conversation history
            conversation_history.append(
                {
                    "role": "assistant",
                    "content": refine_output.content,
                    "reasoning": refine_output.reasoning,
                }
            )

        # Compute reward from test execution (run in thread pool for parallelism)
        reward_output = await self.run_in_code_executor(code_reward_fn, task, current_response)
        test_passed = reward_output.is_correct

        # Assign rewards
        evaluator_correct_count = 0

        for traj in all_trajectories:
            if traj.name == self.GENERATOR_NAME:
                reward = 1.0 if test_passed else 0.0
                traj.reward = reward
                for step in traj.steps:
                    step.reward = reward

            elif traj.name == self.EVALUATOR_NAME:
                metadata = getattr(traj, "metadata", {}) or {}
                verdict = metadata.get("verdict", "unknown")

                evaluator_correct = (verdict == "correct" and test_passed) or (
                    verdict == "incorrect" and not test_passed
                )
                reward = 1.0 if evaluator_correct else 0.0
                traj.reward = reward
                for step in traj.steps:
                    step.reward = reward
                if evaluator_correct:
                    evaluator_correct_count += 1

        # If use_final_outcome_reward is enabled, propagate the final reward
        # to all trajectories (both generator and evaluator, all iterations)
        if self.use_final_outcome_reward:
            broadcast_final_reward(all_trajectories, reward_output.reward)

        # Compute metrics
        reward_metadata = reward_output.metadata or {}
        passed_tests = reward_metadata.get("passed_tests", 0)
        total_tests = reward_metadata.get("total_tests", 0)

        metrics = {
            f"{self.EVALUATOR_NAME}_acc": (
                evaluator_correct_count / total_evaluator_predictions if total_evaluator_predictions > 0 else 0.0
            ),
            "test_rounds": 1,
            "test_passed": int(test_passed),
            "total_iterations": total_evaluator_predictions,
            "success": int(test_passed),
            f"{self.GENERATOR_NAME}_attempts": total_generator_attempts,
            f"{self.EVALUATOR_NAME}_predictions": total_evaluator_predictions,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "pass_rate": (passed_tests / total_tests if total_tests > 0 else 0.0),
        }

        return Episode(
            id=uid,
            task=task,
            trajectories=all_trajectories,
            is_correct=test_passed,
            metrics=metrics,
        )

    def extract_response(self, model_output) -> str:
        """Extract code content from model output."""
        content = model_output.content or model_output.text or ""
        return content
