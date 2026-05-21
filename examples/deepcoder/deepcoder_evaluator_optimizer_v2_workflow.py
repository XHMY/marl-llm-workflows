"""Deepcoder Evaluator-Optimizer V2 Workflow with proper multi-turn conversation.

This workflow implements a 2-agent pattern for code generation with:
- Inner loop: Evaluator-optimizer (generate -> evaluate logic -> refine)
- Outer loop (optional): Test execution (run tests -> if fail, new eval-opt round)

Key differences from V1 (deepcoder_evaluator_optimizer_workflow.py):
- Agent-specific conversation histories (no shared history mixing both agents)
- Proper user/assistant alternation (no consecutive assistant messages)
- Evaluator prompt is self-contained: embeds problem + code directly
- Refinement prompt embeds reviewer feedback, encourages incremental fixes
- Generator sees its prior code as assistant turns (can reference previous logic)
"""

import json
import re
from typing import Any, Dict

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine.rollout.rollout_engine import RolloutEngine
from rllm.rewards.reward_fn import code_reward_fn
from rllm.rewards.reward_types import RewardOutput
from rllm.workflows.code_test_loop_mixin import CodeTestLoopMixin
from rllm.workflows.evaluator_optimizer_workflow import EvaluationResult
from rllm.workflows.evaluator_optimizer_v2_workflow import (
    EvaluatorOptimizerV2Workflow,
)


class DeepcodeEvaluatorOptimizerV2Workflow(CodeTestLoopMixin, EvaluatorOptimizerV2Workflow):
    """Deepcoder V2 workflow with evaluator-optimizer pattern.

    Supports two modes:
    - Single-pass (enable_test_loop=False, default):
      Run eval-opt loop once, compute reward from final code test
    - Test loop (enable_test_loop=True):
      Run eval-opt loop, test final code, if fail start new round with feedback

    Uses agent-specific multi-turn conversation histories:
    - Generator: [user(problem), asst(code), user(feedback), asst(fix), ...]
    - Evaluator: [user(problem+code), asst(verdict), user(problem+revised_code), ...]
    """

    GENERATOR_NAME = "generator"
    EVALUATOR_NAME = "evaluator"

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        prompts: Dict[str, Any] = None,
        prompt_file: str = "examples/deepcoder/prompt.json",
        max_iterations: int = 2,
        enable_test_loop: bool = False,
        max_test_rounds: int = 2,
        max_tests_to_show: int = 3,
        public_test_only: bool = False,
        use_final_outcome_reward: bool = False,
        **kwargs,
    ):
        """Initialize the Deepcoder evaluator-optimizer V2 workflow.

        Args:
            rollout_engine: Engine for LLM inference
            prompts: Optional pre-loaded prompt templates
            prompt_file: Path to JSON file with prompt templates
            max_iterations: Max eval-opt iterations per test round (inner loop)
            enable_test_loop: Whether to enable test-based refinement (outer loop)
            max_test_rounds: Max test execution rounds (when test loop enabled)
            max_tests_to_show: Max failed tests to include in feedback
            public_test_only: Whether to only show public tests in feedback
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
        self._prompts = prompts
        self._prompt_file = prompt_file

        # CodeTestLoopMixin configuration
        self.enable_test_loop = enable_test_loop
        self.max_test_rounds = max_test_rounds
        self.max_tests_to_show = max_tests_to_show
        self.public_test_only = public_test_only

        # Track test results for reward computation
        self._test_passed = False
        self._current_test_feedback = None

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
        return data.get("deepcoder_evaluator_optimizer_v2_prompts", {})

    # ===== Prompt builders =====

    def build_generator_prompt(self, task: Dict[str, Any]) -> str:
        """Build initial code generation prompt."""
        problem = task["question"]

        if self._current_test_feedback:
            template = self.prompts.get("generator_with_test_feedback", {}).get(
                "template",
                "Your previous solution failed. Fix it based on:\n{test_feedback}",
            )
            return template.format(test_feedback=self._current_test_feedback)
        else:
            template = self.prompts.get("generator_initial", {}).get(
                "template",
                "Solve this programming problem:\n{problem}",
            )
            return template.format(problem=problem)

    def build_evaluator_prompt(
        self,
        task: Dict[str, Any],
        current_response: str,
        iteration: int,
        conversation_history: list,
    ) -> str:
        """Build self-contained code evaluation prompt with problem and code embedded."""
        problem = task["question"]
        template = self.prompts.get("evaluator_critique", {}).get(
            "template",
            "You are a code reviewer checking a solution for correctness.\n\n"
            "Problem:\n{problem}\n\n"
            "Code to review:\n{solution}\n\n"
            "Instructions:\n"
            "- Check the code logic for correctness\n"
            "- Provide your verdict as \\boxed{{Correct}} or \\boxed{{Incorrect}}\n"
            "- If incorrect, point out where the logic goes wrong\n"
            "- Do NOT solve the problem yourself or suggest the correct approach",
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
        """Build refinement prompt embedding reviewer feedback."""
        template = self.prompts.get("generator_refinement", {}).get(
            "template",
            "A reviewer identified an issue in your code:\n\n"
            "{feedback}\n\n"
            "Revise your solution to fix the identified issue.\n\n"
            "Instructions:\n"
            "- Your previous code is shown above in the conversation\n"
            "- Keep the parts that are correct\n"
            "- Focus on correcting the flawed part\n"
            "- Output your code in a markdown code block with ```python",
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
        """Compute generator reward based on test execution."""
        return code_reward_fn(task, response)

    def compute_evaluator_reward(
        self,
        task: Dict[str, Any],
        evaluated_response: str,
        evaluation: EvaluationResult,
        ground_truth_correct: bool,
    ) -> RewardOutput:
        """Compute evaluator reward based on prediction accuracy."""
        evaluator_prediction_correct = (
            evaluation.verdict == "correct" and ground_truth_correct
        ) or (evaluation.verdict == "incorrect" and not ground_truth_correct)

        return RewardOutput(
            reward=1.0 if evaluator_prediction_correct else 0.0,
            is_correct=evaluator_prediction_correct,
            metadata={
                "verdict": evaluation.verdict,
                "ground_truth_correct": ground_truth_correct,
            },
        )

    def extract_response(self, model_output) -> str:
        """Extract code content from model output."""
        return model_output.content or model_output.text or ""

    # ===== Override run() for mode selection =====

    async def run(self, task: Dict[str, Any], uid: str, **kwargs) -> Episode:
        """Execute the evaluator-optimizer V2 workflow.

        Dispatches between single-pass and test-loop modes.
        Both use agent-specific multi-turn conversation histories.
        """
        self.reset(task, uid)

        if self.enable_test_loop:
            return await self._run_with_test_loop(task, uid)
        else:
            return await self._run_single_pass(task, uid)

    async def _run_single_pass(
        self,
        task: Dict[str, Any],
        uid: str,
    ) -> Episode:
        """Execute single-pass eval-opt with V2 multi-turn structure."""
        self._test_passed = False
        self._current_test_feedback = None

        all_trajectories = []
        total_generator_attempts = 0
        total_evaluator_predictions = 0

        # Agent-specific conversation histories
        gen_conversation = []
        eval_conversation = []

        # Step 1: Generator creates initial response
        gen_prompt = self.build_generator_prompt(task)
        gen_messages = [{"role": "user", "content": gen_prompt}]

        gen_output = await self.rollout_engine.get_model_response(
            gen_messages,
            agent_name=self.GENERATOR_NAME,
        )

        current_response = self.extract_response(gen_output)

        gen_trajectory = self._create_trajectory(
            name=self.GENERATOR_NAME,
            messages=gen_messages,
            model_output=gen_output,
            response=current_response,
            reward=0.0,  # Deferred until test execution
        )
        all_trajectories.append(gen_trajectory)
        total_generator_attempts += 1

        # Build generator's conversation history
        gen_conversation = [
            {"role": "user", "content": gen_prompt},
            {"role": "assistant", "content": gen_output.content, "reasoning": gen_output.reasoning},
        ]

        # Inner eval-opt loop
        for iteration in range(self.max_iterations):
            # Step 2: Evaluator reviews current code
            eval_prompt = self.build_evaluator_prompt(
                task, current_response, iteration, eval_conversation,
            )

            eval_messages = eval_conversation + [{"role": "user", "content": eval_prompt}]

            eval_output = await self.rollout_engine.get_model_response(
                eval_messages,
                agent_name=self.EVALUATOR_NAME,
            )

            evaluation = self.parse_evaluation(eval_output.content)

            eval_trajectory = self._create_trajectory(
                name=self.EVALUATOR_NAME,
                messages=eval_messages,
                model_output=eval_output,
                response=eval_output.content,
                reward=0.0,  # Deferred
                action={"verdict": evaluation.verdict, "feedback": evaluation.feedback},
            )
            eval_trajectory.metadata = {
                "iteration": iteration,
                "verdict": evaluation.verdict,
            }
            all_trajectories.append(eval_trajectory)
            total_evaluator_predictions += 1

            # Update evaluator's conversation history
            eval_conversation.extend([
                {"role": "user", "content": eval_prompt},
                {"role": "assistant", "content": eval_output.content},
            ])

            # Check if evaluator is satisfied
            if not self.should_continue(evaluation, iteration, task):
                break

            # Step 3: Generator refines based on feedback
            refine_prompt = self.build_refinement_prompt(
                task, current_response, evaluation, iteration, gen_conversation,
            )

            refine_messages = gen_conversation + [{"role": "user", "content": refine_prompt}]

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
                reward=0.0,  # Deferred
            )
            all_trajectories.append(refine_trajectory)
            total_generator_attempts += 1

            # Update generator's conversation history
            gen_conversation.extend([
                {"role": "user", "content": refine_prompt},
                {"role": "assistant", "content": refine_output.content, "reasoning": refine_output.reasoning},
            ])

        # Compute reward from test execution
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

        if self.use_final_outcome_reward:
            final_reward_value = reward_output.reward
            for trajectory in all_trajectories:
                trajectory.reward = final_reward_value
                for step in trajectory.steps:
                    step.reward = final_reward_value

        # Compute metrics
        reward_metadata = reward_output.metadata or {}
        passed_tests = reward_metadata.get("passed_tests", 0)
        total_tests = reward_metadata.get("total_tests", 0)

        metrics = {
            f"{self.EVALUATOR_NAME}_acc": (
                evaluator_correct_count / total_evaluator_predictions
                if total_evaluator_predictions > 0
                else 0.0
            ),
            "test_rounds": 1,
            "test_passed": int(test_passed),
            "total_iterations": total_evaluator_predictions,
            "success": int(test_passed),
            f"{self.GENERATOR_NAME}_attempts": total_generator_attempts,
            f"{self.EVALUATOR_NAME}_predictions": total_evaluator_predictions,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "pass_rate": (
                passed_tests / total_tests if total_tests > 0 else 0.0
            ),
        }

        return Episode(
            id=uid,
            task=task,
            trajectories=all_trajectories,
            is_correct=test_passed,
            metrics=metrics,
        )

    async def _run_with_test_loop(
        self,
        task: Dict[str, Any],
        uid: str,
    ) -> Episode:
        """Execute eval-opt workflow with test-based refinement loop."""
        all_trajectories = []
        self._test_passed = False
        self._current_test_feedback = None

        final_test_result = None

        total_generator_attempts = 0
        total_evaluator_predictions = 0
        evaluator_correct_count = 0

        for test_round in range(self.max_test_rounds):
            # Reset agent-specific conversation histories for new test round
            gen_conversation = []
            eval_conversation = []

            # Step 1: Generator creates initial response (or refines from test feedback)
            gen_prompt = self.build_generator_prompt(task)
            gen_messages = [{"role": "user", "content": gen_prompt}]

            gen_output = await self.rollout_engine.get_model_response(
                gen_messages,
                agent_name=self.GENERATOR_NAME,
            )

            current_response = self.extract_response(gen_output)

            gen_trajectory = self._create_trajectory(
                name=self.GENERATOR_NAME,
                messages=gen_messages,
                model_output=gen_output,
                response=current_response,
                reward=0.0,  # Deferred
            )
            all_trajectories.append(gen_trajectory)
            total_generator_attempts += 1

            # Build generator's conversation history
            gen_conversation = [
                {"role": "user", "content": gen_prompt},
                {"role": "assistant", "content": gen_output.content, "reasoning": gen_output.reasoning},
            ]

            # Inner eval-opt loop
            for iteration in range(self.max_iterations):
                # Step 2: Evaluator reviews current code
                eval_prompt = self.build_evaluator_prompt(
                    task, current_response, iteration, eval_conversation,
                )

                eval_messages = eval_conversation + [{"role": "user", "content": eval_prompt}]

                eval_output = await self.rollout_engine.get_model_response(
                    eval_messages,
                    agent_name=self.EVALUATOR_NAME,
                )

                evaluation = self.parse_evaluation(eval_output.content)

                eval_trajectory = self._create_trajectory(
                    name=self.EVALUATOR_NAME,
                    messages=eval_messages,
                    model_output=eval_output,
                    response=eval_output.content,
                    reward=0.0,  # Deferred
                    action={"verdict": evaluation.verdict, "feedback": evaluation.feedback},
                )
                eval_trajectory.metadata = {
                    "iteration": iteration,
                    "test_round": test_round,
                    "verdict": evaluation.verdict,
                }
                all_trajectories.append(eval_trajectory)
                total_evaluator_predictions += 1

                # Update evaluator's conversation history
                eval_conversation.extend([
                    {"role": "user", "content": eval_prompt},
                    {"role": "assistant", "content": eval_output.content},
                ])

                # Check if evaluator is satisfied
                if not self.should_continue(evaluation, iteration, task):
                    break

                # Step 3: Generator refines based on feedback
                refine_prompt = self.build_refinement_prompt(
                    task, current_response, evaluation, iteration, gen_conversation,
                )

                refine_messages = gen_conversation + [{"role": "user", "content": refine_prompt}]

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
                    reward=0.0,  # Deferred
                )
                all_trajectories.append(refine_trajectory)
                total_generator_attempts += 1

                # Update generator's conversation history
                gen_conversation.extend([
                    {"role": "user", "content": refine_prompt},
                    {"role": "assistant", "content": refine_output.content, "reasoning": refine_output.reasoning},
                ])

            # End of inner eval-opt loop — run tests
            test_result = await self.run_tests(task, current_response)
            final_test_result = test_result

            if test_result.all_passed:
                self._test_passed = True
                break
            else:
                self._current_test_feedback = test_result.feedback

        # Assign rewards based on test results
        test_passed = final_test_result.all_passed if final_test_result else False

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

        if self.use_final_outcome_reward:
            final_reward = 1.0 if test_passed else 0.0
            for trajectory in all_trajectories:
                trajectory.reward = final_reward
                for step in trajectory.steps:
                    step.reward = final_reward

        # Compute metrics
        metrics = {
            f"{self.EVALUATOR_NAME}_acc": (
                evaluator_correct_count / total_evaluator_predictions
                if total_evaluator_predictions > 0
                else 0.0
            ),
            "test_rounds": test_round + 1,
            "test_passed": int(test_passed),
            "total_iterations": total_evaluator_predictions,
            "success": int(test_passed),
            f"{self.GENERATOR_NAME}_attempts": total_generator_attempts,
            f"{self.EVALUATOR_NAME}_predictions": total_evaluator_predictions,
        }

        if final_test_result:
            metrics["passed_tests"] = final_test_result.passed_tests
            metrics["total_tests"] = final_test_result.total_tests
            metrics["pass_rate"] = (
                final_test_result.passed_tests / final_test_result.total_tests
                if final_test_result.total_tests > 0
                else 0.0
            )

        return Episode(
            id=uid,
            task=task,
            trajectories=all_trajectories,
            is_correct=test_passed,
            metrics=metrics,
        )
