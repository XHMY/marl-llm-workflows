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


class Evaluator:
    """Evaluator agent for critiquing solutions."""

    def __init__(self, rollout_engine: RolloutEngine, prompts: Dict[str, Any], **kwargs):
        self.rollout_engine = rollout_engine
        self.prompts = prompts

    async def evaluate_solution(self, problem: str, solution_content: str, conversation_history: list) -> Trajectory:
        """Evaluate a solution and provide critique."""
        if "evaluator_critique" not in self.prompts:
            raise KeyError("'evaluator_critique' prompt template not found")

        template = self.prompts["evaluator_critique"]["template"]
        critique_prompt = template

        # Build message history: original problem, solution, then critique request
        messages = conversation_history + [{"role": "user", "content": critique_prompt}]

        output: ModelOutput = await self.rollout_engine.get_model_response(messages, agent_name="evaluator")

        return Trajectory(
            name="evaluator",
            steps=[
                Step(
                    chat_completions=messages + [{
                        "role": "assistant",
                        "content": output.content,
                        "reasoning": output.reasoning
                    }],
                    thought=output.reasoning,
                    action=self._parse_verdict(output.content),
                    model_output=output,
                )
            ],
        )

    def _parse_verdict(self, response: str) -> Dict[str, Any]:
        """Parse verdict from \\boxed{Correct/Incorrect} format."""
        verdict_match = re.search(r"\\boxed\{(Correct|Incorrect)\}", response, re.IGNORECASE)
        verdict = "unknown"
        if verdict_match:
            verdict = verdict_match.group(1).lower()

        return {
            "verdict": verdict,
            "feedback": response
        }


class Refiner:
    """Refiner agent for generating improved solutions."""

    def __init__(self, rollout_engine: RolloutEngine, prompts: Dict[str, Any], **kwargs):
        self.rollout_engine = rollout_engine
        self.prompts = prompts

    async def refine_solution(self, conversation_history: list) -> Trajectory:
        """Generate refined solution based on feedback."""
        if "generator_refinement" not in self.prompts:
            raise KeyError("'generator_refinement' prompt template not found")

        template = self.prompts["generator_refinement"]["template"]
        refinement_prompt = template

        # Add refinement prompt to conversation history
        messages = conversation_history + [{"role": "user", "content": refinement_prompt}]

        output: ModelOutput = await self.rollout_engine.get_model_response(messages, agent_name="refiner")

        return Trajectory(
            name="refiner",
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

class MultiAgentMathWorkflow(Workflow):
    """Multi-agent workflow for mathematical problem solving with iterative refinement."""

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        reward_function: RewardFunction = None,
        max_refinement_iterations: int = 3,
        **kwargs
    ):
        super().__init__(rollout_engine, **kwargs)
        self.reward_function = reward_function
        self.max_refinement_iterations = max_refinement_iterations

        # Agents will be initialized with prompts from task data
        self.generator = None
        self.evaluator = None
        self.refiner = None

    def _initialize_agents(self, prompts: Dict[str, Any]):
        """Initialize agents with prompt templates."""
        if self.generator is None:
            self.generator = Generator(self.rollout_engine, prompts)
            self.evaluator = Evaluator(self.rollout_engine, prompts)
            self.refiner = Refiner(self.rollout_engine, prompts)

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        """
        Execute multi-agent math reasoning workflow with iterative refinement.

        Flow:
        1. Generator creates initial solution (reward computed using ground truth)
        2. Evaluator critiques the solution and provides verdict
        3. If Evaluator says "incorrect" and iterations < max: Refiner creates new solution
        4. Repeat steps 2-3 until Evaluator says "correct" or max iterations reached

        Note: Ground truth is used only for reward computation (training signal).
        The workflow flow is controlled by the Evaluator's verdict, allowing agents
        to operate autonomously.

        Args:
            task: Dictionary with 'question', 'ground_truth_answer', 'prompts'
            uid: Unique identifier for this episode

        Returns:
            Episode with all trajectories and metrics
        """
        self.reset(task, uid)

        problem = task["question"]
        task["ground_truth"] = task["final_answer"]
        with open("examples/math_reasoning/prompt.json", "r") as f:
            prompts = json.load(f)["multi_agent_math_prompts"]

        # Initialize agents with prompts
        self._initialize_agents(prompts)

        all_trajectories = []
        conversation_history = []

        # Step 1: Generator creates initial solution
        generator_trajectory = await self.generator.generate_solution(problem)
        current_answer = generator_trajectory.steps[0].action

        # Compute reward for generator using ground truth (for training signal)
        reward_result = self.reward_function(task, current_answer)
        generator_trajectory.steps[0].reward = reward_result.reward
        generator_trajectory.reward = reward_result.reward

        all_trajectories.append(generator_trajectory)

        # Build conversation history for next agents
        conversation_history.extend(generator_trajectory.steps[0].chat_completions)

        # Track metrics
        generator_attempts = 1
        generator_correct_count = int(reward_result.is_correct)
        evaluator_correct_predictions = 0
        evaluator_total_predictions = 0
        refiner_correct_count = 0
        refiner_attempts = 0

        # Control variable: continues based on Evaluator's verdict
        should_continue = True

        # Iterative refinement loop
        iteration = 0
        while should_continue and iteration < self.max_refinement_iterations - 1:
            # Step 2: Evaluator critiques current solution
            evaluator_trajectory = await self.evaluator.evaluate_solution(
                problem,
                current_answer,
                conversation_history
            )

            verdict_info = evaluator_trajectory.steps[0].action
            evaluator_verdict = verdict_info["verdict"]

            # Compute ground truth correctness for evaluator reward
            ground_truth_correct = self.reward_function(task, current_answer).is_correct

            # Compute evaluator accuracy (comparing verdict with ground truth)
            evaluator_correct = (evaluator_verdict == "correct" and ground_truth_correct) or \
                              (evaluator_verdict == "incorrect" and not ground_truth_correct)
            evaluator_trajectory.steps[0].reward = 1.0 if evaluator_correct else 0.0
            evaluator_trajectory.reward = evaluator_trajectory.steps[0].reward
            evaluator_correct_predictions += int(evaluator_correct)
            evaluator_total_predictions += 1

            all_trajectories.append(evaluator_trajectory)

            # Update conversation history
            conversation_history.extend([
                {"role": "assistant", "content": evaluator_trajectory.steps[0].model_output.content}
            ])

            # Step 3: Use Evaluator's verdict to decide whether to refine
            # If Evaluator says "incorrect", continue refining
            if evaluator_verdict == "incorrect":
                refiner_trajectory = await self.refiner.refine_solution(conversation_history)
                current_answer = refiner_trajectory.steps[0].action

                # Compute reward for refined solution using ground truth (for training signal)
                reward_result = self.reward_function(task, current_answer)
                refiner_trajectory.steps[0].reward = reward_result.reward
                refiner_trajectory.reward = refiner_trajectory.steps[0].reward

                refiner_attempts += 1
                refiner_correct_count += int(reward_result.is_correct)

                all_trajectories.append(refiner_trajectory)

                # Update conversation history
                conversation_history.extend(refiner_trajectory.steps[0].chat_completions[-1:])

                # Continue loop - next iteration will ask Evaluator again
                should_continue = True
            else:
                # Evaluator says "correct", stop refining
                should_continue = False

            iteration += 1

        # Final evaluation if we exited due to max iterations
        # (if we exited because Evaluator said "correct", we already have the evaluation)
        if should_continue and iteration == self.max_refinement_iterations - 1:
            # We hit max iterations after a Refiner step, need one more evaluation
            final_evaluator_trajectory = await self.evaluator.evaluate_solution(
                problem,
                current_answer,
                conversation_history
            )

            verdict_info = final_evaluator_trajectory.steps[0].action
            evaluator_verdict = verdict_info["verdict"]

            # Compute ground truth correctness for evaluator reward
            ground_truth_correct = self.reward_function(task, current_answer).is_correct

            evaluator_correct = (evaluator_verdict == "correct" and ground_truth_correct) or \
                              (evaluator_verdict == "incorrect" and not ground_truth_correct)
            final_evaluator_trajectory.steps[0].reward = 1.0 if evaluator_correct else 0.0
            final_evaluator_trajectory.reward = final_evaluator_trajectory.steps[0].reward
            evaluator_correct_predictions += int(evaluator_correct)
            evaluator_total_predictions += 1

            all_trajectories.append(final_evaluator_trajectory)

        # Compute final ground truth correctness for the episode
        final_reward_result = self.reward_function(task, current_answer)
        final_is_correct = final_reward_result.is_correct

        # Compute final metrics
        generator_acc = generator_correct_count / generator_attempts if generator_attempts > 0 else 0.0
        evaluator_acc = evaluator_correct_predictions / evaluator_total_predictions if evaluator_total_predictions > 0 else 0.0
        refiner_acc = refiner_correct_count / refiner_attempts if refiner_attempts > 0 else 0.0

        metrics = {
            "generator_acc": generator_acc,
            "evaluator_acc": evaluator_acc,
            "refiner_acc": refiner_acc,
            "total_iterations": iteration + 1,
            "success": int(final_is_correct),
            "evaluator_total_predictions": evaluator_total_predictions,
            "refiner_attempts": refiner_attempts,
            "generator_attempts": generator_attempts,
        }

        return Episode(
            id=uid,
            task=task,
            trajectories=all_trajectories,
            is_correct=final_is_correct,
            metrics=metrics,
        )
