"""Single-agent workflow for DeepCoder code generation (SA-RL).

Generator creates code, reward computed from tests.

Ported from the single-pass path of fork ``examples/deepcoder/single_agent_deepcoder_workflow.py``;
only the imports, the base class (``MultiAgentWorkflow``), prompt loading (``prompts`` dict or
``load_prompts``, no ``prompt_file``) and the removed test loop differ.
"""

from typing import Any

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine

from marl_workflows.rewards.code_reward import code_reward_fn
from marl_workflows.workflows.base import MultiAgentWorkflow, load_prompts
from marl_workflows.workflows.code import reject_test_loop


class SingleAgentDeepcodeWorkflow(MultiAgentWorkflow):
    """
    Single-agent workflow for code generation.

    Generate code, compute reward from test execution.

    Example:
        workflow = SingleAgentDeepcodeWorkflow(rollout_engine=engine)
        episode = await workflow.run(task, uid)
    """

    GENERATOR_NAME = "generator"

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        prompts: dict[str, Any] | None = None,
        enable_test_loop: bool = False,
        max_test_rounds: int = 2,
        max_tests_to_show: int = 3,
        public_test_only: bool = False,
        **kwargs,
    ):
        """Initialize the single-agent DeepCoder workflow.

        Args:
            rollout_engine: Engine for LLM inference
            prompts: Optional pre-loaded prompt templates (default: the packaged
                ``deepcoder_single_agent_prompts`` section)
            enable_test_loop: Must be False (the test loop is not ported)
            max_test_rounds: Kept for config compatibility; unused
            max_tests_to_show: Kept for config compatibility; unused
            public_test_only: Kept for config compatibility; unused
            **kwargs: Additional arguments passed to parent workflow
        """
        reject_test_loop(enable_test_loop)
        super().__init__(rollout_engine, **kwargs)
        self._prompts = prompts

        self.enable_test_loop = enable_test_loop
        self.max_test_rounds = max_test_rounds
        self.max_tests_to_show = max_tests_to_show
        self.public_test_only = public_test_only

    @property
    def prompts(self) -> dict[str, Any]:
        """Lazy load prompts if not provided."""
        if self._prompts is None:
            self._prompts = load_prompts("deepcoder", "deepcoder_single_agent_prompts")
        return self._prompts

    def _build_initial_prompt(self, problem: str) -> str:
        """Build initial code generation prompt."""
        template = self.prompts.get("generator_initial", {}).get(
            "template",
            "You are an expert competitive programmer. Solve the following "
            "programming problem.\n\nProblem:\n{problem}\n\nRequirements:\n"
            "- Write clean, efficient Python code\n- Handle all edge cases\n"
            "- Output your code in a markdown code block with ```python",
        )
        return template.format(problem=problem)

    def _create_trajectory(
        self,
        messages: list,
        model_output: ModelOutput,
        response: str,
        reward: float,
    ) -> Trajectory:
        """Create a trajectory from generation output."""
        return Trajectory(
            name=self.GENERATOR_NAME,
            steps=[
                Step(
                    chat_completions=messages
                    + [
                        {
                            "role": "assistant",
                            "content": model_output.content,
                            "reasoning": model_output.reasoning,
                        }
                    ],
                    thought=model_output.reasoning,
                    action=response,
                    model_output=model_output,
                    reward=reward,
                )
            ],
            reward=reward,
        )

    async def run(self, task: dict[str, Any], uid: str, **kwargs) -> Episode:
        """
        Execute single-agent workflow.

        - Generate code solution
        - Compute reward from test execution
        - Return single trajectory

        Args:
            task: Dictionary with 'question' (problem) and 'ground_truth' (tests)
            uid: Unique identifier for this episode

        Returns:
            Episode with trajectories and metrics
        """
        self.reset(task, uid)

        problem = task["question"]

        # Single-pass mode: generate once and compute reward
        return await self._run_single_pass(task, uid, problem)

    async def _run_single_pass(
        self,
        task: dict[str, Any],
        uid: str,
        problem: str,
    ) -> Episode:
        """Execute single-pass generation without test loop."""
        prompt_content = self._build_initial_prompt(problem)
        messages = [{"role": "user", "content": prompt_content}]

        # Generate solution
        output: ModelOutput = await self.rollout_engine.get_model_response(messages, agent_name=self.GENERATOR_NAME)

        code = output.content or output.text or ""

        # Compute reward from test execution (run in thread pool for parallelism)
        reward_output = await self.run_in_code_executor(code_reward_fn, task, code)
        reward = reward_output.reward
        is_correct = reward_output.is_correct

        # Create trajectory
        trajectory = self._create_trajectory(
            messages=messages,
            model_output=output,
            response=code,
            reward=reward,
        )

        # Build metrics
        metadata = reward_output.metadata or {}
        passed_tests = metadata.get("passed_tests", 0)
        total_tests = metadata.get("total_tests", 0)

        metrics = {
            "generator_acc": float(is_correct),
            "success": int(is_correct),
            "test_rounds": 1,
            "generator_attempts": 1,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "pass_rate": (passed_tests / total_tests if total_tests > 0 else 0.0),
        }

        # Attach reward metadata (test details) to episode info for debugging
        episode_info = {}
        if metadata and not is_correct:
            test_results = metadata.get("test_results", [])
            if test_results:
                episode_info["reward_test_details"] = test_results[:3]  # first 3 tests

        return Episode(
            id=uid,
            task=task,
            trajectories=[trajectory],
            is_correct=is_correct,
            metrics=metrics,
            info=episode_info,
        )
