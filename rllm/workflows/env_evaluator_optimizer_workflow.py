"""Environment-based Evaluator-Optimizer Workflow.

This module provides a generic multi-agent workflow that combines environment
interaction with the evaluator-optimizer pattern for action decisions.

At each step, instead of directly using the actor's proposed action:
1. Actor proposes an action based on current observation
2. Evaluator reviews the proposed action
3. If evaluator rejects, Actor refines based on feedback
4. Loop until evaluator approves or max iterations reached
5. Execute the final action in the environment

This enables iterative refinement of actions at each step, which can improve
decision quality in complex environments.

Example:
    trainer = AgentTrainer(
        workflow_class=EnvEvaluatorOptimizerWorkflow,
        workflow_args={
            "agent_cls": MiniWobAgent,
            "env_cls": BrowserGymEnv,
            "agent_args": {...},
            "env_args": {...},
            "max_refine_iterations": 2,
        },
        config=config,
        ...
    )
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.workflows.env_single_agent_workflow import EnvSingleAgentWorkflow
from rllm.workflows.workflow import TerminationEvent, TerminationReason


@dataclass
class ActionEvaluation:
    """Result of evaluating a proposed action.

    Attributes:
        is_good: Whether the evaluator approves the action
        feedback: Explanation or reason for the evaluation
        confidence: Confidence score (0.0 to 1.0)
    """

    is_good: bool
    feedback: str
    confidence: float = 1.0


class EnvEvaluatorOptimizerWorkflow(EnvSingleAgentWorkflow):
    """Generic multi-agent workflow with evaluator-optimizer action decisions.

    This workflow extends EnvSingleAgentWorkflow by adding a 2-agent
    evaluator-optimizer loop at each action step. The workflow maintains
    separate trajectories for each agent role:

    - actor: Primary agent that proposes, refines, and executes actions
    - evaluator: Reviews proposed actions and provides feedback

    This follows the same pattern as EvaluatorOptimizerWorkflow where the same
    agent (actor) handles both initial proposals and refinements based on feedback.

    The workflow produces multiple trajectories per episode, enabling
    per-agent LoRA training when configured with separate agent names.

    Attributes:
        max_refine_iterations: Maximum evaluation-refinement iterations per step
        eval_prompt_template: Template for evaluator prompts (optional)
        refine_prompt_template: Template for refinement prompts (optional)
        reward_weights: Weights for distributing rewards to each agent
    """

    # Agent role names (no underscores per naming convention)
    # Note: Only 2 agents, actor handles both initial proposals and refinements
    ACTOR_NAME = "actor"
    EVALUATOR_NAME = "evaluator"

    # Default prompt templates
    DEFAULT_EVAL_PROMPT = """Review the proposed action for the current state.

Current State:
{observation}

Proposed Action: {action}

Is this action appropriate? Consider:
- Does the action make sense given the current state?
- Will this action help progress toward the goal?
- Are there any obvious issues with this action?

Respond with either:
- GOOD: <brief reason why the action is appropriate>
- BAD: <brief explanation of the issue>"""

    DEFAULT_REFINE_PROMPT = """The proposed action was rejected. Please suggest a better action.

Current State:
{observation}

Rejected Action: {action}
Feedback: {feedback}

Propose a better action. Output your reasoning followed by the action."""

    def __init__(
        self,
        agent_cls,
        env_cls,
        agent_args: dict | None = None,
        env_args: dict | None = None,
        max_steps: int = 10,
        max_refine_iterations: int = 2,
        eval_prompt_template: str | None = None,
        refine_prompt_template: str | None = None,
        reward_weights: dict | None = None,
        **kwargs,
    ):
        """Initialize the EnvEvaluatorOptimizerWorkflow.

        Args:
            agent_cls: Agent class (e.g., MiniWobAgent, FrozenLakeAgent)
            env_cls: Environment class (e.g., BrowserGymEnv, FrozenLakeEnv)
            agent_args: Arguments passed to agent constructor
            env_args: Arguments passed to environment constructor
            max_steps: Maximum number of environment steps per episode
            max_refine_iterations: Maximum evaluation-refinement iterations per step
            eval_prompt_template: Custom template for evaluator prompts
            refine_prompt_template: Custom template for refinement prompts
            reward_weights: Dict mapping agent names to reward weights
            **kwargs: Additional arguments passed to parent workflow
        """
        super().__init__(
            agent_cls=agent_cls,
            env_cls=env_cls,
            agent_args=agent_args,
            env_args=env_args,
            max_steps=max_steps,
            **kwargs,
        )

        self.max_refine_iterations = max_refine_iterations
        self.eval_prompt_template = eval_prompt_template or self.DEFAULT_EVAL_PROMPT
        self.refine_prompt_template = refine_prompt_template or self.DEFAULT_REFINE_PROMPT

        # Default reward weights: actor gets full reward, evaluator gets partial
        self.reward_weights = reward_weights or {
            self.ACTOR_NAME: 1.0,
            self.EVALUATOR_NAME: 0.5,
        }

        # Multi-agent trajectory tracking
        self._evaluator_steps: list[Step] = []
        self._actor_refinement_steps: list[Step] = []

    async def run(self, task: dict, uid: str, **kwargs) -> Episode | None:
        """Execute the multi-agent environment workflow.

        Args:
            task: Task dictionary containing problem information
            uid: Unique identifier for this episode

        Returns:
            Episode with trajectories for actor and evaluator
        """
        # Reset and create agent/env
        observation, info = await self.timed_env_call(self.reset, task=task, uid=uid)

        # Initial observation to agent
        self.agent.update_from_env(observation, 0, False, info)

        for step_num in range(1, self.max_steps + 1):
            # Step 1: Actor proposes action via LLM
            actor_output: ModelOutput = await self.timed_llm_call(
                self.agent.chat_completions,
                agent_name=self.ACTOR_NAME,
                application_id=uid,
                **kwargs,
            )

            # Check for max response length exceeded
            if actor_output.finish_reason == "length":
                raise TerminationEvent(TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED)

            # Parse proposed action from actor's response
            proposed_action = self._extract_action_from_response(actor_output.text)

            # Step 2: Evaluator-optimizer loop for action refinement
            final_action, eval_steps, gen_steps = await self._evaluate_and_refine(
                observation=observation,
                proposed_action=proposed_action,
                step_num=step_num,
                uid=uid,
                **kwargs,
            )

            # Track multi-agent steps
            self._evaluator_steps.extend(eval_steps)
            self._actor_refinement_steps.extend(gen_steps)

            # Step 3: Execute final action in environment
            # Use final_action instead of actor's original response
            action = self.agent.update_from_model(final_action)
            next_obs, reward, done, info = await self.timed_env_call(
                self.env.step, action
            )

            # Update agent with environment feedback
            self.agent.update_from_env(next_obs, reward, done, info)
            observation = next_obs

            if done:
                raise TerminationEvent(TerminationReason.ENV_DONE)

        raise TerminationEvent(TerminationReason.MAX_TURNS_EXCEEDED)

    async def _evaluate_and_refine(
        self,
        observation: Any,
        proposed_action: str,
        step_num: int,
        uid: str,
        **kwargs,
    ) -> tuple[str, list[Step], list[Step]]:
        """Run evaluator-optimizer loop to refine action.

        Args:
            observation: Current environment observation
            proposed_action: Initial action proposed by actor
            step_num: Current step number
            uid: Episode unique identifier

        Returns:
            tuple: (final_action, evaluator_steps, refinement_steps)
        """
        evaluator_steps = []
        refinement_steps = []
        current_action = proposed_action

        for iteration in range(self.max_refine_iterations):
            # Build evaluation prompt
            obs_str = self._format_observation(observation)
            eval_prompt = self.eval_prompt_template.format(
                observation=obs_str,
                action=current_action,
            )
            eval_messages = [{"role": "user", "content": eval_prompt}]

            # Get evaluator response
            eval_output: ModelOutput = await self.rollout_engine.get_model_response(
                eval_messages,
                agent_name=self.EVALUATOR_NAME,
                application_id=uid,
            )

            # Parse evaluation
            evaluation = self._parse_evaluation(eval_output.content or eval_output.text)

            # Record evaluator step
            evaluator_steps.append(Step(
                chat_completions=eval_messages + [{
                    "role": "assistant",
                    "content": eval_output.content or eval_output.text,
                    "reasoning": eval_output.reasoning,
                }],
                thought=eval_output.reasoning or "",
                action={"verdict": "good" if evaluation.is_good else "bad"},
                model_output=eval_output,
            ))

            # If evaluator approves, use current action
            if evaluation.is_good:
                break

            # If not approved and not last iteration, actor refines the action
            if iteration < self.max_refine_iterations - 1:
                refine_prompt = self.refine_prompt_template.format(
                    observation=obs_str,
                    action=current_action,
                    feedback=evaluation.feedback,
                )
                refine_messages = [{"role": "user", "content": refine_prompt}]

                refine_output: ModelOutput = await self.rollout_engine.get_model_response(
                    refine_messages,
                    agent_name=self.ACTOR_NAME,
                    application_id=uid,
                )

                # Extract new action from actor's refinement response
                current_action = self._extract_action_from_response(
                    refine_output.content or refine_output.text
                )

                # Record refinement step (still part of actor's work)
                refinement_steps.append(Step(
                    chat_completions=refine_messages + [{
                        "role": "assistant",
                        "content": refine_output.content or refine_output.text,
                        "reasoning": refine_output.reasoning,
                    }],
                    thought=refine_output.reasoning or "",
                    action=current_action,
                    model_output=refine_output,
                ))

        return current_action, evaluator_steps, refinement_steps

    def _extract_action_from_response(self, response: str) -> str:
        """Extract action from model response.

        Uses the agent's parsing method if available, otherwise returns
        the raw response.

        Args:
            response: Raw model response

        Returns:
            Extracted action string
        """
        if hasattr(self.agent, '_parse_model_response'):
            # Try to use agent's parsing method
            result = self.agent._parse_model_response(response)
            # Handle tuple return (thought, action)
            if isinstance(result, tuple):
                return result[1] if len(result) > 1 else result[0]
            return result
        return response

    def _format_observation(self, observation: Any) -> str:
        """Format observation for prompts.

        Args:
            observation: Environment observation

        Returns:
            Formatted string representation
        """
        if isinstance(observation, dict):
            # Try common keys for text representation
            for key in ['axtree_txt', 'text', 'state', 'description']:
                if key in observation:
                    return str(observation[key])
            return str(observation)
        return str(observation)

    def _parse_evaluation(self, response: str) -> ActionEvaluation:
        """Parse evaluator response into ActionEvaluation.

        Args:
            response: Raw evaluator response

        Returns:
            ActionEvaluation with is_good, feedback, and confidence
        """
        response_upper = response.upper()

        # Check for explicit GOOD/BAD markers
        if "GOOD" in response_upper and "BAD" not in response_upper:
            return ActionEvaluation(
                is_good=True,
                feedback=response,
                confidence=1.0,
            )
        elif "BAD" in response_upper:
            # Extract feedback after "BAD:"
            feedback = response
            if "BAD:" in response_upper:
                idx = response_upper.index("BAD:")
                feedback = response[idx + 4:].strip()
            return ActionEvaluation(
                is_good=False,
                feedback=feedback,
                confidence=1.0,
            )
        else:
            # Default to accepting if no clear rejection
            return ActionEvaluation(
                is_good=True,
                feedback=response,
                confidence=0.5,
            )

    def reset(self, task: dict | None = None, uid: str | None = None) -> tuple[Any, dict]:
        """Reset workflow state for a new episode.

        Args:
            task: The task to reset to
            uid: Unique identifier for the episode

        Returns:
            tuple: (observation, info) from environment reset
        """
        # Reset multi-agent step tracking
        self._evaluator_steps = []
        self._actor_refinement_steps = []

        # Call parent reset
        return super().reset(task, uid)

    def collect_trajectories(self) -> Episode:
        """Collect trajectories from all agents.

        Returns:
            Episode: Episode containing actor and evaluator trajectories
        """
        episode = Episode()

        # Actor trajectory (from parent's agent)
        if self.agent is not None and hasattr(self.agent, 'trajectory'):
            actor_traj = deepcopy(self.agent.trajectory)
            actor_traj.name = self.ACTOR_NAME
            episode.trajectories.append(actor_traj)

        # Actor refinement trajectory (refinements are also actor work)
        # We keep this as a separate trajectory for proper LoRA training
        # since refinement uses different prompts than the main agent loop
        if self._actor_refinement_steps:
            refinement_traj = Trajectory(
                name=self.ACTOR_NAME,
                task=self.task,
                steps=deepcopy(self._actor_refinement_steps),
            )
            episode.trajectories.append(refinement_traj)

        # Evaluator trajectory
        if self._evaluator_steps:
            evaluator_traj = Trajectory(
                name=self.EVALUATOR_NAME,
                task=self.task,
                steps=deepcopy(self._evaluator_steps),
            )
            episode.trajectories.append(evaluator_traj)

        return episode

    def compute_trajectory_reward(self, trajectory: Trajectory) -> None:
        """Compute trajectory-level reward with agent-specific weighting.

        Args:
            trajectory: The trajectory to compute reward for
        """
        # Get base reward from actor trajectory
        if trajectory.name == self.ACTOR_NAME:
            # Actor gets full environment reward
            if trajectory.steps:
                trajectory.reward = trajectory.steps[-1].reward
            else:
                trajectory.reward = 0.0
        else:
            # Other agents get weighted reward based on actor's performance
            # Find actor trajectory to get base reward
            actor_reward = 0.0
            for attr_name in dir(self):
                if attr_name.startswith("_"):
                    continue
                attr_value = getattr(self, attr_name)
                if hasattr(attr_value, 'trajectory') and hasattr(attr_value.trajectory, 'name'):
                    if attr_value.trajectory.name == self.ACTOR_NAME:
                        if attr_value.trajectory.steps:
                            actor_reward = attr_value.trajectory.steps[-1].reward
                        break

            # Apply weight
            weight = self.reward_weights.get(trajectory.name, 0.0)
            trajectory.reward = actor_reward * weight

    def collect_metrics(self, episode: Episode) -> None:
        """Collect metrics from the episode.

        Args:
            episode: The episode to collect metrics from
        """
        metrics = {}

        for traj in episode.trajectories:
            name = traj.name
            # Aggregate metrics by agent name (actor may have multiple trajectories)
            prefix = f"{name}"
            if f"{prefix}_reward" in metrics:
                metrics[f"{prefix}_reward"] += traj.reward
                metrics[f"{prefix}_steps"] += len(traj.steps)
            else:
                metrics[f"{prefix}_reward"] = traj.reward
                metrics[f"{prefix}_steps"] = len(traj.steps)

        # Calculate accuracy (positive reward = correct)
        for name in [self.ACTOR_NAME, self.EVALUATOR_NAME]:
            if f"{name}_reward" in metrics:
                metrics[f"{name}_acc"] = 1.0 if metrics[f"{name}_reward"] > 0 else 0.0

        # Add overall metrics
        metrics["total_evaluator_calls"] = len(self._evaluator_steps)
        metrics["total_refinement_calls"] = len(self._actor_refinement_steps)
        metrics["refinement_rate"] = (
            len(self._actor_refinement_steps) / len(self._evaluator_steps)
            if self._evaluator_steps else 0.0
        )

        episode.metrics = metrics
