"""Environment-based Single-Agent Workflow.

This module provides a generic single-agent workflow that works with ANY
agent/environment combination. It serves as the workflow-based equivalent
of the AgentPPOTrainer's agent+env execution loop.

This workflow can be used as:
1. A direct replacement for agent-based training (same behavior, workflow trainer)
2. A foundation for multi-agent workflows (EnvEvaluatorOptimizerWorkflow inherits from this)

Example:
    # Instead of:
    trainer = AgentTrainer(
        agent_class=FrozenLakeAgent,
        env_class=FrozenLakeEnv,
        ...
    )

    # Use:
    trainer = AgentTrainer(
        workflow_class=EnvSingleAgentWorkflow,
        workflow_args={
            "agent_cls": FrozenLakeAgent,
            "env_cls": FrozenLakeEnv,
            "agent_args": {...},
            "env_args": {...},
        },
        ...
    )
"""

from copy import deepcopy
from typing import Any

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.workflows.timing_mixin import TimingTrackingMixin
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow


class EnvSingleAgentWorkflow(TimingTrackingMixin, Workflow):
    """Generic single-agent workflow for environment-based tasks.

    This workflow wraps any agent/environment combination and executes
    the standard agent-env interaction loop:
    1. Reset environment and agent
    2. Agent receives observation
    3. Model generates response
    4. Agent parses response into action
    5. Environment executes action
    6. Repeat until done or max_steps

    The workflow produces a single trajectory containing all steps.

    Attributes:
        agent_cls: The agent class to instantiate
        env_cls: The environment class to instantiate
        agent_args: Arguments passed to agent constructor
        env_args: Arguments passed to environment constructor
        max_steps: Maximum number of steps per episode
        agent: The instantiated agent (created per episode)
        env: The instantiated environment (created per episode)
    """

    # Default agent name for single-agent workflow
    AGENT_NAME = "actor"

    def __init__(
        self,
        agent_cls,
        env_cls,
        agent_args: dict | None = None,
        env_args: dict | None = None,
        max_steps: int = 10,
        **kwargs,
    ):
        """Initialize the EnvSingleAgentWorkflow.

        Args:
            agent_cls: Agent class (e.g., MiniWobAgent, FrozenLakeAgent) or string name
            env_cls: Environment class (e.g., BrowserGymEnv, FrozenLakeEnv) or string name
            agent_args: Arguments passed to agent constructor
            env_args: Arguments passed to environment constructor
            max_steps: Maximum number of steps per episode
            **kwargs: Additional arguments passed to parent Workflow
        """
        from rllm.trainer.env_agent_mappings import AGENT_CLASS_MAPPING, ENV_CLASS_MAPPING

        super().__init__(**kwargs)

        # Resolve string names to classes if needed
        self.agent_cls = AGENT_CLASS_MAPPING[agent_cls] if isinstance(agent_cls, str) else agent_cls
        self.env_cls = ENV_CLASS_MAPPING[env_cls] if isinstance(env_cls, str) else env_cls

        # Store args (make copies to avoid mutation)
        self.agent_args = dict(agent_args) if agent_args is not None else {}
        self.env_args = dict(env_args) if env_args is not None else {}
        self.max_steps = max_steps

        # Agent and environment instances (created per episode in reset)
        self.agent = None
        self.env = None

    def _create_agent(self):
        """Create a new agent instance.

        Override this method in subclasses to customize agent creation.

        Returns:
            BaseAgent: The created agent instance
        """
        return self.agent_cls(**self.agent_args)

    def _create_env(self, task: dict):
        """Create a new environment instance for the given task.

        Override this method in subclasses to customize environment creation.

        Args:
            task: The task dictionary

        Returns:
            BaseEnv: The created environment instance
        """
        return self.env_cls(**self.env_args)

    async def run(self, task: dict, uid: str, **kwargs) -> Episode | None:
        """Execute the single-agent environment workflow.

        Args:
            task: Task dictionary containing problem information
            uid: Unique identifier for this episode

        Returns:
            Episode with single trajectory (or None to use postprocess_episode)
        """
        # Reset and create agent/env
        observation, info = await self.timed_env_call(self.reset, task=task, uid=uid)

        # Initial observation to agent
        self.agent.update_from_env(observation, 0, False, info)

        for step_num in range(1, self.max_steps + 1):
            # Get model response
            output: ModelOutput = await self.timed_llm_call(
                self.agent.chat_completions,
                agent_name=self.AGENT_NAME,
                application_id=uid,
                **kwargs,
            )
            response = output.text

            # Check for max response length exceeded
            if output.finish_reason == "length":
                raise TerminationEvent(TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED)

            # Agent processes response and produces action
            action = self.agent.update_from_model(response)

            # Execute action in environment
            next_obs, reward, done, info = await self.timed_env_call(
                self.env.step, action
            )

            # Update agent with environment feedback
            self.agent.update_from_env(next_obs, reward, done, info)

            if done:
                raise TerminationEvent(TerminationReason.ENV_DONE)

        raise TerminationEvent(TerminationReason.MAX_TURNS_EXCEEDED)

    def reset(self, task: dict | None = None, uid: str | None = None) -> tuple[Any, dict]:
        """Reset workflow state for a new episode.

        Creates new agent and environment instances, resets the environment,
        and prepares for a new episode.

        Args:
            task: The task to reset to
            uid: Unique identifier for the episode

        Returns:
            tuple: (observation, info) from environment reset
        """
        # Call parent reset (initializes uid, task, timing, etc.)
        super().reset(task, uid)

        # Create fresh agent and environment instances
        self.agent = self._create_agent()
        self.env = self._create_env(task)

        # Set trajectory task and name
        if hasattr(self.agent, '_trajectory'):
            self.agent._trajectory.task = task
            self.agent._trajectory.name = self.AGENT_NAME

        # Reset environment and return initial observation
        return self.env.reset(task)

    def collect_trajectories(self) -> Episode:
        """Collect trajectories from the agent.

        Returns:
            Episode: Episode containing the agent's trajectory
        """
        episode = Episode()

        # Get agent's trajectory
        if self.agent is not None and hasattr(self.agent, 'trajectory'):
            trajectory = deepcopy(self.agent.trajectory)
            trajectory.name = self.AGENT_NAME
            episode.trajectories.append(trajectory)

        return episode

    def compute_trajectory_reward(self, trajectory: Trajectory) -> None:
        """Compute trajectory-level reward.

        For single-agent workflows, the trajectory reward is typically
        the final step's reward (environment reward).

        Args:
            trajectory: The trajectory to compute reward for
        """
        if trajectory.steps:
            # Use the last step's reward as trajectory reward
            trajectory.reward = trajectory.steps[-1].reward
        else:
            trajectory.reward = 0.0

    def assign_episode_correctness(self, episode: Episode) -> None:
        """Assign episode-level correctness flag.

        For environment-based workflows, correctness is determined by
        whether the task was completed successfully (positive reward).

        Args:
            episode: The episode to assign correctness to
        """
        total_reward = sum(traj.reward for traj in episode.trajectories)
        episode.is_correct = total_reward > 0

    def is_multithread_safe(self) -> bool:
        """Check if the workflow is multithread safe.

        Returns:
            bool: True if both agent and env are thread-safe
        """
        if self.env is not None and hasattr(self.env, 'is_multithread_safe'):
            return self.env.is_multithread_safe()
        return True
