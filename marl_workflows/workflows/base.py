"""Base class and shared helpers for the multi-agent workflows.

Naming contract (see ``marl_workflows.contracts``, contracts C1-C5)
===================================================================

Every workflow builds ``Trajectory`` objects and calls the rollout engine with an
``agent_name``. Both names are part of the experiment definition:

* ``Trajectory.name`` matches ``contracts.TRAJ_NAME_RE`` (letters, then optional digits; no
  ``_`` or ``:``). rLLM builds ``trajectory_id = f"{task_uuid}_{Trajectory.name}"`` and the
  GRPO group of a row is (task, ``Trajectory.name``), so the name decides which samples are
  normalised together. The role of a row is the name with trailing digits removed
  (``contracts.role_from_name``) and decides which LoRA adapter is updated with it.
* ``agent_name`` (passed to ``rollout_engine.get_model_response``) is the role itself, with
  no digits, and must be one of ``trainer.agent_names``; under IP the engine derives the LoRA
  id from its index (``contracts.lora_int_id_for``).

Names per workflow (C4), as trajectory names -> agent_name values:

* single agent: ``generator`` -> ``generator``.
* Math Voting: ``generator0/1/2`` (one GRPO group per voter index) and ``aggregator`` ->
  ``generator``, ``aggregator``.
* Code Voting: ``generator`` three times (one pooled GRPO group) and ``aggregator`` -> same.
* Eval-Opt: ``generator`` (every turn is its own trajectory) and ``evaluator`` -> same.
* Orch-Workers: ``orchestrator``, ``worker`` (all workers) and ``synthesizer`` -> same.
* voting_v2: ``voterA/B/C`` and ``aggregator`` -> same.
* controlled single-role control: ``generator0..N-1`` -> ``generator``.

Renaming any of these silently changes the experiment (GRPO groups and adapter routing), so
the names below are kept verbatim from the research fork.

``MultiAgentWorkflow`` differs from upstream ``rllm.workflows.workflow.Workflow`` only in
what the fork added to that base class: ``executor`` is optional (the evaluator builds
workflows without one), a dedicated ``code_reward_executor`` (the rllm-free sandbox pool of
``marl_workflows.rewards.code_reward.CodeRewardExecutor``) and ``run_in_code_executor``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterable
from concurrent.futures import Executor
from functools import cache, partial
from importlib import resources
from typing import Any, Literal

from rllm.agents.agent import Trajectory
from rllm.engine.rollout.rollout_engine import RolloutEngine
from rllm.workflows.workflow import Workflow

__all__ = ["MultiAgentWorkflow", "broadcast_final_reward", "load_prompts"]

_PROMPT_FILES: dict[str, str] = {"math": "math.json", "deepcoder": "deepcoder.json"}


class MultiAgentWorkflow(Workflow):
    """Upstream ``Workflow`` with an optional executor and a code-reward executor.

    Upstream extra kwargs (``timeout``, ``gamma``, ``reward_bonus_coeff``, ...) pass through
    ``**kwargs``; unknown kwargs are absorbed by the upstream base, as in the fork.
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        executor: Executor | None = None,
        code_reward_executor: Executor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(rollout_engine, executor, **kwargs)
        self.code_reward_executor = code_reward_executor

    async def run_in_executor(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run ``fn`` in ``self.executor`` (fork ``rllm/workflows/workflow.py:290-302``)."""
        if self.executor is None:
            raise ValueError(
                "Cannot use run_in_executor: no executor was provided during Workflow initialization. "
                "Pass an executor to the Workflow constructor if you need to run functions in a thread pool."
            )
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, partial(fn, *args, **kwargs))

    async def run_in_code_executor(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run ``fn`` in the code-reward executor, falling back to ``self.executor``.

        Fork ``rllm/workflows/workflow.py:305-333`` without the batch-test scheduler and the
        semaphore (execution modes no paper run used).
        """
        executor = self.code_reward_executor or self.executor
        if executor is None:
            raise ValueError("No executor available for run_in_code_executor.")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, partial(fn, *args, **kwargs))


@cache
def _prompt_text(task: str) -> str:
    try:
        filename = _PROMPT_FILES[task]
    except KeyError:
        raise ValueError(f"Unknown prompt task {task!r}; expected one of {list(_PROMPT_FILES)}") from None
    return resources.files("marl_workflows.workflows").joinpath("prompts", filename).read_text(encoding="utf-8")


def load_prompts(task: Literal["math", "deepcoder"], section: str) -> dict[str, Any]:
    """Return one section of the packaged prompt file (a fresh dict on every call).

    The files are byte-identical copies of the fork's ``examples/math_reasoning/prompt.json``
    and ``examples/deepcoder/prompt.json``.
    """
    data = json.loads(_prompt_text(task))
    if section not in data:
        raise KeyError(f"Prompt section {section!r} not in prompts/{_PROMPT_FILES[task]}; have {sorted(data)}")
    return data[section]


def broadcast_final_reward(trajectories: Iterable[Trajectory], value: float) -> None:
    """Give every trajectory and every step the episode's final outcome reward.

    This is the ``use_final_outcome_reward`` broadcast of the fork's workflows: each role is
    trained on the episode outcome, not on its own correctness.
    """
    for trajectory in trajectories:
        trajectory.reward = value
        for step in trajectory.steps:
            step.reward = value
