"""A scripted rollout engine so workflows run on CPU without vLLM.

``FakeRolloutEngine`` returns canned ``ModelOutput``s keyed by the ``agent_name`` of each
call and records every call (messages and kwargs), so tests can check trajectory names,
the ``agent_name`` / ``application_id`` each workflow sends, and the prompts it builds.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine

# A scripted reply: a plain string (used as content), a ready ModelOutput, or a callable
# that receives the call's messages and kwargs and returns either of the two.
Reply = str | ModelOutput | Callable[[list[dict], dict[str, Any]], "str | ModelOutput"]


@dataclass
class FakeCall:
    """One recorded ``get_model_response`` call."""

    messages: list[dict]
    kwargs: dict[str, Any]
    output: ModelOutput

    @property
    def agent_name(self) -> str | None:
        return self.kwargs.get("agent_name")

    @property
    def prompt(self) -> str:
        """Content of the last message (the prompt the workflow just added)."""
        return self.messages[-1]["content"]


def make_output(content: str, reasoning: str | None = None) -> ModelOutput:
    """A ``ModelOutput`` as the verl engine returns it (``text`` and ``content`` both set)."""
    return ModelOutput(
        text=content,
        content=content,
        reasoning=reasoning,
        prompt_ids=[1, 2, 3],
        completion_ids=[4, 5],
        logprobs=[-0.1, -0.2],
        prompt_length=3,
        completion_length=2,
        finish_reason="stop",
    )


@dataclass
class FakeRolloutEngine(RolloutEngine):
    """Replies per ``agent_name`` from a script; records every call.

    ``script[agent_name]`` is a single reply (used for every call of that agent) or a
    sequence of replies consumed in call order; the last one repeats once the sequence is
    exhausted. ``default`` answers agents missing from the script. ``yield_between_calls``
    makes each call await once so that concurrent ``asyncio.gather`` calls interleave.
    """

    script: Mapping[str, Reply | Sequence[Reply]] = field(default_factory=dict)
    default: Reply | None = None
    yield_between_calls: bool = True
    calls: list[FakeCall] = field(default_factory=list)
    _counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def __post_init__(self) -> None:
        super().__init__()

    def _next_reply(self, agent_name: str | None) -> Reply:
        key = agent_name if agent_name is not None else ""
        if key in self.script:
            entry = self.script[key]
        elif self.default is not None:
            entry = self.default
        else:
            raise KeyError(f"FakeRolloutEngine has no reply for agent_name={agent_name!r}")
        if isinstance(entry, str | ModelOutput) or callable(entry):
            return entry
        index = self._counters[key]
        self._counters[key] += 1
        return entry[min(index, len(entry) - 1)]

    async def get_model_response(self, messages: list[dict], **kwargs: Any) -> ModelOutput:
        if self.yield_between_calls:
            await asyncio.sleep(0)
        reply = self._next_reply(kwargs.get("agent_name"))
        if callable(reply) and not isinstance(reply, ModelOutput):
            reply = reply(messages, kwargs)
        output = reply if isinstance(reply, ModelOutput) else make_output(reply)
        # Copy the messages: workflows extend shared history lists after the call.
        self.calls.append(FakeCall(messages=[dict(m) for m in messages], kwargs=dict(kwargs), output=output))
        return output

    def calls_for(self, agent_name: str) -> list[FakeCall]:
        """Recorded calls with this ``agent_name``, in call order."""
        return [c for c in self.calls if c.agent_name == agent_name]

    @property
    def agent_names(self) -> list[str | None]:
        """``agent_name`` of every call, in call order."""
        return [c.agent_name for c in self.calls]
