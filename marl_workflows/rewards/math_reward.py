"""Math reward: upstream ``RewardMathFn`` graded on the text after the last ``</think>``.

Upstream (rllm@186d0bbb ``rllm/rewards/math_reward.py:51``) grades
``response.split("</think>")[1]``, the text between the first and the second delimiter. A
response that repeats the delimiter is then graded on an intermediate segment. The fork
changed that line to ``rsplit("</think>", 1)[1]`` (fork ``rllm/rewards/math_reward.py:49-52``)
and every Math number of the paper was computed with it.

Instead of copying ``RewardMathFn``, the subclass below rewrites the action so that the
upstream ``split(...)[1]`` yields exactly the text after the last delimiter: everything up
to the last ``</think>`` is replaced by a single delimiter. Grading (mathd OR sympy) and the
reward values are upstream's.
"""

from __future__ import annotations

from typing import Any

from rllm.agents.agent import Action
from rllm.globals import THOUGHT_DELIMITER_END
from rllm.rewards.math_reward import RewardMathFn
from rllm.rewards.reward_types import RewardConfig, RewardOutput

__all__ = ["LastThinkRewardMathFn", "math_reward_fn"]


class LastThinkRewardMathFn(RewardMathFn):
    """``RewardMathFn`` that grades the text after the last ``</think>``."""

    def __call__(self, task_info: dict, action: str) -> RewardOutput:
        if action and THOUGHT_DELIMITER_END in action:
            # Upstream takes split(delim)[1]; with one leading delimiter that is the tail
            # after the last delimiter in the original response.
            action = THOUGHT_DELIMITER_END + action.rsplit(THOUGHT_DELIMITER_END, 1)[1]
        return super().__call__(task_info, action)


def math_reward_fn(task_info: dict, action: Any) -> RewardOutput:
    """Math reward function (mirrors rllm@186d0bbb ``rllm/rewards/reward_fn.py:47-62``)."""
    reward_fn = LastThinkRewardMathFn(RewardConfig())
    if isinstance(action, Action):
        action = action.action
    return reward_fn(task_info, action)
