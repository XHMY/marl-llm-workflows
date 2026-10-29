"""Math reward: port of fork ``tests/rewards/test_math_reward.py`` against our subclass.

Every case runs through ``LastThinkRewardMathFn`` (and, with the default config, through
``math_reward_fn``), which must grade the text after the last ``</think>``.
"""

from __future__ import annotations

import pytest
from rllm.agents.agent import Action
from rllm.rewards.math_reward import RewardMathFn
from rllm.rewards.reward_types import RewardConfig, RewardType

from marl_workflows.rewards.math_reward import LastThinkRewardMathFn, math_reward_fn

pytestmark = pytest.mark.stack


def _task(ground_truth, **extra) -> dict:
    return {
        "problem": "",
        "problem_type": RewardType.MATH,
        "data_source": "test",
        "ground_truth": ground_truth,
        **extra,
    }


# (model_response, ground_truth, expected is_correct, expected reward) with the default config.
DEFAULT_CASES = {
    "correct_simple": ("<think>2 + 2 = 4</think>\nThe answer is \\boxed{4}.", "4", True, 1.0),
    "multiple_ground_truths": (
        "<think>The answer could be written as a fraction or decimal</think>\nThe answer is \\boxed{0.5}.",
        ["0.5", "1/2", "one half"],
        True,
        1.0,
    ),
    "incorrect": ("<think>2 + 2 = 5</think>\nThe answer is \\boxed{5}.", "4", False, 0.0),
    "format_no_delimiter_no_box": ("The answer is 4.", "4", False, 0.0),
    "format_no_boxed_answer": ("<think>2 + 2 = 4</think>\nThe answer is 4.", "4", False, 0.0),
    "no_ground_truth": ("<think>2 + 2 = 4</think>\nThe answer is \\boxed{4}.", None, False, 0.0),
    "wrong_problem_type": (
        "<think>This is not a math problem</think>\nSome code here.",
        "def add(a, b): return a + b",
        False,
        0.0,
    ),
    "complex_expression": (
        "<think>Let me solve this quadratic equation</think>\nThe answer is \\boxed{x = 2, x = -3}.",
        "x = 2, x = -3",
        True,
        1.0,
    ),
    "latex_in_ground_truth": (
        "<think>Let me calculate this carefully</think>\nThe answer is \\boxed{24}.",
        "\\boxed{24}",
        True,
        1.0,
    ),
    "numerical": (
        "<think>The answer is approximately pi</think>\nThe answer is \\boxed{3.14159}.",
        "3.14159",
        True,
        1.0,
    ),
    "fraction_equivalence": (
        "<think>Let me simplify this fraction</think>\nThe answer is \\boxed{\\frac{1}{2}}.",
        "1/2",
        True,
        1.0,
    ),
    "empty_ground_truth_list": (
        "<think>Calculating the answer</think>\nThe answer is \\boxed{42}.",
        ["\\boxed{}", ""],
        False,
        0.0,
    ),
    # The fork's change: a repeated delimiter is graded on the text after the last one.
    "repeated_delimiter": (
        "</think>\n\n</think>\n\nOkay, let's solve it again.\n\nThe answer is \\boxed{100}.",
        "100",
        True,
        1.0,
    ),
    "repeated_delimiter_answer_before_last": (
        "<think>draft \\boxed{100}</think>\n\n</think>\n\nThe solution continues without a final answer.",
        "100",
        False,
        0.0,
    ),
    "answer_between_delimiters_only": (
        "<think>a</think>\\boxed{7}</think>\nfinal: \\boxed{100}",
        "100",
        True,
        1.0,
    ),
    "empty_response": ("", "4", False, 0.0),
    "delimiter_at_end": ("<think>\\boxed{4}</think>", "4", False, 0.0),
}


@pytest.mark.parametrize("case", sorted(DEFAULT_CASES))
def test_default_config_cases(case: str) -> None:
    response, ground_truth, is_correct, reward = DEFAULT_CASES[case]
    task = _task(ground_truth)
    for fn in (LastThinkRewardMathFn(RewardConfig()), math_reward_fn):
        out = fn(dict(task), response)
        assert bool(out.is_correct) is is_correct, (case, fn)
        assert out.reward == reward, (case, fn)


def test_upstream_differs_on_repeated_delimiter() -> None:
    """Upstream grades the first segment; this is why the subclass exists."""
    response = DEFAULT_CASES["repeated_delimiter"][0]
    assert not RewardMathFn(RewardConfig())(_task("100"), response).is_correct
    assert LastThinkRewardMathFn(RewardConfig())(_task("100"), response).is_correct


def test_single_delimiter_matches_upstream() -> None:
    for case, (response, ground_truth, _, _) in DEFAULT_CASES.items():
        if response.count("</think>") > 1:
            continue
        ours = LastThinkRewardMathFn(RewardConfig())(_task(ground_truth), response)
        upstream = RewardMathFn(RewardConfig())(_task(ground_truth), response)
        assert (ours.is_correct, ours.reward) == (upstream.is_correct, upstream.reward), case


def test_toolcall_bonus() -> None:
    config = RewardConfig()
    config.toolcall_bonus = 0.5
    config.correct_reward = 1.0
    out = LastThinkRewardMathFn(config)(
        _task("4", has_toolcall=True), "<think>I'll use calculation tools</think>\nThe answer is \\boxed{4}."
    )
    assert out.is_correct is True
    assert out.reward == 1.5


def test_custom_config_values() -> None:
    config = RewardConfig()
    config.correct_reward = 2.0
    config.incorrect_reward = -1.0
    config.format_error_reward = -0.5
    fn = LastThinkRewardMathFn(config)
    task = _task("10")
    assert fn(task, "<think>5 + 5 = 10</think>\nThe answer is \\boxed{10}.").reward == 2.0
    assert fn(task, "<think>5 + 5 = 11</think>\nThe answer is \\boxed{11}.").reward == -1.0
    out = fn(task, "The answer is 10.")
    assert not out.is_correct and out.reward == -0.5


def test_format_reward_requires_delimiter() -> None:
    config = RewardConfig()
    config.apply_format_reward = True
    fn = LastThinkRewardMathFn(config)
    assert not fn(_task("4"), "The answer is \\boxed{4}.").is_correct
    assert fn(_task("4"), "x</think>y</think>The answer is \\boxed{4}.").is_correct


def test_action_object_is_unwrapped() -> None:
    out = math_reward_fn(_task("4"), Action(action="<think>t</think>\\boxed{4}"))
    assert out.is_correct is True
