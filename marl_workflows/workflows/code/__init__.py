"""Code (DeepCoder) workflow classes named by ``marl_workflows.recipes``.

Each module is a thin task binding over a generic workflow in ``marl_workflows.workflows``:
prompts (``prompts/deepcoder.json``), the sandboxed code reward
(``rewards.code_reward.code_reward_fn``, run through ``run_in_code_executor``) and answer
parsing. Ported from the single-pass paths of the fork's ``examples/deepcoder/*_workflow.py``;
the test-feedback loop (``CodeTestLoopMixin``) is not ported, since every paper run had
``enable_test_loop=False``. The constructors keep its kwargs so that configs written for the
fork still load, and reject ``enable_test_loop=True``.
"""


def reject_test_loop(enable_test_loop: bool) -> None:
    """Raise if the dropped test-feedback loop is requested."""
    if enable_test_loop:
        raise ValueError(
            "enable_test_loop=True is not supported: the code test-feedback loop was not ported "
            "(no paper run used it). Use enable_test_loop=False."
        )
