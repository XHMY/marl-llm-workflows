"""Training entry point for the Evaluator-Optimizer Math Workflow.

This script trains a 2-agent evaluator-optimizer workflow for mathematical
problem solving using reinforcement learning.

Usage:
    python -m examples.math_reasoning.train_evaluator_optimizer_math \
        trainer.agent_names=['generator','evaluator'] \
        +rllm.workflow.max_iterations=3
"""

import hydra

from examples.math_reasoning.evaluator_optimizer_math_workflow import (
    EvaluatorOptimizerMathWorkflow,
)
from rllm.data.dataset import DatasetRegistry
from rllm.rewards.reward_fn import math_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="multi_agent_ppo_trainer",
    version_base=None,
)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("dapo_math", "train")
    test_dataset = DatasetRegistry.load_dataset("aime2025", "test")

    assert train_dataset is not None, (
        "Failed to load train dataset. "
        "Please run examples/math_reasoning/prepare_dapo_math_data.py first."
    )
    assert test_dataset is not None, (
        "Failed to load test dataset. "
        "Please run examples/math_reasoning/prepare_aime_test_data.py first."
    )

    # Get workflow config options
    max_iterations = 3
    use_final_outcome_reward = False
    if hasattr(config, "rllm") and hasattr(config.rllm, "workflow"):
        max_iterations = getattr(config.rllm.workflow, "max_iterations", 3)
        use_final_outcome_reward = getattr(
            config.rllm.workflow, "use_final_outcome_reward", False
        )

    # Get initial_lora_weights from config if specified
    initial_lora_weights = None
    if hasattr(config, "rllm") and hasattr(config.rllm, "workflow"):
        initial_lora_weights_cfg = getattr(
            config.rllm.workflow, "initial_lora_weights", None
        )
        if initial_lora_weights_cfg is not None:
            initial_lora_weights = dict(initial_lora_weights_cfg)

    trainer = AgentTrainer(
        workflow_class=EvaluatorOptimizerMathWorkflow,
        workflow_args={
            "max_iterations": max_iterations,
            "reward_function": math_reward_fn,
            "use_final_outcome_reward": use_final_outcome_reward,
            "initial_lora_weights": initial_lora_weights,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
