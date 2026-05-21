"""Training entry point for Deepcoder Evaluator-Optimizer Workflow.

This script trains a 2-agent evaluator-optimizer workflow for code generation
using reinforcement learning with nested test loops.

The workflow pattern:
- Inner loop: Generator -> Evaluator (logic review) -> Refine
- Outer loop: Run tests -> If fail, new inner loop with test feedback

Usage:
    python -m examples.deepcoder.train_deepcoder_evaluator_optimizer \
        trainer.agent_names=['generator','evaluator'] \
        +rllm.workflow.max_iterations=2 \
        +rllm.workflow.max_test_rounds=2
"""

import hydra

from examples.deepcoder.deepcoder_evaluator_optimizer_workflow import (
    DeepcodeEvaluatorOptimizerWorkflow,
)
from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="multi_agent_ppo_trainer",
    version_base=None,
)
def main(config):
    # Load datasets
    dataset_name = getattr(config.data, "dataset_name", "deepcoder")
    train_dataset = DatasetRegistry.load_dataset(dataset_name, "train")
    test_dataset = DatasetRegistry.load_dataset(dataset_name, "test")

    assert train_dataset is not None, (
        "Failed to load train dataset. "
        "Please run examples/deepcoder/prepare_deepcoder_data.py first."
    )
    assert test_dataset is not None, (
        "Failed to load test dataset. "
        "Please run examples/deepcoder/prepare_deepcoder_data.py first."
    )

    # Get workflow config from hydra config
    max_iterations = 2  # Inner eval-opt loop
    enable_test_loop = False  # Default: disabled
    max_test_rounds = 2  # Outer test loop (when enable_test_loop=True)
    max_tests_to_show = 3
    public_test_only = False
    use_final_outcome_reward = False

    if hasattr(config, "rllm") and hasattr(config.rllm, "workflow"):
        max_iterations = getattr(config.rllm.workflow, "max_iterations", 2)
        enable_test_loop = getattr(config.rllm.workflow, "enable_test_loop", False)
        max_test_rounds = getattr(config.rllm.workflow, "max_test_rounds", 2)
        max_tests_to_show = getattr(config.rllm.workflow, "max_tests_to_show", 3)
        public_test_only = getattr(config.rllm.workflow, "public_test_only", False)
        use_final_outcome_reward = getattr(config.rllm.workflow, "use_final_outcome_reward", False)

    trainer = AgentTrainer(
        workflow_class=DeepcodeEvaluatorOptimizerWorkflow,
        workflow_args={
            "max_iterations": max_iterations,
            "enable_test_loop": enable_test_loop,
            "max_test_rounds": max_test_rounds,
            "max_tests_to_show": max_tests_to_show,
            "public_test_only": public_test_only,
            "use_final_outcome_reward": use_final_outcome_reward,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
