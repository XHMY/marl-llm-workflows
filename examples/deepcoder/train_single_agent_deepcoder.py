"""Training entry point for Single-Agent DeepCoder Workflow.

This script trains a single-agent workflow for code generation
using reinforcement learning. It's a simpler baseline compared to
the multi-agent evaluator-optimizer and voting workflows.

The workflow pattern:
- Generator: Creates code solution (with optional test loop for refinement)
- Reward: Computed via test execution (code_reward_fn)

Usage:
    python -m examples.deepcoder.train_single_agent_deepcoder \
        trainer.share_policy=True \
        +rllm.workflow.max_test_rounds=2
"""

import hydra

from examples.deepcoder.single_agent_deepcoder_workflow import (
    SingleAgentDeepcodeWorkflow,
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
    enable_test_loop = False  # Default: disabled
    max_test_rounds = 2
    max_tests_to_show = 3
    public_test_only = False

    if hasattr(config, "rllm") and hasattr(config.rllm, "workflow"):
        enable_test_loop = getattr(config.rllm.workflow, "enable_test_loop", False)
        max_test_rounds = getattr(config.rllm.workflow, "max_test_rounds", 2)
        max_tests_to_show = getattr(config.rllm.workflow, "max_tests_to_show", 3)
        public_test_only = getattr(config.rllm.workflow, "public_test_only", False)

    trainer = AgentTrainer(
        workflow_class=SingleAgentDeepcodeWorkflow,
        workflow_args={
            "enable_test_loop": enable_test_loop,
            "max_test_rounds": max_test_rounds,
            "max_tests_to_show": max_tests_to_show,
            "public_test_only": public_test_only,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
