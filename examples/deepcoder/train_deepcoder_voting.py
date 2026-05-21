"""Training entry point for Deepcoder Voting Workflow.

This script trains a 2-agent voting workflow for code generation
using reinforcement learning with parallel generation and test loop.

The workflow pattern:
- Generator: Creates N code solutions in parallel
- Aggregator: Reviews all solutions and selects the best one
- Test execution: If tests fail, regenerate with test feedback
- Loop until tests pass or max_test_rounds reached

Usage:
    python -m examples.deepcoder.train_deepcoder_voting \
        trainer.agent_names=['generator','aggregator'] \
        +rllm.workflow.n_votes=3 \
        +rllm.workflow.max_test_rounds=2
"""

import hydra

from examples.deepcoder.deepcoder_voting_workflow import DeepcodeVotingWorkflow
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
    n_votes = 3
    enable_test_loop = False  # Default: disabled
    max_test_rounds = 2
    max_tests_to_show = 3
    public_test_only = False
    use_final_outcome_reward = False

    if hasattr(config, "rllm") and hasattr(config.rllm, "workflow"):
        n_votes = getattr(config.rllm.workflow, "n_votes", 3)
        enable_test_loop = getattr(config.rllm.workflow, "enable_test_loop", False)
        max_test_rounds = getattr(config.rllm.workflow, "max_test_rounds", 2)
        max_tests_to_show = getattr(config.rllm.workflow, "max_tests_to_show", 3)
        public_test_only = getattr(config.rllm.workflow, "public_test_only", False)
        use_final_outcome_reward = getattr(config.rllm.workflow, "use_final_outcome_reward", False)

    trainer = AgentTrainer(
        workflow_class=DeepcodeVotingWorkflow,
        workflow_args={
            "n_votes": n_votes,
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
