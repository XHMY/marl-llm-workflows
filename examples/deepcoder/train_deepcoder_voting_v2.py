"""Training entry point for Deepcoder Voting V2 Workflow.

This script trains a multi-agent voting v2 workflow for code generation
where each voter uses a distinct LoRA adapter.

Usage:
    python -m examples.deepcoder.train_deepcoder_voting_v2 \
        trainer.agent_names=['voterA','voterB','voterC','aggregator'] \
        +rllm.workflow.n_votes=3 \
        +rllm.workflow.enable_test_loop=False
"""

import hydra

from examples.deepcoder.deepcoder_voting_v2_workflow import DeepcodeVotingV2Workflow
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
    voter_names = None
    enable_test_loop = False
    max_test_rounds = 2
    max_tests_to_show = 3
    public_test_only = False
    use_final_outcome_reward = False
    use_rubric_reward = False

    if hasattr(config, "rllm") and hasattr(config.rllm, "workflow"):
        n_votes = getattr(config.rllm.workflow, "n_votes", 3)
        enable_test_loop = getattr(config.rllm.workflow, "enable_test_loop", False)
        max_test_rounds = getattr(config.rllm.workflow, "max_test_rounds", 2)
        max_tests_to_show = getattr(config.rllm.workflow, "max_tests_to_show", 3)
        public_test_only = getattr(config.rllm.workflow, "public_test_only", False)
        use_final_outcome_reward = getattr(config.rllm.workflow, "use_final_outcome_reward", False)
        use_rubric_reward = getattr(config.rllm.workflow, "use_rubric_reward", False)

        voter_names_cfg = getattr(config.rllm.workflow, "voter_names", None)
        if voter_names_cfg is not None:
            voter_names = list(voter_names_cfg)

    trainer = AgentTrainer(
        workflow_class=DeepcodeVotingV2Workflow,
        workflow_args={
            "n_votes": n_votes,
            "voter_names": voter_names,
            "enable_test_loop": enable_test_loop,
            "max_test_rounds": max_test_rounds,
            "max_tests_to_show": max_tests_to_show,
            "public_test_only": public_test_only,
            "use_final_outcome_reward": use_final_outcome_reward,
            "use_rubric_reward": use_rubric_reward,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
