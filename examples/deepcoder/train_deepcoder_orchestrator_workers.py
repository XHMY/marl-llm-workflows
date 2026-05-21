"""Training entry point for the Deepcoder Orchestrator-Workers Workflow.

This script trains a 3-agent orchestrator-workers workflow for code generation
using reinforcement learning with strategy proposal.

Usage:
    python -m examples.deepcoder.train_deepcoder_orchestrator_workers \
        trainer.agent_names=['orchestrator','worker','synthesizer'] \
        +rllm.workflow.max_subtasks=3
"""

import hydra

from examples.deepcoder.deepcoder_orchestrator_workers_workflow import (
    DeepcodeOrchestratorWorkersWorkflow,
)
from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="multi_agent_ppo_trainer",
    version_base=None,
)
def main(config):
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

    # Get workflow config parameters
    max_subtasks = 3
    use_final_outcome_reward = True
    if hasattr(config, "rllm") and hasattr(config.rllm, "workflow"):
        max_subtasks = getattr(config.rllm.workflow, "max_subtasks", 3)
        use_final_outcome_reward = getattr(
            config.rllm.workflow, "use_final_outcome_reward", True
        )

    # Get initial_lora_weights from config if specified
    # Remap "generator" -> "worker" since checkpoints use "generator" but
    # this workflow's agent name is "worker"
    initial_lora_weights = None
    if hasattr(config, "rllm") and hasattr(config.rllm, "workflow"):
        initial_lora_weights_cfg = getattr(
            config.rllm.workflow, "initial_lora_weights", None
        )
        if initial_lora_weights_cfg is not None:
            initial_lora_weights = dict(initial_lora_weights_cfg)
            # Remap: checkpoint adapter "generator" -> workflow agent "worker"
            if "generator" in initial_lora_weights and "worker" not in initial_lora_weights:
                initial_lora_weights["worker"] = initial_lora_weights.pop("generator")

    trainer = AgentTrainer(
        workflow_class=DeepcodeOrchestratorWorkersWorkflow,
        workflow_args={
            "max_subtasks": max_subtasks,
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
