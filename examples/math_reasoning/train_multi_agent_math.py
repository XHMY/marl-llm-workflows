import hydra

from examples.math_reasoning.multi_agent_math_workflow import MultiAgentMathWorkflow
from rllm.data.dataset import DatasetRegistry
from rllm.rewards.reward_fn import math_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="multi_agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("dapo_math", "train")
    test_dataset = DatasetRegistry.load_dataset("dapo_math", "test")

    assert train_dataset is not None, (
        "Failed to load train dataset. "
        "Please run examples/math_reasoning/prepare_dapo_math_data.py first."
    )
    assert test_dataset is not None, (
        "Failed to load test dataset. "
        "Please run examples/math_reasoning/prepare_dapo_math_data.py first."
    )

    trainer = AgentTrainer(
        workflow_class=MultiAgentMathWorkflow,
        workflow_args={
            "max_refinement_iterations": 3,
            "reward_function": math_reward_fn,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
