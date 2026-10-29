"""Training entry point: ``python -m marl_workflows.trainer.train marl.task=<T> marl.workflow=<W> ...``.

Replaces rLLM's ``AgentTrainer._train_verl`` (rllm@186d0bbb ``rllm/trainer/agent_trainer.py:124-154``)
and the research fork's per-workflow ``examples/*/train_*.py`` scripts: the recipe named by
``marl.task``/``marl.workflow`` supplies the workflow class, its kwargs and the datasets.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, open_dict

from marl_workflows.data.registry import MarlDatasetRegistry
from marl_workflows.recipes import build_workflow_args, get_recipe, load_object
from marl_workflows.trainer.config_injection import inject_multi_agent_config

__all__ = ["main", "run"]


@hydra.main(config_path="pkg://marl_workflows.configs", config_name="multi_agent_ppo_trainer", version_base=None)
def main(config: DictConfig) -> None:
    run(config)


def run(config: DictConfig) -> None:
    """Inject the multi-agent keys, point verl at the prepared datasets and run the Ray task runner."""
    import ray
    from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env

    from marl_workflows.trainer.task_runner import MultiAgentTaskRunner

    if not config.marl.get("task") or not config.marl.get("workflow"):
        raise ValueError("Set marl.task (math|deepcoder) and marl.workflow (see marl_workflows.recipes.RECIPES)")
    recipe = get_recipe(config.marl.task, config.marl.workflow)
    inject_multi_agent_config(config, recipe)

    train_files = MarlDatasetRegistry.verl_path(config.marl.train_dataset, "train")
    val_files = MarlDatasetRegistry.verl_path(config.marl.val_dataset, "test")
    with open_dict(config):
        config.data.train_files = train_files
        config.data.val_files = val_files

    workflow_cls = load_object(recipe.workflow_cls)
    workflow_args = build_workflow_args(recipe, config)

    if not ray.is_initialized():
        # read off all the `ray_init` settings from the config (as upstream _train_verl)
        ray_init_settings = {k: v for k, v in (config.get("ray_init") or {}).items() if v is not None}
        ray.init(runtime_env=get_ppo_ray_runtime_env(), **ray_init_settings)

    runner = MultiAgentTaskRunner.remote()
    ray.get(runner.run.remote(config=config, workflow_class=workflow_cls, workflow_args=workflow_args))


if __name__ == "__main__":
    main()
