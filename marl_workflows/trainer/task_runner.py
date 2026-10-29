"""Ray task runner that builds the worker groups and runs :class:`MultiAgentWorkflowTrainer`.

``MultiAgentTaskRunner.run`` is a bounded copy of rllm@186d0bbb
``rllm/trainer/verl/train_agent_ppo.py`` ``TaskRunner.run`` (lines 63-154, 169-190, 219-223;
workflow branch only), with every change fenced by ``# >>> MARL(<id>)`` / ``# <<< MARL(<id>)``:

- T1: register the multi-LoRA vLLM replica (``patches.apply_driver_patches``) first.
- T2: FSDP only; the actor/rollout worker is ``MultiAgentAsyncActorRolloutRefWorker`` (async rollout).
- T3: the trainer is ``MultiAgentWorkflowTrainer``.
- T4: the agent, environment and SDK branches are dropped.

The WP2 modules are imported inside ``run`` so that importing this module stays light.
"""

from __future__ import annotations

import os
import socket

import ray
from verl.trainer.ppo.reward import load_reward_manager

__all__ = ["MultiAgentTaskRunner"]


# COPIED FROM rllm@186d0bbb rllm/trainer/verl/train_agent_ppo.py:63-154, 169-190, 219-223; diff regions marked MARL(...).
# fmt: off
@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class MultiAgentTaskRunner:
    """Ray actor running one training job (see the module docstring for the changes from upstream)."""

    # >>> MARL(T4): workflow-only signature
    def run(self, config, workflow_class, workflow_args=None):
        # <<< MARL(T4)
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf
        from verl.single_controller.ray import RayWorkerGroup
        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        OmegaConf.register_new_resolver("mul", lambda x, y: int(x) * int(y))
        OmegaConf.resolve(config)
        pprint(OmegaConf.to_container(config))

        # >>> MARL(T1): replicas must spawn the multi-LoRA vLLM server before any worker group exists
        from marl_workflows.verl_ext.patches import apply_driver_patches

        apply_driver_patches()
        # <<< MARL(T1)

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Starting from verl 0.6.1, this cls has been standardized for both fsdp and megatron backends.
        ray_worker_group_cls = RayWorkerGroup
        # Define worker classes based on the actor strategy.
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}
            # >>> MARL(T2): multi-adapter worker; the reference worker (only built without LoRA) stays upstream's
            from verl.workers.fsdp_workers import ActorRolloutRefWorker

            from marl_workflows.verl_ext.workers import MultiAgentAsyncActorRolloutRefWorker
            # <<< MARL(T2)

            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                # import warnings
                # warnings.warn(f"Legacy worker impl is going to be deprecated, will be removed in the future. \
                #   Please set trainer.use_legacy_worker_impl = false to switch to the new worker implementation.")
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import CriticWorker

                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

            # >>> MARL(T2): only the async rollout carries per-request LoRA ids
            if config.actor_rollout_ref.rollout.mode != "async":
                raise ValueError("actor_rollout_ref.rollout.mode must be 'async'")
            actor_rollout_cls = MultiAgentAsyncActorRolloutRefWorker
        else:
            raise NotImplementedError(f"actor strategy {config.actor_rollout_ref.actor.strategy!r} is not supported (fsdp/fsdp2 only)")
        # <<< MARL(T2)

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # Map roles to their corresponding remote worker classes.
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        # Define the resource pool specification.
        # Map roles to the resource pool.
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # Add a reference policy worker if KL loss or KL reward is used.
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # Load the reward manager for training and validation.
        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {}))
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # >>> MARL(T3): workflow branch only, with the multi-agent trainer
        from marl_workflows.trainer.trainer import MultiAgentWorkflowTrainer

        if workflow_class is not None:
            # <<< MARL(T3)
            workflow_args = workflow_args or {}
            if config.rllm.workflow.get("workflow_args") is not None:
                for key, value in config.rllm.workflow.get("workflow_args").items():
                    if value is not None:
                        if key in workflow_args and isinstance(workflow_args[key], dict):
                            workflow_args[key].update(value)
                        else:
                            workflow_args[key] = value

            # >>> MARL(T3)
            trainer = MultiAgentWorkflowTrainer(
                # <<< MARL(T3)
                config=config,
                tokenizer=tokenizer,
                processor=processor,
                role_worker_mapping=role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
                workflow_class=workflow_class,
                workflow_args=workflow_args,
            )

        # >>> MARL(T4): no agent/env or SDK trainer
        else:
            raise ValueError("workflow_class is required")
        # <<< MARL(T4)

        trainer.init_workers()
        try:
            trainer.fit_agent()
        finally:
            trainer.shutdown()
# fmt: on
