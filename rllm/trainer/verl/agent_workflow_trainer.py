import asyncio
import math
import os
import re
import shutil
import threading
import uuid
from collections import Counter, defaultdict
from functools import reduce
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.core_algos import (
    AdvantageEstimator,
    agg_loss,
)
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    apply_kl_penalty,
    compute_advantage,
)
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.debug import marked_timer

from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout.verl_engine import VerlEngine
from rllm.utils.episode_logger import EpisodeLogger
from rllm.workflows.workflow import TerminationReason


class AgentWorkflowPPOTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        workflow_class=None,
        workflow_args=None,
    ):
        super().__init__(config=config, tokenizer=tokenizer, processor=processor, role_worker_mapping=role_worker_mapping, resource_pool_manager=resource_pool_manager, ray_worker_group_cls=ray_worker_group_cls, reward_fn=reward_fn, val_reward_fn=val_reward_fn)

        self.workflow_class = workflow_class
        self.workflow_args = workflow_args or {}
        self._validate_config()

        # Initialize teacher engine if distillation is enabled
        self.distill_enabled = self.config.rllm.get("distill", {}).get("enable", False)
        if self.distill_enabled:
            print("Distillation is enabled, will ignore rewards returned in episodes.")
        self.teacher_engine = None
        self.teacher_tokenizer = None
        if self.distill_enabled:
            from transformers import AutoTokenizer

            from rllm.engine.rollout.openai_engine import OpenAIEngine

            teacher_rollout_args = self.config.rllm.distill.get("teacher_rollout_args", {})
            teacher_model = teacher_rollout_args.get("model", "")
            if not teacher_model:
                raise ValueError("model must be specified in rllm.distill.teacher_rollout_args when distillation is enabled")

            self.teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model)
            self.teacher_engine = OpenAIEngine(
                **teacher_rollout_args,
                tokenizer=self.teacher_tokenizer,
            )

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def _validate_config(self):
        assert self.workflow_class is not None, "workflow_class is required for agent workflow trainer"
        assert self.config.actor_rollout_ref.hybrid_engine is True, "Only hybrid engine is supported"
        assert self.config.actor_rollout_ref.rollout.mode == "async", "Only async rollout mode is supported"
        assert self.use_rm is False, "Reward models are not supported. Rewards should be assigned using a reward function in the workflow or environment."
        if self.config.rllm.rejection_sample.multiplier != 1:
            assert self.config.rllm.rejection_sample.enable is True, "rejection sampling is disabled, but rejection_sample.multiplier is not 1"

        # TODO: revisit whether this is now supported by Verl
        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
            raise NotImplementedError("REMAX is not supported yet")

    def init_workers(self):
        super().init_workers()

        rollout_engine = VerlEngine(
            config=self.config,
            rollout_manager=self.async_rollout_manager,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )

        # Create episode logger if enabled in config
        episode_logger = None
        if self.config.trainer.get("log_episodes", False):
            # Get episode log directory from config, default to "logs/my_project/my_experiment"
            episode_log_dir = self.config.trainer.get("episode_log_dir", f"logs/{self.config.trainer.project_name}/{self.config.trainer.experiment_name}")
            episode_logger = EpisodeLogger(base_dir=episode_log_dir, subdirectory="episodes")

        self.agent_execution_engine = AgentWorkflowEngine(
            workflow_cls=self.workflow_class,
            workflow_args=self.workflow_args,
            rollout_engine=rollout_engine,
            config=self.config,
            n_parallel_tasks=self.config.rllm.workflow.n_parallel_tasks,
            retry_limit=self.config.rllm.workflow.retry_limit,
            episode_logger=episode_logger,
            code_executor_workers=getattr(self.config.rllm.workflow, "code_executor_workers", 0),
            max_concurrent_code_execs=getattr(self.config.rllm.workflow, "max_concurrent_code_execs", 0),
        )

        # init workflow workers
        asyncio.run_coroutine_threadsafe(self.agent_execution_engine.initialize_pool(), self._loop).result()

    def _split_batch_by_agent(self, batch: DataProto) -> dict[str, tuple[DataProto, list[int]]]:
        """Split batch by agent name extracted from trajectory_ids.

        Agent names are embedded in trajectory_ids as a suffix (e.g., "traj_123_agent_0").
        Extract via traj_id.rsplit("_", 1)[1].

        Each agent's sub-batch is padded to be divisible by world_size for distributed
        training. The padding is transparent to callers since scatter-back logic only
        iterates over original indices.

        Args:
            batch: The full batch with trajectory_ids in non_tensor_batch

        Returns:
            Dict mapping agent_name -> (sub_batch, original_indices)
            Note: sub_batch may contain padding samples, but original_indices only
            contains the non-padded sample indices for scatter-back operations.
        """
        trajectory_ids = batch.non_tensor_batch["trajectory_ids"]
        # Extract agent name from trajectory_id (format: {uid}_{agent_name})
        # Strip numeric suffix to get base agent name for LoRA routing
        # (e.g., "generator0" → "generator" for VotingWorkflow)
        raw_names = [traj_id.rsplit("_", 1)[1] for traj_id in trajectory_ids]
        agent_names = [re.sub(r'\d+$', '', name) for name in raw_names]

        # Group indices by agent name
        agent_to_indices = defaultdict(list)
        for idx, agent_name in enumerate(agent_names):
            agent_to_indices[agent_name].append(idx)

        # Get world_size for padding
        world_size = self.actor_rollout_wg.world_size

        # Create sub-batches with padding
        result = {}
        for agent_name, indices in agent_to_indices.items():
            sub_batch = batch.select_idxs(indices)

            # Pad sub_batch to be divisible by world_size
            if world_size > 0 and len(sub_batch) % world_size != 0:
                sub_batch, _ = pad_dataproto_to_divisor(sub_batch, world_size)

            # Return original indices (without padding) for scatter-back
            result[agent_name] = (sub_batch, indices)

        return result

    def fit_agent(self):
        """
        The training loop of PPO. Adapted to train the underlying model of agent.
        """
        from rllm.utils.tracking import Tracking

        self.global_steps = 0

        # Load checkpoint first to restore global_steps
        self._load_checkpoint()

        # Determine resume parameters for wandb logging
        resume_run_id = None
        resume_mode = None

        if self.global_steps > 0:
            # We're resuming from a checkpoint, try to resume wandb logging
            metadata = self._load_training_metadata()
            resume_run_id = metadata.get("wandb_run_id")
            resume_mode = "allow"  # Use "allow" to handle deleted runs gracefully

            if resume_run_id:
                print(f"Resuming wandb run: {resume_run_id}")
            else:
                print("No wandb run_id in metadata, will attempt to find by experiment_name")

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
            resume_run_id=resume_run_id,
            resume_mode=resume_mode,
        )

        # Store logger reference for metadata saving
        self._tracking_logger = logger

        # perform validation before training
        import time

        start_time = time.time()
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            self.agent_execution_engine.set_training_step(self.global_steps, mode="val", epoch=0)
            val_metrics = self._validate_agent()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        print(f"Time taken to validate agent: {time.time() - start_time}")
        # we start from step 1
        self.global_steps += 1

        batch = None
        solve_none = 0
        solve_all = 0
        solve_partial = 0
        num_tasks = 0
        termination_counts = Counter()
        workflow_metrics = defaultdict(list)
        metrics = {}
        timing_raw = {}

        for epoch in range(self.config.trainer.total_epochs):
            pprint(f"epoch {epoch}, step {self.global_steps} started")
            for batch_dict in self.train_dataloader:
                do_profile = self.global_steps in self.config.trainer.profile_steps if self.config.trainer.get("profile_steps") is not None else False
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(do_profile)

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_tasks += len(new_batch.batch)

                new_batch.non_tensor_batch["task_ids"] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n)

                new_batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"], non_tensor_batch_keys=["raw_prompt_ids"])

                # Update training step in engine for episode logging
                self.agent_execution_engine.set_training_step(self.global_steps, mode="train", epoch=epoch)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate trajectories
                    final_gen_batch_output = self.generate_trajectories(batch=new_batch, timing_raw=timing_raw)

                    # need to repeat to make shape match
                    repeat_counts = final_gen_batch_output.meta_info["repeat_counts"]
                    new_batch = new_batch.sample_level_repeat(repeat_counts)
                    final_gen_batch_output.meta_info.pop("repeat_counts", None)  # no longer needed after this
                    new_batch = new_batch.union(final_gen_batch_output)

                    # rejection sampling
                    # we do rejection sampling at the episode level instead of the traj/step level
                    uids = new_batch.non_tensor_batch["task_ids"]
                    unique_uids = np.unique(uids)
                    is_correct = new_batch.non_tensor_batch["is_correct"]
                    drop_uids = set()

                    for uid in unique_uids:
                        candidate_rows = uids == uid
                        candidate_is_correct = is_correct[candidate_rows]

                        # Check if all episodes are correct or incorrect
                        if not candidate_is_correct.any():
                            drop_uids.add(uid)
                            solve_none += 1
                        elif candidate_is_correct.all():
                            drop_uids.add(uid)
                            solve_all += 1
                        else:
                            solve_partial += 1

                    # Build a view with a single item per episode_id for metrics/logging
                    seen_episodes = set()
                    episode_unique_idxs = []
                    for i, episode_id in enumerate(new_batch.non_tensor_batch["episode_ids"]):
                        if episode_id not in seen_episodes:
                            seen_episodes.add(episode_id)
                            episode_unique_idxs.append(i)
                    episode_unique_batch = new_batch.select_idxs(episode_unique_idxs)

                    # log metrics returned by workflows
                    for metric_dict in episode_unique_batch.non_tensor_batch["metrics"]:
                        for key, value in metric_dict.items():
                            workflow_metrics[key].append(value)

                    # collect and log termination reasons
                    termination_reasons = episode_unique_batch.non_tensor_batch["termination_reasons"]
                    termination_counts.update(termination_reasons)

                    # If no valid samples remain, skip this batch and get a new one
                    # if len(drop_uids) == len(unique_uids):
                    #     print("No valid samples remain, skipping batch")
                    #     continue

                    if not self.config.rllm.rejection_sample.enable:
                        batch = new_batch
                    else:
                        rejection_mask = np.isin(uids, list(drop_uids))
                        new_batch = new_batch[~rejection_mask]
                        if batch is None:
                            batch = new_batch
                        else:
                            batch = DataProto.concat([batch, new_batch])

                        if solve_partial < self.config.data.train_batch_size:
                            continue
                        else:
                            # randomly select bsz task uids from batch, then filter batch to only contain these tasks
                            # TODO: add heuristic for selecting train_batch_size uids
                            uids = batch.non_tensor_batch["task_ids"]
                            unique_uids = np.unique(uids)
                            assert len(unique_uids) >= self.config.data.train_batch_size, "Not enough unique uids to sample from"
                            selected_uids = np.random.choice(unique_uids, size=self.config.data.train_batch_size, replace=False)
                            selected_mask = np.isin(uids, selected_uids)
                            batch = batch[selected_mask]

                    if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "broadcast":
                        # need to make sure both number of last steps (number of uids) and number of total steps in the batch
                        # (batch size after processing) are both multiples of world size

                        # first we split the batch in two: one with only the last steps of each trajectory and the other with the remaining steps
                        is_last_step = batch.non_tensor_batch["is_last_step"]
                        valid_last_step_indices = np.where(is_last_step == True)[0]
                        not_last_step_indices = np.where(is_last_step == False)[0]
                        last_step_batch = batch.select_idxs(valid_last_step_indices)  # This batch only has valid last steps
                        non_last_step_batch = batch.select_idxs(not_last_step_indices)

                        # round down last_step_batch to make sure its multiple of world size
                        num_trainer_replicas = self.actor_rollout_wg.world_size
                        max_batch_size = (last_step_batch.batch["input_ids"].shape[0] // num_trainer_replicas) * num_trainer_replicas

                        size_mask = torch.zeros(last_step_batch.batch["input_ids"].shape[0], dtype=torch.bool)
                        size_mask[:max_batch_size] = True
                        last_step_batch = last_step_batch[size_mask]  # filtered last steps

                        # now we go through all the non_last_step_batch and keep everything that has same trajectory_id that exists in the filtered last steps
                        valid_last_step_trajectory_ids = last_step_batch.non_tensor_batch["trajectory_ids"]
                        non_last_step_trajectory_ids = non_last_step_batch.non_tensor_batch["trajectory_ids"]
                        non_last_step_mask = np.isin(non_last_step_trajectory_ids, valid_last_step_trajectory_ids)
                        non_last_step_batch = non_last_step_batch[non_last_step_mask]

                        # concatenate then pad
                        batch = DataProto.concat([last_step_batch, non_last_step_batch])
                        batch = self._pad_dataproto_to_world_size(batch)

                    else:
                        # then we just pad the batch size to a multiple of world size
                        batch = self._pad_dataproto_to_world_size(batch=batch)

                    # recompute old_log_probs - per agent if multi-agent mode
                    with marked_timer("old_log_prob", timing_raw, color="blue"):

                        if not self.config.trainer.share_policy:
                            # Multi-agent mode: compute log probs per agent
                            agent_batches = self._split_batch_by_agent(batch)

                            # Pre-allocate result tensors with response length (not full sequence length)
                            # compute_log_prob returns log probs only for response tokens
                            response_length = batch.batch["responses"].shape[1]
                            batch_size = batch.batch["attention_mask"].shape[0]
                            device = batch.batch["attention_mask"].device
                            all_old_log_probs = torch.zeros(batch_size, response_length, dtype=torch.float32, device=device)
                            all_entropys = torch.zeros(batch_size, response_length, dtype=torch.float32, device=device)

                            for agent_name, (sub_batch, indices) in agent_batches.items():
                                if len(sub_batch) == 0:
                                    continue  # Skip empty batches silently

                                # Switch to agent's LoRA adapter
                                self.actor_rollout_wg.set_active_lora(agent_role=agent_name, lora_config={})

                                # Compute log probs for this agent's samples
                                sub_log_prob = self.actor_rollout_wg.compute_log_prob(sub_batch)

                                # Scatter results back to original positions
                                for i, orig_idx in enumerate(indices):
                                    all_old_log_probs[orig_idx] = sub_log_prob.batch["old_log_probs"][i]
                                    all_entropys[orig_idx] = sub_log_prob.batch["entropys"][i]

                            # Store aggregated results
                            batch.batch["old_log_probs"] = all_old_log_probs
                            entropys = all_entropys
                        else:
                            # Single-agent mode: original behavior
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)

                        # Compute entropy metrics (common to both paths)
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)

                        if "rollout_log_probs" in batch.batch.keys():
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            debug_metrics = calculate_debug_metrics(batch)
                            metrics.update(debug_metrics)

                    if self.use_reference_policy:
                        # compute reference log_prob - per agent if multi-agent mode
                        with marked_timer("ref", timing_raw, color="olive"):

                            if not self.config.trainer.share_policy:
                                # Multi-agent mode: compute ref log probs per agent
                                agent_batches = self._split_batch_by_agent(batch)

                                # Pre-allocate result tensor with response length (not full sequence length)
                                # compute_ref_log_prob returns log probs only for response tokens
                                response_length = batch.batch["responses"].shape[1]
                                batch_size = batch.batch["attention_mask"].shape[0]
                                device = batch.batch["attention_mask"].device
                                all_ref_log_probs = torch.zeros(batch_size, response_length, dtype=torch.float32, device=device)

                                for agent_name, (sub_batch, indices) in agent_batches.items():
                                    if len(sub_batch) == 0:
                                        continue  # Skip empty batches silently

                                    # For reference policy, compute without LoRA adapter
                                    if not self.ref_in_actor:
                                        sub_ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(sub_batch)
                                    else:
                                        sub_ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(sub_batch)

                                    # Scatter results back
                                    for i, orig_idx in enumerate(indices):
                                        all_ref_log_probs[orig_idx] = sub_ref_log_prob.batch["ref_log_prob"][i]

                                batch.batch["ref_log_prob"] = all_ref_log_probs
                            else:
                                # Single-agent mode: original behavior
                                if not self.ref_in_actor:
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                else:
                                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # step_ids is safe to always use for advantage computation
                        # if we're not using computing advantages stepwise (i.e., for cumulative agents or single turn workflows)
                        # then step_ids == trajectory_ids
                        batch.non_tensor_batch["uid"] = batch.non_tensor_batch["step_ids"]

                        if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "per_step":
                            batch.batch["token_level_scores"] = batch.batch["step_rewards"]
                        else:
                            batch.batch["token_level_scores"] = batch.batch["traj_rewards"]

                        if self.distill_enabled:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                            batch = self._remove_padding(batch)
                            distill_advantages = asyncio.run_coroutine_threadsafe(self._compute_distill_advantages(batch), self._loop).result()
                            batch.batch["advantages"] = distill_advantages
                            batch.batch["returns"] = distill_advantages
                        else:
                            # apply_kl_penalty if available
                            if self.config.algorithm.use_kl_in_reward:
                                batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                            if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "broadcast":
                                is_last_step = batch.non_tensor_batch["is_last_step"]
                                last_step_indices = np.where(is_last_step == True)[0]
                                not_last_step_indices = np.where(is_last_step == False)[0]
                                non_last_step_batch = batch.select_idxs(not_last_step_indices)
                                batch = batch.select_idxs(last_step_indices)  # This batch only has last steps
                                # last_step_batch contains no padded steps as it was rounded down (not padded) to a multiple of world size
                            else:
                                batch = self._remove_padding(batch)  # compute advantages over non-padded steps only

                            # compute advantages, executed on the driver process
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=self.config.algorithm.norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )

                            if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "broadcast":
                                # Merging the separated out steps using the advantage from last steps
                                self._stepwise_advantage_broadcast(batch, non_last_step_batch)
                                batch = DataProto.concat([batch, non_last_step_batch])

                    # remove invalid items filtered out due to compact filtering
                    is_valid = batch.non_tensor_batch["is_valid"]
                    valid_idxs = np.where(is_valid == True)[0]
                    batch = batch.select_idxs(valid_idxs)

                    # for backward compatibility
                    if self.config.rllm.mask_truncated_samples:
                        mask = batch.batch["attention_mask"][:, -1] == 1
                        batch = batch[~mask]

                    # re-pad batch size to world size for gradient update
                    batch = self._pad_dataproto_to_world_size(batch=batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.get("temperature")

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor - per agent if multi-agent mode
                        with marked_timer("update_actor", timing_raw, color="red"):
                            if not self.config.trainer.share_policy:
                                # Multi-agent mode: update each agent's LoRA adapter
                                agent_batches = self._split_batch_by_agent(batch)
                                all_actor_metrics = {}

                                # Per-role overrides for L1 interventions (kept outside the actor
                                # dataclass namespace to avoid hydra-instantiate field rejection):
                                #   I1 — `+trainer.kl_loss_coef_per_role.<role>=<beta>` per-role KL anchor.
                                #        Requires `actor_rollout_ref.actor.use_kl_loss=true`; the reference
                                #        policy worker is enabled by `train_agent_ppo.py` whenever set.
                                #   I6 — `+trainer.coherence_reg.{enable,ema_window,eps}` and
                                #        `+trainer.coherence_reg.per_role_weights.<role>=<lambda>`
                                #        gradient-direction coherence regularizer (step-scaling form).
                                kl_coef_per_role = self.config.trainer.get("kl_loss_coef_per_role", None)
                                coherence_reg = self.config.trainer.get("coherence_reg", None)
                                coherence_enabled = bool(coherence_reg) and bool(coherence_reg.get("enable", False))
                                coherence_role_weights = (
                                    coherence_reg.get("per_role_weights", {}) if coherence_enabled else {}
                                )

                                for agent_name, (sub_batch, indices) in agent_batches.items():
                                    if len(sub_batch) == 0:
                                        continue  # Skip empty batches silently

                                    # Switch to agent's LoRA adapter for update
                                    self.actor_rollout_wg.set_active_lora(agent_role=agent_name, lora_config={})

                                    # Per-role L1 overrides. Copy meta_info to avoid mutating the parent
                                    # batch's dict (sub_batch.meta_info is shared by reference after
                                    # select_idxs).
                                    overrides: dict = {}
                                    if kl_coef_per_role is not None and agent_name in kl_coef_per_role:
                                        overrides["kl_loss_coef_override"] = float(kl_coef_per_role[agent_name])
                                    if coherence_enabled and agent_name in coherence_role_weights:
                                        overrides.update({
                                            "coherence_reg_role": agent_name,
                                            "coherence_reg_weight": float(coherence_role_weights[agent_name]),
                                            "coherence_reg_ema_window": int(coherence_reg.get("ema_window", 128)),
                                            "coherence_reg_eps": float(coherence_reg.get("eps", 1e-8)),
                                        })
                                    if overrides:
                                        sub_batch.meta_info = {**sub_batch.meta_info, **overrides}

                                    # Update this agent's adapter
                                    sub_actor_output = self.actor_rollout_wg.update_actor(sub_batch)

                                    # Collect metrics with agent prefix
                                    for key, value in reduce_metrics(sub_actor_output.meta_info.get("metrics", {})).items():
                                        all_actor_metrics[f"actor/{agent_name}/{key.split('/')[-1]}"] = value

                                metrics.update(all_actor_metrics)
                            else:
                                # Single-agent mode: original behavior
                                actor_output = self.actor_rollout_wg.update_actor(batch)
                                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                                metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                        with marked_timer("testing", timing_raw, color="green"):
                            self.agent_execution_engine.set_training_step(self.global_steps, mode="val", epoch=epoch)
                            val_metrics: dict = self._validate_agent()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                    # Visualize some sample trajectories
                    if batch is not None and len(batch) > 0:
                        # Randomly select a few samples to visualize
                        batch_size = len(batch)
                        num_samples = min(2, batch_size)  # Visualize up to 2 samples
                        if num_samples > 0:
                            sample_indices = np.random.choice(batch_size, size=num_samples, replace=False)
                            for idx in sample_indices:
                                self.visualize_trajectory_last_step(batch, sample_idx=idx, max_samples=1)

                with marked_timer("stop_profile", timing_raw):
                    self._stop_profiling(do_profile)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                metrics["batch/solve_none"] = solve_none / num_tasks
                metrics["batch/solve_all"] = solve_all / num_tasks
                metrics["batch/solve_partial"] = solve_partial / num_tasks

                for key, value in workflow_metrics.items():
                    metrics[f"batch/{key}"] = np.mean(value)

                for r in TerminationReason:
                    metrics[f"batch/{r.value}"] = termination_counts[r.value] / len(set(new_batch.non_tensor_batch["episode_ids"]))

                metrics["batch/num_tasks"] = num_tasks

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                batch = None
                solve_none = 0
                solve_all = 0
                solve_partial = 0
                num_tasks = 0
                termination_counts = Counter()
                workflow_metrics = defaultdict(list)
                metrics = {}
                timing_raw = {}

                self.global_steps += 1

                if is_last_step:
                    # perform validation after training
                    if self.val_reward_fn is not None:
                        self.agent_execution_engine.set_training_step(self.global_steps, mode="val", epoch=epoch)
                        val_metrics = self._validate_agent()
                        pprint(f"Final validation metrics: {val_metrics}")
                        logger.log(data=val_metrics, step=self.global_steps)

                    try:
                        logger.finish()
                    except Exception:
                        pass  # skip errors during cleanup

                    return

    def _validate_agent(self):
        is_correct_lst = []
        data_source_lst = []
        uid_lst = []
        workflow_metrics_by_source = defaultdict(lambda: defaultdict(list))
        batches_for_distill = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch.non_tensor_batch["task_ids"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object)

            n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)

            test_batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"], non_tensor_batch_keys=["raw_prompt_ids"])  # these are not needed for environment based interaction
            test_batch.meta_info = {"validate": True}

            test_output_gen_batch = self.generate_trajectories(batch=test_batch)
            repeat_counts = test_output_gen_batch.meta_info["repeat_counts"]
            # need to repeat to make shape match
            test_batch = test_batch.sample_level_repeat(repeat_counts)
            test_output_gen_batch.meta_info.pop("repeat_counts", None)  # no longer needed after this
            test_batch = test_batch.union(test_output_gen_batch)

            if self.distill_enabled:
                batches_for_distill.append(test_batch)

            seen_episodes = set()
            selected_idxs = []
            for i, episode_id in enumerate(test_batch.non_tensor_batch["episode_ids"]):
                if episode_id not in seen_episodes:
                    seen_episodes.add(episode_id)
                    selected_idxs.append(i)
            test_batch = test_batch.select_idxs(selected_idxs)

            is_correct_lst.extend(test_batch.non_tensor_batch["is_correct"])
            uid_lst.extend(test_batch.non_tensor_batch["task_ids"])

            data_sources = test_batch.non_tensor_batch.get("data_source", None)
            if data_sources is None:
                data_sources = ["unknown"] * len(test_batch)
            data_source_lst.extend(data_sources)

            # Collect workflow metrics per episode and data source
            for i, data_source in enumerate(data_sources):
                episode_metrics = test_batch.non_tensor_batch["metrics"][i]
                if episode_metrics is not None:
                    for key, value in episode_metrics.items():
                        workflow_metrics_by_source[data_source][key].append(float(value))

        metrics = {}
        is_correct_array = np.array(is_correct_lst)
        uid_array = np.array(uid_lst)
        data_source_array = np.array(data_source_lst)

        for data_source in np.unique(data_source_array):
            pass_rates = defaultdict(list)

            data_source_mask = data_source_array == data_source
            is_correct_data_source = is_correct_array[data_source_mask]
            uids_data_source = uid_array[data_source_mask]

            for is_correct, uid in zip(is_correct_data_source, uids_data_source, strict=False):
                pass_rates[uid].append(is_correct)

            metrics[f"val/{data_source}/pass@1"] = np.mean(is_correct_data_source)
            metrics[f"val/{data_source}/pass@{n_val_samples}"] = np.mean([1 if any(pass_rate) else 0 for pass_rate in pass_rates.values()])

            # Add workflow metrics for this data source
            if data_source in workflow_metrics_by_source:
                for key, values in workflow_metrics_by_source[data_source].items():
                    if values:  # Only add if we have values
                        metrics[f"val/{data_source}/{key}"] = np.mean(values)

        # Compute distillation metrics if enabled
        if self.distill_enabled and batches_for_distill:
            try:
                # Concatenate all validation batches
                combined_batch = DataProto.concat(batches_for_distill)

                # Compute old_log_probs for distillation
                old_log_prob = self.actor_rollout_wg.compute_log_prob(combined_batch)
                combined_batch = combined_batch.union(old_log_prob)

                # Compute distillation advantages
                distill_advantages = asyncio.run_coroutine_threadsafe(self._compute_distill_advantages(combined_batch), self._loop).result()

                # Extract distillation metrics
                response_mask = combined_batch.batch["response_mask"]
                valid_advantages = distill_advantages[response_mask.bool()]
                if len(valid_advantages) > 0:
                    metrics["val/distill/mean_advantage"] = valid_advantages.mean().item()
                    metrics["val/distill/std_advantage"] = valid_advantages.std().item()
                    metrics["val/distill/min_advantage"] = valid_advantages.min().item()
                    metrics["val/distill/max_advantage"] = valid_advantages.max().item()
            except Exception as e:
                print(f"Warning: Failed to compute distillation metrics during validation: {e}")
                import traceback

                traceback.print_exc()

        return metrics

    async def _compute_distill_advantages(self, batch: DataProto) -> torch.Tensor:
        """
        Compute distillation advantages by querying teacher and aligning logprobs.

        For each sample in the batch:
        1. Extract student completion_ids, student logprobs, and (if needed) chat_completions
        2. Query teacher for logprobs on the same completion
        3. Align teacher logprobs to student tokens (byte-level alignment if different tokenizers)
        4. Compute per-token advantages: teacher_logprob - student_logprob

        Args:
            batch: DataProto containing student responses, logprobs, and optionally chat_completions
                  (chat_completions only needed when shared_tokenizer=False)

        Returns:
            Tensor of shape (batch_size, max_response_length) with per-token advantages.
            Padded positions have advantages = 0.0
        """
        shared_tokenizer = self.config.rllm.distill.shared_tokenizer

        prompts = batch.batch["prompts"]  # (batch_size, max_prompt_length)
        responses = batch.batch["responses"]  # (batch_size, max_response_length)
        response_mask = batch.batch["response_mask"]  # (batch_size, max_response_length)
        attention_mask = batch.batch["attention_mask"]  # (batch_size, max_prompt_length + max_response_length)
        rollout_log_probs = batch.batch["rollout_log_probs"]  # (batch_size, max_response_length)

        batch_size, max_prompt_length = prompts.shape
        _, max_response_length = responses.shape
        advantages = torch.zeros((batch_size, max_response_length), dtype=torch.float32)

        # Only need chat_completions and chat_parser when tokenizers differ
        if not shared_tokenizer:
            from rllm.trainer.distill import align_teacher_logprobs

            chat_completions = batch.non_tensor_batch.get("chat_completions", None)
            if chat_completions is None:
                raise ValueError("chat_completions not found in batch, cannot perform distillation.")

            if not hasattr(self.teacher_engine, "chat_parser") or self.teacher_engine.chat_parser is None:
                raise ValueError("Teacher engine does not have a chat_parser.")
            teacher_chat_parser = self.teacher_engine.chat_parser

        async def get_teacher_logprobs(prompt: str | list[int], prompt_length: int, sample_idx: int) -> list[float]:
            """Query teacher for logprobs
            Note: We assume the teacher engine is vLLM, not SGLang.
            This is because SGLang does not support prompt_logprobs through the completions endpoint.
            Though we could support this through the echo and logprobs params if needed.
            You can still use SGLang to serve the policy, but you should ensure the temperature is 1.0 and the top_p is 1.0."""

            teacher_resp = await self.teacher_engine.completion(
                prompt,
                max_tokens=1,
                extra_body={"prompt_logprobs": 1},
            )
            if not teacher_resp.prompt_logprobs:
                raise ValueError(f"Teacher missing prompt_logprobs for sample {sample_idx}.")

            return teacher_resp.prompt_logprobs[prompt_length:]

        async def process_sample(sample_idx: int) -> None:
            """Process a single sample: query teacher, align, compute advantages."""
            try:
                student_prompt_length = attention_mask[sample_idx, :max_prompt_length].sum().item()
                if student_prompt_length == 0:
                    raise ValueError(f"Sample {sample_idx} has no valid prompt tokens.")

                student_response_length = attention_mask[sample_idx, -max_response_length:].sum().item()
                if student_response_length == 0:
                    raise ValueError(f"Sample {sample_idx} has no valid response tokens.")

                student_prompt_ids = prompts[sample_idx, -student_prompt_length:].tolist()
                student_response_ids = responses[sample_idx, :student_response_length].tolist()
                student_logprobs = rollout_log_probs[sample_idx, :student_response_length].tolist()

                if shared_tokenizer:
                    # Fast path: student and teacher use the same tokenizer
                    # Directly use student token IDs for teacher query
                    teacher_ids = student_prompt_ids + student_response_ids
                    aligned_teacher_logprobs = await get_teacher_logprobs(teacher_ids, student_prompt_length, sample_idx)

                else:
                    # Slow path: different tokenizers, need byte-level alignment
                    sample_chat_completions = chat_completions[sample_idx]
                    if sample_chat_completions is None or len(sample_chat_completions) == 0:
                        raise ValueError(f"Sample {sample_idx} has no chat_completions for distillation.")

                    teacher_prompt_messages = sample_chat_completions[:-1]
                    teacher_completion_messages = sample_chat_completions[-1:]

                    reasoning_str = teacher_completion_messages[0].get("reasoning", "")
                    content_str = teacher_completion_messages[0].get("content", "")
                    if not reasoning_str and not content_str:
                        raise ValueError(f"Sample {sample_idx} has no reasoning or content in teacher completion message.")

                    # Build teacher prompt and completion
                    teacher_prompt = teacher_chat_parser.parse(
                        teacher_prompt_messages,
                        is_first_msg=True,
                        add_generation_prompt=True,
                        tools=[],
                        accumulate_reasoning=False,
                    )
                    teacher_prompt_ids = self.teacher_tokenizer.encode(teacher_prompt, add_special_tokens=False)

                    teacher_completion = teacher_chat_parser.parse(
                        teacher_completion_messages,
                        is_first_msg=False,
                        add_generation_prompt=False,
                        tools=[],
                        accumulate_reasoning=True,
                    )
                    if teacher_completion.startswith(teacher_chat_parser.generation_prompt):
                        teacher_completion = teacher_completion[len(teacher_chat_parser.generation_prompt) :]
                    teacher_completion_ids = self.teacher_tokenizer.encode(teacher_completion, add_special_tokens=False)

                    teacher_full_prompt = teacher_prompt + teacher_completion
                    teacher_prompt_length = len(teacher_prompt_ids)
                    teacher_logprobs = await get_teacher_logprobs(teacher_full_prompt, teacher_prompt_length, sample_idx)

                    # Align teacher logprobs to student tokens using fast byte-level alignment algorithm
                    aligned_teacher_logprobs = align_teacher_logprobs(
                        student_ids=student_response_ids,
                        student_tokenizer=self.tokenizer,
                        teacher_ids=teacher_completion_ids,
                        teacher_tokenizer=self.teacher_tokenizer,
                        teacher_logprobs=teacher_logprobs,
                        student_logprobs=student_logprobs,
                        reasoning_str=reasoning_str,
                        content_str=content_str,
                    )

                    # Visualize first sample for debugging alignment
                    if sample_idx == 0:
                        from rllm.trainer.distill import visualize_alignment

                        visualize_alignment(
                            student_ids=student_response_ids,
                            student_tokenizer=self.tokenizer,
                            teacher_ids=teacher_completion_ids,
                            teacher_tokenizer=self.teacher_tokenizer,
                            teacher_logprobs=teacher_logprobs,
                            student_logprobs=student_logprobs,
                            reasoning_str=reasoning_str,
                            content_str=content_str,
                            max_tokens=150,
                        )

                # reverse kl: teacher_logprob - student_logprob
                sample_advantages = [t_lp - s_lp for t_lp, s_lp in zip(aligned_teacher_logprobs, student_logprobs, strict=False)]

                advantages[sample_idx, :student_response_length] = torch.tensor(sample_advantages, dtype=torch.float32)

            except Exception as e:
                print(f"Error processing sample {sample_idx} for distillation: {e}")
                import traceback

                traceback.print_exc()
                batch.non_tensor_batch["is_valid"][sample_idx] = False  # drop the item from the batch

        await asyncio.gather(*[process_sample(i) for i in range(batch_size)])
        advantages *= response_mask.float()
        return advantages

    def generate_trajectories(self, batch, timing_raw=None, **kwargs):
        """
        Generates trajectories asynchronously using the agent execution engine's excute tasks method.
        Post-processing is done in the engine as well.

        Args:
            batch: The input batch for trajectory generation
            timing_raw: Dictionary to store timing information for profiling
            **kwargs: Additional arguments to pass to trajectory_generator

        Returns:
            list: List of collected processed trajectories
        """
        if timing_raw is None:
            timing_raw = {}

        with marked_timer("generate_trajectories", timing_raw, color="red"):
            coro = self.agent_execution_engine.execute_tasks_verl(batch, **kwargs)
            final_gen_batch_output = asyncio.run_coroutine_threadsafe(coro, self._loop).result()

        return final_gen_batch_output

    def _stepwise_advantage_broadcast(self, last_step_batch, non_last_step_batch):
        """
        Broadcast the advantage from last_step_batch to all other steps within the same episode and trajectory.
        """

        # NOTE: Currently takes the average of advantages. For GRPO, advantage and returns is uniform for each token so this makes no difference.
        # NOTE: For simplicity, assumes advantage and return is the same, which also holds for GRPO variants

        src_traj_ids = last_step_batch.non_tensor_batch["trajectory_ids"]
        src_eps_ids = last_step_batch.non_tensor_batch["episode_ids"]
        src_steps = last_step_batch.non_tensor_batch["step_nums"]
        src_mask = last_step_batch.batch["response_mask"]
        src_advantages = last_step_batch.batch["advantages"]

        tgt_traj_ids = non_last_step_batch.non_tensor_batch["trajectory_ids"]
        tgt_eps_ids = non_last_step_batch.non_tensor_batch["episode_ids"]
        tgt_mask = non_last_step_batch.batch["response_mask"]

        # Build id -> scalar advantage
        traj_ep_to_scalar_adv = {}
        for i, (traj_id, eps_id) in enumerate(zip(src_traj_ids, src_eps_ids, strict=False)):
            mask = src_mask[i].bool()
            scalar = src_advantages[i][mask].mean()

            if self.config.rllm.stepwise_advantage.normalize_by_steps:
                # normalize the advantage against number of steps
                scalar = scalar / src_steps[i]
                # reassign the normalized advantage to last_step_batch as well
                last_step_batch.batch["advantages"][i][mask] = scalar

            traj_ep_to_scalar_adv[(traj_id, eps_id)] = scalar

        # Create new tensor for non_last_step_batch with per-token assignment
        scalar_rows = torch.stack([torch.full_like(tgt_mask[i], fill_value=traj_ep_to_scalar_adv[(traj_id, eps_id)], dtype=torch.float32) for i, (traj_id, eps_id) in enumerate(zip(tgt_traj_ids, tgt_eps_ids, strict=False))])  # shape: (N2, T)

        # Apply the response mask of the target batch
        final_advantage = scalar_rows * tgt_mask

        # Assignment
        non_last_step_batch.batch["advantages"] = final_advantage
        non_last_step_batch.batch["returns"] = final_advantage

    def _pad_dataproto_to_world_size(self, batch):
        world_sizes = []
        if self.use_critic and self.critic_wg.world_size != 0:
            world_sizes.append(self.critic_wg.world_size)
        if self.use_reference_policy and not self.ref_in_actor and self.ref_policy_wg.world_size != 0:
            world_sizes.append(self.ref_policy_wg.world_size)
        if self.use_rm and self.rm_wg.world_size != 0:
            world_sizes.append(self.rm_wg.world_size)
        if self.hybrid_engine:
            if self.actor_rollout_wg.world_size != 0:
                world_sizes.append(self.actor_rollout_wg.world_size)
        else:
            if self.actor_wg.world_size != 0:
                world_sizes.append(self.actor_wg.world_size)
            if hasattr(self, "rollout_wg") and self.rollout_wg.world_size != 0:
                world_sizes.append(self.rollout_wg.world_size)
        if not world_sizes:
            return batch

        world_size = reduce(math.lcm, world_sizes)

        batch = self._remove_padding(batch)  # Remove any padded steps from the batch (just in case)
        original_batch_size = batch.batch["prompts"].shape[0]
        batch, pad_size = pad_dataproto_to_divisor(batch, world_size)

        # for the padded dataproto, make the traj mask to 0. is_last_step also False
        for i in range(pad_size):
            idx = original_batch_size + i
            batch.non_tensor_batch["is_last_step"][idx] = False
            batch.non_tensor_batch["is_pad_step"][idx] = True
            batch.non_tensor_batch["is_valid"][idx] = False

        return batch

    def _remove_padding(self, batch):
        """Removes padded steps from the batch"""
        is_pad_step = batch.non_tensor_batch["is_pad_step"]
        non_pad_step_indices = np.where(is_pad_step == False)[0]
        batch = batch.select_idxs(non_pad_step_indices)  # This batch only has non_pad steps
        return batch

    def _save_checkpoint(self):
        """
        Override parent checkpoint saving to handle multi-agent LoRA adapters.

        In multi-agent mode (share_policy=False), each agent has its own LoRA adapter.
        The parent class only saves the "default" adapter, so we need to:
        1. Call parent to save the full model checkpoint
        2. Remove the default lora_adapter directory
        3. Save each agent's LoRA adapter to its own directory
        """
        # Call parent to save the full model checkpoint (including all adapters in model_*.pt)
        super()._save_checkpoint()

        # Save training metadata (including wandb run ID for resume) - always save regardless of mode
        self._save_training_metadata()

        # If not multi-agent mode, the parent's save is sufficient
        if self.config.trainer.share_policy:
            return

        # Get the checkpoint path
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        actor_local_path = os.path.join(local_global_step_folder, "actor")
        default_lora_path = os.path.join(actor_local_path, "lora_adapter")

        # Remove the default lora_adapter directory (it only contains "default" adapter)
        if os.path.exists(default_lora_path):
            shutil.rmtree(default_lora_path)
            print(f"Removed default lora_adapter directory: {default_lora_path}")

        # Save each agent's LoRA adapter to its own directory
        agent_names = self.config.trainer.get("agent_names", [])
        if not agent_names:
            print("Warning: share_policy=False but no agent_names configured, skipping multi-agent LoRA save")
            return

        for agent_name in agent_names:
            agent_lora_path = os.path.join(actor_local_path, f"lora_adapter_{agent_name}")
            self.actor_rollout_wg.save_single_lora_adapter(
                agent_name=agent_name,
                save_path=agent_lora_path,
                global_step=self.global_steps,
            )
            print(f"Saved LoRA adapter for agent '{agent_name}' to: {agent_lora_path}")

    def _save_training_metadata(self):
        """Save training metadata to enable resuming wandb logging from checkpoint.

        Loads existing metadata first so that dashboard-written fields
        (e.g. slurm_config, n_gpus, cpus_per_gpu, mem_per_gpu) are preserved.
        """
        import json

        metadata_path = os.path.join(
            self.config.trainer.default_local_dir, "training_metadata.json"
        )

        # Load existing metadata to preserve dashboard-written fields
        metadata = self._load_training_metadata()

        # Selectively update training-owned fields
        metadata["global_steps"] = self.global_steps
        metadata["total_training_steps"] = self.total_training_steps
        metadata["experiment_name"] = self.config.trainer.experiment_name
        metadata["project_name"] = self.config.trainer.project_name
        metadata["slurm_job_id"] = os.environ.get("SLURM_JOB_ID", None)

        # Save wandb run ID if available
        if hasattr(self, "_tracking_logger") and self._tracking_logger is not None:
            wandb_run_id = getattr(self._tracking_logger, "wandb_run_id", None)
            if wandb_run_id:
                metadata["wandb_run_id"] = wandb_run_id

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved training metadata to: {metadata_path}")

    def _load_training_metadata(self) -> dict:
        """Load training metadata from checkpoint directory.

        Returns:
            Dictionary containing saved metadata, or empty dict if not found.
        """
        import json

        metadata_path = os.path.join(
            self.config.trainer.default_local_dir, "training_metadata.json"
        )
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                return json.load(f)
        return {}

    def shutdown(self):
        """A cleanup method to gracefully stop the background event loop."""
        if hasattr(self, "agent_execution_engine") and self.agent_execution_engine is not None:
            self.agent_execution_engine.shutdown()
            self.agent_execution_engine = None
        if hasattr(self, "_loop") and self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if hasattr(self, "_thread") and self._thread is not None:
            self._thread.join()

    def visualize_trajectory_last_step(self, tensor_batch, sample_idx=0, max_samples=1):
        """
        Visualize last steps from a workflow rollout using the shared visualization utility.
        """
        from rllm.utils.visualization import visualize_trajectories

        # Select only last steps if stepwise-advantage is enabled
        if "is_last_step" in tensor_batch.non_tensor_batch:
            is_last = tensor_batch.non_tensor_batch["is_last_step"]
            if is_last is not None and len(is_last) == len(tensor_batch):
                tensor_batch = tensor_batch[is_last]

        if len(tensor_batch) == 0:
            return

        end_idx = min(sample_idx + max_samples, len(tensor_batch))
        indices = list(range(sample_idx, end_idx))

        visualize_trajectories(
            batch=tensor_batch,
            tokenizer=self.tokenizer,
            sample_indices=indices,
            mask_key="response_mask",
            reward_key="step_rewards" if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "per_step" else "traj_rewards",
            show_workflow_metadata=True,
        )
