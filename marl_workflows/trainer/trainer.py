# ruff: noqa: E712  (the bounded copies keep upstream's `== True` comparisons verbatim)
"""``MultiAgentWorkflowTrainer``: rLLM's workflow trainer with per-role LoRA adapters.

Base: rllm@186d0bbb ``rllm/trainer/verl/agent_workflow_trainer.py:40`` (``AgentWorkflowPPOTrainer``).

The per-role computations (log-probs, reference log-probs, actor updates, metric renaming,
per-role KL and loss weights) live in :class:`~marl_workflows.trainer.role_routing.RoleRoutedWorkerGroup`,
which replaces ``self.actor_rollout_wg``; the training loop itself is upstream code. Three
methods are bounded copies of upstream with every change fenced by ``# >>> MARL(<id>)`` /
``# <<< MARL(<id>)`` (``tests/test_fit_agent_copy.py`` checks that nothing else differs):

- ``init_workers`` (upstream :98-126): W1 proxy, W2 multi-LoRA engine, W3 code-executor pool.
- ``fit_agent`` (upstream :128-499): R1-R7, see the region comments.
- ``_pad_dataproto_to_world_size`` (upstream :831-863): P1.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import uuid
import warnings
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import Executor
from functools import reduce
from pprint import pprint
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.trainer.verl.agent_workflow_trainer import AgentWorkflowPPOTrainer
from rllm.utils.episode_logger import EpisodeLogger
from rllm.workflows.workflow import TerminationReason
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, apply_kl_penalty, compute_advantage
from verl.utils.debug import marked_timer

from marl_workflows.contracts import (
    RUN_COMPLETE_FMT,
    STEP_DIR_FMT,
    TRAINING_METADATA_FILE,
    WALLCLOCK_STOP_FMT,
    actor_ckpt_dir,
    adapter_dirname,
    expected_lora_int_ids,
)
from marl_workflows.trainer.role_routing import RoleRoutedWorkerGroup
from marl_workflows.trainer.tracking import ResumableTracking
from marl_workflows.trainer.wallclock import WallclockStop
from marl_workflows.utils import allocated_cpu_count

__all__ = ["MultiAgentWorkflowTrainer", "ValDropAccounting"]


class ValDropAccounting:
    """Validation episodes dropped before their first step (``repeat_counts == 0``).

    Upstream at the pin excludes such episodes from ``val/*/pass@k``, which inflates the
    metric. ``metrics(..., correct=True)`` applies rLLM e9be9ec6's semantics (a dropped
    episode counts as a failure); ``correct=False`` (default, paper behaviour) only reports
    ``val/{ds}/dropped_episodes``. Data sources follow upstream ``_validate_agent``
    (``data_source`` column, else ``"unknown"``).
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._episodes: Counter[str] = Counter()
        self._dropped: Counter[str] = Counter()
        self._uids: defaultdict[str, set[str]] = defaultdict(set)
        self._uids_seen: defaultdict[str, set[str]] = defaultdict(set)

    def record(self, batch: Any, repeat_counts: Sequence[int]) -> None:
        """Account one validation input batch (one row per episode) and its ``repeat_counts``."""
        n = len(repeat_counts)
        sources = batch.non_tensor_batch.get("data_source", None)
        if sources is None:
            sources = ["unknown"] * n
        uids = batch.non_tensor_batch.get("task_ids", None)
        if uids is None:
            uids = [str(i) for i in range(n)]
        for src, uid, count in zip(sources, uids, repeat_counts, strict=True):
            src, uid = str(src), str(uid)
            self._episodes[src] += 1
            self._uids[src].add(uid)
            if int(count) == 0:
                self._dropped[src] += 1
            else:
                self._uids_seen[src].add(uid)

    def metrics(self, upstream: Mapping[str, Any], *, n_val: int, correct: bool) -> dict[str, float]:
        """Metrics to merge into the upstream validation metrics (replacing pass@k when ``correct``)."""
        out: dict[str, float] = {}
        for src in sorted(self._episodes):
            dropped = self._dropped[src]
            out[f"val/{src}/dropped_episodes"] = float(dropped)
            if not correct:
                continue
            total = self._episodes[src]
            uid_total = len(self._uids[src])
            k1, kn = f"val/{src}/pass@1", f"val/{src}/pass@{n_val}"
            if kn in upstream and kn != k1:
                out[kn] = float(upstream[kn]) * len(self._uids_seen[src]) / uid_total
            elif kn != k1:
                out[kn] = 0.0
            if k1 in upstream:
                # With n_val == 1 upstream overwrites pass@1 with the per-uid value; both equal
                # the per-episode mean, so the per-episode correction is the right one.
                out[k1] = float(upstream[k1]) * (total - dropped) / total
            else:
                out[k1] = 0.0
        return out


# COPIED FROM rllm@186d0bbb rllm/trainer/verl/agent_workflow_trainer.py (methods init_workers :98-126,
# fit_agent :128-499, _pad_dataproto_to_world_size :831-863); diff regions marked MARL(...).
class MultiAgentWorkflowTrainer(AgentWorkflowPPOTrainer):
    """Workflow PPO trainer with one LoRA adapter per role (IP) or one shared adapter (SP)."""

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping,
        resource_pool_manager,
        ray_worker_group_cls=RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        workflow_class=None,
        workflow_args=None,
    ) -> None:
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            workflow_class=workflow_class,
            workflow_args=workflow_args,
        )
        self._last_saved_step = 0
        self._lora_state_checked = False
        self._val_acct = ValDropAccounting()
        self._code_executor: Executor | None = None
        self._tracking_logger: ResumableTracking | None = None

    # ------------------------------------------------------------------ config

    @property
    def _agent_names(self) -> list[str]:
        return [str(n) for n in self.config.trainer.agent_names]

    @property
    def _share_policy(self) -> bool:
        return bool(self.config.trainer.share_policy)

    def _validate_config(self) -> None:
        super()._validate_config()
        trainer = self.config.trainer
        sp = bool(trainer.share_policy)
        if not sp and int(self.config.actor_rollout_ref.model.get("lora_rank", 0) or 0) <= 0:
            raise ValueError("share_policy=False (one adapter per role) requires actor_rollout_ref.model.lora_rank > 0")
        if trainer.get("kl_loss_coef_per_role"):
            if sp:
                raise ValueError("trainer.kl_loss_coef_per_role needs share_policy=False (one adapter per role)")
            if not self.config.actor_rollout_ref.actor.get("use_kl_loss", False):
                raise ValueError("trainer.kl_loss_coef_per_role needs actor_rollout_ref.actor.use_kl_loss=true")
        lw = trainer.get("loss_weight_per_role") or {}
        if not sp and str(lw.get("mode", "none")) != "none":
            warnings.warn("trainer.loss_weight_per_role applies to share_policy=True only; ignored", stacklevel=2)

    def _uses_code_executor(self) -> bool:
        if int(self.config.rllm.workflow.get("code_executor_workers", 0) or 0) > 0:
            return True
        marl = self.config.get("marl") or {}
        if marl.get("task") and marl.get("workflow"):
            from marl_workflows.recipes import get_recipe

            return get_recipe(marl.task, marl.workflow).uses_code_executor
        return False

    # ------------------------------------------------------------------ bounded copies
    # Kept in upstream formatting so that the drift guard compares line by line.
    # fmt: off

    def init_workers(self):
        # >>> MARL(W1): build the worker groups without the upstream engine, then route actor calls per role
        RayPPOTrainer.init_workers(self)
        self.actor_rollout_wg = RoleRoutedWorkerGroup(
            self.actor_rollout_wg,
            agent_names=self._agent_names,
            share_policy=self._share_policy,
            trainer_config=self.config.trainer,
            rollout_temperature=self.config.actor_rollout_ref.rollout.temperature,
        )
        # <<< MARL(W1)

        # >>> MARL(W2): rollout engine that maps agent_name to the role's vLLM LoRA id
        from marl_workflows.engine.verl_engine import MultiLoRAVerlEngine

        rollout_engine = MultiLoRAVerlEngine(
            config=self.config,
            rollout_manager=self.async_rollout_manager,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        # <<< MARL(W2)

        # Create episode logger if enabled in config
        episode_logger = None
        if self.config.trainer.get("log_episodes", False):
            # Get episode log directory from config, default to "logs/my_project/my_experiment"
            episode_log_dir = self.config.trainer.get("episode_log_dir", f"logs/{self.config.trainer.project_name}/{self.config.trainer.experiment_name}")
            episode_logger = EpisodeLogger(base_dir=episode_log_dir, subdirectory="episodes")

        # >>> MARL(W3): code-reward executor (rllm-free spawn pool), handed to every workflow through workflow_args
        if self._uses_code_executor():
            from marl_workflows.rewards.code_reward import CodeRewardExecutor

            n_workers = int(self.config.rllm.workflow.get("code_executor_workers", 0) or 0) or allocated_cpu_count()
            self._code_executor = CodeRewardExecutor(max_workers=n_workers)
            self.workflow_args = {**self.workflow_args, "code_reward_executor": self._code_executor}
        # <<< MARL(W3)

        self.agent_execution_engine = AgentWorkflowEngine(
            workflow_cls=self.workflow_class,
            workflow_args=self.workflow_args,
            rollout_engine=rollout_engine,
            config=self.config,
            n_parallel_tasks=self.config.rllm.workflow.n_parallel_tasks,
            retry_limit=self.config.rllm.workflow.retry_limit,
            episode_logger=episode_logger,
        )

        # init workflow workers
        asyncio.run_coroutine_threadsafe(self.agent_execution_engine.initialize_pool(), self._loop).result()

    def fit_agent(self):
        """
        The training loop of PPO. Adapted to train the underlying model of agent.
        """
        # >>> MARL(R1): load the checkpoint before wandb init so a resumed segment continues the same run;
        # stop at once on a finished run (RUN_COMPLETE); set up the optional wall-clock stop
        self.global_steps = 0
        self._load_checkpoint()
        if self.global_steps > 0 and self.global_steps >= self.total_training_steps:
            print(RUN_COMPLETE_FMT.format(step=self.global_steps), flush=True)
            return
        self._last_saved_step = self.global_steps
        wallclock = WallclockStop.from_config(self.config.trainer.get("wallclock_stop"))
        resume_run_id, resume_mode = None, None
        if self.global_steps > 0:
            resume_run_id = self._load_training_metadata().get("wandb_run_id")
            resume_mode = "allow"  # "allow" also covers a run deleted from wandb
        logger = ResumableTracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
            resume_run_id=resume_run_id,
            resume_mode=resume_mode,
        )
        self._tracking_logger = logger
        # <<< MARL(R1)

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
                # >>> MARL(R2): wall-clock stop before any step of this segment completed (nothing new to save)
                iter_start = time.time()
                if wallclock is not None and not wallclock.has_completed_step and wallclock.should_stop():
                    print(WALLCLOCK_STOP_FMT.format(step=self.global_steps - 1, saved="none (no step in this segment)"), flush=True)
                    try:
                        logger.finish()
                    except Exception:
                        pass  # skip errors during cleanup
                    return
                # <<< MARL(R2)
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

                # >>> MARL(R3): train total_training_steps steps (upstream stops after total-1); renamed from
                # the fork's is_last_step, which the broadcast-advantage branches below overwrite
                is_final_step = self.global_steps >= self.total_training_steps
                # <<< MARL(R3)

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

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            debug_metrics = calculate_debug_metrics(batch)
                            metrics.update(debug_metrics)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
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
                    # >>> MARL(R4): verl's update_policy reads the temperature from meta_info; set it explicitly
                    batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                    # <<< MARL(R4)

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                        with marked_timer("testing", timing_raw, color="green"):
                            self.agent_execution_engine.set_training_step(self.global_steps, mode="val", epoch=epoch)
                            val_metrics: dict = self._validate_agent()
                        metrics.update(val_metrics)

                    # >>> MARL(R5): also save on the final step
                    if self.config.trainer.save_freq > 0 and (is_final_step or self.global_steps % self.config.trainer.save_freq == 0):
                        # <<< MARL(R5)
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
                # >>> MARL(R6): keep the step time for the wall-clock estimate (timing_raw is reset below)
                step_time = timing_raw.get("step", None)
                # <<< MARL(R6)

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

                # >>> MARL(R7): finish after the final step (decided before it ran); otherwise stop before the
                # job's wall-clock limit, saving the last completed step if it is not saved yet
                if is_final_step:
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

                if wallclock is not None:
                    wallclock.record(step_time, time.time() - iter_start)
                    if wallclock.should_stop():
                        completed = self.global_steps - 1
                        if self._last_saved_step != completed:
                            self.global_steps = completed
                            try:
                                self._save_checkpoint()
                            finally:
                                self.global_steps = completed + 1
                        saved = os.path.join(self.config.trainer.default_local_dir, STEP_DIR_FMT.format(step=completed))
                        print(WALLCLOCK_STOP_FMT.format(step=completed, saved=saved), flush=True)
                        try:
                            logger.finish()
                        except Exception:
                            pass  # skip errors during cleanup
                        return
                # <<< MARL(R7)

    def _pad_dataproto_to_world_size(self, batch):
        world_sizes = []
        if self.use_critic and self.critic_wg.world_size != 0:
            world_sizes.append(self.critic_wg.world_size)
        # >>> MARL(P1): with LoRA the reference policy lives in the actor (no ref_policy_wg exists)
        if self.use_reference_policy and not self.ref_in_actor and self.ref_policy_wg.world_size != 0:
            # <<< MARL(P1)
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

    # fmt: on
    # ------------------------------------------------------------------ wrappers

    def generate_trajectories(self, batch, timing_raw=None, **kwargs):
        """Upstream generation plus dropped-episode accounting (validation) and a one-time LoRA id check."""
        out = super().generate_trajectories(batch, timing_raw=timing_raw, **kwargs)
        if batch.meta_info.get("validate"):
            self._val_acct.record(batch, out.meta_info["repeat_counts"])
        elif not self._lora_state_checked:
            self._check_lora_registration()
        return out

    def _check_lora_registration(self) -> None:
        """After the first training rollout, every rank must have registered exactly the expected LoRA ids (C7)."""
        self._lora_state_checked = True
        if not self.config.trainer.get("check_lora_registration", True):
            return
        if int(self.config.actor_rollout_ref.model.get("lora_rank", 0) or 0) <= 0:
            return
        expected = expected_lora_int_ids(self._agent_names, self._share_policy)
        states = self.actor_rollout_wg.probe_rollout_lora_state()
        states = states if isinstance(states, list) else [states]
        bad = [s for s in states if not s.get("rollout_lora_registered") or list(s.get("lora_int_ids", [])) != expected]
        if bad:
            raise RuntimeError(f"vLLM LoRA registration mismatch: expected ids {expected} on every rank, got {states}")
        print(f"[marl] vLLM LoRA ids registered on {len(states)} rank(s): {expected}", flush=True)

    def _validate_agent(self):
        self._val_acct.reset()
        metrics = super()._validate_agent()
        metrics.update(
            self._val_acct.metrics(
                metrics,
                n_val=int(self.config.actor_rollout_ref.rollout.val_kwargs.n),
                correct=bool(self.config.trainer.get("val_count_dropped_episodes", False)),
            )
        )
        return metrics

    # ------------------------------------------------------------------ checkpoints

    def _save_checkpoint(self) -> None:
        """Upstream save, then metadata and (IP) one PEFT adapter directory per role (C14)."""
        super()._save_checkpoint()
        self._last_saved_step = self.global_steps
        self._save_training_metadata()
        if self._share_policy:
            return
        actor_dir = actor_ckpt_dir(self.config.trainer.default_local_dir, self.global_steps)
        # Under IP the worker skips upstream's lora_adapter/ export (there is no "default" adapter); its presence
        # would make evaluation read this checkpoint as SP (eval.checkpoints.detect_adapter_layout).
        shared_dir = os.path.join(actor_dir, adapter_dirname(None))
        if os.path.exists(shared_dir):
            raise RuntimeError(f"IP checkpoint must not contain the SP adapter directory {shared_dir}")
        for role in self._agent_names:
            path = os.path.join(actor_dir, adapter_dirname(role))
            self.actor_rollout_wg.save_single_lora_adapter(
                agent_name=role, save_path=path, global_step=self.global_steps
            )
            print(f"Saved LoRA adapter for agent '{role}' to: {path}")

    def _metadata_path(self) -> str:
        return os.path.join(self.config.trainer.default_local_dir, TRAINING_METADATA_FILE)

    def _load_training_metadata(self) -> dict:
        """``training_metadata.json`` from ``default_local_dir``, or ``{}``."""
        path = self._metadata_path()
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    def _save_training_metadata(self) -> None:
        """Merge the training-owned fields into ``training_metadata.json`` (keeps fields written by others)."""
        metadata = self._load_training_metadata()
        metadata["global_steps"] = self.global_steps
        metadata["total_training_steps"] = self.total_training_steps
        metadata["experiment_name"] = self.config.trainer.experiment_name
        metadata["project_name"] = self.config.trainer.project_name
        metadata["slurm_job_id"] = os.environ.get("SLURM_JOB_ID", None)
        metadata["actor_model_path"] = self.config.actor_rollout_ref.model.path
        base_model = self.config.trainer.get("base_model_tag")
        if base_model:
            metadata["base_model"] = base_model
        run_id = getattr(self._tracking_logger, "wandb_run_id", None)
        if run_id:
            metadata["wandb_run_id"] = run_id
        path = self._metadata_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved training metadata to: {path}")

    def shutdown(self) -> None:
        super().shutdown()
        if self._code_executor is not None:
            self._code_executor.shutdown(wait=True)
            self._code_executor = None
