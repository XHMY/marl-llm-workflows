# I1 — Per-Role KL Anchor

## Overview

I1 adds `β_role · D_KL(π_θ^role || π_ref^role)` to the per-agent policy loss,
with **independent β per role**. A role that drives terminal instability
(e.g. the high-frequency role under voting / orchestrator-workers, or the
evaluator under evaluator-optimizer) can be anchored toward the base model
while other roles remain unconstrained.

I1 is activated via Hydra; no code changes are required to opt in.

```bash
actor_rollout_ref.actor.use_kl_loss=true
actor_rollout_ref.actor.kl_loss_coef=0.0
actor_rollout_ref.actor.kl_loss_type=low_var_kl
+trainer.kl_loss_coef_per_role.<anchor_role>=<beta>
+trainer.kl_loss_coef_per_role.<other_role>=0.0
```

The global `kl_loss_coef=0.0` makes any role not listed in
`trainer.kl_loss_coef_per_role` fall through with KL weight 0, so the
per-role overrides are the only place KL applies.

## Files modified

| File | Change |
|---|---|
| `rllm/trainer/verl/agent_workflow_trainer.py` | Stamp per-role KL coef onto `sub_batch.meta_info` before `update_actor`. Also gate `_pad_dataproto_to_world_size` access to `ref_policy_wg` on `not self.ref_in_actor`. |
| `verl/verl/workers/actor/dp_actor.py` | In the KL-loss block, read coef from `data.meta_info["kl_loss_coef_override"]` (falls through to `config.kl_loss_coef`). Logs `actor/kl_coef`. |

## Implementation

### Per-role config namespace: `trainer.kl_loss_coef_per_role`

Per-role coefficients live under `trainer.*`, not
`actor_rollout_ref.actor.*`. The actor config is a frozen `@dataclass`
(`verl/verl/workers/config/actor.py`) instantiated via
`omega_conf_to_dataclass(...)`, which strictly rejects unknown fields. The
trainer config block has no structured-dataclass enforcement and accepts the
`+trainer.kl_loss_coef_per_role.<role>=<beta>` Hydra add-syntax verbatim.

### Trainer side: stamp coef onto `meta_info` per agent

In `agent_workflow_trainer.py` at the `update_actor` step inside `fit_agent`:

```python
kl_coef_per_role = self.config.trainer.get("kl_loss_coef_per_role", None)

for agent_name, (sub_batch, indices) in agent_batches.items():
    if len(sub_batch) == 0:
        continue
    self.actor_rollout_wg.set_active_lora(agent_role=agent_name, lora_config={})

    overrides: dict = {}
    if kl_coef_per_role is not None and agent_name in kl_coef_per_role:
        overrides["kl_loss_coef_override"] = float(kl_coef_per_role[agent_name])
    if overrides:
        # Clone meta_info — select_idxs returns a new DataProto but shares
        # the parent's meta_info dict by reference.
        sub_batch.meta_info = {**sub_batch.meta_info, **overrides}

    sub_actor_output = self.actor_rollout_wg.update_actor(sub_batch)
```

The clone (`{**sub_batch.meta_info, **overrides}`) is mandatory:
`DataProto.select_idxs` shares `meta_info` by reference, and an in-place
mutation would leak the previous role's override into the next role's
sub-batch.

### Actor side: read coef from `meta_info`

In `dp_actor.py`, inside the KL-loss block of `update_policy()`:

```python
if self.config.use_kl_loss:
    ref_log_prob = model_inputs["ref_log_prob"]
    kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob,
                     kl_penalty=self.config.kl_loss_type)
    kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask,
                       loss_agg_mode=loss_agg_mode)

    # Per-role override (I1). Falls through to global config when absent.
    kl_coef = float(data.meta_info.get("kl_loss_coef_override",
                                        self.config.kl_loss_coef))
    policy_loss = policy_loss + kl_loss * kl_coef
    micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
    micro_batch_metrics["actor/kl_coef"] = kl_coef
```

`actor/kl_coef` is logged per microbatch so wandb shows the effective β for
each role.

### Reference-policy worker activation under LoRA

Setting `actor_rollout_ref.actor.use_kl_loss=true` is sufficient.
`rllm/trainer/verl/train_agent_ppo.py:154` registers `Role.RefPolicy`:

```python
if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
    role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
    mapping[Role.RefPolicy] = global_pool_id
```

Under LoRA (`lora_rank > 0`), `RayPPOTrainer.__init__` sets
`self.ref_in_actor = True` and no separate `ref_policy_wg` is created
(`verl/verl/trainer/ppo/ray_trainer.py:753`). The actor worker computes
reference log-probs by un-applying the LoRA adapter
(`fsdp_workers.py:compute_ref_log_prob` with `is_lora=True`).

Per-agent reference log-prob computation handles this case
(`agent_workflow_trainer.py:425–461`):

```python
if not self.ref_in_actor:
    sub_ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(sub_batch)
else:
    sub_ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(sub_batch)
```

### World-size pad fix (required under LoRA)

The first I1 launch crashed with
`AttributeError: 'AgentWorkflowPPOTrainer' object has no attribute 'ref_policy_wg'`
after the first rollout. Root cause: `_pad_dataproto_to_world_size`
(around line 1027 of `agent_workflow_trainer.py`) referenced
`ref_policy_wg` unconditionally. Gate on `not self.ref_in_actor`:

```python
# before
if self.use_reference_policy and self.ref_policy_wg.world_size != 0:
    world_sizes.append(self.ref_policy_wg.world_size)

# after
if self.use_reference_policy and not self.ref_in_actor and self.ref_policy_wg.world_size != 0:
    world_sizes.append(self.ref_policy_wg.world_size)
```

The actor worker's world size is already counted via
`actor_rollout_wg.world_size` later in the same function, so under LoRA
there is no missing world-size contribution.

## Backward compatibility

- When `+trainer.kl_loss_coef_per_role` is unset,
  `data.meta_info.get("kl_loss_coef_override", self.config.kl_loss_coef)`
  returns the global config value. Pre-existing runs are unaffected.
- The `_pad_dataproto_to_world_size` fix is a strict bug fix: any run that
  set `use_kl_loss=true` under LoRA (not just I1) would have hit the same
  `AttributeError`. The guard makes the function correct in the LoRA path
  while leaving the non-LoRA path identical.
- I1 only fires when the trainer enters the multi-agent branch
  (`not self.config.trainer.share_policy`). Shared-policy runs see the
  global `kl_loss_coef` only.
