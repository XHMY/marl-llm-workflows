# I6 — Gradient-Direction Coherence Regularizer

## Overview

I6 scales a role's accumulated gradient by `1 - λ · max(0, cos(g, ḡ))`,
where `cos` is a single scalar computed over the concatenated active-adapter
gradient vector and `ḡ` is its exponential moving average. Updates whose
overall cosine with history is non-positive pass through unchanged; aligned
updates are uniformly damped by the same scalar across every parameter
shard. This is the **step-scaling form** of the regularizer; a per-parameter
projection (PCGrad-style component surgery) is not implemented and would be
a separate variant.

The implementation lives at the actor's mini-batch boundary — gradient
surgery between `loss.backward()` (accumulated across micro-batches) and
`optimizer.step()`. It is conditional, per-role, and zero-overhead when
disabled.

## Key concepts

- **Step-scaling formulation.** Step-scaling does not need a Hessian-vector
  product and integrates cleanly with FSDP.
- **Active-adapter scoping.** Under multi-LoRA, only the currently-active
  adapter has `requires_grad=True` (set by `fsdp_workers.set_active_lora`).
  Iterating `named_parameters()` and filtering on
  `requires_grad and grad is not None` yields exactly the active role's
  parameters, so the regularizer is implicitly per-role.
- **Per-role EMA.** EMA tensors are stored in
  `self._coherence_ema[role][param_name]`, keyed by role and parameter name.
  Lazy-initialized on first sight of a role to a clone of the current grad
  shard.
- **Sharded EMA, scalar communication.** EMA shards live alongside the
  FSDP-sharded gradient; only three scalars (`g·g`, `ḡ·ḡ`, `g·ḡ`) are
  all-reduced per call.
- **Post-surgery EMA update.** The EMA is updated toward the *post-surgery*
  gradient. Updating toward the pre-surgery gradient makes the EMA
  self-reinforcing and defeats the regularizer over time.
- **One-sided clamp.** `cos` is clamped to `[0, 1]` before scaling so
  anti-aligned (exploratory) steps are not damped; only persistent alignment
  with history is attacked.
- **Per-rank EMA checkpointing.** EMA shards are saved alongside the actor
  checkpoint at `<actor_local_path>/coherence_ema/rank{R}.pt`, one file per
  FSDP rank. Save runs only when the EMA dict is non-empty (non-I6 runs and
  λ=0 roles incur zero I/O). On `world_size` mismatch or a missing file,
  the EMA cold-starts.

## Files modified

| File | Change |
|---|---|
| `verl/verl/workers/actor/dp_actor.py` | Added `self._coherence_ema` dict in `__init__`, `_apply_coherence_regularizer()`, `save_coherence_ema_state()` / `load_coherence_ema_state()`, and a call site in `update_policy()` between the micro-batch backward loop and `_optimizer_step()`. |
| `verl/verl/workers/fsdp_workers.py` | Wired EMA save / load into `FSDPWorker.save_checkpoint` / `load_checkpoint` — per-rank shard files under `<actor_local_path>/coherence_ema/`. |
| `rllm/trainer/verl/agent_workflow_trainer.py` | Extended the per-role override block in the multi-agent actor-update loop to merge I1 (`kl_loss_coef_per_role`) and I6 (`coherence_reg.per_role_weights`) into one `overrides` dict, then thread into `sub_batch.meta_info` per agent. |

## Implementation

### `dp_actor.py` — `_apply_coherence_regularizer()`

Two-pass over the active adapter's parameters with an all-reduce between
passes for the global cosine similarity.

**Pass 1 — accumulate dot products.** For each `requires_grad` parameter
with a non-None grad:

- Extract the local shard:
  `g_local = grad.to_local() if isinstance(grad, DTensor) else grad`.
- Lazy-init `ema_local` from `g_local.detach().clone()` on first sight of
  this `(role, param_name)`.
- Accumulate `gg += (g · g)`, `ee += (ḡ · ḡ)`, `ge += (g · ḡ)` in
  `float32` (bf16 underflows on small dot products).
- Stash `(g_local, ema_local)` for pass 2.

**All-reduce.** Stack `(gg, ee, ge)` and `all_reduce(SUM)` across the world.
Cost: 12 bytes per scalar per rank.

**Pass 2 — apply surgery + update EMA.**

- `cos = ge / (sqrt(gg) · sqrt(ee) + eps)`
- `scale = 1 - weight · clamp(cos, 0, 1)`
- For each stashed `(g_local, ema_local)`:
  - `g_local.mul_(scale)` — surgery.
  - `ema_local.mul_(τ).add_(g_local, alpha=1 - τ)` — update toward
    post-surgery grad, with `τ = 1 - 1/ema_window`.

**Metrics emitted** (the trainer wraps them with `actor/{role}/...`):

- `actor/coherence_cos` — global cosine similarity.
- `actor/coherence_scale` — applied scale factor.
- `actor/coherence_ema_norm` — `sqrt(ee)`, sanity-checks the EMA is being
  updated.

**Call-site** in `update_policy()`:

```python
# After micro-batch backward accumulation, before clip + step.
if data.meta_info.get("coherence_reg_weight", 0.0) > 0:
    coh_metrics: dict = {}
    self._apply_coherence_regularizer(
        role=str(data.meta_info["coherence_reg_role"]),
        weight=float(data.meta_info["coherence_reg_weight"]),
        ema_window=int(data.meta_info.get("coherence_reg_ema_window", 128)),
        eps=float(data.meta_info.get("coherence_reg_eps", 1e-8)),
        metrics=coh_metrics,
    )
    append_to_dict(metrics, coh_metrics)

grad_norm = self._optimizer_step()
```

The regularizer runs **before** grad clipping so any pathological scale
spikes from `(1 - cos)` are bounded by the existing clip path.

### `agent_workflow_trainer.py` — config read + meta_info injection

In the multi-agent actor-update loop (only when `not share_policy`), the
trainer reads both I1 and I6 config blocks once per epoch and injects
per-agent overrides into `sub_batch.meta_info`:

```python
kl_coef_per_role = self.config.trainer.get("kl_loss_coef_per_role", None)
coherence_reg = self.config.trainer.get("coherence_reg", None)
coherence_enabled = bool(coherence_reg) and bool(coherence_reg.get("enable", False))
coherence_role_weights = (
    coherence_reg.get("per_role_weights", {}) if coherence_enabled else {}
)

for agent_name, (sub_batch, indices) in agent_batches.items():
    if len(sub_batch) == 0:
        continue
    self.actor_rollout_wg.set_active_lora(agent_role=agent_name, lora_config={})

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

    sub_actor_output = self.actor_rollout_wg.update_actor(sub_batch)
```

Two correctness invariants:

1. **Copy `meta_info`, do not mutate.** `sub_batch.meta_info` is shared by
   reference across siblings after `select_idxs`; mutating in place would
   cross-contaminate another role's overrides.
2. **`weight=0` short-circuits.** `_apply_coherence_regularizer` returns
   early when `weight <= 0`. A role with `λ=0` pays zero overhead — no EMA
   dict is created.

### EMA checkpointing

EMA shards live alongside the FSDP-sharded gradient, so each rank persists
its own file. On-disk layout under each `global_step_N/actor/`:

```
coherence_ema/
├── rank0.pt
└── rank1.pt    # one file per FSDP rank
```

Each file is
`torch.save({"world_size": int, "rank": int, "ema": {role: {param_name: cpu_tensor}}})`.

- **Save** (`FSDPWorker.save_checkpoint`, after the actor + LoRA-adapter
  writes): no-op when `_coherence_ema` is empty. Otherwise rank R writes
  `coherence_ema/rank{R}.pt`, with all tensors moved to CPU first. A
  `dist.barrier()` follows so the dump is observed-complete before the
  worker offloads parameters.
- **Load** (`FSDPWorker.load_checkpoint`, after the actor checkpoint
  manager runs): looks for `coherence_ema/rank{R}.pt`. On hit, the saved
  `world_size` is checked against the current world. On miss or mismatch
  the load returns `False` and the EMA stays empty — the regularizer
  cold-starts on the next step exactly as in a fresh run.
- **No cross-rank reshuffling.** Re-running with a different
  `n_gpus_per_node` (changing world_size) intentionally cold-starts.
- **No optimizer-state coupling.** EMA reload happens after the existing
  checkpoint manager call; failures here do not corrupt optimizer state.

## Configuration

I6 is plumbed via Hydra dynamic fields rooted at `trainer.coherence_reg`.
Keeping these outside the actor dataclass namespace avoids hydra-instantiate
field rejection.

```
+trainer.coherence_reg.enable=true
+trainer.coherence_reg.ema_window=128         # τ = 1 - 1/window
+trainer.coherence_reg.eps=1e-8
+trainer.coherence_reg.per_role_weights.<role>=<lambda>   # 1.0 = full surgery, 0.0 = off
```

If `coherence_reg.enable` is false or absent, the regularizer is a no-op
everywhere — no extra memory, no extra communication.

### Recommended operating range

Operate at λ ≤ 0.3 so `scale ≥ 0.7` even when `cos = 1`. The EMA continues
to track the live gradient and the regularizer behaves as intended damping
rather than blocking. λ ∈ {0.1, 0.3} with `ema_window ∈ {32, 128}` is the
recommended scan range.

## Diagnostics

Read `actor/{role}/coherence_cos` in wandb:

- Persistent positive cosine → I6 is fighting genuine drift along a
  preferred direction.
- Cosine near 0 → drift is incoherent at this EMA window; the regularizer
  is mostly inert.
- Cosine swinging between positive and negative → reasonable training noise.
- Cosine pinned at ≈ 1.0 across many steps with
  `coherence_scale ≈ 1 - λ` → the freeze regime (see Pitfalls).

## Cold-start behavior

On step 1 the EMA equals the current gradient (lazy init), so `cos = 1`
exactly and `scale = 1 - λ`. With `λ=1.0` the first step is fully damped.
In the small-λ regime (≤ 0.3) this is one acceptable step of partial
movement and the EMA decorrelates from the live grad over the next few
steps as new orthogonal-noise components enter the EMA. In the λ → 1.0
regime the cold-start interacts badly with the steady-state freeze
described below — the EMA never picks up a direction other than the
lazy-init one.

## Pitfalls

### Steady-state freeze at λ → 1.0 with genuinely persistent drift

The intent of I6 is to damp persistent gradient direction. With the
lazy-init EMA + post-surgery EMA update + λ near 1.0, the regularizer can
*freeze* the role rather than merely damp it:

```
Step 1: g₁ = grad, EMA ← g₁ (lazy init)
        cos(g₁, g₁) = 1, scale = 1 − λ ≈ 0
        post-surgery g₁ ← 0
        EMA ← τ·g₁ + (1−τ)·0 = τ·g₁           # direction = direction(g₁)
Step 2: g₂ persistent ⇒ direction(g₂) ≈ direction(g₁)
        cos(g₂, τ·g₁) ≈ 1
        scale ≈ 0, post-surgery g₂ ← 0
        EMA ← τ²·g₁                            # direction unchanged
Step n: EMA ← τⁿ·g₁ — magnitude decays, direction is still direction(g₁).
```

The EMA's *direction* never updates because the only nonzero contribution
to it is the lazy-init clone of g₁; subsequent updates contribute zero.
The role's update stays clamped at zero until mini-batch noise misaligns
the live grad enough for the cosine to drop below 1. Mitigations:

1. **Operate at λ ≤ 0.3.** The EMA continues to track the live grad.
   This is the recommended setting.
2. **Pre-surgery EMA update** as an alternative — update `ema` toward
   `g_pre` instead of `g_post`. Not implemented; would decouple the freeze
   risk from λ but loses the "self-reinforcing-EMA" guarantee.
3. **Min-scale floor / linear warmup** — clamp `scale ≥ scale_min` or warm
   λ from 0 to its target over `ema_window` steps. Not implemented.

### Global scalar, not per-parameter projection

The implementation accumulates `gg`, `ee`, `ge` as sums across *all*
`requires_grad` parameters, all-reduces those three scalars, computes one
global cosine, and broadcasts the resulting scalar onto every parameter
shard. A parameter tensor whose own gradient is locally orthogonal to its
own EMA is still damped by the global factor. This is what makes I6 cheap
(three-scalar all-reduce vs per-tensor reductions); do not interpret the
"surgery" language as PCGrad-style component projection.

## Backward compatibility

- Default `coherence_reg.enable=false` (or absent) — pre-I6 runs are
  unaffected.
- Per-role weight `0.0` short-circuits before any EMA allocation; a role
  with λ=0 pays zero overhead.
- I6 is independent of I1 — they can run together (both populate
  `overrides`) or alone.
- Single-agent / `share_policy=true` runs do not enter the multi-agent
  override block, so I6 has no effect there. Extending to shared-policy
  runs would require threading `coherence_reg_role` through a single-role
  channel.
