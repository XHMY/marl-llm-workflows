# Multi-Agent LoRA

This page documents how per-role LoRA adapters are implemented in this fork:
initialization, vLLM weight synchronization, per-role activation during
training, on-disk save, and post-hoc extraction.

## Key concepts

- **PEFT adapter names** (`"generator"`, `"evaluator"`, …) identify adapters
  within the PEFT model. They come from `trainer.agent_names`.
- **vLLM `lora_int_id`** (integers like `100`, `101`) route inference
  requests inside vLLM. The trainer maintains the string → int mapping.
- Agent names are extracted from `trajectory_ids` via
  `traj_id.rsplit("_", 1)[1]`. **Agent names must not contain underscores.**

## Files modified versus upstream

### verl backend

| File | Change |
|---|---|
| `verl/verl/utils/fsdp_utils.py` | Added `adapter_name` parameter to `layered_summon_lora_params()` and `collect_lora_params()`. |
| `verl/verl/workers/fsdp_workers.py` | Multi-agent LoRA initialization, `set_active_lora()` method, `save_single_lora_adapter()` method. |
| `verl/verl/workers/sharding_manager/fsdp_vllm.py` | Changed PEFT config access to support arbitrary adapter names. |
| `verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py` | Dynamic `max_loras` based on `agent_names` config; added `remove_lora()` and `list_loras()`. |
| `verl/verl/experimental/agent_loop/agent_loop.py` | Added `remove_lora()` to `AsyncLLMServerManager`. |
| `verl/verl/utils/dataset/rl_dataset.py` | Added JSON parsing for string `extra_info`. |

### rllm trainer

| File | Change |
|---|---|
| `rllm/trainer/verl/agent_workflow_trainer.py` | Per-agent log prob computation, per-agent reference log prob, per-agent actor updates. |

## Three guard sites under `share_policy=false`

When isolated-policy training is enabled, three independent locations in
`verl/verl/workers/fsdp_workers.py` gate on `len(agent_names)`:

1. **Init (around line 445)** — creates named adapters
   (e.g. `"generator"`, `"evaluator"`) via `add_adapter()`.
2. **vLLM weight sync (around line 692)** — decides between
   `_sync_multi_agent_lora_weights` and `_sync_single_agent_weights`.
3. **`set_active_lora()` (around line 975)** — switches the active adapter
   for the next training mini-batch.

All three must use `>= 1` (not `> 1`) so that a single-agent isolated-policy
run still exercises the multi-adapter code path. If the vLLM sync guard
(#2) is missed, trained weights never reach vLLM — the sync falls back to
collecting the stale `"default"` adapter without a `lora_int_id`, while
inference requests a specific `lora_int_id` and silently serves the base
model.

## Trainer changes — `agent_workflow_trainer.py`

### `_split_batch_by_agent()` helper

Extracts agent names from `trajectory_ids` via `traj_id.rsplit("_", 1)[1]`,
strips trailing digits so `generator0` / `generator1` route to the
`generator` adapter, groups samples by agent name, and returns sub-batches
with their original indices so the per-agent results can be scattered back
into a full batch.

### `compute_log_prob`

Multi-agent mode iterates over agents, calls `set_active_lora(role)` to swap
the adapter, computes log probs on that role's sub-batch, and scatters
results back to the original positions.

### `compute_ref_log_prob`

Under LoRA, the reference policy lives in the same actor worker; the worker
computes reference log probs by un-applying the active LoRA adapter
(`fsdp_workers.py:compute_ref_log_prob` with `is_lora=True`). No separate
reference-policy worker is created when `lora_rank > 0`.

### `update_actor`

Iterates over agents, calls `set_active_lora(role)`, runs the optimizer step
on that role's sub-batch, and collects metrics with agent-specific prefixes
(`actor/generator/loss`, `actor/evaluator/grad_norm`, …).

## Configuration

```yaml
trainer:
  agent_names: ['generator', 'evaluator']
  share_policy: false      # one adapter per agent

actor_rollout_ref:
  model:
    lora_rank: 64
    lora_alpha: 32
    target_modules: all-linear
```

When `agent_names` is empty or has one entry, the original (single-adapter)
code path is used; `adapter_name="default"` is the default parameter in the
verl helper functions.

## Checkpoint layout

Per-role LoRA adapters land under each checkpoint step:

```
checkpoints/<project>/<experiment>/global_step_N/
├── actor/
│   ├── lora_adapter_generator/
│   ├── lora_adapter_evaluator/
│   └── ...
└── data.pt
```

Only the last ~7-10 checkpoint steps keep full FSDP weights; older steps are
LoRA-only and not directly resumable. Check for `model_world_size_*.pt`
before picking a `resume_from_path`.

## Post-hoc LoRA extraction

`scripts/extract_lora_from_checkpoint.py` extracts per-role LoRA adapters
from a checkpoint's FSDP shards into a flat HuggingFace-PEFT-compatible
directory. Use this when you want to serve a trained adapter outside the
training stack or merge it into the base model.
