# Policy Sharing (IP vs SP)

Policy sharing is the central design knob this fork studies. Given a workflow
with several specialized roles, there are two natural ways to train them:

- **Shared-Policy (SP)** — one set of trainable parameters across all roles.
  Every role uses the same LoRA adapter; per-role behavior comes from the
  prompt alone.
- **Isolated-Policy (IP)** — one set of trainable parameters per role. Each
  role gets its own LoRA adapter, optimized independently against its own
  rollouts and advantages.

LoRA is the implementation substrate; the framework reasoning is about
*policy sharing* as a structural design choice. The paper reports when each
choice is preferable.

## Configuration

A single flag selects the mode:

```yaml
trainer:
  agent_names: ['generator', 'evaluator']
  share_policy: true     # SP: one adapter for all roles
  # share_policy: false  # IP: one adapter per role
```

`share_policy=true` initializes a single LoRA adapter and routes every role's
update into it. `share_policy=false` initializes one named adapter per entry
in `agent_names` and routes per-role rollouts to per-role adapters.

## How each maps onto LoRA

| Mode | Adapters | vLLM serving | Backward path |
|---|---|---|---|
| SP (`share_policy=true`) | One adapter (default name). | One `lora_int_id` for all roles. | All role rollouts contribute to the same adapter gradient. |
| IP (`share_policy=false`) | N adapters, one per `agent_names[i]`. | N `lora_int_id`s; inference routes by role. | Per-role mini-batch contributes to that role's adapter only; other adapters are frozen on that step. |

Under IP, the trainer iterates over roles, calls `set_active_lora(role)` to
swap the active adapter, computes log probs and updates the optimizer for
that role's sub-batch, then moves to the next role. Per-agent metrics
(`actor/{role}/training_ppl`, `actor/{role}/grad_norm`, etc.) are only
emitted in this mode; SP runs see only the global `actor/*` metrics.

## When to use which

The paper's empirical finding is that the IP-vs-SP choice has a non-trivial
interaction with workflow, task, and scale. As a rough orientation (see the
paper for the full result):

- IP often reaches a higher *peak* validation accuracy.
- IP is more susceptible to terminal accuracy cliffs in workflows with N>1
  same-role agents on the same prompt.
- SP avoids the per-role gradient-amplification mode but introduces its own
  failure modes when per-role gradient mass is asymmetric across roles.

For mechanistic detail on how each mode interacts with role-level gradient
dynamics, see the paper. For the implementation walkthrough, see
[Multi-Agent LoRA](lora-implementation.md).

## Backward compatibility

`share_policy=false` with `len(agent_names) == 1` (or empty) falls back to
single-adapter behavior — useful for single-agent baselines that still want
the per-role-adapter code path exercised.
