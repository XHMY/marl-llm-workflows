# Training

This page walks through the multi-agent PPO trainer config and the per-agent
training loop. The single source-of-truth config is
`rllm/trainer/config/multi_agent_ppo_trainer.yaml`; the trainer that consumes
it is `rllm/trainer/verl/agent_workflow_trainer.py`.

## Algorithm

GRPO is the default advantage estimator. The trainer's per-agent loop
computes log probs, reference log probs, and advantages per role under
isolated-policy training; under shared-policy training the loop folds into a
single update over all roles.

```yaml
algorithm:
  adv_estimator: grpo
```

## Config walkthrough

### Model and LoRA

```yaml
actor_rollout_ref:
  model:
    lora_rank: 64
    lora_alpha: 32
    target_modules: all-linear
    use_remove_padding: True
```

`target_modules: all-linear` applies LoRA to every linear layer.
`use_remove_padding=True` packs sequences for the actor forward pass.

### Optimizer / PPO

```yaml
  actor:
    ppo_mini_batch_size: 64
    loss_agg_mode: seq-mean-token-mean
    use_dynamic_bsz: True
    ppo_max_token_len_per_gpu: 51200
    ppo_micro_batch_size_per_gpu: 4
    clip_ratio_high: 0.28
    ppo_epochs: 1
    optim:
      lr: 2e-5
      lr_warmup_steps: 15
      warmup_style: cosine
```

One PPO epoch per training step. Mini-batches of 64 are split into
micro-batches of 4 per GPU; dynamic-batch-size packing fills each micro-batch
up to `ppo_max_token_len_per_gpu` tokens. Sequence-mean / token-mean loss
aggregation keeps gradient magnitude comparable across responses of
different lengths.

### Rollout

```yaml
  rollout:
    mode: async
    name: vllm
    n: 8
    temperature: 0.7
    gpu_memory_utilization: 0.85
    enable_prefix_caching: True
    enable_chunked_prefill: True
```

Async vLLM rollout with `n=8` samples per prompt at temperature 0.7. Prefix
caching and chunked prefill improve throughput when multiple roles share
prompt context.

### Multi-agent trainer settings

```yaml
trainer:
  agent_names: ['default']
  share_policy: False
  total_training_steps: 500
  save_freq: 5
  test_freq: 10
```

- `agent_names` — the list of roles in the workflow. Override per workflow:
  `['generator', 'aggregator']` for voting, `['generator', 'evaluator']` for
  evaluator-optimizer, `['orchestrator', 'worker', 'synthesizer']` for
  orchestrator-workers.
- `share_policy` — see [Policy Sharing](policy-sharing.md).
- `save_freq` — checkpoint every N training steps. The on-disk LoRA layout
  is in [Multi-Agent LoRA](lora-implementation.md).
- `test_freq` — validation every N training steps.

### Workflow knobs

```yaml
rllm:
  workflow:
    use_workflow: True
    use_final_outcome_reward: True
    n_parallel_tasks: 512
    retry_limit: 3
    code_executor_workers: 0
    max_concurrent_code_execs: 0
```

- `use_final_outcome_reward=True` broadcasts the final workflow outcome
  reward to every trajectory in the episode (see the reward-attribution
  caveat below).
- `code_executor_workers` and `max_concurrent_code_execs` select among the
  three code-execution modes documented in
  [Trajectories and Evaluation](trajectories-and-eval.md#code-execution-modes).

### Stepwise advantage

```yaml
rllm:
  stepwise_advantage:
    enable: True
    mode: per_step
    normalize_by_steps: False
```

Multi-agent episodes are sequences of steps with role-specific rewards;
`stepwise_advantage` configures how per-step rewards are folded into the GRPO
advantage estimate.

## Per-agent training loop

Under `share_policy=false`, each training step iterates:

1. Roll out one episode per task using the workflow (parallel up to
   `n_parallel_tasks`).
2. Compute per-agent log probs:
   - Iterate over agents.
   - `set_active_lora(role)` swaps the adapter.
   - Compute log probs on that role's sub-batch.
   - Scatter results back into the full batch.
3. Compute per-agent reference log probs (under LoRA, the reference is the
   base model — reached by un-applying the active adapter).
4. Compute GRPO advantages.
5. Update each adapter:
   - `set_active_lora(role)`.
   - Run one optimizer step on that role's sub-batch.
   - Emit per-agent metrics with the `actor/{role}/*` prefix.

## Reward attribution caveat

All non-v2 workflows are launched with `rllm.workflow.use_final_outcome_reward=true`.
In `rllm/workflows/voting_workflow.py:324-331` (and the analogous sites in
`evaluator_optimizer_workflow.py` and `orchestrator_workers_workflow.py`),
this copies the aggregator's (or synthesizer's) scalar reward onto every
generator / worker trajectory. So under default settings, every per-agent
training-time reward in voting / eval-opt / orchestrator-workers is
structurally identical across same-role agents.

The same flag also applies at evaluation time when reading the trajectory
dumps: `dashboard/task_configs.py` (math: lines 43-66; DeepCoder:
lines 108-138) overrides `voting`, `evaluator_optimizer`, and
`orchestrator_workers_propose` to `use_final_outcome_reward=True` at eval
time, so per-agent `reward` in `evaluation_trajectories/<workflow>/.../eval_*.json`
is the broadcast workflow outcome, *not* per-agent correctness.

If you want per-agent training rewards, use the v2 workflow variants
(`voting_v2_workflow.py`, `evaluator_optimizer_v2_workflow.py`) or set
`use_final_outcome_reward=false` explicitly.

To recover per-agent correctness after the fact, regrade each agent's
parsed answer against ground truth with the same verifier used at training
time (math: a `grade_answer_*` function from the math verifier; code: the
code-execution reward function on the parsed program against the ground-truth
test cases).

The top-level `traj_dump['is_correct']` and the
aggregator / synthesizer / final-iter trajectory's `reward` field DO carry
the workflow outcome correctly; those are safe to read as workflow pass@1.

## Common training configuration

```yaml
trainer:
  agent_names: ['generator', 'evaluator']
  share_policy: True
  n_gpus_per_node: 4

actor_rollout_ref:
  model:
    path: Qwen/Qwen3-1.7B
    lora_rank: 64
    lora_alpha: 32

data:
  max_prompt_length: 15360
  max_response_length: 5120
```

## Launching a training run

The end-to-end command:

```bash
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

python3 -m examples.math_reasoning.train_voting_math \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    trainer.agent_names=['generator','aggregator'] \
    trainer.share_policy=false \
    +rllm.workflow.n_votes=3
```

For SLURM submission across the paper's full experimental matrix, use
`dashboard/launch_experiment.sh`. See
[Running Experiments](running-experiments.md).

## Key configuration parameters

| Parameter | Description |
|---|---|
| `trainer.agent_names` | Role list. Used to allocate per-role adapters. |
| `trainer.share_policy` | `true` = SP (one adapter), `false` = IP (per-role). |
| `trainer.n_gpus_per_node` | GPUs per node. |
| `trainer.total_training_steps` | Total training steps. |
| `data.max_prompt_length` | Max prompt tokens. |
| `data.max_response_length` | Max response tokens per role rollout. |
| `actor_rollout_ref.model.path` | Base model checkpoint. |
| `actor_rollout_ref.model.lora_rank` | LoRA rank. |
| `actor_rollout_ref.model.lora_alpha` | LoRA alpha. |
| `rllm.workflow.n_parallel_tasks` | Parallel workflow instances per step. |
