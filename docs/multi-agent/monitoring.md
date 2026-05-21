# Monitoring

Operational reference for wandb logs and checkpoint structure produced by
multi-agent training.

## Checkpoint structure

```
checkpoints/
├── <project>/
│   └── <experiment>/
│       ├── global_step_5/
│       │   ├── actor/
│       │   │   ├── lora_adapter_generator/   # per-role LoRA weights (IP)
│       │   │   └── lora_adapter_aggregator/
│       │   └── data.pt                       # training state
│       ├── global_step_10/
│       ├── ...
│       ├── latest_checkpointed_iteration.txt # current step number
│       ├── training_metadata.json            # wandb_run_id, slurm_job_id, total_steps
│       └── eval_results.jsonl                # per-checkpoint evaluation results
└── init_weight/                              # base model weights
```

### `eval_results.jsonl`

One JSON object per checkpoint evaluation, with fields:

| Field | Meaning |
|---|---|
| `checkpoint_step` | Training step the checkpoint was saved at. |
| `accuracy` | Per-checkpoint accuracy on the evaluation dataset. |
| `dataset` | Evaluation dataset name (`dapo_math`, `deepcoder_primeintellect`, …). |
| `workflow_type` | Workflow used at evaluation time. |
| `share_policy` | `True` (SP) or `False` (IP). |
| `eval_mode` | `training_validation`, `trained_checkpoint`, or `base_model`. |
| `pass_at_k` | Pass@k scores when applicable. |

### FSDP retention

Only the last ~7-10 checkpoint steps retain full FSDP weights; older steps
keep only the LoRA adapter directories and are not directly resumable.
Before picking a `resume_from_path`, verify that
`model_world_size_*.pt` exists in that step's `actor/` directory.

## Wandb on-disk layout

```
wandb/
├── run-{YYYYMMDD_HHMMSS}-{run_id}/
│   └── files/
│       ├── output.log           # training metrics, one line per step
│       ├── config.yaml          # full training config
│       ├── wandb-metadata.json  # run metadata
│       └── wandb-summary.json   # final metrics
```

Runs resume across SLURM preemptions, producing multiple `run-*-{same_id}`
directories per experiment. Stitch them by sorting directories that share a
`run_id`.

## Fetching wandb metrics

Fetch per-step history with the standard `wandb` Python client. The history
streamed by `scan_history` contains one row per logged step; each row is a
sparse dict keyed by metric name.

```python
import wandb

api = wandb.Api()
run = api.run(f"{entity}/{project}/{run_id}")
for row in run.scan_history(keys=[
    "actor/generator/training_ppl",
    "actor/generator/chi2_token",
    "_step",
]):
    step = row["_step"]
    gen_ppl = row.get("actor/generator/training_ppl")
    gen_chi2 = row.get("actor/generator/chi2_token")
```

For large runs, prefer caching the result on disk (parquet works well) to
avoid re-streaming on every analysis pass.

### Resume-segment caveat

Post-resume wandb segments may log fewer keys than segment 1; some metric
keys (`batch/*`, `chi2_token`, `grad_norm` in some runs) are dropped from
later segments and appear as missing keys in per-step row dicts. Check key
presence per-step before concluding that a metric plateaued.

## Key metrics

### Global metrics (all runs)

| Metric | Meaning |
|---|---|
| `actor/entropy` | Token entropy averaged across all roles. Not per-agent. |
| `critic/score/mean` | Per-trajectory reward mean. |
| `critic/rewards/mean` | Per-trajectory reward (identical to `batch/success` in voting / orchestrator-workers with fixed trajectory count; differs by 3-7% in eval-opt due to variable trajectories per problem). |
| `batch/success` | Per-problem accuracy. Preferred for task accuracy. |
| `training/rollout_probs_diff_mean` | Policy drift between training and rollout. |

### Per-agent metrics (IP runs only)

| Metric | Meaning |
|---|---|
| `actor/{role}/training_ppl` | Per-agent perplexity. |
| `actor/{role}/chi2_token` | Per-token chi-squared divergence between training and rollout policies. |
| `actor/{role}/grad_norm` | Gradient norm. Supervisory roles often have 5-8× larger grad norms than generators. |
| `actor/{role}/kl` | KL divergence from the rollout policy. |
| `actor/{role}/pg_clipfrac` | PPO clip fraction. |
| `actor/{role}/pg_loss` | Policy-gradient loss. |
| `actor/{role}/ppl_ratio` | Training / rollout PPL ratio. |

Role names: `generator`, `aggregator`, `evaluator`, `orchestrator`, `worker`, `synthesizer`.

Under shared-policy training there is no per-agent breakdown; only the
global `actor/*` metrics are emitted.

## Validation accuracy

The default validation dataset for math training entry points is AIME2025
(~30 problems), logged as `val/unknown/pass@1`. This is *not* the full
DAPO-Math validation set (~1412 problems) — for the canonical validation
accuracy in the paper, evaluate via `dashboard/evaluate_checkpoints.py`
against `dapo_math` rather than relying on `val/unknown/pass@1` for math.

For DeepCoder, `val/unknown/pass@1` is computed on the canonical
PrimeIntellect validation split and can be used directly.

For canonical paper-side accuracy numbers, the source-priority chain is:
(1) the per-checkpoint `eval_results.jsonl` written by
`dashboard/evaluate_checkpoints.py` against `dapo_math` (math) or
`deepcoder_primeintellect` (code); (2) wandb `val/unknown/pass@1` as a
fallback for DeepCoder cells that lack a per-checkpoint evaluation;
(3) the repo-level `eval_results.jsonl` rows tagged `eval_mode="base_model"`
for the reference line. Math has no wandb fallback because
`val/unknown/pass@1` on math runs is AIME2025, not the canonical
DAPO-Math validator.

## `batch/success` versus `critic/rewards/mean`

`batch/success` is per-problem accuracy; `critic/rewards/mean` is
per-trajectory reward. They are identical in voting and orchestrator-workers
where every problem produces a fixed number of trajectories. They differ by
3-7% in evaluator-optimizer because iterative refinement produces a variable
number of trajectories per problem. Use `batch/success` when reporting task
accuracy.
