# Running Experiments

End-to-end recipe for launching the multi-agent training cells documented in
this fork and evaluating the resulting checkpoints.

The experimental matrix is `workflow × policy × scale × task`:

- Workflows: voting, evaluator-optimizer, orchestrator-workers, single-agent baseline.
- Policies: IP (`share_policy=false`), SP (`share_policy=true`).
- Scales: Qwen3 0.6B, 1.7B, 4B.
- Tasks: math (DAPO-Math-17K), code (DeepCoder-PrimeIntellect).

Each cell is named `{Workflow}-{Policy}-{Scale}-{Task}` (e.g. `Voting-IP-1.7B-Math`).

## 1. Install

```bash
git clone --recurse-submodules <repo-url>
cd rllm_0.2.1
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .[verl]
```

## 2. Configure SLURM for your cluster

Copy `dashboard/slurm_config_template.conf` and fill in the placeholders for
your cluster (partition, account, time limit, GPU constraint). The `# META:
GPU_TYPE=` line selects the per-GPU PPO token budget; set it to `H100`,
`L40s`, `A40`, or `RTX8000` depending on the GPU model your node provisions.

## 3. Launch a training cell

`dashboard/launch_experiment.sh` generates a single sbatch script from CLI
arguments and submits it.

```bash
# Voting × IP × 1.7B × Math
bash dashboard/launch_experiment.sh \
    --workflow voting \
    --model 1.7B \
    --share-policy false \
    --slurm-config dashboard/slurm_config_template.conf \
    --n-gpus 2 --cpus-per-gpu 8 --mem-per-gpu 128G
```

Preview the generated sbatch script without submitting:

```bash
bash dashboard/launch_experiment.sh \
    --workflow voting --model 1.7B --share-policy false \
    --slurm-config dashboard/slurm_config_template.conf \
    --n-gpus 2 --cpus-per-gpu 8 --mem-per-gpu 128G \
    --dry-run
```

### CLI flags

| Flag | Values |
|---|---|
| `--workflow` | `voting`, `evaluator_optimizer`, `orchestrator_workers_propose`, `single_agent` |
| `--model` | `0.6B`, `1.7B`, `4B` |
| `--share-policy` | `true` (SP), `false` (IP) |
| `--task-type` | `math` (default) or `deepcoder` |
| `--slurm-config` | Path to a `.conf` file modelled on `dashboard/slurm_config_template.conf` |
| `--n-gpus` / `--cpus-per-gpu` / `--mem-per-gpu` | SLURM resources injected into the sbatch script |
| `--workflow-params` | Override workflow-specific config (e.g. `n_votes`, `max_iterations`). |
| `--dry-run` | Print the sbatch script without submitting. |

For DeepCoder cells:

```bash
bash dashboard/launch_experiment.sh \
    --workflow voting --model 1.7B --share-policy false \
    --task-type deepcoder \
    --slurm-config dashboard/slurm_config_template.conf \
    --n-gpus 2 --cpus-per-gpu 8 --mem-per-gpu 128G
```

DeepCoder cells require enough CPU cores for the code-execution pool; see
[Trajectories and Evaluation §code-execution-modes](trajectories-and-eval.md#code-execution-modes).

## 4. Evaluate checkpoints

After training, evaluate every saved checkpoint off-policy with the canonical
validation dataset:

```bash
python dashboard/evaluate_checkpoints.py \
    --experiment-dir checkpoints/<project>/<experiment>/ \
    --dataset dapo_math \
    --use-training-lengths \
    --max-samples 1412 \
    --trajectory-output-dir evaluation_trajectories
```

- `--use-training-lengths` keeps prompt and response length caps consistent
  with training. Always pass this.
- `--dataset` is `dapo_math` for math cells and
  `deepcoder_primeintellect` for code cells.
- Per-checkpoint results are appended to `<experiment_dir>/eval_results.jsonl`
  and trajectory dumps land under
  `evaluation_trajectories/<experiment>/<dataset>/step_<N>/`.

The output layout is documented in
[Trajectories and Evaluation](trajectories-and-eval.md#output-layout).

## 5. Per-agent correctness from trajectory dumps

Under the default training setting (`use_final_outcome_reward=true`), the
per-agent `reward` field in `evaluation_trajectories/<workflow>/.../eval_*.json`
is the broadcast workflow outcome, not per-agent correctness. The top-level
`traj_dump['is_correct']` and the aggregator / synthesizer / final-iter
trajectory's `reward` field do carry the workflow outcome correctly — those
are safe to read as workflow pass@1. To recover per-agent correctness after
the fact, regrade each agent's parsed answer against ground truth with the
same verifier used at training time (math: a `grade_answer_*` function from
the math verifier; code: the code-execution reward function).

## 6. Interventions

The two interventions reported in the paper:

- [I1 — Per-role KL anchor](interventions/i1-per-role-kl-anchor.md)
- [I6 — Coherence regularizer](interventions/i6-coherence-regularizer.md)

Each intervention page documents its config switches and the code path it
touches.
