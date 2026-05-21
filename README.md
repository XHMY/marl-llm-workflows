# When Does Multi-Agent RL Training Improve LLM Workflows?

Code and experiments for the arxiv paper **"When Does Multi-Agent RL Training Improve LLM Workflows?"** ([arxiv link TODO]).

We study end-to-end RL training of multi-agent LLM workflows: when does jointly training roles in a workflow (voting, evaluator-optimizer, orchestrator-workers) improve over the base model, and when does training peak and then collapse? The paper presents an empirical map across workflows × scales × tasks × policy-sharing strategies, and explains the strongest patterns through role-level gradient mechanisms.

## Built on rllm

This repository is a research fork of [rllm v0.2.1](https://github.com/rllm-org/rllm) ([Tan et al., 2025](https://pretty-radio-b75.notion.site/rLLM-A-Framework-for-Post-Training-Language-Agents-21b81902c146819db63cd98a54ba5f31)). It adds:

- Three multi-agent workflows (voting, evaluator-optimizer, orchestrator-workers) plus a single-agent baseline.
- Multi-agent LoRA: per-role adapters with a shared-policy / isolated-policy toggle.
- A multi-agent PPO trainer (GRPO + per-agent advantages + per-agent metrics).
- An experiment launcher and checkpoint evaluator under `dashboard/`.

The training backend (`verl/`) is a git submodule of [our verl fork](https://github.com/XHMY/verl), pinned to the `rllm-0.2.1-local` branch — upstream verl `v0.6.1` plus one commit that adds the multi-agent LoRA and agent-loop changes used in this repository (see [docs/multi-agent/lora-implementation.md](docs/multi-agent/lora-implementation.md)).

## Installation

Requires Python 3.11 and a CUDA-capable GPU. The recommended path is `uv`:

```bash
git clone --recurse-submodules <repo-url>
cd rllm_0.2.1

uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .[verl]
```

If you cloned without `--recurse-submodules`, initialize the submodules separately:

```bash
git submodule update --init --recursive
```

For alternate install paths (direct pip install from git, Docker, or Tinker as a training backend), see the [upstream rllm installation guide](https://rllm-project.readthedocs.io/en/latest/getting-started/installation).

## Quick start

Launch one cell of the experimental matrix — voting workflow, isolated policy, 1.7B model, math task — on a 2× H100 node:

```bash
bash dashboard/launch_experiment.sh \
    --workflow voting \
    --model 1.7B \
    --share-policy false \
    --slurm-config dashboard/slurm_config_template.conf \
    --n-gpus 2 --cpus-per-gpu 8 --mem-per-gpu 128G
```

Copy `dashboard/slurm_config_template.conf` first and fill in the placeholders (partition, account, time limit, GPU constraint) for your cluster. Add `--dry-run` to preview the generated sbatch script without submitting. See [docs/multi-agent/running-experiments.md](docs/multi-agent/running-experiments.md) for the full set of CLI flags and the procedure for training every cell in the matrix.

## Multi-Agent RL extensions

The implementation diff against upstream rllm is documented under `docs/multi-agent/`:

| Topic | Page |
|---|---|
| Overview of what this fork adds | [docs/multi-agent/index.md](docs/multi-agent/index.md) |
| The three multi-agent workflows + single-agent baseline | [docs/multi-agent/workflows.md](docs/multi-agent/workflows.md) |
| Isolated-Policy (IP) vs Shared-Policy (SP) | [docs/multi-agent/policy-sharing.md](docs/multi-agent/policy-sharing.md) |
| Multi-agent LoRA implementation | [docs/multi-agent/lora-implementation.md](docs/multi-agent/lora-implementation.md) |
| Training loop and PPO config | [docs/multi-agent/training.md](docs/multi-agent/training.md) |
| Trajectory dumps, checkpoint evaluation, code-execution modes | [docs/multi-agent/trajectories-and-eval.md](docs/multi-agent/trajectories-and-eval.md) |
| wandb metrics and checkpoint structure | [docs/multi-agent/monitoring.md](docs/multi-agent/monitoring.md) |
| End-to-end recipe (install → train → eval) | [docs/multi-agent/running-experiments.md](docs/multi-agent/running-experiments.md) |
| Intervention design notes (per-role KL anchor, coherence regularizer) | [docs/multi-agent/interventions/](docs/multi-agent/interventions/i1-per-role-kl-anchor.md) |

## Repository layout

| Path | Purpose |
|---|---|
| `rllm/` | rllm framework (forked from upstream); multi-agent workflows live in `rllm/workflows/` and the multi-agent trainer in `rllm/trainer/verl/agent_workflow_trainer.py`. |
| `verl/` | Training backend (git submodule). Multi-agent LoRA support in `verl/verl/workers/fsdp_workers.py`. |
| `examples/math_reasoning/` | Math experiment entry points (voting / eval-opt / orch-workers / single-agent baseline). |
| `examples/deepcoder/` | DeepCoder code-generation entry points. |
| `dashboard/` | Experiment launcher (`launch_experiment.sh`), checkpoint evaluator (`evaluate_checkpoints.py`), SLURM config template. |
| `scripts/` | General utilities (LoRA extraction / merging, checkpoint verification, dataset prep). |

## Running the experiments

1. **Install** as above.
2. **Configure SLURM** by copying `dashboard/slurm_config_template.conf` and filling in the placeholders (partition, account, time limit, GPU type) for your cluster.
3. **Launch experiments** from `dashboard/launch_experiment.sh` — one cell per invocation across the workflow × policy × scale × task matrix.
4. **Evaluate checkpoints** with `python dashboard/evaluate_checkpoints.py --use-training-lengths`. Trajectories land under `evaluation_trajectories/` and per-checkpoint accuracy in `<experiment_dir>/eval_results.jsonl`.

The complete step-by-step recipe is in [docs/multi-agent/running-experiments.md](docs/multi-agent/running-experiments.md).

## Citation

If you use this code or the experiments, please cite the paper:

```bibtex
@misc{multi_agent_rl_workflows_2026,
  title  = {When Does Multi-Agent RL Training Improve LLM Workflows?},
  author = {TODO},
  year   = {2026},
  note   = {arxiv preprint; [arxiv link TODO]}
}
```

## Acknowledgments

Built on the [rllm framework](https://github.com/rllm-org/rllm) .

## License

This repository inherits the upstream rllm license — see `LICENSE`.
