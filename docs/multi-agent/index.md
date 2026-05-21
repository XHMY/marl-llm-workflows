# Multi-Agent RL

This section documents the multi-agent reinforcement learning extensions in
this fork. They are not part of upstream
[rllm](https://github.com/rllm-org/rllm) and were built for the experiments
behind the paper *"When Does Multi-Agent RL Training Improve LLM Workflows?"*
([arxiv link TODO]).

## What this fork adds

| Component | Description | Files |
|---|---|---|
| Multi-agent workflows | Three workflow topologies (voting, evaluator-optimizer, orchestrator-workers) that route a single task through specialized roles. | `rllm/workflows/voting_workflow.py`, `evaluator_optimizer_workflow.py`, `orchestrator_workers_workflow.py` |
| Multi-agent LoRA | Per-role LoRA adapters with a shared-policy / isolated-policy toggle. Each role can update its own adapter or share one across roles. | `verl/verl/workers/fsdp_workers.py`, `rllm/trainer/verl/agent_workflow_trainer.py` |
| Multi-agent PPO trainer | GRPO-based trainer config tuned for multi-role rollouts; per-agent advantages, per-agent log probs, per-agent metrics. | `rllm/trainer/config/multi_agent_ppo_trainer.yaml` |
| Experiment launcher | Single entry point that submits a SLURM job for any (workflow, model, policy, task) cell. | `dashboard/launch_experiment.sh` |
| Checkpoint evaluator | Batched off-policy evaluation across checkpoints, with trajectory dumps. | `dashboard/evaluate_checkpoints.py` |

## Experimental design

The paper sweeps four factors:

- **Workflow** — voting, evaluator-optimizer, orchestrator-workers, plus a
  single-agent baseline.
- **Policy sharing** — *Isolated-Policy (IP)*: one LoRA adapter per role
  (`share_policy=false`). *Shared-Policy (SP)*: one adapter across roles
  (`share_policy=true`).
- **Scale** — Qwen3 0.6B, 1.7B, 4B.
- **Task** — DAPO-Math (math reasoning), DeepCoder-PrimeIntellect (code generation).

Each experiment is identified by the config name `{Workflow}-{Policy}-{Scale}-{Task}`,
for example `Voting-IP-1.7B-Math` or `Eval-Opt-SP-4B-DC`.

## Documentation map

- [Workflows](workflows.md) — role rosters, prompt routing, output aggregation
  for each of the four workflows.
- [Policy Sharing (IP vs SP)](policy-sharing.md) — the central design knob and
  how it maps onto LoRA adapters.
- [Multi-Agent LoRA](lora-implementation.md) — per-role adapter lifecycle:
  initialization, vLLM sync, save/load, post-hoc extraction.
- [Training](training.md) — GRPO, per-agent advantages, the
  `multi_agent_ppo_trainer.yaml` config, the reward-attribution caveat.
- [Trajectories and Evaluation](trajectories-and-eval.md) — trajectory dump
  layout, canonical datasets, the checkpoint-evaluation harness, the three
  code-execution modes for DeepCoder.
- [Monitoring](monitoring.md) — wandb log layout, per-agent metric keys,
  checkpoint structure on disk.
- [Running Experiments](running-experiments.md) — end-to-end commands to launch
  a training cell and evaluate the resulting checkpoints.
- [Interventions](interventions/i1-per-role-kl-anchor.md) — design notes for
  the two interventions reported in the paper.
