# Architecture

`marl-llm-workflows` adds multi-agent workflow training to two unmodified upstream projects:
[rLLM](https://github.com/rllm-org/rllm), which runs agent workflows and trains them with its
workflow trainer, and [verl](https://github.com/volcengine/verl), which provides the FSDP actor,
the vLLM rollout servers and the PPO/GRPO machinery. Both are git submodules under `third_party/`.
Everything in this package is a subclass, a registry hook, or a bounded copy of an upstream function
marked with `# >>> MARL(<id>)` / `# <<< MARL(<id>)`. Nothing is patched at import time.

- [design.md](design.md): the module map (module, upstream class it builds on, role) and the contract
  table of every name that crosses a module boundary (trajectory names, LoRA ids, metric names,
  checkpoint layout, experiment names).
- [upstream-patches.md](upstream-patches.md): the complete list of runtime alterations of upstream
  code, where each is applied, and the upstream classes that are subclassed instead of patched.

## What a training step does

1. **Rollout.** For each problem in the batch, the trainer runs the workflow (for example Voting:
   three generator calls, then one aggregator call). Each model call names its role (`agent_name`).
   The engine maps the role to a LoRA id (one adapter per role under IP, the single shared adapter
   under SP) and the vLLM server applies that adapter to the request.
2. **Reward.** The workflow grades the episode (Math: the boxed answer after the reasoning; Code: unit
   tests in sandboxed processes). By default every role receives the episode outcome.
3. **Advantages.** Each role's sequences form their own GRPO groups across the samples of a problem.
4. **Update.** Under IP the batch is split by role and each role's adapter is updated only on its own
   sequences (one optimizer call per role). Under SP one update covers all roles.
5. **Sync.** The updated adapters are loaded into the vLLM servers under their LoRA ids before the
   next rollout.

## Layers

| Layer | Package modules | Built on |
|---|---|---|
| Entry point | `trainer/train.py` (Hydra main), `trainer/config_injection.py`, `trainer/task_runner.py` | rLLM's `TaskRunner` (bounded copy, workflow path only); the Hydra config `configs/multi_agent_ppo_trainer.yaml` extends verl's `ppo_trainer`. The recipe (`recipes.py`) chosen by `marl.task` / `marl.workflow` supplies the workflow class, role names and reward. |
| Trainer | `trainer/trainer.py`, `trainer/role_routing.py`, `trainer/loss_weight.py`, `trainer/wallclock.py`, `trainer/tracking.py` | rLLM's `AgentWorkflowPPOTrainer`. The training loop is a bounded copy so that it trains exactly the configured number of steps, stops cleanly before a time limit, and saves per-role adapters and run metadata. Role routing wraps the verl worker group: it splits the batch by role, activates the role's adapter and renames metrics to `actor/<role>/...`. |
| Actor workers | `verl_ext/workers.py`, `verl_ext/actor.py`, `verl_ext/lora_utils.py` | verl's FSDP `AsyncActorRolloutRefWorker` and `DataParallelPPOActor`. Adds one PEFT adapter per role, switches the trainable adapter, syncs every adapter to vLLM, saves each adapter separately, supports a per-role KL coefficient and counts optimizer steps. |
| Rollout | `verl_ext/vllm_rollout.py`, `verl_ext/vllm_server.py`, `engine/verl_engine.py` | verl's async vLLM rollout, HTTP server and server manager; rLLM's `VerlEngine`. Allows several LoRA adapters in vLLM, routes each request to its role's adapter, and gives concurrent calls of one episode distinct request ids. |
| Workflows | `workflows/base.py`, `workflows/{voting,voting_v2,evaluator_optimizer,orchestrator_workers}.py`, `workflows/math/*`, `workflows/code/*`, `workflows/prompts/*.json` | rLLM's `Workflow`. Trajectory names encode the role (`generator0`, `aggregator`, ...), which is how the trainer recovers the role of every sequence. |
| Rewards | `rewards/math_reward.py`, `rewards/code_reward.py` | rLLM's math and code reward functions; the code executor runs tests in separate processes with CPU-time and memory limits. |
| Data | `data/registry.py`, `data/rl_dataset.py`, `data/prepare_*.py` | rLLM's `DatasetRegistry` (storage moved to `$MARL_DATA_DIR`) and verl's `RLHFDataset` (polars reader). |
| Evaluation | `eval/evaluate_checkpoints.py`, `eval/checkpoints.py`, `eval/vllm_server.py`, `engine/openai_engine.py` | rLLM's `OpenAIEngine` and workflow engine against a LoRA-enabled vLLM OpenAI server; one adapter per role is served under its own model name. |
| Launcher | `launch.py`, `scripts/launch.py`, `configs/paper_runs.yaml`, `configs/slurm/*.conf`, `scripts/slurm/*.tmpl` | Pure string building: maps a cell and a control to Hydra overrides, renders SLURM scripts, reproduces paper runs. |

## Shared names

`marl_workflows/contracts.py` holds every literal that one layer produces and another consumes:
the trajectory-name grammar and role parsing, the LoRA id rule (`123 + index of the role`, matching
verl's single-adapter id 123 under SP), adapter directory names, metric names, stdout markers and the
experiment-name format. `contracts.py`, `recipes.py`, `utils.py` and `launch.py` import neither torch
nor rLLM nor verl, so the launcher and the tests of these modules run without the training stack.
