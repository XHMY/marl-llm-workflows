# Design

`marl-llm-workflows` trains multi-agent LLM workflows (Voting, Evaluator-Optimizer,
Orchestrator-Workers, and the single-agent baseline) with GRPO, either with one LoRA adapter
shared by all roles (SP, `share_policy`) or one adapter per role (IP, `multi_lora`). It is a
thin layer over two **unmodified** upstream projects, pinned as git submodules:

| Submodule | Upstream | Pin |
|---|---|---|
| `third_party/rllm` | https://github.com/rllm-org/rllm | `186d0bbbb5bfe18f8f6b8e4eac6db9379cf27738` |
| `third_party/verl` | https://github.com/volcengine/verl | tag `v0.6.1` (`d62da4950573d7a4b7ef2362337952e7ab59e78d`) |

All extensions are subclasses, registry hooks or bounded, marked copies (`# >>> MARL(<id>)` /
`# <<< MARL(<id>)`); nothing is patched at import time, and nothing writes inside
`third_party/` (datasets live under `$MARL_DATA_DIR`).

## Using the repository

Two equivalent ways to put the three packages on the import path:

1. **No install (PYTHONPATH).** In an environment that already has the pinned stack
   (`requirements/stack.txt`):
   ```bash
   source scripts/env.sh   # PYTHONPATH=<repo>:<repo>/third_party/rllm:<repo>/third_party/verl:$PYTHONPATH
   python -c "import verl, rllm; print(verl.__file__, rllm.__file__)"   # both inside third_party/
   ```
   `env.sh` prepends, so the submodules take precedence over any rllm/verl installed in the env.
2. **Editable installs**, in this order (rllm's `verl` extra would otherwise reinstall verl
   0.6.1 from PyPI over the submodule):
   ```bash
   pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
   pip install -r requirements/stack.txt
   pip install --no-deps -e third_party/verl
   pip install --no-deps -e third_party/rllm
   pip install -e ".[dev]"
   ```

`requirements/stack.txt` holds the exact pins of the direct dependencies (equal to
`pyproject.toml` `dependencies`, enforced by `tests/test_upstream_pins.py`);
`requirements/lock-fork-env.txt` is the full `pip freeze` of the environment that produced the
paper runs (informational).

Environment variables: `MARL_DATA_DIR` (dataset registry root, default
`~/.cache/marl-llm-workflows`), `MARL_INIT_WEIGHT_DIR` (merged `*_s300` init weights, default
`checkpoints/init_weight`), `MARL_LORA_STRICT` (default strict; `0` restores the research
fork's silent base-model fallback for an unknown or missing LoRA id).

## Module map

Importing `marl_workflows`, `marl_workflows.contracts`, `marl_workflows.recipes` or
`marl_workflows.utils` imports neither torch nor rllm nor verl.

| Module | Builds on (upstream) | Role |
|---|---|---|
| `contracts.py` | constants mirror verl `verl/workers/rollout/vllm_rollout/utils.py` (`VLLM_LORA_INT_ID=123`, `VLLM_LORA_PATH`) and rLLM `trajectory_id = f"{task_uuid}_{name}"` | Single source of every name crossing a boundary (table below) |
| `recipes.py` | — | `(task, workflow)` table: agent names, prompt/response lengths, workflow kwargs, class paths; `recipe_for_experiment` |
| `utils.py` | — | `marl_data_dir()`, `init_weight_dir()`, `allocated_cpu_count()`, `repo_root()` |
| `data/registry.py` | rLLM `rllm.data.dataset.DatasetRegistry` | `MarlDatasetRegistry`: storage relocated to `$MARL_DATA_DIR`; `data_path` / `verl_path` |
| `data/rl_dataset.py` | verl `verl.utils.dataset.rl_dataset.RLHFDataset` | `PolarsRLHFDataset` (`data.custom_cls`), polars parquet reader |
| `data/prepare_*.py` | rLLM `DatasetRegistry.register_dataset` | `dapo_math`, `aime2025`, `aime2024`, `math500`, `deepcoder_primeintellect`, `smoke_math` |
| `verl_ext/workers.py` | verl `verl.workers.fsdp_workers.AsyncActorRolloutRefWorker` | `MultiAgentAsyncActorRolloutRefWorker`: one PEFT adapter per role, per-role weight sync to vLLM (ids 123..123+N-1), `set_active_lora`, `save_single_lora_adapter`, `probe_rollout_lora_state` |
| `verl_ext/actor.py` | verl `verl.workers.actor.dp_actor.DataParallelPPOActor` | `MultiAgentDPActor`: per-role KL coefficient override, optimizer-step counters |
| `verl_ext/lora_utils.py` | verl `verl.utils.fsdp_utils` (`layered_summon_lora_params`, `collect_lora_params`) | Adapter-aware copies (`adapter_name` argument) |
| `verl_ext/vllm_rollout.py` | verl `vllm_rollout_spmd.vLLMAsyncRollout` | `MultiLoRAvLLMAsyncRollout`: `max_loras` from config, per-id adapter add/remove |
| `verl_ext/vllm_server.py` | verl `vllm_async_server.vLLMHttpServerBase`, `vLLMReplica` | `MultiLoRAvLLMHttpServer(Base)`, `MultiLoRAvLLMReplica`: per-request LoRA id routing |
| `verl_ext/patches.py` | verl `fsdp_workers.get_peft_model`, `rollout.base._ROLLOUT_REGISTRY`, `RolloutReplicaRegistry` | The only runtime-patch module: P1 (scoped), P2 (worker), H1 (driver registry hook) |
| `engine/verl_engine.py` | verl `agent_loop.AsyncLLMServerManager`; rLLM `rllm.engine.rollout.verl_engine.VerlEngine` | `MultiLoRAServerManager` (unique vLLM request ids, LoRA id forwarding), `MultiLoRAVerlEngine` (`agent_name` -> LoRA id) |
| `engine/openai_engine.py` | rLLM `rllm.engine.rollout.openai_engine.OpenAIEngine` | `LoRAOpenAIEngine`: per-call model routing for evaluation |
| `trainer/trainer.py` | rLLM `rllm.trainer.verl.agent_workflow_trainer.AgentWorkflowPPOTrainer` | `MultiAgentWorkflowTrainer`: bounded copies of `fit_agent`, `init_workers`, `_pad_dataproto_to_world_size`; per-role checkpointing, metadata, wall-clock stop |
| `trainer/role_routing.py` | verl `RayWorkerGroup`, `DataProto` | `split_batch_by_role`, `RoleRoutedWorkerGroup`: per-role log-prob and update calls under IP |
| `trainer/loss_weight.py`, `wallclock.py` | — | Per-role loss weighting (SP control), `WallclockStop` |
| `trainer/tracking.py` | rLLM `rllm.utils.tracking.Tracking` | `ResumableTracking`: resume the same wandb run |
| `trainer/task_runner.py`, `train.py`, `config_injection.py` | rLLM `rllm.trainer.verl.train_agent_ppo.TaskRunner` (bounded copy) | Hydra entry, Ray task runner, injection of agent names / `max_loras` / warm-start adapters |
| `configs/multi_agent_ppo_trainer.yaml` | rLLM `agent_ppo_trainer.yaml` (extends verl `ppo_trainer.yaml`) | Primary Hydra config |
| `workflows/base.py` and workflow modules | rLLM `rllm.workflows.workflow.Workflow` | `MultiAgentWorkflow` base and the ported workflows (`workflows/math/*`, `workflows/code/*`) |
| `rewards/math_reward.py` | rLLM `rllm.rewards.math_reward.RewardMathFn` | `LastThinkRewardMathFn`: grade the text after the last `</think>` |
| `rewards/code_reward.py` | rLLM `rllm.rewards.code_reward` (research-fork rewrite) | Sandboxed test execution (Pipe IPC, RLIMIT_AS/CPU, direct mode) |
| `eval/*` | rLLM `OpenAIEngine`, `AgentWorkflowEngine` | Checkpoint discovery and evaluation against a LoRA-enabled vLLM server |
| `launch.py` | — | Hydra override / sbatch builder, paper-run registry |

## Recipes

`marl_workflows/recipes.py` is the authority for agent names, lengths and workflow kwargs per
`(task, workflow)`; values were checked against the research fork's
`dashboard/task_configs.py`, `dashboard/launch_experiment.sh` and `examples/*/train_*.py`.
Training kwargs are read as `config.rllm.workflow.get(name, default)`, so paper command lines
such as `+rllm.workflow.n_votes=3` keep working. Two defaults differ on purpose from the fork's
entry scripts: the orchestrator's `max_subtasks` default is 3 (the value all paper runs passed),
and `initial_lora_weights` is honoured for every recipe by the trainer (orchestrator recipes
remap `generator` -> `worker`).

## Contract table

| # | Contract | Exact value / rule | Producer | Consumer | Enforced in |
|---|---|---|---|---|---|
| C1 | Trajectory name grammar | `^[A-Za-z]+[0-9]*$`; no `_` or `:` | workflows | rLLM engine `trajectory_id=f"{task_uuid}_{name}"` (rllm@186d0bbb engine/agent_workflow_engine.py:268) | `contracts.validate_trajectory_name`, workflow tests |
| C2 | Role of a row | `re.sub(r"\d+$","", trajectory_id.rsplit("_",1)[1])` | — | role routing, T1 | `contracts.role_from_trajectory_id` |
| C3 | GRPO group | `uid = step_ids = f"{trajectory_id}_step{k}"` (per_step; rllm engine :346, trainer :335); group = (task, Trajectory.name) | upstream | verl `compute_advantage` | unchanged upstream; workflow name tests |
| C4 | Names per workflow | Math Voting: `generator0/1/2`, `aggregator`. Code Voting: `generator0/1/2`, `aggregator` (the paper's Code runs used `generator`×3, one pooled group). Eval-Opt: `generator` (every turn), `evaluator`. Orch: `orchestrator`, `worker` (all), `synthesizer`. voting_v2: `voterA/B/C`, `aggregator`. SA: `generator`. Controlled: `generator0..N-1` | workflows | C2/C3 | `tests/test_workflows_*` |
| C5 | `agent_name` kwarg | = role (no digits) ∈ `trainer.agent_names`; required under IP | workflows | `MultiLoRAVerlEngine` / `LoRAOpenAIEngine` | `contracts.lora_int_id_for` (ValueError) |
| C6 | LoRA id | IP: `agent_names.index(role) + 123`; SP: `None` → server uses 123; vLLM `LoRARequest(lora_name=str(id), lora_int_id=id, lora_path="simon_lora_path")` | VerlEngine / worker sync | vLLM server / vLLM worker | `contracts`, test asserts 123 == verl `VLLM_LORA_INT_ID` |
| C7 | Registered ids after each sync | IP: `[123..123+N-1]` (all adapters, every sync incl. first); SP: `[123]` every sync incl. first | worker `rollout_mode` | `probe_rollout_lora_state`, trainer first-step assertion | `verl_ext/workers.py` (rollout_mode), `trainer/trainer.py` |
| C8 | Worker RPCs | `set_active_lora(agent_role: str, lora_config: dict \| None=None) -> None`; `save_single_lora_adapter(agent_name: str, save_path: str, global_step: int=0) -> None`; `probe_rollout_lora_state() -> dict` (list over ranks); all `Dispatch.ONE_TO_ALL` | WP2 | WP3 proxy / trainer | `verl_ext/workers.py` |
| C9 | Guards | (1) init: `len(agent_names) >= 1 and not share_policy`; (2) sync: `hasattr(peft,"peft_config") and len(agent_names) >= 1 and not share_policy`; (3) switch: `not _is_lora or not names or len(names) < 1` → return; non-actor → return | — | — | `verl_ext/workers.py` |
| C10 | Injected config | `actor_rollout_ref.agent_names`, `actor_rollout_ref.share_policy`, `actor_rollout_ref.initial_lora_adapters` (IP dict), `actor_rollout_ref.model.lora_adapter_path` (SP warm start), `actor_rollout_ref.rollout.engine_kwargs.vllm.max_loras=len(agent_names)`, `data.dataset_name` (code) | `config_injection` | workers / server / rollout | `trainer/config_injection.py` |
| C11 | meta_info | `temperature` (always set on update batch); `kl_loss_coef_override` (IP per-role, from `trainer.kl_loss_coef_per_role.<role>`) | proxy / trainer | `MultiAgentDPActor` | `verl_ext/actor.py`, `trainer/role_routing.py` |
| C12 | Per-role update order | roles in order of first appearance in the (balanced) batch; one `update_actor` per role (one LR-scheduler step per role per training step) | proxy | verl | `trainer/role_routing.py` |
| C13 | Metric names | IP: `actor/{role}/{leaf}` (leaf = last `/` component of each verl key, includes `pg_loss`, `grad_norm`, `lr`, `kl_loss`, `kl_coef`, `training_ppl`, `chi2_token`, `num_mini_batches`, `optim_steps_this_call`, `optim_step_count`, …); SP: verl names `actor/*` incl. the 3 counters; T1: `actor/loss_weight/{role}`, `…/{role}_mean_resp_len`, `…/{role}_rows`, `actor/loss_weight/row_mean`; `actor/entropy` global; validation `val/{ds}/pass@1`, `val/{ds}/pass@{n}`, NEW `val/{ds}/dropped_episodes` | proxy / actor / trainer | wandb, analysis | `verl_ext/actor.py`, `trainer/trainer.py` |
| C14 | Checkpoint layout | `{default_local_dir}/global_step_{N}/actor/` + `lora_adapter_{role}/{adapter_model.safetensors,adapter_config.json}` (IP; keys without adapter name) or `lora_adapter/` (SP); FSDP shards incl. dead `default` adapter; `data.pt`; `{default_local_dir}/latest_checkpointed_iteration.txt`; `{default_local_dir}/training_metadata.json` {global_steps, total_training_steps, experiment_name, project_name, slurm_job_id, wandb_run_id, base_model?, actor_model_path} | trainer / worker | eval, resume, merge script | `verl_ext/workers.py`, `trainer/trainer.py` |
| C15 | Stdout markers | `RUN_COMPLETE global_step=N`; `WALLCLOCK_STOP completed_step=N saved=<path\|none (no step in this segment)>` | trainer | SLURM chain scripts | `trainer/trainer.py` |
| C16 | Experiment name | `{prefix}-{model_tag}-{multi_lora\|share_policy}-{math\|deepcoder_primeintellect}[-{suffix}]`, `model_tag = qwen3_{size}[ _sNNN]` (exceptions recorded in paper_runs.yaml, e.g. `…-multi_lora-l8192-math`) | launcher | eval discovery | `contracts.experiment_name`, `eval/checkpoints.parse_experiment_name` |
| C17 | Reward semantics | Math: grade text after the last `</think>` (mathd OR sympy). Code: 3 s/test, global `min((3+1)·n+5, 120)` s; each sample graded in a fresh `python -I` interpreter that never imports rllm or torch (own session, killed as a group on timeout) with RLIMIT_AS and RLIMIT_DATA 4 GiB (`RLLM_CODE_REWARD_MAX_MEMORY_GB/BYTES`), RLIMIT_CPU `min((t+1)n+5,120)`, RLIMIT_FSIZE, RLIMIT_CORE 0; submitted through `CodeRewardExecutor`, a spawn pool whose workers import only `code_sandbox` | rewards | workflows, eval | `rewards/`, reward tests |
| C18 | Datasets | `dapo_math` train/test(1412), `aime2025` test, `aime2024` test, `math500` test, `deepcoder_primeintellect` train/test(1000), `smoke_math`; stored under `$MARL_DATA_DIR` | data prep | train, eval | `data/`, checksums |
| C19 | Eval output | `eval_results.jsonl` keys = fork `EvalResult.to_dict` (timestamp, hostname, dataset, experiment_name, checkpoint_step, workflow_type, model_size, model_name, share_policy, accuracy, num_correct, num_total, eval_duration_seconds, eval_mode, length_mode, max_prompt_length, max_response_length, n_rollouts, mean_accuracy, std_accuracy, pass_at_n, [per_problem_n_correct], [pass_at_k], [eval_variant]); trajectories `{dir}/{experiment}/{dataset}/step_{N}/{episode_id}.json` | eval | analysis | `eval/` |
| C20 | Internal carriers | `sampling_params["__marl_lora_int_id__"]` (VerlEngine → server manager); `sampling_params["__marl_model__"]` (LoRAOpenAIEngine → client proxy); never reach vLLM/OpenAI | engines | engines | `engine/verl_engine.py`, `engine/openai_engine.py` |
| C21 | Env vars | `MARL_DATA_DIR`, `MARL_INIT_WEIGHT_DIR`, `MARL_LORA_STRICT` (default strict), `RLLM_CODE_REWARD_MAX_MEMORY_GB/BYTES`, `SLURM_JOB_END_TIME` | — | — | — |
