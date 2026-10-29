# Configuration

Training is configured with Hydra. The primary config is
`marl_workflows/configs/multi_agent_ppo_trainer.yaml`; it extends verl's `ppo_trainer` config, and
any verl key can be overridden on the command line as usual. This page lists what the package adds
on top of verl and rLLM. The values every workflow needs are set by the launcher
(`scripts/launch.py`), so most runs never set these keys by hand.

Hydra rule to remember: a key that is not declared in the yaml must be passed with a leading `+`
(`+rllm.workflow.n_votes=3`); a declared key must not. The per-role trainer options below are declared
as empty dicts so that their leaves are always passed with `+`.

## Recipe selection

| Key | Default | Meaning |
|---|---|---|
| `marl.task` | `null` (required) | `math` or `deepcoder`. |
| `marl.workflow` | `null` (required) | Recipe id: `single_agent`, `voting`, `evaluator_optimizer`, `orchestrator_workers_propose`, `voting_v2`, `voting_prompt_het` (Math), `controlled_sa_hga` (Math). The recipe (`marl_workflows/recipes.py`) fixes the workflow class, role names, reward function and default length caps. |
| `marl.train_dataset` | `null` | Registered dataset for training (`train` split); `null` uses the recipe's (`dapo_math`, `deepcoder_primeintellect`). |
| `marl.val_dataset` | `null` | Registered dataset for in-training validation (`test` split); `null` uses the recipe's (`aime2025` for Math, `deepcoder_primeintellect` for Code). |
| `data.dataset_name` | not declared | Code only: dataset registry name (`+data.dataset_name=deepcoder_primeintellect`, set by the launcher). When given, it replaces the Code train and validation dataset names. |
| `data.custom_cls` | `PolarsRLHFDataset` | Dataset class that reads the registered parquet files with polars (robust to nested `extra_info`). |

## Roles and policy sharing

| Key | Default | Meaning |
|---|---|---|
| `trainer.agent_names` | `['default']` | Role names. Left at the default, they are replaced by the recipe's roles. Each role gets its own LoRA adapter under IP. Names are letters only; under IP no name may be `default` or a substring of another. |
| `trainer.share_policy` | `false` | `false` (IP): one LoRA adapter per role, each updated only on its own role's samples. `true` (SP): one adapter for all roles. |
| `rllm.workflow.initial_lora_weights.<role>` | `{}` | Initialize a role's adapter from a saved LoRA adapter directory. Under SP the `generator` entry (or the first one) initializes the shared adapter. Orchestrator-Workers maps a `generator` entry to `worker`. |
| `trainer.base_model_tag` | `null` | Tag of merged init weights (e.g. `qwen3_1.7b_s300`), written to `training_metadata.json` so that evaluation loads the same base model. Set by the launcher's `--base-model`. |
| `trainer.check_lora_registration` | `true` | After the first training rollout, assert that vLLM holds exactly the expected LoRA ids (one per role under IP, one under SP). |

Keys written by the trainer at start-up (do not set them): `actor_rollout_ref.agent_names`,
`actor_rollout_ref.share_policy`, `actor_rollout_ref.initial_lora_adapters` (IP),
`actor_rollout_ref.model.lora_adapter_path` (SP warm start), and
`actor_rollout_ref.rollout.engine_kwargs.vllm.max_loras` (number of roles).

## Workflow parameters

Read by the recipes with a default; pass them with `+` except `use_final_outcome_reward`.

| Key | Default | Used by | Meaning |
|---|---|---|---|
| `rllm.workflow.use_final_outcome_reward` | `true` | all multi-agent workflows | Every role is rewarded with the episode outcome. `false`: each role is rewarded for its own output (per-role correctness). |
| `+rllm.workflow.n_votes` | 3 | voting, voting_v2, voting_prompt_het | Number of voters per problem. |
| `+rllm.workflow.max_iterations` | 3 (Math), 2 (Code) | evaluator_optimizer | Maximum generator attempts per problem. |
| `+rllm.workflow.max_subtasks` | 3 | orchestrator_workers_propose | Number of workers the orchestrator assigns. |
| `+rllm.workflow.use_rubric_reward` | `false` | voting_v2 | Reward each voter with a rubric score instead of the outcome. |
| `+rllm.workflow.voter_names` | `null` | voting_v2 | Override the voter role names (`voterA`, `voterB`, `voterC`). |
| `+rllm.workflow.n_virtual_generators` | 3 | controlled_sa_hga | Independent generator samples per problem, all on one adapter, without an aggregator. |
| `rllm.workflow.code_executor_workers` | 0 | Code | Size of the process pool that runs the unit tests. 0 or less: one process per allocated CPU (`SLURM_CPUS_ON_NODE`, else the CPU affinity mask). The launcher sets GPUs x CPUs per GPU. |
| `rllm.workflow.workflow_args.*` | rLLM defaults | all | Extra keyword arguments passed to the workflow class. |

## Trainer options

| Key | Default | Meaning |
|---|---|---|
| `+trainer.loss_weight_per_role.mode` | off | SP only. `length`: scale each sequence's policy-gradient loss by its role's mean response length divided by the batch mean; `static`: use `+trainer.loss_weight_per_role.weights.<role>`. Weights are normalized to mean 1 over non-empty responses. Logged as `actor/loss_weight/*`. |
| `+trainer.kl_loss_coef_per_role.<role>` | off | IP only, with `actor_rollout_ref.actor.use_kl_loss=true`. KL-to-reference coefficient for that role's adapter update; roles not listed use `actor_rollout_ref.actor.kl_loss_coef`. |
| `+trainer.wallclock_stop.enable` | off | Stop cleanly before the job's time limit: when the next step would not finish `margin_s` before the deadline, save the last completed step and exit 0 (`WALLCLOCK_STOP completed_step=N saved=<path>`). |
| `+trainer.wallclock_stop.deadline_epoch` | `$SLURM_JOB_END_TIME` | Deadline as a Unix time; outside SLURM without it the stop is disabled. |
| `+trainer.wallclock_stop.margin_s` | 900 | Safety margin in seconds. |
| `trainer.val_count_dropped_episodes` | `false` | Count validation episodes that failed before their first step as wrong in the logged pass@k (and log `val/<dataset>/dropped_episodes`). `false` reproduces the paper runs. |
| `trainer.total_training_steps` | 500 | Number of training steps. A run that already reached it prints `RUN_COMPLETE global_step=N` and exits 0. |
| `trainer.project_name` | `marl-llm-workflows` | wandb project and checkpoint directory (`checkpoints/<project>/<experiment>`). |
| `+actor_rollout_ref.rollout.engine_kwargs.vllm.logprobs_mode` | vLLM default | `processed_logprobs` makes vLLM return log-probabilities after temperature scaling, so the logged per-role log-prob statistics compare trainer and sampler at the same temperature. Off in the paper's grid runs. |

## Defaults that differ from verl or rLLM

Copied from the configuration of the paper runs: GRPO advantages; LoRA rank 64, alpha 32, all linear
layers; learning rate 2e-5 with a constant schedule after a linear warmup of
`actor_rollout_ref.actor.optim.lr_warmup_steps` (15 in the yaml; most paper runs used 0, see
`configs/paper_runs.yaml`); one PPO epoch, mini-batch 64, `clip_ratio_high` 0.28,
`seq-mean-token-mean` loss aggregation; 8 samples per problem at temperature 0.7; 64 problems per
step; validation every 10 steps and checkpoints every 5. The per-role update performs one optimizer
call per role per training step, so under IP the learning-rate scheduler advances once per role.

## Launcher settings

`scripts/launch.py` options that are not Hydra keys:

| Option | Default | Meaning |
|---|---|---|
| `--task`, `--workflow`, `--policy`, `--scale` | required | The cell. `--task code` is an alias of `deepcoder`. |
| `--control` | none | A named variation (see [reproducing.md](reproducing.md)). |
| `--paper-run` | none | Reproduce a run from `configs/paper_runs.yaml`. |
| `--gpus` | 2 (4 for `cap_8192`) | `trainer.n_gpus_per_node` and the SLURM GPU count. |
| `--gpu-type` | H100, or the conf's `GPU_TYPE` | Selects the per-GPU token budget (`ppo_max_token_len_per_gpu`). |
| `--cpus-per-gpu`, `--mem-per-gpu` | conf META, else 8 / 64G (Math), 16 / 80G (Code on H100) | SLURM resources; Code also sizes the code-execution pool. |
| `--base-model` | none | Merged init weights `$MARL_INIT_WEIGHT_DIR/<task>/<tag>`. |
| `--init-adapter role=path` | none | Sets `+rllm.workflow.initial_lora_weights.<role>`. |
| `--resume-from` | none | Resume from a `global_step_N` directory (`trainer.resume_mode=resume_path`). |
| `--project`, `--name-suffix`, `--experiment-name` | `marl-llm-workflows`, none, derived | Naming; the experiment name is `{workflow}-{model}-{multi_lora or share_policy}-{math or deepcoder_primeintellect}[-suffix]`. |
| `--processed-logprobs` | off | Adds the `logprobs_mode` override above. |
| `--extra "k=v ..."`, positional overrides | none | Extra Hydra overrides; a key given again replaces the earlier value. |
| `--slurm-config`, `--time`, `--submit`, `--chain K` | none, conf, off, 1 | Render or submit a SLURM job; `--chain` submits K segments with `afterany` dependencies. |
| `--env-activate` | `source scripts/env.sh` | Shell line run in the job before training (e.g. `conda activate marl && source scripts/env.sh`). |

SLURM confs (`configs/slurm/*.conf`) hold `#SBATCH` lines plus `# META:` lines read by the launcher:
`GPU_TYPE`, and `CPUS_PER_GPU_{MATH,CODE}` / `MEM_PER_GPU_{MATH,CODE}`. `configs/slurm/example.conf`
is a template for other clusters.

## Code reward sandbox

Each Code solution is graded in a fresh Python interpreter (`marl_workflows/rewards/code_sandbox.py`
starts `code_sandbox_runner.py` with `python -I`, in its own session and with a minimal environment).
The runner loads only the LiveCodeBench test harness from `third_party/rllm` (not the `rllm` package,
so not torch) plus numpy and pandas, which the harness imports into every solution; it starts at about
300 MiB of address space. It then runs all tests of the problem in sequence: 3 s per test, stop at the
first failure, and a global timeout of `min(4 * n_tests + 5, 120)` s, at which the whole session is
killed, including any processes the solution started. Limits of the runner: address space and data
segment 4 GiB (`RLLM_CODE_REWARD_MAX_MEMORY_GB` or `_BYTES`; `0` disables), CPU time equal to the
global timeout, files of at most 64 MiB, no core dumps. In training, the reward function runs on
threads of the trainer and each grading goes to a spawn process pool of `code_executor_workers`
workers that import only the sandbox module (tens of MB each). A fresh interpreter is used because
a memory limit set in a child forked from a process that has imported torch counts the inherited
address space of several GB. The paper's Code runs used such a forked sandbox, whose effective memory
limit was tens of MB.

## Voting advantage groups

The Voting workflows (`voting`, `voting_prompt_het`) name their voters `generator0`, `generator1`,
`generator2` on both tasks, so each voter's `rollout.n` samples of a problem form their own GRPO
group, and all voters train the one `generator` adapter (the role is the name without trailing
digits). The paper's Code Voting runs named every voter `generator`, one pooled group of
`n_votes × rollout.n` samples; under the broadcast reward this is equivalent up to the constant
advantage factor sqrt(3(n-1)/(3n-1)) = 0.9555 for n = 8 (`tests/test_vote_grouping_advantages.py`).

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `MARL_DATA_DIR` | `~/.cache/marl-llm-workflows` | Dataset registry root (prepared parquet files). |
| `MARL_INIT_WEIGHT_DIR` | `checkpoints/init_weight` | Root of merged init weights (`<task>/<tag>`). |
| `MARL_LORA_STRICT` | strict | A request for an unknown role, or a missing role under IP, raises. `0` falls back to the base model instead. |
| `RLLM_CODE_REWARD_MAX_MEMORY_GB` / `_BYTES` | 4 GiB | Address-space and data-segment limit of the code-reward sandbox process (see Code reward sandbox). |
| `SLURM_JOB_END_TIME` | set by SLURM | Deadline for the wall-clock stop. |
