# Trajectories and Evaluation

This page covers off-policy checkpoint evaluation and the trajectory dumps
produced by `dashboard/evaluate_checkpoints.py`. The training-time wandb
metrics live separately — see [Monitoring](monitoring.md).

## Output layout

All trajectories produced by `dashboard/evaluate_checkpoints.py` land at:

```
evaluation_trajectories/<experiment_name>/<dataset>/step_<N>/<episode_id>.json
```

- `<experiment_name>` — taken from the checkpoint directory; for paper
  experiments this is the `{Workflow}-{Policy}-{Scale}-{Task}` config name.
  Base-model and single-agent-transfer runs use synthetic checkpoint names.
- `<dataset>` — `args.dataset`. Pass `--dataset` explicitly; the default
  comes from `task_configs.default_eval_dataset` and may not match the paper's
  canonical evaluation split.
- `<N>` — checkpoint step. `0` is reserved for base-model and
  single-agent-transfer runs.
- `<episode_id>` — `eval_{i}.json` for single-rollout runs, or
  `eval_{i}_run_{j}.json` when `--n-rollouts > 1`.

When submitting a job, pass `--trajectory-output-dir evaluation_trajectories`
and let `evaluate_checkpoints.py` nest the rest. Do not add an extra
campaign-name subdirectory; analysis scripts assume the layout above.

## Canonical evaluation datasets

Use the task's canonical evaluation split:

- **Math** — DAPO-Math test split: `--dataset dapo_math --max-samples 100`
- **Code / DeepCoder** — PrimeIntellect test split:
  `--dataset deepcoder_primeintellect --max-samples 100`

Always pass `--use-training-lengths` so prompt and response length caps
match the training configuration. Skipping this flag silently truncates
responses to a different cap and changes the measured behavior.

## Step selection for trajectory analysis

When collecting trajectories for analysis, three checkpoints per experiment
are sufficient for most patterns:

- **Early training** — checkpoint near the midpoint between the first
  nonzero training checkpoint and the best training step.
- **Best** — checkpoint with the highest training-validation `pass@1`
  (from the experiment log or `eval_results.jsonl`).
- **Terminal / degeneration** — checkpoint where the failure-mode signature
  is visible (drawn from the training log, not guessed).

Step 0 is not the early-training step. Collect step-0 trajectories
separately as a shared base-model baseline if you need it.

## Code execution modes

Code evaluation runs LLM-generated code against test cases using subprocess
isolation: each test runs in its own process with `RLIMIT_AS`, `RLIMIT_CPU`,
and `reliability_guard()` so untrusted code cannot corrupt the parent.

Three execution modes control how these subprocesses are scheduled across
problems. They are selected by the `rllm.workflow.code_executor_workers` and
`rllm.workflow.max_concurrent_code_execs` config values.

### Mode 1: Direct (ProcessPoolExecutor)

```yaml
rllm.workflow.code_executor_workers: 48
rllm.workflow.max_concurrent_code_execs: 0
```

Each ProcessPoolExecutor worker runs all tests for one problem sequentially
in a single subprocess. Tests run one by one with early exit on the first
failure.

- Used for training where `_USE_DIRECT_EXECUTION=True` is set in worker init.
- Fast for wrong code (~200 ms; exits on test 0 failure).
- Slow for correct code with slow tests (sequential sum of all test times).
- Global timeout: `min((timeout + 1) * num_tests + 5, 120)` seconds.

### Mode 2: Per-Problem Parallel

```yaml
rllm.workflow.code_executor_workers: 0
rllm.workflow.max_concurrent_code_execs: 0
```

Each problem spawns all N test subprocesses at once. Results are collected
via `select.poll()` with early failure detection (kill remaining on first
failure).

- Used when no concurrency controls are set (standalone evaluation).
- Fast for correct code: wall time is the max single test, not the sum.
- Risk: N problems × M tests = N×M subprocesses simultaneously.

### Mode 3: BatchTestScheduler

```yaml
rllm.workflow.code_executor_workers: 0
rllm.workflow.max_concurrent_code_execs: 48
```

A shared subprocess pool (sized to CPU count) runs tests from *different*
problems. Three behaviors define the pipeline:

1. **Cross-problem scheduling** — test #0 for all problems runs first;
   failures are eliminated before test #1.
2. **Immediate re-queue** — when a test passes, the next test for that
   problem is queued without waiting for other problems.
3. **Speculative parallel** — when the pool is underutilized, remaining
   tests for surviving problems launch in parallel; if an earlier test
   fails, the speculative tests are cancelled.

Performance comparison (32 problems × 20 tests, 64 CPUs):

| Scenario | Mode 2 | Mode 3 | Speedup |
|---|---|---|---|
| 80% incorrect | 4.8 s, 640 forks | 1.4 s, 197 forks | 3.4× |
| All incorrect | 4.9 s, 640 forks | 0.5 s, 64 forks | 9.4× |
| All correct | 4.9 s, 640 forks | 4.3 s, 640 forks | 1.2× |
| Slow-correct long-tail | 0.7 s, 40 forks | 0.7 s, 40 forks | 1.0× |

`launch_experiment.sh` selects Mode 3 automatically for DeepCoder tasks,
sizing the pool to `CPUS_PER_GPU * N_GPUS`. For DeepCoder training,
always set `code_executor_workers` equal to the CPU count of the node —
the executor pool is the bottleneck and undersizing it stalls training.

### Subprocess isolation

Every test case runs inside `_temp_run()`, which:

1. Closes inherited file descriptors (`os.closerange`) to prevent FD leaks
   from Ray / CUDA.
2. Sets `RLIMIT_AS` (4 GB) to cap memory.
3. Sets `RLIMIT_CPU` as a kernel-level timeout (catches C-level GIL
   holders like pathological regex).
4. Calls `lcb_run_test()`, which invokes `reliability_guard()`
   (disables `os.fork`, `os.getcwd`, etc.).
5. Sends the result via `multiprocessing.Pipe`.
6. Calls `os._exit(0)` to skip atexit handlers.

### Supported code-evaluation datasets

| Dataset | Test format | Execution modes |
|---|---|---|
| livecodebench, codeforces, primeintellect | LCB (stdin / functional) | All 3 modes |
| taco, apps, code_contests | TACO → converted to LCB | All 3 modes |
| leetcode | Custom checker | Fallback only |
| kodcode | pytest-style | Fallback only |
| humanevalplus | Custom checker | Fallback only |

Non-LCB datasets fall back to running `code_reward_fn` directly in a
thread executor when using the BatchTestScheduler.

### Code-execution dispatch

```
Workflow.run_in_code_executor(code_reward_fn, task, code)
  │
  ├─ BatchTestScheduler present? ──yes──→ scheduler.submit(task, code)
  │                                         │
  │                                         ├─ LCB dataset? ──yes──→ Pipeline execution
  │                                         │                         (speculative parallel)
  │                                         └─ Other dataset? ──→ Fallback thread executor
  │
  └─ No scheduler ──→ loop.run_in_executor(executor, code_reward_fn)
                        │
                        ├─ ProcessPoolExecutor (_USE_DIRECT_EXECUTION=True)
                        │   → lcb_check_correctness_direct (Mode 1)
                        │
                        └─ ThreadPoolExecutor (_USE_DIRECT_EXECUTION=False)
                            → lcb_check_correctness_v2 (Mode 2)
```

The relevant code paths are:

- `rllm/rewards/code_reward.py` — `RewardCodeFn`, `lcb_check_correctness_direct`, `lcb_check_correctness_v2`, `_temp_run`.
- `rllm/rewards/batch_code_executor.py` — `BatchTestScheduler` cross-problem pipeline scheduler.
- `rllm/workflows/workflow.py` — `run_in_code_executor()` dispatch point.
- `rllm/workflows/code_test_loop_mixin.py` — `CodeTestLoopMixin.run_tests()` caller interface.
- `rllm/engine/agent_workflow_engine.py` — creates the executor or scheduler based on config.
