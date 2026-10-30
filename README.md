# marl-llm-workflows

**Project page:** https://xhmy.github.io/marl-llm-workflows/

Multi-agent reinforcement learning for LLM workflows. The package trains the roles of a workflow
(Voting, Evaluator-Optimizer, Orchestrator-Workers) with GRPO from the episode outcome, either with
one LoRA adapter per role (**IP**, isolated policies) or with one adapter shared by all roles
(**SP**, shared policy), and compares them with single-agent RL of the same model (**SA-RL**).

It is the code of the paper *When Does Multi-Agent RL Improve LLM Workflows? Workflow, Scale, and
Policy-Sharing Tradeoffs*. The experiments train Qwen3-0.6B, 1.7B and 4B on Math (DAPO-Math-17K)
and Code (DeepCoder) problems.

The package is a thin layer over two unmodified upstream projects, pinned as git submodules:
[rLLM](https://github.com/rllm-org/rllm) (workflows and the workflow trainer) and
[verl](https://github.com/volcengine/verl) (FSDP training and vLLM rollouts). All extensions are
subclasses, registry hooks, or bounded and marked copies of upstream functions; see
[docs/architecture.md](docs/architecture.md) and [docs/upstream-patches.md](docs/upstream-patches.md).

## Install

```bash
git clone --recurse-submodules https://github.com/XHMY/marl-llm-workflows.git && cd marl-llm-workflows
conda create -n marl python=3.11 -y && conda activate marl
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements/stack.txt
```

Then either use the checkout directly:

```bash
source scripts/env.sh        # puts the package and both submodules first on PYTHONPATH
```

or install editable packages, in this order:

```bash
pip install --no-deps -e third_party/verl
pip install --no-deps -e third_party/rllm
pip install -e ".[dev]"
```

Datasets are stored under `$MARL_DATA_DIR` (default `~/.cache/marl-llm-workflows`). Details,
including flash-attn and the GPU requirements, are in [docs/installation.md](docs/installation.md).

## Smoke test

On one GPU (about 20 minutes), train Voting with Qwen3-0.6B for two steps on a 40-problem Math
fixture, first with one adapter per role, then with a shared adapter:

```bash
export MARL_DATA_DIR=$PWD/data_cache
bash scripts/smoke_gpu.sh
```

The script checks the per-role training metrics, the saved adapters and the checkpoint metadata,
and that a second launch of a finished run exits at once. CPU tests: `pytest tests/ -q`.

## Train a cell

A cell is a task, a workflow, a policy and a scale. `scripts/launch.py` builds the training command:

```bash
python -m marl_workflows.data.prepare_dapo_math
python -m marl_workflows.data.prepare_aime
python scripts/launch.py --task math --workflow voting --policy ip --scale 1.7B --print
```

which prints

```
python -m marl_workflows.trainer.train marl.task=math marl.workflow=voting data.max_prompt_length=20480 \
  data.max_response_length=5120 actor_rollout_ref.model.path=Qwen/Qwen3-1.7B trainer.project_name=marl-llm-workflows \
  trainer.experiment_name=voting-qwen3_1.7b-multi_lora-math trainer.n_gpus_per_node=2 \
  'trainer.agent_names=[generator,aggregator]' trainer.share_policy=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 +rllm.workflow.n_votes=3 \
  rllm.workflow.use_final_outcome_reward=true
```

Run it directly (after exporting the vLLM variables listed in [examples/math](examples/math/README.md)),
or let the launcher write and submit a SLURM job, which sets them:

```bash
python scripts/launch.py --task math --workflow voting --policy ip --scale 1.7B \
    --slurm-config configs/slurm/dgxh_preempt_h100.conf --submit --chain 3
```

`--workflow` is one of `voting`, `evaluator_optimizer`, `orchestrator_workers_propose` or
`single_agent`; `--policy` is `ip` or `sp`; `--task` is `math` or `deepcoder`. Controls such as the
per-role reward or the length-weighted shared-policy loss are added with `--control`, and every run
of the paper can be reproduced with `--paper-run <id>` from `configs/paper_runs.yaml`.
[docs/reproducing.md](docs/reproducing.md) has the full table of cells and controls, and
[docs/configuration.md](docs/configuration.md) lists every configuration key the package adds.
Copy-paste commands for every cell are in [examples/math](examples/math/README.md) and
[examples/deepcoder](examples/deepcoder/README.md).

## Evaluate

Every saved checkpoint is evaluated off-policy on the validation split, with the training length
caps:

```bash
python -m marl_workflows.eval.evaluate_checkpoints --task-type math \
    --checkpoints-dir checkpoints/marl-llm-workflows --dataset dapo_math --use-training-lengths \
    --output-json eval_results.jsonl --trajectory-output-dir evaluation_trajectories
```

The evaluator starts a LoRA-enabled vLLM server, loads each role's adapter, runs the same workflow
classes, and appends one line per checkpoint to `eval_results.jsonl`. Use `--dataset
deepcoder_primeintellect --task-type deepcoder` for Code. See
[docs/reproducing.md](docs/reproducing.md#4-evaluate-checkpoints) for the other evaluation modes and
SLURM submission.

## Layout

```
marl_workflows/
  contracts.py  recipes.py  utils.py  launch.py    shared names, (task, workflow) recipes, launcher
  configs/multi_agent_ppo_trainer.yaml             Hydra config of the trainer
  trainer/       entry point, trainer, per-role routing, wall-clock stop, wandb resume
  verl_ext/      multi-adapter FSDP worker, actor, vLLM rollout and server, the patch list
  engine/        LoRA-aware rollout engine (training) and OpenAI-client engine (evaluation)
  workflows/     Voting, Evaluator-Optimizer, Orchestrator-Workers, single agent; prompts
  rewards/       Math and Code rewards
  data/          dataset registry, preparation scripts, smoke fixture
  eval/          checkpoint discovery and evaluation
configs/         paper_runs.yaml (paper-run registry), slurm/*.conf (cluster settings)
scripts/         launch.py, env.sh, smoke_gpu.sh, export_paper_runs.py, merge_lora_to_base.py, slurm/ templates
examples/        copy-paste commands per cell
docs/            installation, reproducing, configuration, architecture, design, upstream-patches;
                 index.html is the project page (GitHub Pages serves this folder)
third_party/     rllm and verl submodules (unmodified)
tests/           CPU tests; tests/gpu needs a GPU
```

## Upstream pins

| Submodule | Upstream | Pin |
|---|---|---|
| `third_party/rllm` | https://github.com/rllm-org/rllm | `186d0bbbb5bfe18f8f6b8e4eac6db9379cf27738` |
| `third_party/verl` | https://github.com/volcengine/verl | tag `v0.6.1` (`d62da4950573d7a4b7ef2362337952e7ab59e78d`) |

The training stack is pinned in `requirements/stack.txt` (torch 2.8.0, vLLM 0.11.0, transformers
4.56.1, peft 0.18.1, ray 2.54.0).

## License and attribution

Apache License 2.0 (see `LICENSE`). The package builds on rLLM and verl, both Apache-2.0; the
submodules keep their own licenses. `NOTICE` lists every file derived from upstream code; each such
file keeps the upstream copyright header and states its modifications.

## Citation

```bibtex
@misc{zeng2026doesmultiagentrlimprove,
  title         = {When Does Multi-Agent RL Improve LLM Workflows? Workflow, Scale, and Policy-Sharing Tradeoffs},
  author        = {Yifan Zeng and Yiran Wu and Yaolun Zhang and Wentian Zhao and Kun Wan and Qingyun Wu and Huazheng Wang},
  year          = {2026},
  eprint        = {2605.24202},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2605.24202},
}
```
