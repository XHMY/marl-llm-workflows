# Code (DeepCoder, PrimeIntellect subset)

Commands for every cell of this task, as printed by `python scripts/launch.py ... --print`
(2 x H100 defaults; on 48 GB GPUs add `--gpu-type L40S`, which lowers the per-GPU token budget).
Run them from the repository root after `source scripts/env.sh` (or the editable installs).

Prepare the data once:

```bash
export MARL_DATA_DIR=$PWD/data_cache
python -m marl_workflows.data.prepare_deepcoder --verify
```

Code runs execute unit tests in a process pool with one worker per allocated CPU; the launcher
sets `rllm.workflow.code_executor_workers` to GPUs x CPUs per GPU (16 by default) and the SLURM
script raises the open-file limit.

Outside SLURM, set the environment the job script would set:

```bash
export VLLM_USE_V1=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    VLLM_ATTENTION_BACKEND=FLASH_ATTN PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
    VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000 RAY_TMPDIR=/tmp/ray_$USER
```

To submit instead of running in the foreground, replace `--print` with
`--slurm-config configs/slurm/dgxh_preempt_h100.conf --submit` (add `--chain K` for K resumable segments).

## Voting

### IP, Qwen3-0.6B

`python scripts/launch.py --task deepcoder --workflow voting --policy ip --scale 0.6B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=deepcoder \
    marl.workflow=voting \
    data.max_prompt_length=10240 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=voting-qwen3_0.6b-multi_lora-deepcoder_primeintellect \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[generator,aggregator]' \
    trainer.share_policy=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=51712 \
    +rllm.workflow.n_votes=3 \
    rllm.workflow.use_final_outcome_reward=true \
    rllm.workflow.code_executor_workers=32 \
    +data.dataset_name=deepcoder_primeintellect
```

### IP, Qwen3-1.7B

`python scripts/launch.py --task deepcoder --workflow voting --policy ip --scale 1.7B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=deepcoder \
    marl.workflow=voting \
    data.max_prompt_length=10240 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=voting-qwen3_1.7b-multi_lora-deepcoder_primeintellect \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[generator,aggregator]' \
    trainer.share_policy=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    +rllm.workflow.n_votes=3 \
    rllm.workflow.use_final_outcome_reward=true \
    rllm.workflow.code_executor_workers=32 \
    +data.dataset_name=deepcoder_primeintellect
```

### IP, Qwen3-4B

`python scripts/launch.py --task deepcoder --workflow voting --policy ip --scale 4B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=deepcoder \
    marl.workflow=voting \
    data.max_prompt_length=10240 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=voting-qwen3_4b-multi_lora-deepcoder_primeintellect \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[generator,aggregator]' \
    trainer.share_policy=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    +rllm.workflow.n_votes=3 \
    rllm.workflow.use_final_outcome_reward=true \
    rllm.workflow.code_executor_workers=32 \
    +data.dataset_name=deepcoder_primeintellect
```

### SP, Qwen3-0.6B

`python scripts/launch.py --task deepcoder --workflow voting --policy sp --scale 0.6B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=deepcoder \
    marl.workflow=voting \
    data.max_prompt_length=10240 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=voting-qwen3_0.6b-share_policy-deepcoder_primeintellect \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[generator,aggregator]' \
    trainer.share_policy=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=51712 \
    +rllm.workflow.n_votes=3 \
    rllm.workflow.use_final_outcome_reward=true \
    rllm.workflow.code_executor_workers=32 \
    +data.dataset_name=deepcoder_primeintellect
```

### SP, Qwen3-1.7B

`python scripts/launch.py --task deepcoder --workflow voting --policy sp --scale 1.7B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=deepcoder \
    marl.workflow=voting \
    data.max_prompt_length=10240 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=voting-qwen3_1.7b-share_policy-deepcoder_primeintellect \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[generator,aggregator]' \
    trainer.share_policy=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    +rllm.workflow.n_votes=3 \
    rllm.workflow.use_final_outcome_reward=true \
    rllm.workflow.code_executor_workers=32 \
    +data.dataset_name=deepcoder_primeintellect
```

### SP, Qwen3-4B

`python scripts/launch.py --task deepcoder --workflow voting --policy sp --scale 4B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=deepcoder \
    marl.workflow=voting \
    data.max_prompt_length=10240 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=voting-qwen3_4b-share_policy-deepcoder_primeintellect \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[generator,aggregator]' \
    trainer.share_policy=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    +rllm.workflow.n_votes=3 \
    rllm.workflow.use_final_outcome_reward=true \
    rllm.workflow.code_executor_workers=32 \
    +data.dataset_name=deepcoder_primeintellect
```

## Evaluator-Optimizer

### IP, Qwen3-0.6B

`python scripts/launch.py --task deepcoder --workflow evaluator_optimizer --policy ip --scale 0.6B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=deepcoder \
    marl.workflow=evaluator_optimizer \
    data.max_prompt_length=10240 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=evaluator_optimizer-qwen3_0.6b-multi_lora-deepcoder_primeintellect \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[generator,evaluator]' \
    trainer.share_policy=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=51712 \
    +rllm.workflow.max_iterations=2 \
    rllm.workflow.use_final_outcome_reward=true \
    rllm.workflow.code_executor_workers=32 \
    +data.dataset_name=deepcoder_primeintellect
```

### IP, Qwen3-1.7B

`python scripts/launch.py --task deepcoder --workflow evaluator_optimizer --policy ip --scale 1.7B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=deepcoder \
    marl.workflow=evaluator_optimizer \
    data.max_prompt_length=10240 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=evaluator_optimizer-qwen3_1.7b-multi_lora-deepcoder_primeintellect \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[generator,evaluator]' \
    trainer.share_policy=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    +rllm.workflow.max_iterations=2 \
    rllm.workflow.use_final_outcome_reward=true \
    rllm.workflow.code_executor_workers=32 \
    +data.dataset_name=deepcoder_primeintellect
```

### IP, Qwen3-4B

`python scripts/launch.py --task deepcoder --workflow evaluator_optimizer --policy ip --scale 4B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=deepcoder \
    marl.workflow=evaluator_optimizer \
    data.max_prompt_length=10240 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=evaluator_optimizer-qwen3_4b-multi_lora-deepcoder_primeintellect \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[generator,evaluator]' \
    trainer.share_policy=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    +rllm.workflow.max_iterations=2 \
    rllm.workflow.use_final_outcome_reward=true \
    rllm.workflow.code_executor_workers=32 \
    +data.dataset_name=deepcoder_primeintellect
```

### SP, Qwen3-0.6B

`python scripts/launch.py --task deepcoder --workflow evaluator_optimizer --policy sp --scale 0.6B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=deepcoder \
    marl.workflow=evaluator_optimizer \
    data.max_prompt_length=10240 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=evaluator_optimizer-qwen3_0.6b-share_policy-deepcoder_primeintellect \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[generator,evaluator]' \
    trainer.share_policy=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=51712 \
    +rllm.workflow.max_iterations=2 \
    rllm.workflow.use_final_outcome_reward=true \
    rllm.workflow.code_executor_workers=32 \
    +data.dataset_name=deepcoder_primeintellect
```

### SP, Qwen3-1.7B

`python scripts/launch.py --task deepcoder --workflow evaluator_optimizer --policy sp --scale 1.7B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=deepcoder \
    marl.workflow=evaluator_optimizer \
    data.max_prompt_length=10240 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=evaluator_optimizer-qwen3_1.7b-share_policy-deepcoder_primeintellect \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[generator,evaluator]' \
    trainer.share_policy=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    +rllm.workflow.max_iterations=2 \
    rllm.workflow.use_final_outcome_reward=true \
    rllm.workflow.code_executor_workers=32 \
    +data.dataset_name=deepcoder_primeintellect
```

### SP, Qwen3-4B

`python scripts/launch.py --task deepcoder --workflow evaluator_optimizer --policy sp --scale 4B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=deepcoder \
    marl.workflow=evaluator_optimizer \
    data.max_prompt_length=10240 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=evaluator_optimizer-qwen3_4b-share_policy-deepcoder_primeintellect \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[generator,evaluator]' \
    trainer.share_policy=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    +rllm.workflow.max_iterations=2 \
    rllm.workflow.use_final_outcome_reward=true \
    rllm.workflow.code_executor_workers=32 \
    +data.dataset_name=deepcoder_primeintellect
```

## Orchestrator-Workers

### IP, Qwen3-0.6B

`python scripts/launch.py --task deepcoder --workflow orchestrator_workers_propose --policy ip --scale 0.6B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=deepcoder \
    marl.workflow=orchestrator_workers_propose \
    data.max_prompt_length=10240 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=orchestrator_workers_propose-qwen3_0.6b-multi_lora-deepcoder_primeintellect \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[orchestrator,worker,synthesizer]' \
    trainer.share_policy=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=51712 \
    +rllm.workflow.max_subtasks=3 \
    rllm.workflow.use_final_outcome_reward=true \
    rllm.workflow.code_executor_workers=32 \
    +data.dataset_name=deepcoder_primeintellect
```

### IP, Qwen3-1.7B

`python scripts/launch.py --task deepcoder --workflow orchestrator_workers_propose --policy ip --scale 1.7B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=deepcoder \
    marl.workflow=orchestrator_workers_propose \
    data.max_prompt_length=10240 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=orchestrator_workers_propose-qwen3_1.7b-multi_lora-deepcoder_primeintellect \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[orchestrator,worker,synthesizer]' \
    trainer.share_policy=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    +rllm.workflow.max_subtasks=3 \
    rllm.workflow.use_final_outcome_reward=true \
    rllm.workflow.code_executor_workers=32 \
    +data.dataset_name=deepcoder_primeintellect
```

### IP, Qwen3-4B

`python scripts/launch.py --task deepcoder --workflow orchestrator_workers_propose --policy ip --scale 4B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=deepcoder \
    marl.workflow=orchestrator_workers_propose \
    data.max_prompt_length=10240 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=orchestrator_workers_propose-qwen3_4b-multi_lora-deepcoder_primeintellect \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[orchestrator,worker,synthesizer]' \
    trainer.share_policy=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    +rllm.workflow.max_subtasks=3 \
    rllm.workflow.use_final_outcome_reward=true \
    rllm.workflow.code_executor_workers=32 \
    +data.dataset_name=deepcoder_primeintellect
```

### SP, Qwen3-0.6B

`python scripts/launch.py --task deepcoder --workflow orchestrator_workers_propose --policy sp --scale 0.6B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=deepcoder \
    marl.workflow=orchestrator_workers_propose \
    data.max_prompt_length=10240 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=orchestrator_workers_propose-qwen3_0.6b-share_policy-deepcoder_primeintellect \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[orchestrator,worker,synthesizer]' \
    trainer.share_policy=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=51712 \
    +rllm.workflow.max_subtasks=3 \
    rllm.workflow.use_final_outcome_reward=true \
    rllm.workflow.code_executor_workers=32 \
    +data.dataset_name=deepcoder_primeintellect
```

### SP, Qwen3-1.7B

`python scripts/launch.py --task deepcoder --workflow orchestrator_workers_propose --policy sp --scale 1.7B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=deepcoder \
    marl.workflow=orchestrator_workers_propose \
    data.max_prompt_length=10240 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=orchestrator_workers_propose-qwen3_1.7b-share_policy-deepcoder_primeintellect \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[orchestrator,worker,synthesizer]' \
    trainer.share_policy=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    +rllm.workflow.max_subtasks=3 \
    rllm.workflow.use_final_outcome_reward=true \
    rllm.workflow.code_executor_workers=32 \
    +data.dataset_name=deepcoder_primeintellect
```

### SP, Qwen3-4B

`python scripts/launch.py --task deepcoder --workflow orchestrator_workers_propose --policy sp --scale 4B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=deepcoder \
    marl.workflow=orchestrator_workers_propose \
    data.max_prompt_length=10240 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=orchestrator_workers_propose-qwen3_4b-share_policy-deepcoder_primeintellect \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[orchestrator,worker,synthesizer]' \
    trainer.share_policy=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    +rllm.workflow.max_subtasks=3 \
    rllm.workflow.use_final_outcome_reward=true \
    rllm.workflow.code_executor_workers=32 \
    +data.dataset_name=deepcoder_primeintellect
```

## Single-agent RL (SA-RL)

### IP, Qwen3-0.6B

`python scripts/launch.py --task deepcoder --workflow single_agent --policy ip --scale 0.6B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=deepcoder \
    marl.workflow=single_agent \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=single_agent-qwen3_0.6b-multi_lora-deepcoder_primeintellect \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[generator]' \
    trainer.share_policy=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=51712 \
    rllm.workflow.code_executor_workers=32 \
    +data.dataset_name=deepcoder_primeintellect
```

### IP, Qwen3-1.7B

`python scripts/launch.py --task deepcoder --workflow single_agent --policy ip --scale 1.7B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=deepcoder \
    marl.workflow=single_agent \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=single_agent-qwen3_1.7b-multi_lora-deepcoder_primeintellect \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[generator]' \
    trainer.share_policy=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    rllm.workflow.code_executor_workers=32 \
    +data.dataset_name=deepcoder_primeintellect
```

### IP, Qwen3-4B

`python scripts/launch.py --task deepcoder --workflow single_agent --policy ip --scale 4B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=deepcoder \
    marl.workflow=single_agent \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=single_agent-qwen3_4b-multi_lora-deepcoder_primeintellect \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[generator]' \
    trainer.share_policy=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    rllm.workflow.code_executor_workers=32 \
    +data.dataset_name=deepcoder_primeintellect
```

## Evaluate

```bash
python -m marl_workflows.eval.evaluate_checkpoints --task-type deepcoder \
    --checkpoints-dir checkpoints/marl-llm-workflows --dataset deepcoder_primeintellect --use-training-lengths \
    --output-json eval_results.jsonl --trajectory-output-dir evaluation_trajectories
```

See [docs/reproducing.md](../../docs/reproducing.md) for the controls and for reproducing the paper runs exactly.
