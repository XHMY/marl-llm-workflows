# Math (DAPO-Math-17K)

Commands for every cell of this task, as printed by `python scripts/launch.py ... --print`
(2 x H100 defaults; on 48 GB GPUs add `--gpu-type L40S`, which lowers the per-GPU token budget).
Run them from the repository root after `source scripts/env.sh` (or the editable installs).

Prepare the data once:

```bash
export MARL_DATA_DIR=$PWD/data_cache
python -m marl_workflows.data.prepare_dapo_math --verify
python -m marl_workflows.data.prepare_aime --verify
```

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

`python scripts/launch.py --task math --workflow voting --policy ip --scale 0.6B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=math \
    marl.workflow=voting \
    data.max_prompt_length=20480 \
    data.max_response_length=5120 \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=voting-qwen3_0.6b-multi_lora-math \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[generator,aggregator]' \
    trainer.share_policy=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=51712 \
    +rllm.workflow.n_votes=3 \
    rllm.workflow.use_final_outcome_reward=true
```

### IP, Qwen3-1.7B

`python scripts/launch.py --task math --workflow voting --policy ip --scale 1.7B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=math \
    marl.workflow=voting \
    data.max_prompt_length=20480 \
    data.max_response_length=5120 \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=voting-qwen3_1.7b-multi_lora-math \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[generator,aggregator]' \
    trainer.share_policy=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    +rllm.workflow.n_votes=3 \
    rllm.workflow.use_final_outcome_reward=true
```

### IP, Qwen3-4B

`python scripts/launch.py --task math --workflow voting --policy ip --scale 4B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=math \
    marl.workflow=voting \
    data.max_prompt_length=20480 \
    data.max_response_length=5120 \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=voting-qwen3_4b-multi_lora-math \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[generator,aggregator]' \
    trainer.share_policy=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    +rllm.workflow.n_votes=3 \
    rllm.workflow.use_final_outcome_reward=true
```

### SP, Qwen3-0.6B

`python scripts/launch.py --task math --workflow voting --policy sp --scale 0.6B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=math \
    marl.workflow=voting \
    data.max_prompt_length=20480 \
    data.max_response_length=5120 \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=voting-qwen3_0.6b-share_policy-math \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[generator,aggregator]' \
    trainer.share_policy=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=51712 \
    +rllm.workflow.n_votes=3 \
    rllm.workflow.use_final_outcome_reward=true
```

### SP, Qwen3-1.7B

`python scripts/launch.py --task math --workflow voting --policy sp --scale 1.7B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=math \
    marl.workflow=voting \
    data.max_prompt_length=20480 \
    data.max_response_length=5120 \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=voting-qwen3_1.7b-share_policy-math \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[generator,aggregator]' \
    trainer.share_policy=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    +rllm.workflow.n_votes=3 \
    rllm.workflow.use_final_outcome_reward=true
```

### SP, Qwen3-4B

`python scripts/launch.py --task math --workflow voting --policy sp --scale 4B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=math \
    marl.workflow=voting \
    data.max_prompt_length=20480 \
    data.max_response_length=5120 \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=voting-qwen3_4b-share_policy-math \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[generator,aggregator]' \
    trainer.share_policy=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    +rllm.workflow.n_votes=3 \
    rllm.workflow.use_final_outcome_reward=true
```

## Evaluator-Optimizer

### IP, Qwen3-0.6B

`python scripts/launch.py --task math --workflow evaluator_optimizer --policy ip --scale 0.6B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=math \
    marl.workflow=evaluator_optimizer \
    data.max_prompt_length=30720 \
    data.max_response_length=5120 \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=evaluator_optimizer-qwen3_0.6b-multi_lora-math \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[generator,evaluator]' \
    trainer.share_policy=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=51712 \
    +rllm.workflow.max_iterations=3 \
    rllm.workflow.use_final_outcome_reward=true
```

### IP, Qwen3-1.7B

`python scripts/launch.py --task math --workflow evaluator_optimizer --policy ip --scale 1.7B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=math \
    marl.workflow=evaluator_optimizer \
    data.max_prompt_length=30720 \
    data.max_response_length=5120 \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=evaluator_optimizer-qwen3_1.7b-multi_lora-math \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[generator,evaluator]' \
    trainer.share_policy=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    +rllm.workflow.max_iterations=3 \
    rllm.workflow.use_final_outcome_reward=true
```

### IP, Qwen3-4B

`python scripts/launch.py --task math --workflow evaluator_optimizer --policy ip --scale 4B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=math \
    marl.workflow=evaluator_optimizer \
    data.max_prompt_length=30720 \
    data.max_response_length=5120 \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=evaluator_optimizer-qwen3_4b-multi_lora-math \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[generator,evaluator]' \
    trainer.share_policy=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    +rllm.workflow.max_iterations=3 \
    rllm.workflow.use_final_outcome_reward=true
```

### SP, Qwen3-0.6B

`python scripts/launch.py --task math --workflow evaluator_optimizer --policy sp --scale 0.6B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=math \
    marl.workflow=evaluator_optimizer \
    data.max_prompt_length=30720 \
    data.max_response_length=5120 \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=evaluator_optimizer-qwen3_0.6b-share_policy-math \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[generator,evaluator]' \
    trainer.share_policy=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=51712 \
    +rllm.workflow.max_iterations=3 \
    rllm.workflow.use_final_outcome_reward=true
```

### SP, Qwen3-1.7B

`python scripts/launch.py --task math --workflow evaluator_optimizer --policy sp --scale 1.7B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=math \
    marl.workflow=evaluator_optimizer \
    data.max_prompt_length=30720 \
    data.max_response_length=5120 \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=evaluator_optimizer-qwen3_1.7b-share_policy-math \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[generator,evaluator]' \
    trainer.share_policy=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    +rllm.workflow.max_iterations=3 \
    rllm.workflow.use_final_outcome_reward=true
```

### SP, Qwen3-4B

`python scripts/launch.py --task math --workflow evaluator_optimizer --policy sp --scale 4B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=math \
    marl.workflow=evaluator_optimizer \
    data.max_prompt_length=30720 \
    data.max_response_length=5120 \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=evaluator_optimizer-qwen3_4b-share_policy-math \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[generator,evaluator]' \
    trainer.share_policy=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    +rllm.workflow.max_iterations=3 \
    rllm.workflow.use_final_outcome_reward=true
```

## Orchestrator-Workers

### IP, Qwen3-0.6B

`python scripts/launch.py --task math --workflow orchestrator_workers_propose --policy ip --scale 0.6B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=math \
    marl.workflow=orchestrator_workers_propose \
    data.max_prompt_length=20480 \
    data.max_response_length=5120 \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=orchestrator_workers_propose-qwen3_0.6b-multi_lora-math \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[orchestrator,worker,synthesizer]' \
    trainer.share_policy=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=51712 \
    +rllm.workflow.max_subtasks=3 \
    rllm.workflow.use_final_outcome_reward=true
```

### IP, Qwen3-1.7B

`python scripts/launch.py --task math --workflow orchestrator_workers_propose --policy ip --scale 1.7B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=math \
    marl.workflow=orchestrator_workers_propose \
    data.max_prompt_length=20480 \
    data.max_response_length=5120 \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=orchestrator_workers_propose-qwen3_1.7b-multi_lora-math \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[orchestrator,worker,synthesizer]' \
    trainer.share_policy=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    +rllm.workflow.max_subtasks=3 \
    rllm.workflow.use_final_outcome_reward=true
```

### IP, Qwen3-4B

`python scripts/launch.py --task math --workflow orchestrator_workers_propose --policy ip --scale 4B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=math \
    marl.workflow=orchestrator_workers_propose \
    data.max_prompt_length=20480 \
    data.max_response_length=5120 \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=orchestrator_workers_propose-qwen3_4b-multi_lora-math \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[orchestrator,worker,synthesizer]' \
    trainer.share_policy=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    +rllm.workflow.max_subtasks=3 \
    rllm.workflow.use_final_outcome_reward=true
```

### SP, Qwen3-0.6B

`python scripts/launch.py --task math --workflow orchestrator_workers_propose --policy sp --scale 0.6B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=math \
    marl.workflow=orchestrator_workers_propose \
    data.max_prompt_length=20480 \
    data.max_response_length=5120 \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=orchestrator_workers_propose-qwen3_0.6b-share_policy-math \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[orchestrator,worker,synthesizer]' \
    trainer.share_policy=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=51712 \
    +rllm.workflow.max_subtasks=3 \
    rllm.workflow.use_final_outcome_reward=true
```

### SP, Qwen3-1.7B

`python scripts/launch.py --task math --workflow orchestrator_workers_propose --policy sp --scale 1.7B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=math \
    marl.workflow=orchestrator_workers_propose \
    data.max_prompt_length=20480 \
    data.max_response_length=5120 \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=orchestrator_workers_propose-qwen3_1.7b-share_policy-math \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[orchestrator,worker,synthesizer]' \
    trainer.share_policy=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    +rllm.workflow.max_subtasks=3 \
    rllm.workflow.use_final_outcome_reward=true
```

### SP, Qwen3-4B

`python scripts/launch.py --task math --workflow orchestrator_workers_propose --policy sp --scale 4B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=math \
    marl.workflow=orchestrator_workers_propose \
    data.max_prompt_length=20480 \
    data.max_response_length=5120 \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=orchestrator_workers_propose-qwen3_4b-share_policy-math \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[orchestrator,worker,synthesizer]' \
    trainer.share_policy=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    +rllm.workflow.max_subtasks=3 \
    rllm.workflow.use_final_outcome_reward=true
```

## Single-agent RL (SA-RL)

### IP, Qwen3-0.6B

`python scripts/launch.py --task math --workflow single_agent --policy ip --scale 0.6B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=math \
    marl.workflow=single_agent \
    data.max_prompt_length=15360 \
    data.max_response_length=5120 \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=single_agent-qwen3_0.6b-multi_lora-math \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[generator]' \
    trainer.share_policy=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=51712
```

### IP, Qwen3-1.7B

`python scripts/launch.py --task math --workflow single_agent --policy ip --scale 1.7B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=math \
    marl.workflow=single_agent \
    data.max_prompt_length=15360 \
    data.max_response_length=5120 \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=single_agent-qwen3_1.7b-multi_lora-math \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[generator]' \
    trainer.share_policy=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960
```

### IP, Qwen3-4B

`python scripts/launch.py --task math --workflow single_agent --policy ip --scale 4B --print`

```bash
python -m marl_workflows.trainer.train \
    marl.task=math \
    marl.workflow=single_agent \
    data.max_prompt_length=15360 \
    data.max_response_length=5120 \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    trainer.project_name=marl-llm-workflows \
    trainer.experiment_name=single_agent-qwen3_4b-multi_lora-math \
    trainer.n_gpus_per_node=2 \
    'trainer.agent_names=[generator]' \
    trainer.share_policy=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960
```

## Evaluate

```bash
python -m marl_workflows.eval.evaluate_checkpoints --task-type math \
    --checkpoints-dir checkpoints/marl-llm-workflows --dataset dapo_math --use-training-lengths \
    --output-json eval_results.jsonl --trajectory-output-dir evaluation_trajectories
```

See [docs/reproducing.md](../../docs/reproducing.md) for the controls and for reproducing the paper runs exactly.
