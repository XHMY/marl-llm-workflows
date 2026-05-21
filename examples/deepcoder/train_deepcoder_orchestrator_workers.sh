#!/bin/bash
# Training script for Deepcoder Orchestrator-Workers Workflow (3-agent decomposition pattern)
#
# This script trains a 3-agent workflow for code generation:
# - Orchestrator: Decomposes coding problem into subtasks
# - Worker: Solves individual subtasks in parallel
# - Synthesizer: Combines worker solutions into final code
#
# Pattern: Decompose → Workers (parallel) → Synthesize
#
# Usage: bash examples/deepcoder/train_deepcoder_orchestrator_workers.sh

set -x

ulimit -n 1048576
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_LOGGING_LEVEL=INFO
export VERL_LOGGING_LEVEL=INFO

python3 -m examples.deepcoder.train_deepcoder_orchestrator_workers \
    data.max_prompt_length=10240 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=checkpoints/init_weight/qwen3_4b_s300_deepcoder \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    trainer.project_name='rllm-workflow-MARL' \
    trainer.experiment_name='orchestrator_workers-qwen3_4b_s300-deepcoder' \
    trainer.n_gpus_per_node=2 \
    trainer.share_policy=True \
    trainer.agent_names=['orchestrator','worker','synthesizer'] \
    rllm.workflow.use_final_outcome_reward=true \
    +rllm.workflow.max_subtasks=3

pkill -9 -f 'ray::WorkerDict'
