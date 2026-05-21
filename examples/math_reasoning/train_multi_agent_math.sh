set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_LOGGING_LEVEL=INFO
export VERL_LOGGING_LEVEL=INFO


python3 -m examples.math_reasoning.train_multi_agent_math \
    data.max_prompt_length=15360 \
    data.max_response_length=5120 \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    trainer.project_name='multi-agent-math-reasoning' \
    trainer.experiment_name='qwen3_1.7b-math_3agents' \
    trainer.n_gpus_per_node=2 \
    trainer.agent_names=['generator','evaluator','refiner'] \
    +rllm.workflow.max_refinement_iterations=3

pkill -9 -f 'ray::WorkerDict'
