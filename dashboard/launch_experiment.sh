#!/bin/bash
# Unified experiment launcher for math and deepcoder workflows.
# Generates and submits a single sbatch job from CLI arguments.
#
# A SLURM config file describes the cluster-specific #SBATCH directives
# (partition, account, time limit, GPU constraint, etc.) plus a
# `# META: GPU_TYPE=<H100|L40s|A40|RTX8000>` line that selects the per-GPU
# memory budget. See dashboard/slurm_config_template.conf for the format.
#
# Usage:
#   bash dashboard/launch_experiment.sh \
#       --workflow voting --model 1.7B --share-policy true \
#       --slurm-config dashboard/slurm_config_template.conf \
#       --n-gpus 2 --cpus-per-gpu 8 --mem-per-gpu 128G
#
#   # Deepcoder
#   bash dashboard/launch_experiment.sh \
#       --workflow voting --model 1.7B --share-policy true \
#       --slurm-config dashboard/slurm_config_template.conf --task-type deepcoder \
#       --n-gpus 2 --cpus-per-gpu 8 --mem-per-gpu 128G
#
#   # Preview without submitting
#   bash dashboard/launch_experiment.sh \
#       --workflow voting --model 1.7B --share-policy true \
#       --slurm-config dashboard/slurm_config_template.conf --dry-run \
#       --n-gpus 2 --cpus-per-gpu 8 --mem-per-gpu 128G

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
WORKFLOW=""
MODEL=""
SHARE_POLICY=""
SLURM_CONFIG=""
SBATCH_DIRECTIVES=""
DRY_RUN=false
PROJECT_NAME="rllm-workflow-MARL-v2"
EXTRA_ARGS=""
TASK_TYPE="math"
N_GPUS=""
CPUS_PER_GPU=""
MEM_PER_GPU=""
ENTRY_POINT=""
AGENT_NAMES_OVERRIDE=""
MODEL_PATH_OVERRIDE=""
MAX_PROMPT=""
MAX_RESPONSE=""
WORKFLOW_PARAMS_OVERRIDE=""
DATASET_NAME=""
TIME_LIMIT=""
NAME_SUFFIX=""
BASE_MODEL=""

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workflow)      WORKFLOW="$2";       shift 2 ;;
        --model)         MODEL="$2";          shift 2 ;;
        --share-policy)  SHARE_POLICY="$2";   shift 2 ;;
        --slurm-config)  SLURM_CONFIG="$2";   shift 2 ;;
        --dry-run)       DRY_RUN=true;        shift   ;;
        --project-name)  PROJECT_NAME="$2";   shift 2 ;;
        --extra-args)    EXTRA_ARGS="$2";     shift 2 ;;
        --task-type)     TASK_TYPE="$2";      shift 2 ;;
        --n-gpus)        N_GPUS="$2";                shift 2 ;;
        --cpus-per-gpu)  CPUS_PER_GPU="$2";          shift 2 ;;
        --mem-per-gpu)   MEM_PER_GPU="$2";           shift 2 ;;
        --entry-point)       ENTRY_POINT="$2";              shift 2 ;;
        --agent-names)       AGENT_NAMES_OVERRIDE="$2";     shift 2 ;;
        --model-path)        MODEL_PATH_OVERRIDE="$2";      shift 2 ;;
        --max-prompt)        MAX_PROMPT="$2";               shift 2 ;;
        --max-response)      MAX_RESPONSE="$2";             shift 2 ;;
        --workflow-params)   WORKFLOW_PARAMS_OVERRIDE="$2";  shift 2 ;;
        --dataset)           DATASET_NAME="$2";             shift 2 ;;
        --time-limit)        TIME_LIMIT="$2";              shift 2 ;;
        --name-suffix)       NAME_SUFFIX="$2";             shift 2 ;;
        --base-model)        BASE_MODEL="$2";              shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Validate ─────────────────────────────────────────────────────────────────
if [[ -z "$WORKFLOW" || -z "$MODEL" || -z "$SHARE_POLICY" ]]; then
    echo "ERROR: --workflow, --model, and --share-policy are required."
    echo "Usage: bash $0 --workflow <workflow> --model <model> --share-policy <true|false> --slurm-config <path> [--task-type <math|deepcoder>]"
    echo ""
    echo "  --workflow      single_agent | evaluator_optimizer_v2 | voting_v2 | orchestrator_workers_propose | evaluator_optimizer | voting"
    echo "  --model         0.6B | 1.7B | 4B"
    echo "  --share-policy  true | false"
    echo "  --slurm-config  Path to a .conf file with #SBATCH directives (see dashboard/slurm_config_template.conf)"
    echo "  --n-gpus            Number of GPUs per node"
    echo "  --cpus-per-gpu      CPUs per GPU"
    echo "  --mem-per-gpu       Memory per GPU (e.g. 80G)"
    echo "  --task-type     math (default) | deepcoder"
    echo "  --dry-run       Print sbatch script without submitting"
    echo "  --project-name  Project name (default: rllm-workflow-MARL-v2)"
    echo "  --extra-args    Extra hydra overrides passed through verbatim"
    exit 1
fi
if [[ -z "$SLURM_CONFIG" ]]; then
    echo "ERROR: --slurm-config is required. See dashboard/slurm_config_template.conf."
    exit 1
fi

# ── Fallback lookup functions (used for direct CLI invocation; dashboard path passes values via CLI args) ──

get_entry_point() {
    local task="$1" wf="$2"
    case "${task}_${wf}" in
        math_evaluator_optimizer)      echo "examples.math_reasoning.train_evaluator_optimizer_math" ;;
        math_voting)                   echo "examples.math_reasoning.train_voting_math" ;;
        math_orchestrator_workers_propose) echo "examples.math_reasoning.train_orchestrator_workers_math" ;;
        math_single_agent)             echo "examples.math_reasoning.train_single_agent_math" ;;
        math_voting_v2)                echo "examples.math_reasoning.train_voting_v2_math" ;;
        math_evaluator_optimizer_v2)   echo "examples.math_reasoning.train_evaluator_optimizer_v2_math" ;;
        deepcoder_evaluator_optimizer) echo "examples.deepcoder.train_deepcoder_evaluator_optimizer" ;;
        deepcoder_voting)              echo "examples.deepcoder.train_deepcoder_voting" ;;
        deepcoder_orchestrator_workers_propose) echo "examples.deepcoder.train_deepcoder_orchestrator_workers" ;;
        deepcoder_single_agent)        echo "examples.deepcoder.train_single_agent_deepcoder" ;;
        deepcoder_voting_v2)           echo "examples.deepcoder.train_deepcoder_voting_v2" ;;
        deepcoder_evaluator_optimizer_v2) echo "examples.deepcoder.train_deepcoder_evaluator_optimizer_v2" ;;
        *) echo "UNKNOWN"; return 1 ;;
    esac
}

get_agent_names() {
    case "$1" in
        evaluator_optimizer|evaluator_optimizer_v2)  echo "['generator','evaluator']" ;;
        voting)               echo "['generator','aggregator']" ;;
        voting_v2)            echo "['voterA','voterB','voterC','aggregator']" ;;
        orchestrator_workers_propose) echo "['orchestrator','worker','synthesizer']" ;;
        single_agent)         echo "['generator']" ;;
    esac
}

get_workflow_params() {
    local task="$1" wf="$2"
    case "${task}_${wf}" in
        math_evaluator_optimizer)
            echo "+rllm.workflow.max_iterations=3 rllm.workflow.use_final_outcome_reward=true" ;;
        math_voting)
            echo "+rllm.workflow.n_votes=3 rllm.workflow.use_final_outcome_reward=true" ;;
        math_voting_v2)
            echo "+rllm.workflow.n_votes=3 +rllm.workflow.use_rubric_reward=true rllm.workflow.use_final_outcome_reward=false" ;;
        math_orchestrator_workers_propose)
            echo "+rllm.workflow.max_subtasks=3 rllm.workflow.use_final_outcome_reward=true" ;;
        math_single_agent)
            echo "" ;;
        math_evaluator_optimizer_v2)
            echo "+rllm.workflow.max_iterations=3 rllm.workflow.use_final_outcome_reward=true" ;;
        deepcoder_evaluator_optimizer)
            echo "+rllm.workflow.max_iterations=2 rllm.workflow.use_final_outcome_reward=true +rllm.workflow.enable_test_loop=False" ;;
        deepcoder_voting)
            echo "+rllm.workflow.n_votes=3 rllm.workflow.use_final_outcome_reward=true +rllm.workflow.enable_test_loop=False" ;;
        deepcoder_voting_v2)
            echo "+rllm.workflow.n_votes=3 +rllm.workflow.use_rubric_reward=true rllm.workflow.use_final_outcome_reward=false +rllm.workflow.enable_test_loop=False" ;;
        deepcoder_orchestrator_workers_propose)
            echo "+rllm.workflow.max_subtasks=3 rllm.workflow.use_final_outcome_reward=true +rllm.workflow.enable_test_loop=False" ;;
        deepcoder_single_agent)
            echo "+rllm.workflow.enable_test_loop=False" ;;
        deepcoder_evaluator_optimizer_v2)
            echo "+rllm.workflow.max_iterations=2 rllm.workflow.use_final_outcome_reward=true +rllm.workflow.enable_test_loop=False" ;;
        *)
            echo "" ;;
    esac
}

get_prompt_response_len() {
    local task="$1" wf="$2"
    case "${task}_${wf}" in
        math_evaluator_optimizer|math_evaluator_optimizer_v2)  echo "30720 5120" ;;
        math_voting|math_voting_v2)                           echo "20480 5120" ;;
        math_orchestrator_workers_propose)                     echo "20480 5120" ;;
        math_single_agent)                                     echo "15360 5120" ;;
        deepcoder_evaluator_optimizer|deepcoder_evaluator_optimizer_v2|deepcoder_voting|deepcoder_voting_v2|deepcoder_orchestrator_workers_propose)
            echo "10240 2048" ;;
        deepcoder_single_agent)                                echo "4096 2048" ;;
        *)                                            echo "15360 5120" ;;
    esac
}

get_model_path() {
    case "$1" in
        0.6B) echo "Qwen/Qwen3-0.6B" ;;
        1.7B) echo "Qwen/Qwen3-1.7B" ;;
        4B)   echo "Qwen/Qwen3-4B" ;;
        *) echo "UNKNOWN"; return 1 ;;
    esac
}

get_ppo_max_token_len() {
    local model="$1" gpu_type="$2"
    case "${model}_${gpu_type}" in
        0.6B_L40s|0.6B_A40|0.6B_RTX8000) echo 28672 ;;
        0.6B_H100)                       echo 51712 ;;
        1.7B_L40s|1.7B_A40|1.7B_RTX8000) echo 23554 ;;
        1.7B_H100)                       echo 40960 ;;
        4B_L40s|4B_A40|4B_RTX8000)     echo 10240 ;;
        4B_H100)                         echo 40960 ;;
        *) echo "UNKNOWN"; return 1 ;;
    esac
}

# ── Config-file parsing helpers ───────────────────────────────────────────────
parse_gpu_type() {
    grep -oP '^#\s*META:\s*GPU_TYPE=\K\S+' "$1"
}
read_sbatch_directives() {
    # Return #SBATCH lines, excluding job-name/output/error and GPU/CPU/memory (injected from args)
    grep '^#SBATCH' "$1" | grep -v -E '(--job-name|--output|--error|--gres=gpu|--cpus-per-gpu|--mem-per-gpu)'
}

# ── Read SLURM config ───────────────────────────────────────────────────────
if [[ ! -f "$SLURM_CONFIG" ]]; then
    echo "ERROR: Config not found: $SLURM_CONFIG"
    exit 1
fi
if [[ -z "$N_GPUS" || -z "$CPUS_PER_GPU" || -z "$MEM_PER_GPU" ]]; then
    echo "ERROR: --n-gpus, --cpus-per-gpu, and --mem-per-gpu are required."
    exit 1
fi
GPU_TYPE=$(parse_gpu_type "$SLURM_CONFIG")
SBATCH_DIRECTIVES=$(read_sbatch_directives "$SLURM_CONFIG")
# Override --time if user provided a custom time limit
if [[ -n "$TIME_LIMIT" ]]; then
    SBATCH_DIRECTIVES=$(echo "$SBATCH_DIRECTIVES" | grep -v -- '--time=')
    SBATCH_DIRECTIVES+=$'\n'"#SBATCH --time=${TIME_LIMIT}"
fi


# ── Build experiment name ────────────────────────────────────────────────────
if [[ "$SHARE_POLICY" == "true" ]]; then
    policy_suffix="share_policy"
else
    policy_suffix="multi_lora"
fi
model_lower=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')
if [[ -n "$BASE_MODEL" ]]; then
    model_tag="${BASE_MODEL}"
else
    model_tag="qwen3_${model_lower}"
fi
exp_name_suffix="${DATASET_NAME:-${TASK_TYPE}}"
exp_name="${WORKFLOW}-${model_tag}-${policy_suffix}-${exp_name_suffix}${NAME_SUFFIX:+-${NAME_SUFFIX}}"

# ── Resolve parameters (prefer CLI args from dashboard, fall back to shell lookups) ──
entry_point="${ENTRY_POINT:-$(get_entry_point "$TASK_TYPE" "$WORKFLOW")}"
agent_names="${AGENT_NAMES_OVERRIDE:-$(get_agent_names "$WORKFLOW")}"
model_path="${MODEL_PATH_OVERRIDE:-$(get_model_path "$MODEL")}"
if [[ -n "$MAX_PROMPT" && -n "$MAX_RESPONSE" ]]; then
    max_prompt="$MAX_PROMPT"; max_response="$MAX_RESPONSE"
else
    read -r max_prompt max_response <<< "$(get_prompt_response_len "$TASK_TYPE" "$WORKFLOW")"
fi
workflow_params="${WORKFLOW_PARAMS_OVERRIDE:-$(get_workflow_params "$TASK_TYPE" "$WORKFLOW")}"
ppo_max_token_len=$(get_ppo_max_token_len "$MODEL" "$GPU_TYPE")

# ── Build sbatch script ─────────────────────────────────────────────────────
sbatch_script="#!/bin/bash
#SBATCH --job-name=${exp_name}
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err"

# Inject GPU/CPU/memory from CLI args, plus directives from the SLURM config file
sbatch_script+="
#SBATCH --gres=gpu:${N_GPUS}
#SBATCH --cpus-per-gpu=${CPUS_PER_GPU}
#SBATCH --mem-per-gpu=${MEM_PER_GPU}
${SBATCH_DIRECTIVES}"

sbatch_script+="

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
source ~/.bashrc && conda activate rllm
set -x"

# Add task-type specific sbatch commands
if [[ "$TASK_TYPE" == "deepcoder" ]]; then
    TOTAL_CPUS=$(( CPUS_PER_GPU * N_GPUS ))
    # Mode 1 (ProcessPoolExecutor) — ProcessPool with code_executor_workers > 0 and
    # max_concurrent_code_execs = 0 — gives every problem a single subprocess with a
    # ~120 s CPU budget shared across its tests; this is the default used by the paper.
    workflow_params+=" rllm.workflow.code_executor_workers=${TOTAL_CPUS}"
    workflow_params+=" rllm.workflow.max_concurrent_code_execs=0"
    sbatch_script+="

ulimit -n 1048576"
fi

sbatch_script+="

export RAY_TMPDIR=/tmp/ray_\${USER}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF=\"expandable_segments:False\"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_LOGGING_LEVEL=INFO
export VERL_LOGGING_LEVEL=INFO

python3 -m ${entry_point} \\
    data.max_prompt_length=${max_prompt} \\
    data.max_response_length=${max_response} \\
    actor_rollout_ref.model.path=${model_path} \\
    trainer.project_name='${PROJECT_NAME}' \\
    trainer.experiment_name='${exp_name}' \\
    trainer.n_gpus_per_node=${N_GPUS} \\
    trainer.agent_names=${agent_names} \\
    trainer.share_policy=${SHARE_POLICY^} \\
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len} \\
    ${workflow_params}"

if [[ -n "$EXTRA_ARGS" ]]; then
    sbatch_script+=" \\
    ${EXTRA_ARGS}"
fi

sbatch_script+="
"

# ── Submit or print ──────────────────────────────────────────────────────────
if $DRY_RUN; then
    echo "DRY RUN: ${exp_name}"
    echo "================================================================================"
    echo "$sbatch_script"
else
    tmpfile=$(mktemp /tmp/launch_exp_XXXXXX.sh)
    echo "$sbatch_script" > "$tmpfile"
    mkdir -p logs
    if ! output=$(sbatch "$tmpfile" 2>&1); then
        rm -f "$tmpfile"
        echo "ERROR: sbatch failed: ${output}" >&2
        exit 1
    fi
    job_id=$(echo "$output" | grep -oP '\d+' | tail -1)
    cp "$tmpfile" "logs/${job_id}_${exp_name}.sbatch"
    rm -f "$tmpfile"

    echo "Submitted ${exp_name} → Job ${job_id}"
fi
