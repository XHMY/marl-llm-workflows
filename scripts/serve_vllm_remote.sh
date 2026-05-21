#!/bin/bash
# serve_vllm_remote.sh - Start vLLM serving on the remote GPU server
# Usage: bash serve_vllm_remote.sh <model> [port] [max-model-len] [extra-args...]
#
# On remote: bash /home/yokey/hpc-share/workspace/rllm_0.2.1/scripts/serve_vllm_remote.sh Qwen/Qwen3-1.7B 8000 16384
# Via SSH:   ssh hw4090 "tmux new-session -d -s vllm 'bash /home/yokey/hpc-share/workspace/rllm_0.2.1/scripts/serve_vllm_remote.sh Qwen/Qwen3-1.7B 8000 16384'"
set -e

MODEL="${1:?Usage: serve_vllm_remote.sh <model> [port] [max-model-len] [extra-args...]}"
PORT="${2:-8000}"
MAX_MODEL_LEN="${3:-8192}"
shift 3 2>/dev/null || true
EXTRA_ARGS="$@"

# Activate conda environment
source /home/yokey/hpc-share/miniconda3/etc/profile.d/conda.sh
conda activate rllm

# Set vLLM environment variables
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1

# Check if port is already in use
if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
    echo "ERROR: Port ${PORT} is already in use"
    ss -tlnp | grep ":${PORT} "
    exit 1
fi

echo "=========================================="
echo "Starting vLLM server"
echo "  Model:         ${MODEL}"
echo "  Port:          ${PORT}"
echo "  Max model len: ${MAX_MODEL_LEN}"
echo "  Extra args:    ${EXTRA_ARGS}"
echo "=========================================="

exec vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code \
    $EXTRA_ARGS
