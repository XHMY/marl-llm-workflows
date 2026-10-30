#!/usr/bin/env bash
# GPU smoke test on one GPU (about 20 minutes): two training steps of Voting with Qwen3-0.6B on the
# small Math fixture, first with one adapter per role (IP), then with one shared adapter (SP).
#
#   bash scripts/smoke_gpu.sh            # IP then SP
#   SMOKE_VARIANTS=ip bash scripts/smoke_gpu.sh
#   SMOKE_EVAL=1 bash scripts/smoke_gpu.sh   # also evaluate the IP checkpoints on 8 problems
#   SMOKE_GPUS=2 SMOKE_BATCH=8 bash scripts/smoke_gpu.sh   # two GPUs (FSDP sharding, per-rank adapter sync)
#
# Checks: exit code 0; per-role metrics in the console log under IP (the generator takes 3 optimizer
# steps per training step and the aggregator 1) and global actor metrics under SP; one adapter directory per role (IP) or one shared
# adapter (SP) at steps 1 and 2; training_metadata.json and latest_checkpointed_iteration.txt at 2;
# a second launch of the same command prints RUN_COMPLETE and exits 0.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"
# shellcheck source=/dev/null
source scripts/env.sh

export MARL_DATA_DIR="${MARL_DATA_DIR:-${REPO}/data_cache}"
export VLLM_USE_V1=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_${USER}}"
OUT="${SMOKE_OUT:-${REPO}/outputs/smoke}"
VARIANTS="${SMOKE_VARIANTS:-ip sp}"
GPUS="${SMOKE_GPUS:-1}"          # GPUs per node for the training processes
BATCH="${SMOKE_BATCH:-4}"        # problems per training step (also the PPO mini-batch)
mkdir -p "${OUT}"

python -m marl_workflows.data.prepare_smoke

SMOKE_OVERRIDES=(
    marl.train_dataset=smoke_math marl.val_dataset=smoke_math
    trainer.total_training_steps=2 trainer.save_freq=1 trainer.test_freq=-1 trainer.val_before_train=False
    "trainer.logger=[console]"
    "data.train_batch_size=${BATCH}" data.val_batch_size=8 data.max_prompt_length=2048 data.max_response_length=512
    actor_rollout_ref.rollout.n=2 "actor_rollout_ref.actor.ppo_mini_batch_size=${BATCH}"
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 actor_rollout_ref.rollout.max_num_seqs=64
    rllm.workflow.n_parallel_tasks=16
)

fail() { echo "SMOKE FAIL ($1): $2" >&2; exit 1; }

run_variant() {
    local policy="$1" tag exp ckpt log cmd
    if [[ "${policy}" == ip ]]; then tag=multi_lora; else tag=share_policy; fi
    exp="voting-qwen3_0.6b-${tag}-math-smoke"
    ckpt="${OUT}/${exp}"
    log="${OUT}/${exp}.log"
    rm -rf "${ckpt}"

    cmd="$(python scripts/launch.py --task math --workflow voting --policy "${policy}" --scale 0.6B --gpus "${GPUS}" \
        --project smoke --experiment-name "${exp}" --print \
        "${SMOKE_OVERRIDES[@]}" "trainer.default_local_dir=${ckpt}")"
    echo "+ ${cmd}"

    eval "${cmd}" 2>&1 | tee "${log}" || fail "${policy}" "training exited non-zero (log: ${log})"

    if [[ "${policy}" == ip ]]; then
        # One adapter per role: the batch is split by role, so the actor metrics carry the role name.
        grep -q "actor/generator/pg_loss" "${log}" || fail ip "no actor/generator/pg_loss in the console log"
        grep -q "actor/aggregator/pg_loss" "${log}" || fail ip "no actor/aggregator/pg_loss in the console log"
        grep -Eq "actor/generator/optim_steps_this_call:3(\.0)?( |$)" "${log}" \
            || fail ip "generator did not take 3 optimizer steps per training step"
        grep -Eq "actor/aggregator/optim_steps_this_call:1(\.0)?( |$)" "${log}" \
            || fail ip "aggregator did not take 1 optimizer step per training step"
    else
        # One shared adapter: a single update over the whole batch, so the metrics are global.
        grep -q "actor/pg_loss" "${log}" || fail sp "no actor/pg_loss in the console log"
        grep -q "actor/optim_steps_this_call" "${log}" || fail sp "no actor/optim_steps_this_call in the console log"
        ! grep -q "actor/generator/pg_loss" "${log}" || fail sp "unexpected per-role actor metrics under a shared policy"
    fi
    grep -Eq "actor/(generator/)?pg_loss:-?[0-9.e-]+" "${log}" || fail "${policy}" "pg_loss missing"
    if ! grep -Eq "batch/success:0\.[0-9]+" "${log}"; then
        echo "SMOKE WARN (${policy}): batch/success never strictly between 0 and 1; advantages were all zero" >&2
    fi

    for step in 1 2; do
        local actor="${ckpt}/global_step_${step}/actor"
        if [[ "${policy}" == ip ]]; then
            for role in generator aggregator; do
                [[ -f "${actor}/lora_adapter_${role}/adapter_model.safetensors" ]] \
                    || fail ip "missing ${actor}/lora_adapter_${role}/adapter_model.safetensors"
                [[ -f "${actor}/lora_adapter_${role}/adapter_config.json" ]] \
                    || fail ip "missing ${actor}/lora_adapter_${role}/adapter_config.json"
            done
            [[ ! -d "${actor}/lora_adapter" ]] || fail ip "unexpected shared adapter ${actor}/lora_adapter"
        else
            [[ -f "${actor}/lora_adapter/adapter_model.safetensors" ]] \
                || fail sp "missing ${actor}/lora_adapter/adapter_model.safetensors"
        fi
    done
    [[ "$(cat "${ckpt}/latest_checkpointed_iteration.txt")" == 2 ]] || fail "${policy}" "latest checkpoint is not 2"
    python - "${ckpt}/training_metadata.json" <<'PY' || fail "${policy}" "training_metadata.json global_steps != 2"
import json, sys
sys.exit(0 if json.load(open(sys.argv[1])).get("global_steps") == 2 else 1)
PY

    eval "${cmd}" 2>&1 | tee "${log}.rerun" || fail "${policy}" "re-launch exited non-zero"
    grep -q "RUN_COMPLETE global_step=2" "${log}.rerun" || fail "${policy}" "re-launch did not print RUN_COMPLETE"
    echo "SMOKE PASS (${policy})"
}

for v in ${VARIANTS}; do
    run_variant "${v}"
done

if [[ "${SMOKE_EVAL:-0}" == 1 ]]; then
    python -m marl_workflows.eval.evaluate_checkpoints --task-type math --checkpoints-dir "${OUT}" \
        --experiment-filter voting-qwen3_0.6b-multi_lora-math-smoke --dataset smoke_math --use-training-lengths \
        --eval-interval 1 --max-samples 8 --trajectory-output-dir "${OUT}/traj" --output-json "${OUT}/eval.jsonl"
    echo "SMOKE PASS (eval): results in ${OUT}/eval.jsonl"
fi
