# `scripts/` — general utilities

Operational helpers for working with multi-agent LoRA checkpoints,
benchmarking the code-execution path, and preparing datasets for the
example training entry points.

## LoRA / checkpoint surgery

| Script | Purpose |
|---|---|
| `extract_lora_from_checkpoint.py` | Extract individual LoRA adapters from multi-agent checkpoint shards (post-hoc surgery). Useful when a checkpoint has FSDP shards but no separate adapter files. |
| `merge_lora_to_base.py` | Merge a LoRA adapter into the base model via PEFT. Produces a standalone HF-compatible model dir for inference. |
| `convert_share_policy_to_multi_lora.py` | Migrate a `share_policy=true` checkpoint to the per-role (`multi_lora`) format. |
| `verify_merge.py` | Numerical verification of LoRA merge correctness: load base + adapter + merged weights and check forward-pass parity on a held-out batch. |

## Inference helpers

| Script | Purpose |
|---|---|
| `serve_vllm_remote.sh` | Start vLLM on a remote GPU server with port / model arguments. |
| `launch_litellm.sh` | Launch the litellm proxy (OpenAI-compatible API wrapper) in front of a local vLLM. |

## Benchmarking

| Script | Purpose |
|---|---|
| `benchmark_batch_executor.py` | Microbenchmark for the BatchTestScheduler code-execution backend used by DeepCoder. |
| `benchmark_test_execution.py` | End-to-end timing of the per-problem code-execution path on a held-out batch of DeepCoder problems. |
| `benchmark/` | CodeForces ELO computation (`cf_elo_calc.py`) and inference-provider API benchmarks (`open_ai.py`, `together_ai.py`). |

## Trajectory and config utilities

| Script | Purpose |
|---|---|
| `clean_trajectory_json.py` | Strip token IDs and per-token logprobs from trajectory JSON dumps to shrink size for archival. Idempotent. |
| `dump_cfg.py` | Hydra config dumper. Resolves the layered Hydra config and emits `_generated_agent_ppo_trainer.yaml` — useful when debugging an unexpected config flag. |
| `test_code_reward.py` | Unit-test harness for `rllm/rewards/code_reward.py`. |

## Subdirectories

- `agent/math/` — math agent shell entry points used by `examples/math_reasoning/`.
- `train/` — per-model training launch shells (`simple_math.sh`, `async_math.sh`, `deepcoder/*.sh`, debug variants).
- `data/` — dataset providers and downloaders for the kept examples: `code_dataset.py` (DeepCoder), `dedupe_dataset.py`, `download_datasets.py`.
