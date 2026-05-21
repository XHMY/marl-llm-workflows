"""Evaluate checkpoints for math reasoning and deepcoder tasks.

This script evaluates checkpoints using a simplified vLLM pipeline
with native data parallelism and dynamic LoRA loading.

Supports four evaluation modes:
1. trained_checkpoint: Evaluate checkpoints with their trained LoRA adapters
2. base_model: Evaluate workflows using base model only (no LoRA)
3. single_agent_transfer: Use a single-agent checkpoint's LoRA for ALL agents
4. single_agent_generator_transfer: Use a single-agent checkpoint's LoRA only
   for the generator/worker role; other roles use the base model.

Usage Examples:

    # Case 1: Trained math checkpoints (last checkpoint only)
    python -m dashboard.evaluate_checkpoints \\
        --eval-mode trained_checkpoint \\
        --checkpoints-dir /path/to/checkpoints \\
        --last-checkpoint-only \\
        --output-json eval_results.jsonl

    # Case 2: Deepcoder checkpoints
    python -m dashboard.evaluate_checkpoints \\
        --task-type deepcoder \\
        --eval-mode trained_checkpoint \\
        --checkpoints-dir /path/to/checkpoints \\
        --output-json eval_results.jsonl

    # Case 3: Base model only
    python -m dashboard.evaluate_checkpoints \\
        --eval-mode base_model \\
        --base-model Qwen/Qwen3-0.6B \\
        --workflow-types voting evaluator_optimizer orchestrator_workers_propose \\
        --output-json eval_results.jsonl

    # Case 4: Single-agent transfer (all roles share the SA LoRA)
    python -m dashboard.evaluate_checkpoints \\
        --eval-mode single_agent_transfer \\
        --base-model Qwen/Qwen3-0.6B \\
        --single-agent-lora-path /path/to/checkpoint/lora_adapter \\
        --workflow-types voting evaluator_optimizer orchestrator_workers_propose \\
        --output-json eval_results.jsonl

    # Case 5: Single-agent generator-only transfer (generator/worker uses SA
    # LoRA; supervisor roles like evaluator/aggregator/orchestrator/synthesizer
    # use the base model)
    python -m dashboard.evaluate_checkpoints \\
        --eval-mode single_agent_generator_transfer \\
        --base-model Qwen/Qwen3-0.6B \\
        --single-agent-lora-path /path/to/checkpoint/lora_adapter \\
        --workflow-types voting evaluator_optimizer orchestrator_workers_propose \\
        --output-json eval_results.jsonl
"""

import argparse
import asyncio
import importlib
import json
import math
import multiprocessing
import os
import re
import signal
import socket
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm
from transformers import AutoTokenizer

from dashboard.task_configs import TASK_CONFIGS, AGENT_NAMES_MAP, MODEL_MAP, INIT_WEIGHT_DIR, infer_task_type
from rllm.agents.agent import Episode
from rllm.data.dataset import DatasetRegistry
from rllm.engine.rollout.openai_engine import OpenAIEngine
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.workflows.workflow import TerminationReason

os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"


def _code_executor_init():
    """Initializer for ProcessPoolExecutor workers used for code reward evaluation."""
    from rllm.rewards.code_reward import set_direct_execution

    set_direct_execution(True)


def _allocated_cpu_count() -> int:
    """Return the CPU count allocated by Slurm, falling back to system CPU count."""
    for env_var in ("SLURM_CPUS_ON_NODE", "SLURM_CPUS_PER_TASK", "SLURM_CPUS_PER_GPU"):
        value = os.environ.get(env_var)
        if value:
            try:
                return max(int(value), 1)
            except ValueError:
                pass
    return os.cpu_count() or 1


# ============================================================================
# Enums
# ============================================================================


class EvalMode(Enum):
    """Evaluation mode for checkpoint evaluation."""

    TRAINED_CHECKPOINT = "trained_checkpoint"  # Evaluate with trained LoRA adapters
    BASE_MODEL = "base_model"  # Evaluate using base model only (no LoRA)
    SINGLE_AGENT_TRANSFER = "single_agent_transfer"  # Use single-agent LoRA for all agents
    SINGLE_AGENT_GENERATOR_TRANSFER = "single_agent_generator_transfer"  # Use single-agent LoRA only for generator/worker roles


# Generator/worker role names per workflow. Roles not listed here (evaluator,
# aggregator, orchestrator, synthesizer) are treated as supervisor roles and
# fall back to the base model under SINGLE_AGENT_GENERATOR_TRANSFER.
GENERATOR_AGENT_NAMES_MAP = {
    "single_agent": ["generator"],
    "evaluator_optimizer": ["generator"],
    "evaluator_optimizer_v2": ["generator"],
    "voting": ["generator"],
    "voting_v2": ["voterA", "voterB", "voterC"],
    "orchestrator_workers_propose": ["worker"],
}


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class CheckpointInfo:
    """Information about a discovered checkpoint."""

    experiment_name: str  # e.g., "voting-qwen3_0.6b-math"
    workflow_type: str  # "single_agent", "evaluator_optimizer", "voting"
    model_size: str  # "0.6b", "1.7b"
    model_name: str  # "qwen3_0.6b", "qwen3_1.7b_s430"
    base_model: str  # "Qwen/Qwen3-0.6B"
    checkpoint_step: int  # e.g., 100
    actor_path: str  # Directory containing lora_adapter or lora_adapter_{agent} subdirs (may be actor/ or step dir)
    share_policy: bool  # Whether share_policy mode was used


@dataclass
class EvalResult:
    """Evaluation result for a single checkpoint."""

    experiment_name: str
    checkpoint_step: int
    dataset: str
    workflow_type: str
    model_size: str
    model_name: str
    share_policy: bool
    accuracy: float
    num_correct: int
    num_total: int
    eval_duration_seconds: float
    eval_mode: str = "trained_checkpoint"  # Which evaluation mode was used
    length_mode: str = "eval"  # "eval" = eval_prompt_response_lengths, "training" = prompt_response_lengths
    max_prompt_length: int = 0
    max_response_length: int = 0
    problem_results: list[dict] = field(default_factory=list)
    hostname: str = field(default_factory=lambda: socket.gethostname())
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    n_rollouts: int = 1
    mean_accuracy: float = 0.0
    std_accuracy: float = 0.0
    pass_at_n: float = 0.0
    per_problem_n_correct: list[int] = field(default_factory=list)
    pass_at_k: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        d = {
            "timestamp": self.timestamp,
            "hostname": self.hostname,
            "dataset": self.dataset,
            "experiment_name": self.experiment_name,
            "checkpoint_step": int(self.checkpoint_step),
            "workflow_type": self.workflow_type,
            "model_size": self.model_size,
            "model_name": self.model_name,
            "share_policy": bool(self.share_policy),
            "accuracy": round(float(self.accuracy), 4),
            "num_correct": int(self.num_correct),
            "num_total": int(self.num_total),
            "eval_duration_seconds": round(self.eval_duration_seconds, 1),
            "eval_mode": self.eval_mode,
            "length_mode": self.length_mode,
            "max_prompt_length": self.max_prompt_length,
            "max_response_length": self.max_response_length,
            "n_rollouts": self.n_rollouts,
            "mean_accuracy": round(self.mean_accuracy, 4),
            "std_accuracy": round(self.std_accuracy, 4),
            "pass_at_n": round(self.pass_at_n, 4),
        }
        if self.per_problem_n_correct:
            d["per_problem_n_correct"] = [int(x) for x in self.per_problem_n_correct]
        if self.pass_at_k:
            d["pass_at_k"] = {str(k): round(v, 4) for k, v in self.pass_at_k.items()}
        return d


@dataclass
class EvalConfig:
    """Configuration for evaluation mode."""

    mode: EvalMode
    base_model: str = None  # Required for base_model and single_agent_transfer modes
    workflow_types: list[str] = None  # Workflow types to evaluate
    single_agent_lora_path: str = None  # Path to single-agent LoRA for transfer mode


# ============================================================================
# Auto-scaling n_parallel
# ============================================================================

# GPU base concurrency (L40s + 1.7B = 512 baseline)
_GPU_BASE = {
    "RTX8000": 256,
    "A40": 384,
    "L40s": 512,
    "H100": 1024,
    "H200": 1280,
}

# Model size multiplier (1.7B = 1.0x baseline)
_MODEL_SCALE = {
    "0.6b": 1.5,
    "1.7b": 1.0,
    "4b": 0.5,
}

# Workflow type multiplier (voting generates N parallel candidates per instance)
_WORKFLOW_SCALE = {
    "voting": 0.5,
    "voting_v2": 0.5,
}

GPU_TYPE_CHOICES = list(_GPU_BASE.keys())


def detect_gpu_type() -> str | None:
    """Auto-detect GPU type from nvidia-smi on the current node."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        gpu_name = result.stdout.strip().split("\n")[0].upper()
        if "H200" in gpu_name:
            return "H200"
        if "H100" in gpu_name:
            return "H100"
        if "L40S" in gpu_name:
            return "L40s"
        if "A40" in gpu_name:
            return "A40"
        if "RTX 8000" in gpu_name or "RTX8000" in gpu_name:
            return "RTX8000"
        print(f"Warning: Unrecognized GPU '{gpu_name}', defaulting to L40s scaling")
        return None
    except Exception:
        return None


def compute_n_parallel(gpu_type: str | None, model_size: str, workflow_type: str) -> int:
    """Compute optimal n_parallel based on GPU type, model size, and workflow type.

    Formula: base[gpu] * scale[model] * scale[workflow]
    Baseline: L40s + 1.7B + non-voting = 512
    """
    resolved_gpu = gpu_type or "L40s"
    base = _GPU_BASE.get(resolved_gpu, 512)
    model_mult = _MODEL_SCALE.get(model_size.lower(), 1.0)
    wf_mult = _WORKFLOW_SCALE.get(workflow_type, 1.0)
    n_parallel = max(int(base * model_mult * wf_mult), 32)
    print(f"  Auto n_parallel={n_parallel} (gpu={resolved_gpu}, model={model_size}, workflow={workflow_type})")
    return n_parallel


# ============================================================================
# Constants and Mappings
# ============================================================================

def _import(dotted_path: str):
    """Import a class or function from a dotted import path."""
    mod_path, _, attr = dotted_path.rpartition(".")
    return getattr(importlib.import_module(mod_path), attr)


def get_workflow_map(task_type: str) -> dict:
    """Return a workflow_type -> class mapping for the given task type."""
    return {k: _import(v) for k, v in TASK_CONFIGS[task_type]["workflow_map"].items()}


def get_reward_fn(task_type: str):
    """Return the reward function for the given task type."""
    return _import(TASK_CONFIGS[task_type]["reward_fn"])


# Default to math for backward compatibility (overridden by --task-type)
_active_task_type = "math"


# ============================================================================
# Checkpoint Discovery
# ============================================================================


def parse_experiment_name(experiment_name: str) -> dict | None:
    """Parse experiment name to extract workflow type, model size, and share_policy.

    Handles various naming conventions:
    - Standard: "evaluator_optimizer-qwen3_0.6b-math"
    - Share policy: "evaluator_optimizer-qwen3_0.6b-share_policy-math"
    - Alternative: "qwen3_0.6b-math_single_agent"
    """
    # Check for workflow type - can be at start or elsewhere in the name
    workflow_types = ["single_agent", "evaluator_optimizer_v2", "evaluator_optimizer", "voting_v2", "voting", "orchestrator_workers_propose"]
    workflow_type = None
    for wt in workflow_types:
        if wt in experiment_name:
            workflow_type = wt
            break

    # Backward compat: legacy "orchestrator_workers" (without _propose) maps to "orchestrator_workers_propose"
    if workflow_type is None and "orchestrator_workers" in experiment_name:
        workflow_type = "orchestrator_workers_propose"

    if workflow_type is None:
        return None

    # Extract model size
    model_size_match = re.search(r"qwen3_(\d+\.?\d*b)", experiment_name, re.IGNORECASE)
    if model_size_match:
        model_size = model_size_match.group(1).lower()
    else:
        return None

    base_model = MODEL_MAP.get(model_size)
    if base_model is None:
        return None

    # Extract full model name (e.g., "qwen3_1.7b_s430" or "qwen3_1.7b")
    model_name_match = re.search(r"qwen3_\d+\.?\d*b[^-]*", experiment_name, re.IGNORECASE)
    model_name = model_name_match.group(0).lower() if model_name_match else f"qwen3_{model_size}"

    # Determine if share_policy mode from experiment name (hint only, auto-detected later)
    share_policy = "share_policy" in experiment_name

    return {
        "workflow_type": workflow_type,
        "model_size": model_size,
        "model_name": model_name,
        "base_model": base_model,
        "share_policy": share_policy,
    }


def discover_checkpoints(
    checkpoints_dir: str,
    experiment_filter: str = None,
    step_filter: list[int] = None,
    task_type: str = "math",
) -> list[CheckpointInfo]:
    """Discover checkpoints in the directory for the given task type.

    Supported LoRA directory structures (both are auto-detected):
    - With actor/ subdir:   global_step_N/actor/lora_adapter{,_{agent}}/
    - Without actor/ subdir: global_step_N/lora_adapter{,_{agent}}/
    """
    checkpoints = []
    checkpoints_path = Path(checkpoints_dir)

    if not checkpoints_path.exists():
        raise ValueError(f"Checkpoints directory not found: {checkpoints_dir}")

    task_config = TASK_CONFIGS[task_type]
    filter_include = task_config.get("experiment_filter_include")
    filter_exclude = task_config.get("experiment_filter_exclude")

    for experiment_dir in sorted(checkpoints_path.iterdir()):
        if not experiment_dir.is_dir():
            continue

        experiment_name = experiment_dir.name

        if filter_include and filter_include not in experiment_name.lower():
            continue
        if filter_exclude and filter_exclude in experiment_name.lower():
            continue

        if experiment_filter and not re.search(experiment_filter, experiment_name):
            continue

        parsed = parse_experiment_name(experiment_name)
        if parsed is None:
            print(f"Warning: Could not parse experiment name: {experiment_name}")
            continue

        # Read training_metadata.json to detect custom init weights (e.g. qwen3_1.7b_s300)
        metadata_path = experiment_dir / "training_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            meta_base_model = metadata.get("base_model", "")
            if meta_base_model:
                parsed["model_name"] = meta_base_model
                # If metadata base_model differs from default (has suffix like _s300),
                # resolve to init_weight path
                if meta_base_model != f"qwen3_{parsed['model_size']}":
                    init_weight_path = f"{INIT_WEIGHT_DIR}/{task_type}/{meta_base_model}"
                    if Path(init_weight_path).exists():
                        parsed["base_model"] = init_weight_path

        for step_dir in sorted(experiment_dir.iterdir()):
            if not step_dir.is_dir():
                continue

            step_match = re.match(r"global_step_(\d+)", step_dir.name)
            if not step_match:
                continue

            checkpoint_step = int(step_match.group(1))

            if step_filter and checkpoint_step not in step_filter:
                continue

            # Auto-detect LoRA location: try actor/ subdirectory first, then step dir directly.
            # This handles both old structure (global_step_N/actor/lora_adapter_*/...)
            # and new structure (global_step_N/lora_adapter_*/...) without actor/ subdir.
            agent_names = AGENT_NAMES_MAP.get(parsed["workflow_type"], ["generator"])
            actor_path = None
            detected_share_policy = None
            for candidate in [step_dir / "actor", step_dir]:
                if (candidate / "lora_adapter").exists():
                    actor_path = candidate
                    detected_share_policy = True
                    break
                if all((candidate / f"lora_adapter_{a}").exists() for a in agent_names):
                    actor_path = candidate
                    detected_share_policy = False
                    break

            if actor_path is None:
                print(f"Warning: No valid LoRA adapter found in: {step_dir}")
                continue

            checkpoints.append(
                CheckpointInfo(
                    experiment_name=experiment_name,
                    workflow_type=parsed["workflow_type"],
                    model_size=parsed["model_size"],
                    model_name=parsed["model_name"],
                    base_model=parsed["base_model"],
                    checkpoint_step=checkpoint_step,
                    actor_path=str(actor_path),
                    share_policy=detected_share_policy,
                )
            )

    return checkpoints


def group_checkpoints_by_model(
    checkpoints: list[CheckpointInfo],
) -> dict[str, list[CheckpointInfo]]:
    """Group checkpoints by base model for efficient evaluation."""
    grouped = defaultdict(list)
    for checkpoint in checkpoints:
        grouped[checkpoint.base_model].append(checkpoint)
    return dict(grouped)


def filter_last_checkpoints(checkpoints: list[CheckpointInfo]) -> list[CheckpointInfo]:
    """Keep only the last (highest step) checkpoint per experiment.

    Args:
        checkpoints: List of checkpoint info objects.

    Returns:
        Filtered list with only the highest step checkpoint per experiment.
    """
    last_by_experiment = {}
    for cp in checkpoints:
        if cp.experiment_name not in last_by_experiment:
            last_by_experiment[cp.experiment_name] = cp
        elif cp.checkpoint_step > last_by_experiment[cp.experiment_name].checkpoint_step:
            last_by_experiment[cp.experiment_name] = cp
    return list(last_by_experiment.values())


def extract_model_size(base_model: str) -> str:
    """Extract model size from base model path.

    Args:
        base_model: Model path like "Qwen/Qwen3-0.6B" or "Qwen/Qwen3-1.7B".

    Returns:
        Model size string like "0.6b" or "1.7b".
    """
    match = re.search(r"(\d+\.?\d*)[Bb]", base_model)
    if match:
        return match.group(1).lower() + "b"
    return "unknown"


def synthetic_task_suffix(task_type: str) -> str:
    """Return the canonical task suffix used in synthetic trajectory names."""
    if task_type == "math":
        return "math"
    if task_type == "deepcoder":
        return "deepcoder_primeintellect"
    return task_type


def create_synthetic_checkpoints(eval_config: EvalConfig) -> list[CheckpointInfo]:
    """Create synthetic CheckpointInfo objects for base_model and single_agent_transfer modes.

    Args:
        eval_config: Evaluation configuration with mode, base_model, and workflow_types.

    Returns:
        List of synthetic CheckpointInfo objects (one per workflow type).
    """
    if eval_config.base_model is None:
        raise ValueError("base_model is required for base_model and single_agent_transfer modes")

    if eval_config.workflow_types is None:
        raise ValueError("workflow_types is required for base_model and single_agent_transfer modes")

    model_size = extract_model_size(eval_config.base_model)
    model_name = f"qwen3_{model_size}"
    task_suffix = synthetic_task_suffix(_active_task_type)
    checkpoints = []

    for workflow_type in eval_config.workflow_types:
        if workflow_type not in get_workflow_map(_active_task_type):
            print(f"Warning: Unknown workflow type: {workflow_type}")
            continue

        # Create experiment name based on mode
        if eval_config.mode == EvalMode.BASE_MODEL:
            experiment_name = f"{workflow_type}-qwen3_{model_size}-base_model-{task_suffix}"
            actor_path = None  # No LoRA path for base model
            share_policy = True  # Doesn't matter for base model
        elif eval_config.mode == EvalMode.SINGLE_AGENT_GENERATOR_TRANSFER:
            experiment_name = f"{workflow_type}-qwen3_{model_size}-single_agent_generator_transfer-{task_suffix}"
            actor_path = eval_config.single_agent_lora_path
            share_policy = True  # Generator-only transfer also uses one LoRA (mapped only to generator/worker)
        else:  # SINGLE_AGENT_TRANSFER
            experiment_name = f"{workflow_type}-qwen3_{model_size}-single_agent_transfer-{task_suffix}"
            actor_path = eval_config.single_agent_lora_path
            share_policy = True  # Transfer uses single LoRA for all agents

        checkpoints.append(
            CheckpointInfo(
                experiment_name=experiment_name,
                workflow_type=workflow_type,
                model_size=model_size,
                model_name=model_name,
                base_model=eval_config.base_model,
                checkpoint_step=0,  # Synthetic checkpoints have step 0
                actor_path=actor_path,
                share_policy=share_policy,
            )
        )

    return checkpoints


# ============================================================================
# vLLM Server Management
# ============================================================================


class VLLMServerManager:
    """Manages vLLM server lifecycle with LoRA support."""

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        port: int = 8000,
        max_loras: int = 8,
        max_lora_rank: int = 64,
        max_model_len: int = None,
        enable_lora: bool = True,
    ):
        self.model = model
        self.port = port
        self.base_url = f"http://localhost:{port}"
        self.process = None
        self.loaded_loras: set[str] = set()
        self.enable_lora = enable_lora
        self.config = {
            "tensor_parallel_size": tensor_parallel_size,
            "data_parallel_size": data_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_loras": max_loras,
            "max_lora_rank": max_lora_rank,
            "max_model_len": max_model_len,
        }

    def start(self):
        """Start vLLM server with optional LoRA support."""
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model,
            "--tensor-parallel-size", str(self.config["tensor_parallel_size"]),
            "--gpu-memory-utilization", str(self.config["gpu_memory_utilization"]),
            "--port", str(self.port),
            "--trust-remote-code",
        ]

        # Add LoRA flags only if LoRA is enabled
        if self.enable_lora:
            cmd.extend([
                "--enable-lora",
                "--max-loras", str(self.config["max_loras"]),
                "--max-lora-rank", str(self.config["max_lora_rank"]),
            ])

        # Add data parallel size if > 1
        if self.config["data_parallel_size"] > 1:
            cmd.extend(["--data-parallel-size", str(self.config["data_parallel_size"])])

        # Add max model length if specified
        if self.config["max_model_len"]:
            cmd.extend(["--max-model-len", str(self.config["max_model_len"])])

        print(f"Starting vLLM server: {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd,
            stdout=None,  # Output goes directly to terminal
            stderr=None,  # Errors go directly to terminal
            text=True,
        )
        self._wait_for_ready()

    def _wait_for_ready(self, timeout: int = 600):
        """Wait for server to be ready."""
        print(f"Waiting for vLLM server to be ready (timeout: {timeout}s)...")
        start = time.time()
        while time.time() - start < timeout:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    print(f"vLLM server ready at {self.base_url}")
                    return
            except (requests.ConnectionError, requests.Timeout):
                pass

            # Check if process died
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"vLLM server process died with exit code: {self.process.returncode}. "
                    "Check the terminal output above for details."
                )

            time.sleep(5)

        raise TimeoutError(f"vLLM server failed to start within {timeout}s")

    def load_lora(self, lora_name: str, lora_path: str):
        """Load LoRA adapter dynamically."""
        if not self.enable_lora:
            print(f"Warning: LoRA is disabled, skipping load of {lora_name}")
            return

        if lora_name in self.loaded_loras:
            print(f"LoRA adapter already loaded: {lora_name}")
            return

        response = requests.post(
            f"{self.base_url}/v1/load_lora_adapter",
            json={"lora_name": lora_name, "lora_path": lora_path},
            timeout=60,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Failed to load LoRA {lora_name}: {response.text}")

        self.loaded_loras.add(lora_name)
        print(f"Loaded LoRA adapter: {lora_name} from {lora_path}")

    def unload_lora(self, lora_name: str):
        """Unload LoRA adapter."""
        if not self.enable_lora or lora_name not in self.loaded_loras:
            return

        try:
            response = requests.post(
                f"{self.base_url}/v1/unload_lora_adapter",
                json={"lora_name": lora_name},
                timeout=30,
            )
            if response.status_code == 200:
                self.loaded_loras.discard(lora_name)
                print(f"Unloaded LoRA adapter: {lora_name}")
            else:
                print(f"Warning: Failed to unload LoRA {lora_name}: {response.text}")
        except Exception as e:
            print(f"Warning: Error unloading LoRA {lora_name}: {e}")

    def unload_all_loras(self):
        """Unload all LoRA adapters."""
        for lora_name in list(self.loaded_loras):
            self.unload_lora(lora_name)

    def stop(self):
        """Stop vLLM server."""
        if self.process:
            print("Stopping vLLM server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None
            self.loaded_loras.clear()
            print("vLLM server stopped")


# ============================================================================
# LoRA-Aware OpenAI Engine
# ============================================================================


class LoRAOpenAIEngine(OpenAIEngine):
    """OpenAIEngine with LoRA adapter support via model parameter."""

    def __init__(
        self,
        base_model: str,
        tokenizer,
        base_url: str = "http://localhost:8000/v1",
        lora_names: dict[str, str] = None,
        **kwargs,
    ):
        # Remove parameters not supported by OpenAIEngine
        kwargs.pop("validate", None)

        super().__init__(
            model=base_model,
            tokenizer=tokenizer,
            base_url=base_url,
            api_key="EMPTY",  # vLLM doesn't need API key
            **kwargs,
        )
        self.base_model_name = base_model
        self.lora_names = lora_names or {}

    def set_lora_names(self, lora_names: dict[str, str]):
        """Update LoRA name mapping."""
        self.lora_names = lora_names

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        """Override to use LoRA model name based on agent_name."""
        agent_name = kwargs.pop("agent_name", None)

        # Determine which model/lora to use
        if agent_name and agent_name in self.lora_names:
            # Use the LoRA adapter for this agent
            self.model = self.lora_names[agent_name]
        elif "default" in self.lora_names:
            # Use default LoRA
            self.model = self.lora_names["default"]
        else:
            # Use base model
            self.model = self.base_model_name

        return await super().get_model_response(messages, **kwargs)


# ============================================================================
# Evaluation Functions
# ============================================================================


def setup_lora_for_eval(
    checkpoint: CheckpointInfo,
    server: "VLLMServerManager",
    engine: "LoRAOpenAIEngine",
    eval_mode: EvalMode,
    single_agent_lora_path: str = None,
) -> None:
    """Set up LoRA adapters for evaluation based on mode.

    Args:
        checkpoint: Checkpoint information.
        server: vLLM server manager for loading LoRA adapters.
        engine: LoRA-aware OpenAI engine.
        eval_mode: Evaluation mode (trained_checkpoint, base_model, single_agent_transfer).
        single_agent_lora_path: Path to single-agent LoRA for transfer mode.
    """
    lora_prefix = f"{checkpoint.experiment_name}_step{checkpoint.checkpoint_step}"
    agent_names = AGENT_NAMES_MAP.get(checkpoint.workflow_type, ["generator"])

    if eval_mode == EvalMode.BASE_MODEL:
        # Base model mode: no LoRA adapters, use base model for all agents
        engine.set_lora_names({})
        print(f"Using base model for evaluation (no LoRA)")
        return

    if eval_mode == EvalMode.SINGLE_AGENT_TRANSFER:
        # Single-agent transfer: load one LoRA and map all agents to it
        if not single_agent_lora_path:
            raise ValueError("single_agent_lora_path is required for SINGLE_AGENT_TRANSFER mode")

        lora_name = f"{lora_prefix}_transfer"
        server.load_lora(lora_name, single_agent_lora_path)

        # Map all agents to the same single-agent LoRA
        lora_mapping = {agent: lora_name for agent in agent_names}
        engine.set_lora_names(lora_mapping)
        print(f"Using single-agent transfer LoRA for all agents: {single_agent_lora_path}")
        return

    if eval_mode == EvalMode.SINGLE_AGENT_GENERATOR_TRANSFER:
        # Generator-only transfer: load one LoRA and map only generator/worker roles
        # to it. Supervisor roles (evaluator, aggregator, orchestrator, synthesizer)
        # fall through to the base model via LoRAOpenAIEngine.get_model_response.
        if not single_agent_lora_path:
            raise ValueError(
                "single_agent_lora_path is required for SINGLE_AGENT_GENERATOR_TRANSFER mode"
            )

        generator_agents = GENERATOR_AGENT_NAMES_MAP.get(checkpoint.workflow_type)
        if not generator_agents:
            raise ValueError(
                f"No generator/worker role defined for workflow '{checkpoint.workflow_type}' "
                "in GENERATOR_AGENT_NAMES_MAP"
            )

        lora_name = f"{lora_prefix}_gen_transfer"
        server.load_lora(lora_name, single_agent_lora_path)

        lora_mapping = {agent: lora_name for agent in generator_agents if agent in agent_names}
        engine.set_lora_names(lora_mapping)
        base_only_agents = [a for a in agent_names if a not in lora_mapping]
        print(
            f"Using single-agent generator transfer LoRA for {sorted(lora_mapping)} "
            f"(base model for {base_only_agents}): {single_agent_lora_path}"
        )
        return

    # TRAINED_CHECKPOINT mode: use trained LoRA adapters
    if checkpoint.share_policy:
        # Single shared adapter at lora_adapter/
        lora_name = f"{lora_prefix}_shared"
        lora_path = os.path.join(checkpoint.actor_path, "lora_adapter")
        server.load_lora(lora_name, lora_path)
        # Map all agent names to the same shared adapter
        lora_mapping = {agent: lora_name for agent in agent_names}
        engine.set_lora_names(lora_mapping)
    else:
        # Per-agent adapters at lora_adapter_{agent_name}/
        lora_mapping = {}
        for agent_name in agent_names:
            lora_name = f"{lora_prefix}_{agent_name}"
            agent_lora_path = os.path.join(checkpoint.actor_path, f"lora_adapter_{agent_name}")

            if not os.path.exists(agent_lora_path):
                raise FileNotFoundError(
                    f"LoRA adapter not found for agent '{agent_name}' at: {agent_lora_path}"
                )

            server.load_lora(lora_name, agent_lora_path)
            lora_mapping[agent_name] = lora_name

        engine.set_lora_names(lora_mapping)


async def evaluate_checkpoint(
    checkpoint: CheckpointInfo,
    server: VLLMServerManager,
    engine: LoRAOpenAIEngine,
    dataset: list[dict],
    n_parallel: int = 32,
    eval_mode: EvalMode = EvalMode.TRAINED_CHECKPOINT,
    single_agent_lora_path: str = None,
    trajectory_output_dir: str = None,
    dataset_name: str = None,
    n_rollouts: int = 1,
) -> EvalResult:
    """Evaluate a single checkpoint on the dataset.

    Args:
        checkpoint: Checkpoint information.
        server: vLLM server manager.
        engine: LoRA-aware OpenAI engine.
        dataset: List of tasks to evaluate.
        n_parallel: Number of parallel workflow instances.
        eval_mode: Evaluation mode (trained_checkpoint, base_model, single_agent_transfer).
        single_agent_lora_path: Path to single-agent LoRA for transfer mode.
        trajectory_output_dir: Directory to save detailed trajectory JSON files.
        dataset_name: Name of the dataset being evaluated.
        n_rollouts: Number of independent rollouts per sample.
    Returns:
        EvalResult with accuracy and metrics.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {checkpoint.experiment_name} step {checkpoint.checkpoint_step}")
    print(f"  Workflow: {checkpoint.workflow_type}")
    print(f"  Actor Path: {checkpoint.actor_path}")
    print(f"  Share Policy: {checkpoint.share_policy}")
    print(f"  Eval Mode: {eval_mode.value}")
    print(f"  N Rollouts: {n_rollouts}")
    print(f"{'='*60}")

    start_time = time.time()

    # Set up LoRA adapters based on evaluation mode
    setup_lora_for_eval(
        checkpoint=checkpoint,
        server=server,
        engine=engine,
        eval_mode=eval_mode,
        single_agent_lora_path=single_agent_lora_path,
    )

    # Get workflow class and kwargs from task config
    workflow_map = get_workflow_map(_active_task_type)
    workflow_cls = workflow_map.get(checkpoint.workflow_type)
    if workflow_cls is None:
        raise ValueError(f"Unknown workflow type: {checkpoint.workflow_type}")

    reward_fn = get_reward_fn(_active_task_type)
    task_config = TASK_CONFIGS[_active_task_type]
    workflow_params = task_config["workflow_params"].get(checkpoint.workflow_type, {})

    workflow_kwargs = {"reward_function": reward_fn}
    workflow_kwargs.update(workflow_params)

    # Match DeepCoder training execution mode from dashboard/launch_experiment.sh:
    # code_executor_workers=<allocated CPUs>, max_concurrent_code_execs=0.
    # That selects the ProcessPool/direct-execution path instead of the batch
    # scheduler path, preserving the same code-test timing behavior as training.
    if _active_task_type == "deepcoder":
        code_executor_workers = _allocated_cpu_count()
        print(
            "Using DeepCoder code reward ProcessPoolExecutor "
            f"with {code_executor_workers} workers (max_concurrent_code_execs=0)"
        )
        mp_ctx = multiprocessing.get_context("spawn")
        code_reward_executor = ProcessPoolExecutor(
            max_workers=code_executor_workers,
            mp_context=mp_ctx,
            initializer=_code_executor_init,
        )
        workflow_kwargs["code_reward_executor"] = code_reward_executor
    else:
        code_reward_executor = None

    # Run parallel evaluation with semaphore
    total_tasks = len(dataset) * n_rollouts
    semaphore = asyncio.Semaphore(n_parallel)
    progress_bar = tqdm(total=total_tasks, desc="Evaluating")

    # Pre-compute checkpoint name for trajectory saving
    checkpoint_name = f"{checkpoint.experiment_name}/{dataset_name}/step_{checkpoint.checkpoint_step}" if trajectory_output_dir else None

    async def evaluate_single(task: dict, uid: str) -> Episode | None:
        async with semaphore:
            try:
                workflow = workflow_cls(rollout_engine=engine, **workflow_kwargs)
                result = await workflow.run_with_termination_handling(task, uid)
                # Save trajectory immediately so partial results survive errors
                if trajectory_output_dir and isinstance(result, Episode):
                    save_trajectory_to_json(result, trajectory_output_dir, checkpoint_name)
                return result
            finally:
                progress_bar.update(1)

    # Build all tasks: N rollouts per sample, all run concurrently
    all_tasks = []
    for i, task in enumerate(dataset):
        for j in range(n_rollouts):
            if n_rollouts == 1:
                uid = f"eval_{i}"
            else:
                uid = f"eval_{i}_run_{j}"
            all_tasks.append(evaluate_single(task, uid))

    all_episodes = await asyncio.gather(*all_tasks)

    # Check for inference service errors (TerminationReason.ERROR only — does NOT
    # include length exceeded, timeout, or other normal termination reasons)
    error_episodes = [
        ep for ep in all_episodes
        if isinstance(ep, Episode) and ep.termination_reason == TerminationReason.ERROR
    ]
    if error_episodes:
        progress_bar.close()
        sample_error = error_episodes[0].info.get("error", {}).get("error_message", "unknown")
        raise RuntimeError(
            f"{len(error_episodes)}/{len(all_episodes)} rollouts failed with errors. "
            f"Inference service may be down. Not writing results to output JSONL. "
            f"Sample error: {sample_error}"
        )

    progress_bar.close()

    # Shut down the code reward executor if we created one
    if code_reward_executor is not None:
        code_reward_executor.shutdown(wait=True)

    # Reshape results: episodes_by_problem[i][j] = episode for problem i, rollout j
    episodes_by_problem: list[list[Episode | Exception]] = []
    idx = 0
    for i in range(len(dataset)):
        runs = []
        for j in range(n_rollouts):
            runs.append(all_episodes[idx])
            idx += 1
        episodes_by_problem.append(runs)

    # Compute per-problem results
    problem_results = []
    for i, runs in enumerate(episodes_by_problem):
        per_run = []
        for episode in runs:
            if isinstance(episode, Episode):
                per_run.append(episode.is_correct)
            else:
                per_run.append(False)
        n_correct = int(sum(per_run))
        problem_results.append({
            "uid": f"eval_{i}",
            "n_correct": n_correct,
            "n_rollouts": n_rollouts,
            "pass": any(per_run),
            "per_run": per_run,
        })

    # Compute per-rollout accuracies (for each rollout j, how many problems correct)
    per_rollout_accuracies = []
    for j in range(n_rollouts):
        correct_in_run = sum(
            1 for i in range(len(dataset))
            if problem_results[i]["per_run"][j]
        )
        per_rollout_accuracies.append(correct_in_run / len(dataset) if dataset else 0.0)

    mean_accuracy = sum(per_rollout_accuracies) / len(per_rollout_accuracies)
    if n_rollouts > 1:
        variance = sum((a - mean_accuracy) ** 2 for a in per_rollout_accuracies) / n_rollouts
        std_accuracy = math.sqrt(variance)
    else:
        std_accuracy = 0.0

    # Pass@N: fraction of problems where at least 1 rollout is correct
    pass_at_n = sum(1 for pr in problem_results if pr["pass"]) / len(dataset) if dataset else 0.0

    # Compute pass@k for standard k values using unbiased estimator
    per_problem_n_correct = [pr["n_correct"] for pr in problem_results]
    pass_at_k: dict[int, float] = {}
    for k in [1, 2, 4, 8, 16, 32, 64]:
        if k > n_rollouts:
            break
        vals = []
        for pr in problem_results:
            n, c = pr["n_rollouts"], pr["n_correct"]
            vals.append(1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k))
        pass_at_k[k] = sum(vals) / len(vals) if vals else 0.0

    # For backward compat, num_correct derived from mean accuracy
    num_correct = round(mean_accuracy * len(dataset))

    eval_duration = time.time() - start_time

    print(f"\nResults (N={n_rollouts}):")
    print(f"  Mean Accuracy: {mean_accuracy:.2%}")
    if n_rollouts > 1:
        print(f"  Std Accuracy:  {std_accuracy:.2%}")
        print(f"  Pass@{n_rollouts}:       {pass_at_n:.2%}")
        for k in sorted(pass_at_k.keys()):
            if k != n_rollouts:
                print(f"  Pass@{k}:       {pass_at_k[k]:.2%}")
    print(f"  Duration: {eval_duration:.1f}s")

    return EvalResult(
        experiment_name=checkpoint.experiment_name,
        checkpoint_step=checkpoint.checkpoint_step,
        dataset=dataset_name,
        workflow_type=checkpoint.workflow_type,
        model_size=checkpoint.model_size,
        model_name=checkpoint.model_name,
        share_policy=checkpoint.share_policy,
        accuracy=mean_accuracy,
        num_correct=num_correct,
        num_total=len(dataset),
        eval_duration_seconds=eval_duration,
        eval_mode=eval_mode.value,
        length_mode=getattr(engine, "_length_mode", "eval"),
        max_prompt_length=engine.max_prompt_length,
        max_response_length=engine.max_response_length,
        problem_results=problem_results,
        n_rollouts=n_rollouts,
        mean_accuracy=mean_accuracy,
        std_accuracy=std_accuracy,
        pass_at_n=pass_at_n,
        per_problem_n_correct=per_problem_n_correct,
        pass_at_k=pass_at_k,
    )


# ============================================================================
# JSON Output
# ============================================================================


def save_trajectory_to_json(episode: Episode, output_dir: str, checkpoint_name: str) -> None:
    """Save a single episode's full trajectory to a JSON file.

    Args:
        episode: The Episode containing all prompts and responses.
        output_dir: Base directory for saving trajectories.
        checkpoint_name: Name of the checkpoint being evaluated.
    """
    # Create subdirectory for this checkpoint
    checkpoint_dir = os.path.join(output_dir, checkpoint_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Use episode.id as filename (e.g., "eval_0.json")
    filename = f"{episode.id}.json"
    filepath = os.path.join(checkpoint_dir, filename)

    # Serialize the full episode
    episode_dict = episode.to_dict()

    # Remove large token ID lists from steps to reduce file size
    for traj in episode_dict.get("trajectories", []):
        for step in traj.get("steps", []):
            step.pop("prompt_ids", None)
            step.pop("response_ids", None)
            step.pop("logprobs", None)
            model_output = step.get("model_output")
            if isinstance(model_output, dict):
                model_output.pop("prompt_ids", None)
                model_output.pop("response_ids", None)
                model_output.pop("completion_ids", None)

    with open(filepath, "w") as f:
        json.dump(episode_dict, f, indent=2)


def save_results_to_json(results: list[EvalResult], output_path: str):
    """Append evaluation results to JSON Lines file.

    Each result is written as a single JSON object on its own line.
    This format is easy to append and parse.
    """
    if not results:
        return

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "a") as f:
        for result in results:
            f.write(json.dumps(result.to_dict()) + "\n")

    print(f"\nResults appended to: {output_path}")


def load_existing_results(output_path: str) -> set[tuple]:
    """Load already-evaluated checkpoint keys from a JSONL file.

    Returns a set of (experiment_name, checkpoint_step, dataset, eval_mode, n_rollouts, length_mode) tuples.
    """
    existing = set()
    if not os.path.exists(output_path):
        return existing

    with open(output_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                key = (
                    obj.get("experiment_name", ""),
                    obj.get("checkpoint_step", 0),
                    obj.get("dataset", ""),
                    obj.get("eval_mode", ""),
                    obj.get("n_rollouts", 1),
                    obj.get("length_mode", "eval"),
                )
                existing.add(key)
            except json.JSONDecodeError:
                print(f"Warning: malformed JSON on line {line_num} of {output_path}, skipping")

    return existing


def get_checkpoint_key(checkpoint: CheckpointInfo, dataset: str, eval_mode: str, n_rollouts: int, length_mode: str) -> tuple:
    """Build a dedup key for a checkpoint about to be evaluated."""
    return (
        checkpoint.experiment_name,
        checkpoint.checkpoint_step,
        dataset,
        eval_mode,
        n_rollouts,
        length_mode,
    )


# ============================================================================
# Main Evaluation Loop
# ============================================================================


def main(args):
    """Main evaluation function."""
    global _active_task_type
    _active_task_type = args.task_type

    task_config = TASK_CONFIGS[_active_task_type]

    # Resolve default dataset from task config if not specified
    if args.dataset is None:
        args.dataset = task_config["default_eval_dataset"]

    # Resolve prompt/response lengths from task_configs when not overridden by CLI.
    # --use-training-lengths forces training configs; otherwise uses eval overrides.
    args._user_specified_max_tokens = args.max_tokens is not None
    args._user_specified_max_prompt_length = args.max_prompt_length is not None

    train_lengths = task_config["prompt_response_lengths"]
    eval_overrides = task_config.get("eval_prompt_response_lengths", {})
    if args.use_training_lengths:
        eval_prompt_response_lengths = dict(train_lengths)
        args._length_mode = "training"
    else:
        eval_prompt_response_lengths = {wf: eval_overrides.get(wf, train_lengths[wf]) for wf in train_lengths}
        args._length_mode = "eval"

    if args.max_tokens is None or args.max_prompt_length is None:
        # Use the largest values across all workflows for server sizing
        max_prompt_across = max(p for p, r in eval_prompt_response_lengths.values())
        max_response_across = max(r for p, r in eval_prompt_response_lengths.values())
        if args.max_tokens is None:
            args.max_tokens = max_response_across
        if args.max_prompt_length is None:
            args.max_prompt_length = max_prompt_across
        print(f"Using {_active_task_type} {args._length_mode} config defaults: "
              f"max_prompt_length={args.max_prompt_length}, max_tokens={args.max_tokens}")

    print("=" * 60)
    print(f"Checkpoint Evaluation — {_active_task_type} (vLLM Pipeline)")
    print("=" * 60)

    # Parse evaluation mode
    eval_mode = EvalMode(args.eval_mode)
    print(f"\nEvaluation mode: {eval_mode.value}")

    # Validate --max-samples arguments
    if args.max_samples is not None:
        if not args.trajectory_output_dir:
            raise ValueError("--trajectory-output-dir is required when using --max-samples")
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be a positive integer")

    # Validate arguments based on mode
    if eval_mode == EvalMode.TRAINED_CHECKPOINT:
        if not args.checkpoints_dir:
            raise ValueError("--checkpoints-dir is required for trained_checkpoint mode")
    elif eval_mode in (
        EvalMode.BASE_MODEL,
        EvalMode.SINGLE_AGENT_TRANSFER,
        EvalMode.SINGLE_AGENT_GENERATOR_TRANSFER,
    ):
        if not args.base_model:
            raise ValueError(f"--base-model is required for {eval_mode.value} mode")
        if not args.workflow_types:
            raise ValueError(f"--workflow-types is required for {eval_mode.value} mode")
        if eval_mode in (
            EvalMode.SINGLE_AGENT_TRANSFER,
            EvalMode.SINGLE_AGENT_GENERATOR_TRANSFER,
        ) and not args.single_agent_lora_path:
            raise ValueError(f"--single-agent-lora-path is required for {eval_mode.value} mode")

    # Create or discover checkpoints based on mode
    if eval_mode == EvalMode.TRAINED_CHECKPOINT:
        # Discover checkpoints from directory
        print(f"\nDiscovering checkpoints in: {args.checkpoints_dir}")
        step_filter = args.step_filter if args.step_filter else None
        checkpoints = discover_checkpoints(
            args.checkpoints_dir,
            experiment_filter=args.experiment_filter,
            step_filter=step_filter,
            task_type=_active_task_type,
        )

        # Apply eval-interval filter
        if args.eval_interval and checkpoints:
            before = len(checkpoints)
            checkpoints = [cp for cp in checkpoints if cp.checkpoint_step % args.eval_interval == 0]
            print(f"Eval interval filter ({args.eval_interval}): {before} -> {len(checkpoints)} checkpoints")

        # Apply last-checkpoint-only filter if requested
        if args.last_checkpoint_only and checkpoints:
            print(f"Filtering to keep only last checkpoint per experiment...")
            checkpoints = filter_last_checkpoints(checkpoints)

        if args.base_model:
            print(f"Overriding base model to: {args.base_model}")
            for cp in checkpoints:
                cp.base_model = args.base_model
    else:
        # Create synthetic checkpoints for base_model or single_agent_transfer modes
        eval_config = EvalConfig(
            mode=eval_mode,
            base_model=args.base_model,
            workflow_types=args.workflow_types,
            single_agent_lora_path=args.single_agent_lora_path,
        )
        checkpoints = create_synthetic_checkpoints(eval_config)

    if not checkpoints:
        print("No checkpoints found!")
        return

    print(f"Found {len(checkpoints)} checkpoints")
    for cp in checkpoints:
        print(f"  - {cp.experiment_name} step {cp.checkpoint_step}")

    if eval_mode == EvalMode.TRAINED_CHECKPOINT and args.step_filter and len(args.step_filter) > 1:
        experiment_names = {cp.experiment_name for cp in checkpoints}
        if len(experiment_names) > 1:
            raise ValueError(
                "Multi-step --step-filter runs must be scoped to a single experiment. "
                "Pass an --experiment-filter that matches exactly one experiment."
            )
        print(
            "Multi-step checkpoint evaluation enabled for one experiment: "
            f"{next(iter(experiment_names))}; steps={sorted(args.step_filter)}. "
            "The vLLM server will stay loaded across these checkpoints."
        )

    # Filter out already-evaluated checkpoints (skip in trajectory-only mode)
    if args.max_samples is None:
        existing_keys = load_existing_results(args.output_json)
        if existing_keys:
            original_count = len(checkpoints)
            filtered = []
            for cp in checkpoints:
                key = get_checkpoint_key(cp, args.dataset, eval_mode.value, args.n_rollouts, args._length_mode)
                if key in existing_keys:
                    print(f"  Skipping (already evaluated): {cp.experiment_name} step {cp.checkpoint_step}")
                else:
                    filtered.append(cp)
            checkpoints = filtered
            print(f"Filtered: {original_count} -> {len(checkpoints)} checkpoints ({original_count - len(checkpoints)} already evaluated)")

            if not checkpoints:
                print("All checkpoints already evaluated. Nothing to do.")
                return

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    dataset = DatasetRegistry.load_dataset(args.dataset, "test")
    if dataset is None:
        raise ValueError(f"Dataset not found: {args.dataset}")

    # Convert dataset to list of dicts
    dataset_list = []
    for i in range(len(dataset)):
        task = dict(dataset[i])
        if "ground_truth" not in task and "final_answer" in task:
            task["ground_truth"] = task["final_answer"]
        elif "final_answer" not in task and "ground_truth" in task:
            task["final_answer"] = task["ground_truth"]
        dataset_list.append(task)

    if args.max_samples is not None and args.max_samples < len(dataset_list):
        print(f"Limiting to first {args.max_samples} samples (trajectory-only mode)")
        dataset_list = dataset_list[:args.max_samples]

    print(f"Dataset size: {len(dataset_list)} problems")

    # Auto-detect GPU type once (avoid repeated nvidia-smi calls)
    if args.gpu_type is None and args.n_parallel is None:
        args.gpu_type = detect_gpu_type()
        if args.gpu_type:
            print(f"Auto-detected GPU type: {args.gpu_type}")
        else:
            print("Warning: Could not auto-detect GPU type, will default to L40s scaling")

    # Group checkpoints by model
    checkpoints_by_model = group_checkpoints_by_model(checkpoints)
    print(f"Models to evaluate: {list(checkpoints_by_model.keys())}")

    all_results = []
    server = None

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\nReceived interrupt signal, cleaning up...")
        if server:
            server.stop()
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Process each model group (restart server per model)
    for base_model, model_checkpoints in checkpoints_by_model.items():
        print(f"\n{'#'*60}")
        print(f"Processing model: {base_model}")
        print(f"Checkpoints to evaluate: {len(model_checkpoints)}")
        print(f"{'#'*60}")

        try:
            # Start vLLM server for this model
            # Only enable LoRA if we're not in base_model mode
            enable_lora = eval_mode != EvalMode.BASE_MODEL
            server = VLLMServerManager(
                model=base_model,
                tensor_parallel_size=args.tensor_parallel,
                data_parallel_size=args.data_parallel,
                gpu_memory_utilization=args.gpu_memory_utilization,
                port=args.port,
                max_loras=args.max_loras,
                max_lora_rank=args.max_lora_rank,
                max_model_len=args.max_prompt_length + args.max_tokens,
                enable_lora=enable_lora,
            )
            server.start()

            # Create engine
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            engine = LoRAOpenAIEngine(
                base_model=base_model,
                tokenizer=tokenizer,
                base_url=f"http://localhost:{args.port}/v1",
                max_prompt_length=args.max_prompt_length,
                max_response_length=args.max_tokens,
                sampling_params={
                    "temperature": args.temperature,
                },
                disable_thinking=True,
                api_retries=1,  # Disable retries to surface errors immediately
            )

            # Evaluate all checkpoints for this model (sorted by step number)
            model_checkpoints.sort(key=lambda cp: cp.checkpoint_step)
            for checkpoint in tqdm(model_checkpoints, desc="Evaluating checkpoints", total=len(model_checkpoints)):
                # Apply per-workflow eval prompt/response limits from task_configs
                # when the user hasn't explicitly overridden them via CLI.
                wf_lengths = eval_prompt_response_lengths.get(checkpoint.workflow_type)
                if wf_lengths and not args._user_specified_max_prompt_length:
                    engine.max_prompt_length = wf_lengths[0]
                if wf_lengths and not args._user_specified_max_tokens:
                    engine.max_response_length = wf_lengths[1]
                    engine.max_model_length = engine.max_prompt_length + engine.max_response_length - 1
                engine._length_mode = args._length_mode

                # Compute n_parallel per checkpoint (workflow type may differ)
                if args.n_parallel is not None:
                    effective_n_parallel = args.n_parallel
                else:
                    effective_n_parallel = compute_n_parallel(
                        args.gpu_type, checkpoint.model_size, checkpoint.workflow_type,
                    )

                try:
                    result = asyncio.run(
                        evaluate_checkpoint(
                            checkpoint=checkpoint,
                            server=server,
                            engine=engine,
                            dataset=dataset_list,
                            n_parallel=effective_n_parallel,
                            eval_mode=eval_mode,
                            single_agent_lora_path=args.single_agent_lora_path,
                            trajectory_output_dir=args.trajectory_output_dir,
                            dataset_name=args.dataset,
                            n_rollouts=args.n_rollouts,
                        )
                    )
                    all_results.append(result)

                    # Save intermediate results (only the new result, not the full list)
                    if args.max_samples is None:
                        save_results_to_json([result], args.output_json)

                except Exception as e:
                    print(f"Error evaluating {checkpoint.experiment_name}: {e}")
                    import traceback
                    traceback.print_exc()

                # Unload LoRAs between checkpoints to manage memory
                server.unload_all_loras()

        finally:
            # Stop server before switching models
            if server:
                server.stop()
                server = None

    # Final summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Evaluation mode: {eval_mode.value}")
    print(f"Total checkpoints evaluated: {len(all_results)}")
    if args.max_samples is None:
        print(f"Results appended to: {args.output_json}")
    else:
        print(f"Trajectory-only mode: results NOT saved to {args.output_json}")
        print(f"Trajectories saved to: {args.trajectory_output_dir}")

    if all_results:
        has_multi_rollout = any(r.n_rollouts > 1 for r in all_results)
        print("\nSummary:")
        if has_multi_rollout:
            print(
                f"{'Experiment':<50} {'Step':>6} {'N':>3} "
                f"{'Mean Acc':>10} {'Std Acc':>10} {'Pass@N':>10} {'Mode':<25}"
            )
            print("-" * 118)
        else:
            print(f"{'Experiment':<50} {'Step':>6} {'Accuracy':>10} {'Mode':<25}")
            print("-" * 95)
        for result in sorted(
            all_results, key=lambda x: (x.experiment_name, x.checkpoint_step)
        ):
            if has_multi_rollout:
                print(
                    f"{result.experiment_name:<50} "
                    f"{result.checkpoint_step:>6} "
                    f"{result.n_rollouts:>3} "
                    f"{result.mean_accuracy:>10.2%} "
                    f"{result.std_accuracy:>10.2%} "
                    f"{result.pass_at_n:>10.2%} "
                    f"{result.eval_mode:<25}"
                )
            else:
                print(
                    f"{result.experiment_name:<50} "
                    f"{result.checkpoint_step:>6} "
                    f"{result.accuracy:>10.2%} "
                    f"{result.eval_mode:<25}"
                )


# ============================================================================
# CLI Entry Point
# ============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoints using vLLM pipeline (math or deepcoder)"
    )

    # Task type
    parser.add_argument(
        "--task-type",
        type=str,
        choices=["math", "deepcoder"],
        default="math",
        help="Task type to evaluate: math (default) or deepcoder",
    )

    # Evaluation mode arguments
    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=[
            "trained_checkpoint",
            "base_model",
            "single_agent_transfer",
            "single_agent_generator_transfer",
        ],
        default="trained_checkpoint",
        help=(
            "Evaluation mode: trained_checkpoint (default), base_model, "
            "single_agent_transfer (SA LoRA mapped to all roles), or "
            "single_agent_generator_transfer (SA LoRA mapped only to generator/worker roles)"
        ),
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        required=False,
        default=None,
        help="Path to checkpoints directory (required for trained_checkpoint mode)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model path. Required for base_model/single_agent_transfer modes. "
             "Optional override for trained_checkpoint mode (default: auto-detected from experiment name).",
    )
    parser.add_argument(
        "--workflow-types",
        type=str,
        nargs="+",
        default=None,
        help="Workflow types to evaluate (required for base_model and single_agent_transfer modes)",
    )
    parser.add_argument(
        "--single-agent-lora-path",
        type=str,
        default=None,
        help=(
            "Path to single-agent LoRA adapter (required for single_agent_transfer "
            "and single_agent_generator_transfer modes)"
        ),
    )
    parser.add_argument(
        "--last-checkpoint-only",
        action="store_true",
        help="Only evaluate the last (highest step) checkpoint per experiment",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="eval_results.jsonl",
        help="Output JSON Lines file path (results are appended)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name to evaluate on (default: per task type — aime2025 for math, deepcoder_primeintellect for deepcoder)",
    )
    parser.add_argument(
        "--experiment-filter",
        type=str,
        default=None,
        help="Regex pattern to filter experiments",
    )
    parser.add_argument(
        "--step-filter",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Only evaluate specific checkpoint steps. Multiple steps are supported "
            "when --experiment-filter scopes trained_checkpoint mode to one experiment."
        ),
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10,
        help="Only evaluate checkpoints whose step is divisible by this interval (default: 10)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens for generation. If not set, uses per-workflow values from task_configs.py.",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=None,
        help="Max prompt length in tokens. If not set, uses per-workflow values from task_configs.py.",
    )
    parser.add_argument(
        "--use-training-lengths",
        action="store_true",
        default=False,
        help="Use training prompt/response lengths instead of eval overrides from task_configs.py.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization",
    )
    parser.add_argument(
        "--n-parallel",
        type=int,
        default=None,
        help="Number of parallel workflow instances. If not set, auto-computed from GPU type, model size, and workflow type.",
    )
    parser.add_argument(
        "--gpu-type",
        type=str,
        default=None,
        choices=GPU_TYPE_CHOICES,
        help="GPU type for auto-scaling n_parallel. If not set, auto-detected via nvidia-smi.",
    )
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=1,
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--data-parallel",
        type=int,
        default=1,
        help="Data parallel size for load balancing",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="vLLM server port",
    )
    parser.add_argument(
        "--max-loras",
        type=int,
        default=5,
        help="Maximum number of LoRA adapters to load",
    )
    parser.add_argument(
        "--max-lora-rank",
        type=int,
        default=64,
        help="Maximum LoRA rank",
    )
    parser.add_argument(
        "--trajectory-output-dir",
        type=str,
        default=None,
        help="Directory to save detailed trajectory JSON files (one per problem)",
    )
    parser.add_argument(
        "--n-rollouts",
        type=int,
        default=1,
        help="Number of independent rollouts per sample for computing mean/std accuracy and Pass@N",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit evaluation to first N samples. When set, only saves trajectories "
             "(requires --trajectory-output-dir) and skips writing to eval_results.jsonl. "
             "Useful for trajectory analysis.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
