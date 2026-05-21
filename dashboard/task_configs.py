"""Central task-type registry for math and deepcoder experiments."""

import re
from pathlib import Path

TASK_CONFIGS = {
    "math": {
        "workflow_map": {
            "single_agent": "examples.math_reasoning.single_agent_math_workflow.SingleAgentMathWorkflow",
            "evaluator_optimizer": "examples.math_reasoning.evaluator_optimizer_math_workflow.EvaluatorOptimizerMathWorkflow",
            "voting": "examples.math_reasoning.voting_math_workflow.VotingMathWorkflow",
            "orchestrator_workers_propose": "examples.math_reasoning.orchestrator_workers_math_workflow.OrchestratorWorkersMathWorkflow",
            "voting_v2": "examples.math_reasoning.voting_v2_math_workflow.VotingV2MathWorkflow",
            "evaluator_optimizer_v2": "examples.math_reasoning.evaluator_optimizer_v2_math_workflow.EvaluatorOptimizerV2MathWorkflow",
        },
        "reward_fn": "rllm.rewards.reward_fn.math_reward_fn",
        "entry_points": {
            "evaluator_optimizer": "examples.math_reasoning.train_evaluator_optimizer_math",
            "voting": "examples.math_reasoning.train_voting_math",
            "orchestrator_workers_propose": "examples.math_reasoning.train_orchestrator_workers_math",
            "single_agent": "examples.math_reasoning.train_single_agent_math",
            "voting_v2": "examples.math_reasoning.train_voting_v2_math",
            "evaluator_optimizer_v2": "examples.math_reasoning.train_evaluator_optimizer_v2_math",
        },
        "default_eval_dataset": "aime2025",
        "experiment_suffix": "math",
        "prompt_response_lengths": {
            "evaluator_optimizer": (30720, 5120),
            "voting": (20480, 5120),
            "orchestrator_workers_propose": (20480, 5120),
            "single_agent": (15360, 5120),
            "voting_v2": (20480, 5120),
            "evaluator_optimizer_v2": (30720, 5120),
        },
        "eval_prompt_response_lengths": {
            "evaluator_optimizer": (30720, 8192),
            "voting": (25600, 8192),
            "orchestrator_workers_propose": (25600, 8192),
            "single_agent": (15360, 8192),
            "voting_v2": (25600, 8192),
            "evaluator_optimizer_v2": (30720, 8192),
        },
        "workflow_params": {
            "evaluator_optimizer": {
                "max_iterations": 3,
                "use_final_outcome_reward": True,
            },
            "voting": {
                "n_votes": 3,
                "use_final_outcome_reward": True,
            },
            "orchestrator_workers_propose": {
                "max_subtasks": 3,
                "use_final_outcome_reward": True,
            },
            "single_agent": {},
            "voting_v2": {
                "n_votes": 3,
                "use_rubric_reward": True,
                "use_final_outcome_reward": False,
            },
            "evaluator_optimizer_v2": {
                "max_iterations": 3,
                "use_final_outcome_reward": True,
            },
        },
        "extra_sbatch_cmds": "",
        "experiment_filter_include": "math",
        "experiment_filter_exclude": "deepcoder",
    },
    "deepcoder": {
        "workflow_map": {
            "single_agent": "examples.deepcoder.single_agent_deepcoder_workflow.SingleAgentDeepcodeWorkflow",
            "evaluator_optimizer": "examples.deepcoder.deepcoder_evaluator_optimizer_workflow.DeepcodeEvaluatorOptimizerWorkflow",
            "voting": "examples.deepcoder.deepcoder_voting_workflow.DeepcodeVotingWorkflow",
            "orchestrator_workers_propose": "examples.deepcoder.deepcoder_orchestrator_workers_workflow.DeepcodeOrchestratorWorkersWorkflow",
            "voting_v2": "examples.deepcoder.deepcoder_voting_v2_workflow.DeepcodeVotingV2Workflow",
            "evaluator_optimizer_v2": "examples.deepcoder.deepcoder_evaluator_optimizer_v2_workflow.DeepcodeEvaluatorOptimizerV2Workflow",
        },
        "reward_fn": "rllm.rewards.reward_fn.code_reward_fn",
        "entry_points": {
            "evaluator_optimizer": "examples.deepcoder.train_deepcoder_evaluator_optimizer",
            "voting": "examples.deepcoder.train_deepcoder_voting",
            "orchestrator_workers_propose": "examples.deepcoder.train_deepcoder_orchestrator_workers",
            "single_agent": "examples.deepcoder.train_single_agent_deepcoder",
            "voting_v2": "examples.deepcoder.train_deepcoder_voting_v2",
            "evaluator_optimizer_v2": "examples.deepcoder.train_deepcoder_evaluator_optimizer_v2",
        },
        "default_eval_dataset": "deepcoder_primeintellect",
        "experiment_suffix": "deepcoder",
        "prompt_response_lengths": {
            "evaluator_optimizer": (10240, 2048),
            "voting": (10240, 2048),
            "orchestrator_workers_propose": (10240, 2048),
            "single_agent": (4096, 2048),
            "voting_v2": (10240, 2048),
            "evaluator_optimizer_v2": (10240, 2048),
        },
        # Evaluation-only overrides: larger response length to avoid truncation.
        "eval_prompt_response_lengths": {
            "evaluator_optimizer": (15360, 5120),
            "voting": (20480, 5120),
            "orchestrator_workers_propose": (20480, 5120),
            "single_agent": (4096, 5120),
            "voting_v2": (20480, 5120),
            "evaluator_optimizer_v2": (15360, 5120),
        },
        "workflow_params": {
            "evaluator_optimizer": {
                "max_iterations": 2,
                "use_final_outcome_reward": True,
                "enable_test_loop": False,
            },
            "voting": {
                "n_votes": 3,
                "use_final_outcome_reward": True,
                "enable_test_loop": False,
            },
            "orchestrator_workers_propose": {
                "max_subtasks": 3,
                "use_final_outcome_reward": True,
                "enable_test_loop": False,
            },
            "single_agent": {
                "enable_test_loop": False,
            },
            "voting_v2": {
                "n_votes": 3,
                "use_rubric_reward": True,
                "use_final_outcome_reward": False,
                "enable_test_loop": False,
            },
            "evaluator_optimizer_v2": {
                "max_iterations": 2,
                "use_final_outcome_reward": True,
                "enable_test_loop": False,
            },
        },
        "extra_sbatch_cmds": "ulimit -n 1048576",
        "experiment_filter_include": "deepcoder",
        "experiment_filter_exclude": None,
    },
}

AGENT_NAMES_MAP = {
    "single_agent": ["generator"],
    "evaluator_optimizer": ["generator", "evaluator"],
    "voting": ["generator", "aggregator"],
    "orchestrator_workers_propose": ["orchestrator", "worker", "synthesizer"],
    "voting_v2": ["voterA", "voterB", "voterC", "aggregator"],
    "evaluator_optimizer_v2": ["generator", "evaluator"],
}

MODEL_MAP = {
    "0.6b": "Qwen/Qwen3-0.6B",
    "1.7b": "Qwen/Qwen3-1.7B",
    "4b": "Qwen/Qwen3-4B",
}

INIT_WEIGHT_DIR = "checkpoints/init_weight"

EVAL_DATASETS = {
    "Math": ["dapo_math", "aime2025", "aime2024", "gpqa_diamond"],
    "Code": ["deepcoder_primeintellect", "deepcoder_codeforces"],
}

# Per-dataset evaluation overrides for max_prompt_length and max_tokens.
# Datasets not listed here use the evaluate_checkpoints.py defaults (30720 / 5120).
EVAL_DATASET_OVERRIDES = {}


def extract_model_size(base_model_name: str) -> str:
    """Extract model size (e.g., '1.7b') from a base model name like 'qwen3_1.7b_s300'."""
    m = re.match(r"qwen3_([\d.]+[bB])", base_model_name)
    return m.group(1).lower() if m else ""


def list_init_weights(checkpoint_dir: str = "", task_type: str = "") -> dict[str, list[str]]:
    """Scan init_weight directory and return base models grouped by model size.

    When *task_type* is given (e.g. "math", "deepcoder"), only checkpoints under
    that task subfolder are returned.

    Returns e.g. {"1.7b": ["qwen3_1.7b_s300", "qwen3_1.7b_s430"], "4b": ["qwen3_4b_s300"]}
    """
    init_dir = Path(checkpoint_dir) / INIT_WEIGHT_DIR if checkpoint_dir else Path(INIT_WEIGHT_DIR)
    if task_type:
        init_dir = init_dir / task_type
    if not init_dir.is_dir():
        return {}
    result: dict[str, list[str]] = {}
    for entry in sorted(init_dir.iterdir()):
        if not entry.is_dir():
            continue
        size = extract_model_size(entry.name)
        if size:
            result.setdefault(size, []).append(entry.name)
    return result


def infer_task_type(experiment_name: str) -> str:
    """Infer task type from experiment directory name."""
    name_lower = experiment_name.lower()
    if "deepcoder" in name_lower:
        return "deepcoder"
    return "math"


def get_task_config(task_type: str) -> dict:
    """Return config dict for the given task type."""
    if task_type not in TASK_CONFIGS:
        raise ValueError(f"Unknown task type: {task_type!r}. Expected one of: {list(TASK_CONFIGS.keys())}")
    return TASK_CONFIGS[task_type]


# ── Hydra override helpers ──────────────────────────────────────────────────

# Keys that already exist in the base Hydra config (no '+' prefix needed)
_HYDRA_BASE_KEYS = {"use_final_outcome_reward"}


def workflow_params_to_hydra(params: dict) -> str:
    """Convert a workflow_params dict to a Hydra override string."""
    parts = []
    for key, value in params.items():
        val_str = str(value).lower() if isinstance(value, bool) else str(value)
        prefix = "" if key in _HYDRA_BASE_KEYS else "+"
        parts.append(f"{prefix}rllm.workflow.{key}={val_str}")
    return " ".join(parts)


def resolve_launch_params(task_type: str, workflow: str, model_key: str, base_model: str = "") -> dict:
    """Resolve all launch parameters from the central config.

    Returns a dict with: entry_point, agent_names, workflow_params (Hydra string),
    max_prompt_length, max_response_length, model_path.

    When *base_model* is provided (e.g. "qwen3_1.7b_s300"), the model_path is
    overridden to point at the init_weight directory instead of HuggingFace.
    """
    config = get_task_config(task_type)
    model_lower = model_key.lower()
    if workflow not in config["entry_points"]:
        raise ValueError(f"Unknown workflow {workflow!r} for task {task_type!r}")
    if model_lower not in MODEL_MAP:
        raise ValueError(f"Unknown model {model_key!r}")
    prompt_len, response_len = config["prompt_response_lengths"][workflow]

    if base_model:
        model_path = f"{INIT_WEIGHT_DIR}/{task_type}/{base_model}"
    else:
        model_path = MODEL_MAP[model_lower]

    return {
        "entry_point": config["entry_points"][workflow],
        "agent_names": AGENT_NAMES_MAP[workflow],
        "workflow_params": workflow_params_to_hydra(config["workflow_params"].get(workflow, {})),
        "max_prompt_length": prompt_len,
        "max_response_length": response_len,
        "model_path": model_path,
    }
