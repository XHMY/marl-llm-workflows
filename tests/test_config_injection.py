"""Hydra composition of multi_agent_ppo_trainer with paper command lines, and the injected keys (C10)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.stack

REPO = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO / "marl_workflows" / "configs"
FIXTURES = REPO / "tests" / "fixtures"


def _common(
    task: str, workflow: str, prefix: str, tag: str, sp: bool, names: str, p: int, r: int, tok: int
) -> list[str]:
    """The launcher's common prefix (DESIGN B13.1) without the model download path."""
    policy = "share_policy" if sp else "multi_lora"
    suffix = "math" if task == "math" else "deepcoder_primeintellect"
    return [
        f"marl.task={task}",
        f"marl.workflow={workflow}",
        f"data.max_prompt_length={p}",
        f"data.max_response_length={r}",
        "actor_rollout_ref.model.path=Qwen/Qwen3-1.7B",
        "trainer.project_name=rllm-workflow-MARL-v2",
        f"trainer.experiment_name={prefix}-{tag}-{policy}-{suffix}",
        "trainer.n_gpus_per_node=2",
        f"trainer.agent_names=[{names}]",
        f"trainer.share_policy={sp}",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={tok}",
    ]


# The four sample paper command lines: the reference cell, the T1 SP control, the KL-anchor
# control (per-role KL), and a Code cell with a warm-start adapter.
PAPER_COMMANDS: dict[str, list[str]] = {
    "voting_ip_math": _common(
        "math", "voting", "voting", "qwen3_1.7b", False, "generator,aggregator", 20480, 5120, 40960
    )
    + ["+rllm.workflow.n_votes=3", "rllm.workflow.use_final_outcome_reward=true"],
    "t1_length_weight": _common(
        "math", "voting", "voting", "qwen3_1.7b", True, "generator,aggregator", 20480, 5120, 40960
    )
    + [
        "+rllm.workflow.n_votes=3",
        "rllm.workflow.use_final_outcome_reward=true",
        "+trainer.loss_weight_per_role.mode=length",
        "actor_rollout_ref.actor.optim.lr_warmup_steps=0",
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.logprobs_mode=processed_logprobs",
        "+trainer.wallclock_stop.enable=true",
        "trainer.total_training_steps=300",
    ],
    "kl_anchor_evaluator": _common(
        "math",
        "evaluator_optimizer",
        "evaluator_optimizer",
        "qwen3_1.7b",
        False,
        "generator,evaluator",
        30720,
        5120,
        40960,
    )
    + [
        "+rllm.workflow.max_iterations=3",
        "rllm.workflow.use_final_outcome_reward=true",
        "actor_rollout_ref.actor.use_kl_loss=true",
        "actor_rollout_ref.actor.kl_loss_coef=0.0",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "+trainer.kl_loss_coef_per_role.evaluator=0.01",
        "+trainer.kl_loss_coef_per_role.generator=0.0",
    ],
    "code_orch_sp_init": _common(
        "deepcoder",
        "orchestrator_workers_propose",
        "orchestrator_workers_propose",
        "qwen3_1.7b",
        True,
        "orchestrator,worker,synthesizer",
        10240,
        2048,
        40960,
    )
    + [
        "+rllm.workflow.max_subtasks=3",
        "rllm.workflow.use_final_outcome_reward=true",
        "rllm.workflow.code_executor_workers=16",
        "+data.dataset_name=deepcoder_primeintellect",
        "+rllm.workflow.initial_lora_weights.generator=/ckpt/sa/lora_adapter_generator",
    ],
}

# Keys both configs define whose values the design changes on purpose (DESIGN B9, OQ5, OQ10).
INTENTIONAL_CHANGES = {
    "trainer.project_name": ("marl-llm-workflows", "multi-agent-workflow"),
    "ray_init.include_dashboard": (False, True),
    "data.custom_cls.path": ("pkg://marl_workflows.data.rl_dataset", None),
    "data.custom_cls.name": ("PolarsRLHFDataset", None),
}
# Fork keys dropped on purpose: dead keys, the removed Mode 3 scheduler, fireworks, the public dashboard host.
INTENTIONAL_DROPS = {
    "trainer.lora_adapter_path",
    "trainer.ori_single_policy_no_lora_mode",
    "rllm.workflow.max_concurrent_code_execs",
    "fireworks.deployment_id",
    "fireworks.model_id_prefix",
    "fireworks.concurrency",
    "ray_init.dashboard_host",
}


def _compose(overrides: list[str], config_dir: Path = CONFIG_DIR, name: str = "multi_agent_ppo_trainer"):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name=name, overrides=overrides)


def _inject(cfg) -> Any:
    from marl_workflows.recipes import get_recipe
    from marl_workflows.trainer.config_injection import inject_multi_agent_config

    inject_multi_agent_config(cfg, get_recipe(cfg.marl.task, cfg.marl.workflow))
    return cfg


def _flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    """Leaf values without resolving interpolations (an empty dict is a leaf)."""
    from omegaconf import OmegaConf

    out: dict[str, Any] = {}
    for key in node.keys():
        child = node._get_node(key)
        path = f"{prefix}.{key}" if prefix else str(key)
        if OmegaConf.is_dict(child):
            sub = _flatten(child, path)
            out.update(sub if sub else {path: {}})
        elif OmegaConf.is_list(child):
            out[path] = OmegaConf.to_container(child, resolve=False)
        else:
            out[path] = child._value()
    return out


@pytest.mark.parametrize("name", sorted(PAPER_COMMANDS))
def test_paper_command_lines_compose_and_inject(name: str) -> None:
    from omegaconf import OmegaConf

    cfg = _inject(_compose(PAPER_COMMANDS[name]))
    arr = cfg.actor_rollout_ref
    assert list(arr.agent_names) == list(cfg.trainer.agent_names)
    assert arr.share_policy == cfg.trainer.share_policy
    assert arr.rollout.engine_kwargs.vllm.max_loras == len(cfg.trainer.agent_names)
    # Interpolations stay unresolved on the driver.
    assert OmegaConf.is_interpolation(cfg.data, "gen_batch_size")
    assert OmegaConf.is_interpolation(cfg.trainer, "default_local_dir")


def test_reference_cell_keys() -> None:
    cfg = _inject(_compose(PAPER_COMMANDS["voting_ip_math"]))
    assert list(cfg.actor_rollout_ref.agent_names) == ["generator", "aggregator"]
    assert cfg.actor_rollout_ref.share_policy is False
    assert cfg.rllm.workflow.n_votes == 3
    assert (cfg.marl.train_dataset, cfg.marl.val_dataset) == ("dapo_math", "aime2025")
    assert "initial_lora_adapters" not in cfg.actor_rollout_ref
    assert cfg.actor_rollout_ref.model.get("lora_adapter_path") is None
    assert cfg.data.get("dataset_name") is None
    assert cfg.trainer.loss_weight_per_role == {} and cfg.trainer.kl_loss_coef_per_role == {}


def test_t1_control_keys() -> None:
    cfg = _inject(_compose(PAPER_COMMANDS["t1_length_weight"]))
    assert cfg.actor_rollout_ref.share_policy is True
    assert cfg.trainer.loss_weight_per_role.mode == "length"
    assert cfg.trainer.wallclock_stop.enable is True
    assert cfg.actor_rollout_ref.rollout.engine_kwargs.vllm.logprobs_mode == "processed_logprobs"
    assert cfg.actor_rollout_ref.actor.optim.lr_warmup_steps == 0
    assert cfg.trainer.total_training_steps == 300


def test_kl_anchor_keys() -> None:
    cfg = _inject(_compose(PAPER_COMMANDS["kl_anchor_evaluator"]))
    assert dict(cfg.trainer.kl_loss_coef_per_role) == {"evaluator": 0.01, "generator": 0.0}
    assert cfg.actor_rollout_ref.actor.use_kl_loss is True
    assert list(cfg.actor_rollout_ref.agent_names) == ["generator", "evaluator"]


def test_code_sp_warm_start_remap_and_dataset() -> None:
    cfg = _inject(_compose(PAPER_COMMANDS["code_orch_sp_init"]))
    # SP: the (remapped) SA adapter becomes the shared adapter.
    assert cfg.actor_rollout_ref.model.lora_adapter_path == "/ckpt/sa/lora_adapter_generator"
    assert "initial_lora_adapters" not in cfg.actor_rollout_ref
    assert cfg.actor_rollout_ref.rollout.engine_kwargs.vllm.max_loras == 3
    assert cfg.data.dataset_name == "deepcoder_primeintellect"
    assert (cfg.marl.train_dataset, cfg.marl.val_dataset) == ("deepcoder_primeintellect", "deepcoder_primeintellect")
    assert cfg.rllm.workflow.code_executor_workers == 16


def test_ip_warm_start_orchestrator_remap() -> None:
    cfg = _compose(
        [
            "marl.task=math",
            "marl.workflow=orchestrator_workers_propose",
            "+rllm.workflow.initial_lora_weights.generator=/a",
        ]
    )
    cfg = _inject(cfg)
    assert list(cfg.trainer.agent_names) == ["orchestrator", "worker", "synthesizer"]  # recipe default
    assert dict(cfg.actor_rollout_ref.initial_lora_adapters) == {"worker": "/a"}
    assert cfg.actor_rollout_ref.model.get("lora_adapter_path") is None


def test_ip_warm_start_unknown_role_rejected() -> None:
    cfg = _compose(["marl.task=math", "marl.workflow=voting", "+rllm.workflow.initial_lora_weights.critic=/a"])
    with pytest.raises(ValueError, match="not in trainer.agent_names"):
        _inject(cfg)


def test_deepcoder_executor_and_dataset_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "24")
    cfg = _inject(_compose(["marl.task=deepcoder", "marl.workflow=voting"]))
    assert cfg.rllm.workflow.code_executor_workers == 24
    assert cfg.data.dataset_name == "deepcoder_primeintellect"
    assert cfg.marl.train_dataset == "deepcoder_primeintellect"


def test_marl_dataset_overrides_for_math() -> None:
    cfg = _inject(
        _compose(
            ["marl.task=math", "marl.workflow=voting", "marl.train_dataset=smoke_math", "marl.val_dataset=smoke_math"]
        )
    )
    assert (cfg.marl.train_dataset, cfg.marl.val_dataset) == ("smoke_math", "smoke_math")
    assert cfg.data.get("dataset_name") is None


@pytest.mark.parametrize(
    "overrides, match",
    [
        (["marl.task=math", "marl.workflow=voting", "trainer.agent_names=[default,aggregator]"], "reserved"),
        (["marl.task=math", "marl.workflow=single_agent", "trainer.share_policy=true"], "does not allow"),
        (["marl.task=deepcoder", "marl.workflow=voting", "+rllm.workflow.enable_test_loop=true"], "enable_test_loop"),
        (
            ["marl.task=deepcoder", "marl.workflow=voting", "+rllm.workflow.max_concurrent_code_execs=8"],
            "max_concurrent",
        ),
        (
            ["marl.task=math", "marl.workflow=voting", "+actor_rollout_ref.rollout.engine_kwargs.vllm.logprobs_mode=x"],
            "logprobs_mode",
        ),
    ],
)
def test_rejected_configurations(overrides: list[str], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _inject(_compose(overrides))


def test_equivalence_with_fork_config() -> None:
    ours = _flatten(_compose([]))
    fork = _flatten(_compose([], config_dir=FIXTURES, name="fork_multi_agent_ppo_trainer"))
    changed = {k: (ours[k], fork[k]) for k in sorted(set(ours) & set(fork)) if ours[k] != fork[k]}
    assert changed == INTENTIONAL_CHANGES
    assert set(fork) - set(ours) == INTENTIONAL_DROPS


def test_run_wires_datasets_and_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """``train.run`` without Ray: datasets resolved through the registry, workflow class and args from the recipe."""
    import ray

    from marl_workflows.data.registry import MarlDatasetRegistry
    from marl_workflows.trainer import task_runner, train

    seen: dict[str, Any] = {}

    class FakeRunner:
        @classmethod
        def remote(cls):
            return cls()

        class run:  # mimics the Ray actor-method handle
            @staticmethod
            def remote(**kwargs: Any) -> str:
                seen.update(kwargs)
                return "ref"

    monkeypatch.setattr(MarlDatasetRegistry, "verl_path", classmethod(lambda cls, name, split: f"/data/{name}.{split}"))
    monkeypatch.setattr(train, "load_object", lambda path: f"cls:{path}")
    monkeypatch.setattr(train, "build_workflow_args", lambda recipe, config: {"n_votes": config.rllm.workflow.n_votes})
    monkeypatch.setattr(task_runner, "MultiAgentTaskRunner", FakeRunner)
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray, "get", lambda ref: seen.setdefault("got", ref))

    cfg = _compose(PAPER_COMMANDS["voting_ip_math"])
    train.run(cfg)
    assert seen["got"] == "ref"
    assert seen["workflow_class"] == "cls:marl_workflows.workflows.math.voting.VotingMathWorkflow"
    assert seen["workflow_args"] == {"n_votes": 3}
    assert seen["config"].data.train_files == "/data/dapo_math.train"
    assert seen["config"].data.val_files == "/data/aime2025.test"
    assert seen["config"].actor_rollout_ref.rollout.engine_kwargs.vllm.max_loras == 2
