"""Tests for marl_workflows.contracts (C1, C2, C5-C7, C13, C14, C16, C21)."""

from __future__ import annotations

import pytest

from marl_workflows import contracts as c

# ----------------------------------------------------------------------------- roles


@pytest.mark.parametrize(
    ("name", "role"),
    [
        ("generator0", "generator"),
        ("generator12", "generator"),
        ("generator", "generator"),
        ("aggregator", "aggregator"),
        ("voterA", "voterA"),
        ("worker", "worker"),
        ("evaluator", "evaluator"),
    ],
)
def test_role_from_name(name: str, role: str) -> None:
    assert c.role_from_name(name) == role


@pytest.mark.parametrize(
    ("traj_id", "role"),
    [
        ("3f2b9c1e-8d7a-4c55-9e0b-1a2b3c4d5e6f_generator0", "generator"),
        ("3f2b9c1e-8d7a-4c55-9e0b-1a2b3c4d5e6f_aggregator", "aggregator"),
        ("task_with_underscores_voterB", "voterB"),
        ("abc-def_synthesizer", "synthesizer"),
        ("u_worker2", "worker"),
    ],
)
def test_role_from_trajectory_id(traj_id: str, role: str) -> None:
    assert c.role_from_trajectory_id(traj_id) == role


def test_role_from_trajectory_id_without_separator_raises() -> None:
    with pytest.raises(ValueError):
        c.role_from_trajectory_id("generator0")


def test_roles_from_trajectory_ids_keeps_order() -> None:
    ids = ["a-1_generator0", "a-1_generator1", "a-1_aggregator", "b-2_generator2"]
    assert c.roles_from_trajectory_ids(ids) == ["generator", "generator", "aggregator", "generator"]


@pytest.mark.parametrize("name", ["generator", "generator0", "voterA", "aggregator", "worker12"])
def test_validate_trajectory_name_accepts(name: str) -> None:
    c.validate_trajectory_name(name)


@pytest.mark.parametrize("name", ["gen_0", "gen:0", "0gen", "gen0a", "", "voter-A", "gén"])
def test_validate_trajectory_name_rejects(name: str) -> None:
    with pytest.raises(ValueError):
        c.validate_trajectory_name(name)


# ----------------------------------------------------------------------------- agent names


def test_validate_agent_names_returns_list() -> None:
    assert c.validate_agent_names(("generator", "aggregator"), False) == ["generator", "aggregator"]
    assert c.validate_agent_names(["voterA", "voterB", "voterC", "aggregator"], False) == [
        "voterA",
        "voterB",
        "voterC",
        "aggregator",
    ]


@pytest.mark.parametrize("sp", [False, True])
@pytest.mark.parametrize(
    "names",
    [
        [],
        ["generator", "generator"],
        ["gen_a", "aggregator"],
        ["generator1", "aggregator"],
        ["generator", "agg:r"],
    ],
)
def test_validate_agent_names_rejects_always(names: list[str], sp: bool) -> None:
    with pytest.raises(ValueError):
        c.validate_agent_names(names, sp)


def test_default_forbidden_only_under_ip() -> None:
    with pytest.raises(ValueError, match="reserved"):
        c.validate_agent_names(["default"], False)
    assert c.validate_agent_names(["default"], True) == ["default"]


def test_substring_overlap_forbidden_only_under_ip() -> None:
    with pytest.raises(ValueError, match="substring"):
        c.validate_agent_names(["gen", "generator"], False)
    with pytest.raises(ValueError, match="substring"):
        c.validate_agent_names(["voter", "voterA"], False)
    assert c.validate_agent_names(["gen", "generator"], True) == ["gen", "generator"]


# ----------------------------------------------------------------------------- LoRA ids


def test_lora_int_id_ip() -> None:
    names = ["orchestrator", "worker", "synthesizer"]
    assert [c.lora_int_id_for(n, names, False) for n in names] == [123, 124, 125]


def test_lora_int_id_sp_is_none() -> None:
    assert c.lora_int_id_for("generator", ["generator", "aggregator"], True) is None
    assert c.lora_int_id_for(None, ["generator", "aggregator"], True) is None
    # SP never looks the name up, so even an unknown name maps to the shared adapter.
    assert c.lora_int_id_for("nobody", ["generator"], True) is None


def test_lora_int_id_ip_none_strict_and_lenient() -> None:
    with pytest.raises(ValueError):
        c.lora_int_id_for(None, ["generator", "aggregator"], False)
    assert c.lora_int_id_for(None, ["generator", "aggregator"], False, strict=False) is None


@pytest.mark.parametrize("strict", [True, False])
def test_lora_int_id_unknown_name_raises(strict: bool) -> None:
    with pytest.raises(ValueError, match="Unknown"):
        c.lora_int_id_for("evaluator", ["generator", "aggregator"], False, strict=strict)


def test_expected_lora_int_ids() -> None:
    assert c.expected_lora_int_ids(["generator", "aggregator"], False) == [123, 124]
    assert c.expected_lora_int_ids(["voterA", "voterB", "voterC", "aggregator"], False) == [123, 124, 125, 126]
    assert c.expected_lora_int_ids(["generator", "aggregator"], True) == [123]
    assert c.expected_lora_int_ids(["generator"], False) == [123]


def test_lora_constants() -> None:
    assert c.LORA_INT_ID_BASE == 123
    assert c.LORA_PATH_PLACEHOLDER == "simon_lora_path"
    assert c.SHARED_ADAPTER == "default"


def test_lora_strict_from_env() -> None:
    assert c.lora_strict_from_env({}) is True
    assert c.lora_strict_from_env({c.ENV_STRICT_LORA: "1"}) is True
    assert c.lora_strict_from_env({c.ENV_STRICT_LORA: "0"}) is False


# ----------------------------------------------------------------------------- checkpoints


def test_adapter_dirnames() -> None:
    assert c.adapter_dirname(None) == "lora_adapter"
    assert c.adapter_dirname("generator") == "lora_adapter_generator"
    assert c.adapter_dirname("voterA") == "lora_adapter_voterA"


def test_actor_ckpt_dir() -> None:
    assert c.actor_ckpt_dir("/ckpt/exp", 300) == "/ckpt/exp/global_step_300/actor"
    assert c.STEP_DIR_FMT.format(step=7) == "global_step_7"
    assert c.TRAINING_METADATA_FILE == "training_metadata.json"
    assert c.LATEST_ITER_FILE == "latest_checkpointed_iteration.txt"


# ----------------------------------------------------------------------------- metrics


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("actor/pg_loss", "actor/generator/pg_loss"),
        ("actor/grad_norm", "actor/generator/grad_norm"),
        ("training/rollout_probs_diff/chi2_token", "actor/generator/chi2_token"),
        ("actor/num_mini_batches", "actor/generator/num_mini_batches"),
        ("lr", "actor/generator/lr"),
    ],
)
def test_role_metric_key(key: str, expected: str) -> None:
    assert c.role_metric_key("generator", key) == expected


def test_rename_role_metrics_collision_last_wins() -> None:
    metrics = {"actor/kl_loss": 1.0, "actor/pg_loss": 2.0, "critic/kl_loss": 3.0}
    out = c.rename_role_metrics("aggregator", metrics)
    assert out == {"actor/aggregator/kl_loss": 3.0, "actor/aggregator/pg_loss": 2.0}
    # Reversed insertion order flips the winner: the rule is iteration order, as in the fork.
    out_rev = c.rename_role_metrics("aggregator", dict(reversed(list(metrics.items()))))
    assert out_rev["actor/aggregator/kl_loss"] == 1.0


def test_metric_constants() -> None:
    assert c.OPTIM_COUNTER_KEYS == ("actor/num_mini_batches", "actor/optim_steps_this_call", "actor/optim_step_count")
    assert c.LOSS_WEIGHT_PREFIX == "actor/loss_weight/"
    assert c.META_TEMPERATURE == "temperature"
    assert c.META_KL_OVERRIDE == "kl_loss_coef_override"
    assert c.SAMPLING_LORA_KEY == "__marl_lora_int_id__"
    assert c.OPENAI_MODEL_KEY == "__marl_model__"


def test_stdout_markers() -> None:
    assert c.RUN_COMPLETE_FMT.format(step=300) == "RUN_COMPLETE global_step=300"
    assert (
        c.WALLCLOCK_STOP_FMT.format(step=42, saved="none (no step in this segment)")
        == "WALLCLOCK_STOP completed_step=42 saved=none (no step in this segment)"
    )


# ----------------------------------------------------------------------------- experiment names


def test_experiment_name() -> None:
    assert c.experiment_name("voting", "qwen3_1.7b", False, "math") == "voting-qwen3_1.7b-multi_lora-math"
    assert (
        c.experiment_name("evaluator_optimizer", "qwen3_4b", True, "deepcoder")
        == "evaluator_optimizer-qwen3_4b-share_policy-deepcoder_primeintellect"
    )
    assert (
        c.experiment_name("voting", "qwen3_1.7b", True, "math", "role_len_weight")
        == "voting-qwen3_1.7b-share_policy-math-role_len_weight"
    )
    assert (
        c.experiment_name("voting_v2", "qwen3_1.7b_s300", False, "math") == "voting_v2-qwen3_1.7b_s300-multi_lora-math"
    )
    assert c.experiment_name("voting", "qwen3_0.6b", False, "math", None) == "voting-qwen3_0.6b-multi_lora-math"


def test_experiment_name_unknown_task() -> None:
    with pytest.raises(ValueError):
        c.experiment_name("voting", "qwen3_0.6b", False, "code")


def test_env_var_names() -> None:
    assert c.ENV_DATA_DIR == "MARL_DATA_DIR"
    assert c.ENV_STRICT_LORA == "MARL_LORA_STRICT"
    assert c.ENV_INIT_WEIGHT_DIR == "MARL_INIT_WEIGHT_DIR"
    assert c.TASKS == ("math", "deepcoder")
    assert c.POLICY_TAG == {False: "multi_lora", True: "share_policy"}
