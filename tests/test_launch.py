"""Launcher tests: golden override sets for every cell type and control, experiment names, SLURM rendering.

Regenerate the golden files after an intended change with
``MARL_UPDATE_GOLDEN=1 pytest tests/test_launch.py`` and review the diff.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

import pytest

from marl_workflows import launch
from marl_workflows.contracts import TASK_SUFFIX
from marl_workflows.launch import CONTROLS, LaunchSpec, build_overrides, merge_overrides, parse_override
from marl_workflows.recipes import RECIPES, get_recipe

REPO = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO / "tests" / "fixtures" / "launch_golden"
GRID_WORKFLOWS = ("single_agent", "voting", "evaluator_optimizer", "orchestrator_workers_propose")
SCALES = ("0.6B", "1.7B", "4B")
ROOT_PLACEHOLDER = "<REPO>"


@pytest.fixture(autouse=True)
def _no_init_weight_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MARL_INIT_WEIGHT_DIR", raising=False)


# --------------------------------------------------------------------------------------
# Golden cases
# --------------------------------------------------------------------------------------


def _cell_specs(task: str) -> list[tuple[str, LaunchSpec]]:
    out = []
    for workflow in GRID_WORKFLOWS:
        for sp in get_recipe(task, workflow).allowed_share_policy:
            for scale in SCALES:
                label = f"{task} {workflow} {'sp' if sp else 'ip'} {scale}"
                out.append((label, LaunchSpec(task=task, workflow=workflow, share_policy=sp, scale=scale)))
    return out


def _control_spec(name: str) -> LaunchSpec:
    ctrl = CONTROLS[name]
    task, workflow, policy, scale = ctrl.example.split()
    kwargs: dict = {}
    if ctrl.needs_resume_from:
        kwargs["resume_from"] = "checkpoints/proj/evaluator_optimizer-qwen3_1.7b-multi_lora-math/global_step_320"
    if ctrl.needs_init_adapters:
        kwargs["init_adapters"] = {
            "generator": "checkpoints/init/global_step_300/actor/lora_adapter_generator",
            "aggregator": "checkpoints/init/global_step_300/actor/lora_adapter_generator",
        }
    return LaunchSpec(task=task, workflow=workflow, share_policy=policy == "sp", scale=scale, control=name, **kwargs)


def _option_specs() -> list[tuple[str, LaunchSpec]]:
    return [
        (
            "math voting ip 1.7B --gpu-type L40S",
            LaunchSpec(task="math", workflow="voting", share_policy=False, scale="1.7B", gpu_type="L40S"),
        ),
        (
            "deepcoder voting ip 0.6B --gpu-type L40S --gpus 4",
            LaunchSpec(
                task="deepcoder", workflow="voting", share_policy=False, scale="0.6B", gpu_type="L40S", n_gpus=4
            ),
        ),
        (
            "math orchestrator_workers_propose sp 1.7B --base-model qwen3_1.7b_s300",
            LaunchSpec(
                task="math",
                workflow="orchestrator_workers_propose",
                share_policy=True,
                scale="1.7B",
                base_model_tag="qwen3_1.7b_s300",
            ),
        ),
        (
            "math voting sp 0.6B --processed-logprobs --name-suffix trial --extra",
            LaunchSpec(
                task="math",
                workflow="voting",
                share_policy=True,
                scale="0.6B",
                processed_logprobs=True,
                name_suffix="trial",
                extra=["trainer.total_training_steps=50", "rllm.workflow.use_final_outcome_reward=false"],
            ),
        ),
        (
            "deepcoder evaluator_optimizer ip 4B --project p --experiment-name custom",
            LaunchSpec(
                task="deepcoder",
                workflow="evaluator_optimizer",
                share_policy=False,
                scale="4B",
                project="p",
                experiment_name="custom",
            ),
        ),
        (
            "math voting_v2 sp 0.6B",
            LaunchSpec(task="math", workflow="voting_v2", share_policy=True, scale="0.6B"),
        ),
    ]


def _block(label: str, overrides: list[str]) -> str:
    return f"## {label}\n" + "\n".join(overrides) + "\n"


def _golden_texts() -> dict[str, str]:
    texts = {}
    for task in ("math", "deepcoder"):
        texts[f"cells_{task}.txt"] = "\n".join(_block(lbl, build_overrides(s)) for lbl, s in _cell_specs(task))
    texts["controls.txt"] = "\n".join(
        _block(f"{name} ({CONTROLS[name].example})", build_overrides(_control_spec(name))) for name in CONTROLS
    )
    texts["options.txt"] = "\n".join(_block(lbl, build_overrides(s)) for lbl, s in _option_specs())
    train = launch.render_sbatch(
        LaunchSpec(task="deepcoder", workflow="voting", share_policy=False, scale="1.7B"),
        REPO / "configs" / "slurm" / "preempt_l40s.conf",
        time_limit="0-05:50:00",
        root=ROOT_PLACEHOLDER,
    )
    texts["sbatch_train_deepcoder_voting_ip_1.7B_l40s.txt"] = train
    texts["sbatch_eval_math.txt"] = launch.render_eval_sbatch(
        [
            "--task-type",
            "math",
            "--checkpoints-dir",
            "checkpoints/marl-llm-workflows",
            "--dataset",
            "dapo_math",
            "--use-training-lengths",
        ],
        REPO / "configs" / "slurm" / "dgxh_preempt_h100.conf",
        root=ROOT_PLACEHOLDER,
    )
    return texts


@pytest.mark.parametrize("name", sorted(_golden_texts()))
def test_golden(name: str) -> None:
    text = _golden_texts()[name]
    path = GOLDEN_DIR / name
    if os.environ.get("MARL_UPDATE_GOLDEN") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    assert path.is_file(), f"missing golden file {path}; run with MARL_UPDATE_GOLDEN=1"
    assert text == path.read_text(), f"{name} differs from the golden file"


# --------------------------------------------------------------------------------------
# Semantics
# --------------------------------------------------------------------------------------


def _as_dict(overrides: list[str]) -> dict[str, str | None]:
    return {parse_override(o)[1]: parse_override(o)[2] for o in overrides}


def test_grid_cell_count() -> None:
    n = sum(len(_cell_specs(t)) for t in ("math", "deepcoder"))
    assert n == 2 * (3 * 2 * 3 + 3)  # 3 workflows x 2 policies x 3 scales + SA-RL x 3 scales, per task


@pytest.mark.parametrize("label,spec", _cell_specs("math") + _cell_specs("deepcoder"))
def test_cell_overrides_follow_recipe(label: str, spec: LaunchSpec) -> None:
    recipe = get_recipe(spec.task, spec.workflow)
    ov = _as_dict(build_overrides(spec))
    assert ov["marl.task"] == spec.task and ov["marl.workflow"] == spec.workflow
    assert (int(ov["data.max_prompt_length"]), int(ov["data.max_response_length"])) == recipe.train_lengths
    assert ov["trainer.agent_names"] == "[" + ",".join(recipe.agent_names) + "]"
    assert ov["trainer.share_policy"] == str(spec.share_policy)
    assert ov["trainer.project_name"] == launch.DEFAULT_PROJECT
    model_tag = launch.SCALES[spec.scale][1]
    policy = "share_policy" if spec.share_policy else "multi_lora"
    assert ov["trainer.experiment_name"] == f"{recipe.experiment_prefix}-{model_tag}-{policy}-{TASK_SUFFIX[spec.task]}"
    assert ov["actor_rollout_ref.actor.ppo_max_token_len_per_gpu"] == str(
        launch.PPO_MAX_TOKEN_LEN[(spec.scale, "large")]
    )
    if spec.task == "deepcoder":
        assert ov["data.dataset_name"] == "deepcoder_primeintellect"
        assert ov["rllm.workflow.code_executor_workers"] == "32"  # 2 GPUs x 16 CPUs
    else:
        assert "data.dataset_name" not in ov and "rllm.workflow.code_executor_workers" not in ov
    # Settings left to the yaml are not emitted for a plain cell.
    for key in ("trainer.total_training_steps", "actor_rollout_ref.actor.optim.lr_warmup_steps"):
        assert key not in ov


@pytest.mark.parametrize("key", sorted(launch.WORKFLOW_PARAMS))
def test_workflow_params_match_recipe_defaults(key: tuple[str, str]) -> None:
    recipe = RECIPES[key]
    defaults = dict(recipe.params)
    for ov in launch.WORKFLOW_PARAMS[key]:
        prefix, name, value = parse_override(ov)
        leaf = name.removeprefix("rllm.workflow.")
        assert leaf in defaults, f"{key}: {leaf} is not a recipe param"
        assert str(defaults[leaf]).lower() == value
        # Only use_final_outcome_reward is declared in the yaml; the other leaves need "+".
        assert (prefix == "") == (leaf == "use_final_outcome_reward")


def test_every_workflow_recipe_has_params() -> None:
    assert set(launch.WORKFLOW_PARAMS) == set(RECIPES)


def test_merge_overrides_replaces_in_place() -> None:
    merged = merge_overrides(
        ["a=1", "+b=2", "c=3"],
        ["b=20", "++d=4", "a=10"],
    )
    assert merged == ["a=10", "+b=20", "c=3", "++d=4"]
    assert merge_overrides(["x=1"], ["+x=2"]) == ["+x=2"]
    assert merge_overrides(["~y"]) == ["~y"]


def test_parse_override() -> None:
    assert parse_override("+trainer.kl_loss_coef_per_role.evaluator=0.01") == (
        "+",
        "trainer.kl_loss_coef_per_role.evaluator",
        "0.01",
    )
    assert parse_override("trainer.agent_names=[a,b]") == ("", "trainer.agent_names", "[a,b]")
    with pytest.raises(ValueError):
        parse_override("=1")


@pytest.mark.parametrize("name", sorted(CONTROLS))
def test_control_example_builds(name: str) -> None:
    ov = _as_dict(build_overrides(_control_spec(name)))
    for o in CONTROLS[name].overrides:
        _, key, value = parse_override(o)
        assert ov[key] == value


def test_control_experiment_names() -> None:
    def name(control: str) -> str:
        return _as_dict(build_overrides(_control_spec(control)))["trainer.experiment_name"]

    assert name("per_voter_reward") == "voting-qwen3_1.7b-multi_lora-math-per_agent_reward"
    assert name("evaluator_accuracy_reward") == "evaluator_optimizer-qwen3_1.7b-multi_lora-math-per_agent_reward"
    assert name("prompt_het") == "voting-qwen3_1.7b-multi_lora-math-prompt_het"
    assert name("single_role") == "controlled_sa_hga-qwen3_1.7b-multi_lora-math"
    assert name("voter_adapters_rubric") == "voting_v2-qwen3_1.7b-multi_lora-math"
    assert name("voter_adapters_broadcast_sainit") == "voting_v2-qwen3_1.7b_s300-multi_lora-math"
    assert name("t1_length_weight") == "voting-qwen3_1.7b-share_policy-math-role_len_weight"
    assert name("t2_merged_minibatch") == "voting-qwen3_1.7b-multi_lora-math-merged_minibatch"
    assert name("kl_anchor_evaluator") == "evaluator_optimizer-qwen3_1.7b-multi_lora-math-i1_kl_evaluator_1"
    assert name("cap_8192") == "evaluator_optimizer-qwen3_1.7b-multi_lora-l8192-math"
    assert name("sp_extension") == "voting-qwen3_1.7b-share_policy-math"


def test_base_model_tag_sets_path_and_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    ov = _as_dict(build_overrides(_control_spec("voter_adapters_broadcast_sainit")))
    assert ov["actor_rollout_ref.model.path"] == "checkpoints/init_weight/math/qwen3_1.7b_s300"
    assert ov["trainer.base_model_tag"] == "qwen3_1.7b_s300"
    monkeypatch.setenv("MARL_INIT_WEIGHT_DIR", "/w")
    ov = _as_dict(build_overrides(_control_spec("voter_adapters_broadcast_sainit")))
    assert ov["actor_rollout_ref.model.path"] == "/w/math/qwen3_1.7b_s300"


def test_cap_8192_defaults_to_four_gpus_unless_given() -> None:
    spec = _control_spec("cap_8192")
    assert _as_dict(build_overrides(spec))["trainer.n_gpus_per_node"] == "4"
    spec.n_gpus = 2
    assert _as_dict(build_overrides(spec))["trainer.n_gpus_per_node"] == "2"


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"workflow": "evaluator_optimizer", "control": "per_voter_reward"}, "--workflow voting"),
        ({"control": "t1_length_weight"}, "--policy sp"),
        ({"control": "cap_8192", "workflow": "evaluator_optimizer"}, "--resume-from"),
        ({"control": "init_from_sa_adapters"}, "--init-adapter"),
        ({"control": "nope"}, "Unknown control"),
        ({"task": "deepcoder", "control": "per_voter_reward"}, "--task math"),
        ({"workflow": "single_agent", "share_policy": True}, "share_policy=True"),
    ],
)
def test_invalid_specs(kwargs: dict, message: str) -> None:
    base = {"task": "math", "workflow": "voting", "share_policy": False, "scale": "1.7B"}
    with pytest.raises(ValueError, match=message):
        build_overrides(LaunchSpec(**{**base, **kwargs}))


def test_normalization() -> None:
    assert launch.normalize_scale("qwen3_1.7b") == "1.7B"
    assert launch.normalize_scale("qwen3_4b_s300") == "4B"
    assert launch.normalize_gpu_type("l40s") == "L40S"
    assert launch.normalize_task("code") == "deepcoder"
    with pytest.raises(ValueError):
        launch.normalize_scale("8B")


def test_gpu_policy() -> None:
    launch.check_gpu_policy("4B", "H200")
    launch.check_gpu_policy("1.7B", "L40S")
    with pytest.raises(ValueError):
        launch.check_gpu_policy("4B", "L40S")
    with pytest.raises(ValueError):
        launch.check_gpu_policy("0.6B", "A40")


@pytest.mark.parametrize("conf", sorted((REPO / "configs" / "slurm").glob("*.conf")), ids=lambda p: p.name)
def test_slurm_confs(conf: Path) -> None:
    meta, directives = launch.read_slurm_conf(conf)
    assert launch.normalize_gpu_type(meta["GPU_TYPE"]) in ("H100", "H200", "L40S")
    for key in ("CPUS_PER_GPU_MATH", "MEM_PER_GPU_MATH", "CPUS_PER_GPU_CODE", "MEM_PER_GPU_CODE"):
        assert key in meta
    assert meta["CPUS_PER_GPU_CODE"] == "16"
    assert meta["MEM_PER_GPU_CODE"] == ("64G" if meta["GPU_TYPE"] == "L40S" else "80G")
    assert any(d.startswith("#SBATCH --partition=") for d in directives)
    assert not any("--gres" in d or "--mem-per-gpu" in d for d in directives)


def test_meta_env_exports() -> None:
    assert launch.meta_exports({"GPU_TYPE": "L40S", "ENV_NCCL_P2P_DISABLE": "1"}) == "export NCCL_P2P_DISABLE=1"
    assert launch.meta_exports({"GPU_TYPE": "H100"}) == ""
    spec = LaunchSpec(task="math", workflow="voting", share_policy=False, scale="1.7B")
    l40s = launch.render_sbatch(spec, REPO / "configs" / "slurm" / "preempt_l40s.conf", root=ROOT_PLACEHOLDER)
    h100 = launch.render_sbatch(spec, REPO / "configs" / "slurm" / "dgxh_preempt_h100.conf", root=ROOT_PLACEHOLDER)
    assert "\nexport NCCL_P2P_DISABLE=1\n" in l40s
    assert "NCCL_P2P_DISABLE" not in h100
    assert l40s.index("source scripts/env.sh") < l40s.index("export NCCL_P2P_DISABLE=1") < l40s.index("python -m")


def test_resources_from_conf() -> None:
    spec = LaunchSpec(task="deepcoder", workflow="voting", share_policy=False, scale="1.7B")
    meta, _ = launch.read_slurm_conf(REPO / "configs" / "slurm" / "preempt_l40s.conf")
    spec = launch.apply_slurm_meta(spec, meta)
    assert spec.gpu_type == "L40S"
    assert launch.resolve_resources(spec, meta) == (16, "64G")
    ov = _as_dict(build_overrides(spec))
    assert ov["actor_rollout_ref.actor.ppo_max_token_len_per_gpu"] == "23554"


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def test_cli_print(capsys: pytest.CaptureFixture[str]) -> None:
    rc = launch.cli(
        [
            "--task",
            "code",
            "--workflow",
            "orchestrator_workers_propose",
            "--policy",
            "sp",
            "--scale",
            "0.6B",
            "--extra",
            "trainer.total_training_steps=7 trainer.save_freq=1",
            "--print",
            "trainer.test_freq=-1",
        ]
    )
    assert rc == 0
    argv = shlex.split(capsys.readouterr().out.strip())
    assert argv[:3] == ["python", "-m", launch.ENTRY_MODULE]
    ov = _as_dict(argv[3:])
    assert ov["marl.task"] == "deepcoder"
    assert ov["trainer.total_training_steps"] == "7"
    assert ov["trainer.test_freq"] == "-1"
    assert ov["trainer.agent_names"] == "[orchestrator,worker,synthesizer]"


def test_cli_init_adapter(capsys: pytest.CaptureFixture[str]) -> None:
    rc = launch.cli(
        "--task math --workflow orchestrator_workers_propose --policy ip --scale 1.7B "
        "--init-adapter generator=/a/lora_adapter_generator --print".split()
    )
    assert rc == 0
    ov = _as_dict(shlex.split(capsys.readouterr().out)[3:])
    assert ov["rllm.workflow.initial_lora_weights.generator"] == "/a/lora_adapter_generator"


def test_cli_slurm_refuses_4b_on_l40s(capsys: pytest.CaptureFixture[str]) -> None:
    rc = launch.cli(
        [
            *"--task math --workflow voting --policy ip --scale 4B --slurm-config".split(),
            str(REPO / "configs" / "slurm" / "preempt_l40s.conf"),
        ]
    )
    assert rc == 2
    assert "4B training is not run on L40S" in capsys.readouterr().err


def test_cli_chain_adds_wallclock_stop(capsys: pytest.CaptureFixture[str]) -> None:
    rc = launch.cli(
        [
            *"--task math --workflow voting --policy ip --scale 1.7B --chain 3 --slurm-config".split(),
            str(REPO / "configs" / "slurm" / "dgxh_preempt_h100.conf"),
        ]
    )
    assert rc == 0  # without --submit the script is printed
    out = capsys.readouterr().out
    assert "+trainer.wallclock_stop.enable=true" in out
    assert "#SBATCH --gres=gpu:2" in out and "#SBATCH --mem-per-gpu=64G" in out


def test_cli_submit_needs_slurm_config(capsys: pytest.CaptureFixture[str]) -> None:
    rc = launch.cli("--task math --workflow voting --policy ip --scale 1.7B --submit".split())
    assert rc == 2


def test_cli_requires_cell_or_paper_run() -> None:
    with pytest.raises(SystemExit):
        launch.cli(["--task", "math", "--print"])


def test_cli_eval(capsys: pytest.CaptureFixture[str]) -> None:
    rc = launch.cli(
        [
            "eval",
            "--slurm-config",
            str(REPO / "configs" / "slurm" / "dgxh.conf"),
            "--task",
            "code",
            "--",
            "--task-type",
            "deepcoder",
            "--dataset",
            "deepcoder_primeintellect",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "python -m marl_workflows.eval.evaluate_checkpoints --task-type deepcoder" in out
    assert "#SBATCH --cpus-per-gpu=16" in out and "ulimit -n 1048576" in out


# --------------------------------------------------------------------------------------
# Hydra composition (needs hydra and the verl config package; no GPU)
# --------------------------------------------------------------------------------------


def _all_specs() -> list[tuple[str, LaunchSpec]]:
    specs = _cell_specs("math") + _cell_specs("deepcoder") + _option_specs()
    specs += [(f"control {name}", _control_spec(name)) for name in CONTROLS]
    runs = launch.load_paper_runs(REPO / "configs" / "paper_runs.yaml")
    specs += [(f"paper run {rid}", launch.spec_from_paper_run(rid, runs)) for rid in runs]
    return specs


@pytest.mark.stack
def test_overrides_compose_with_the_trainer_config() -> None:
    hydra = pytest.importorskip("hydra")
    if not (REPO / "marl_workflows" / "configs" / "multi_agent_ppo_trainer.yaml").is_file():
        pytest.skip("trainer config not present")
    with hydra.initialize_config_module(config_module="marl_workflows.configs", version_base=None):
        for label, spec in _all_specs():
            cfg = hydra.compose(config_name="multi_agent_ppo_trainer", overrides=build_overrides(spec))
            assert cfg.marl.task == spec.task, label
            assert list(cfg.trainer.agent_names) == list(get_recipe(spec.task, spec.workflow).agent_names), label
