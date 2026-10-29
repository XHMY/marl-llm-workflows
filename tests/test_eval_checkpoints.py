"""Evaluation: checkpoint discovery, experiment names, the eval_results.jsonl schema, pass@k, the driver."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import pytest

from marl_workflows.eval.checkpoints import (
    CheckpointInfo,
    EvalMode,
    create_synthetic_checkpoints,
    detect_adapter_layout,
    discover_checkpoints,
    filter_last_checkpoints,
    group_checkpoints_by_model,
    parse_experiment_name,
)
from marl_workflows.eval.evaluate_checkpoints import (
    EvalResult,
    compute_n_parallel,
    compute_rollout_metrics,
    get_checkpoint_key,
    load_existing_results,
    resolve_lengths,
    save_results_to_json,
    setup_lora_for_eval,
)
from marl_workflows.eval.vllm_server import VLLMServerManager, lora_modules_for_actor
from marl_workflows.recipes import get_recipe

pytestmark = pytest.mark.stack

# A real line of the research fork's checkpoints/rllm-workflow-MARL-v2/voting-qwen3_0.6b-multi_lora-math/
# eval_results.jsonl (step 300, canonical Math validator), per_problem_n_correct truncated to 5 entries.
FORK_LINE = {
    "timestamp": "2026-05-06T01:35:46.673843",
    "hostname": "cn-w-2.hpc.engr.oregonstate.edu",
    "dataset": "dapo_math",
    "experiment_name": "voting-qwen3_0.6b-multi_lora-math",
    "checkpoint_step": 300,
    "workflow_type": "voting",
    "model_size": "0.6b",
    "model_name": "qwen3_0.6b",
    "share_policy": False,
    "accuracy": 0.2578,
    "num_correct": 364,
    "num_total": 1412,
    "eval_duration_seconds": 9450.5,
    "eval_mode": "trained_checkpoint",
    "length_mode": "training",
    "max_prompt_length": 20480,
    "max_response_length": 5120,
    "n_rollouts": 1,
    "mean_accuracy": 0.2578,
    "std_accuracy": 0.0,
    "pass_at_n": 0.2578,
    "per_problem_n_correct": [0, 0, 0, 0, 0],
    "pass_at_k": {"1": 0.2578},
}
FORK_EVAL_JSONL = Path(
    "/nfs/hpc/share/zengyif/workspace/rllm_0.2.1/checkpoints/rllm-workflow-MARL-v2/"
    "voting-qwen3_0.6b-multi_lora-math/eval_results.jsonl"
)


def _assert_same_schema(produced: dict[str, Any], reference: dict[str, Any]) -> None:
    assert list(produced) == list(reference)  # same keys, same order
    for key, value in reference.items():
        assert type(produced[key]) is type(value), (key, type(produced[key]), type(value))
    for key in ("per_problem_n_correct",):
        assert all(isinstance(x, int) for x in produced[key])
    assert all(isinstance(k, str) and isinstance(v, float) for k, v in produced["pass_at_k"].items())


# --------------------------------------------------------------------------- names


@pytest.mark.parametrize(
    ("name", "workflow", "size", "model_name", "sp"),
    [
        ("voting-qwen3_0.6b-multi_lora-math", "voting", "0.6b", "qwen3_0.6b", False),
        ("voting_v2-qwen3_4b-share_policy-math", "voting_v2", "4b", "qwen3_4b", True),
        ("evaluator_optimizer-qwen3_1.7b-multi_lora-l8192-math", "evaluator_optimizer", "1.7b", "qwen3_1.7b", False),
        (
            "orchestrator_workers-qwen3_1.7b-multi_lora-math",
            "orchestrator_workers_propose",
            "1.7b",
            "qwen3_1.7b",
            False,
        ),
        ("controlled_sa_hga-qwen3_1.7b-multi_lora-math", "single_agent", "1.7b", "qwen3_1.7b", False),
        ("voting-qwen3_1.7b-multi_lora-math-prompt_het", "voting", "1.7b", "qwen3_1.7b", False),
        ("single_agent-qwen3_1.7b_s300-multi_lora-math", "single_agent", "1.7b", "qwen3_1.7b_s300", False),
    ],
)
def test_parse_experiment_name(name: str, workflow: str, size: str, model_name: str, sp: bool) -> None:
    parsed = parse_experiment_name(name)
    assert parsed is not None
    assert (parsed["workflow_type"], parsed["model_size"], parsed["model_name"], parsed["share_policy"]) == (
        workflow,
        size,
        model_name,
        sp,
    )
    assert parsed["base_model"] == {"0.6b": "Qwen/Qwen3-0.6B", "1.7b": "Qwen/Qwen3-1.7B", "4b": "Qwen/Qwen3-4B"}[size]


def test_parse_experiment_name_rejects() -> None:
    assert parse_experiment_name("voting-llama_8b-multi_lora-math") is None
    assert parse_experiment_name("frozenlake-qwen3_0.6b") is None
    assert parse_experiment_name("voting-qwen3_8b-multi_lora-math") is None
    with pytest.warns(UserWarning):
        assert parse_experiment_name("evaluator_optimizer_v2-qwen3_1.7b-multi_lora-math") is None
    code = parse_experiment_name("voting-qwen3_0.6b-multi_lora-deepcoder_primeintellect")
    assert code is not None and code["recipe"].task == "deepcoder"


def test_parse_every_fork_experiment_name(repo_root: Path) -> None:
    names = [
        line.strip()
        for line in (repo_root / "tests" / "fixtures" / "experiment_names.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    parsed = {n: parse_experiment_name(n) for n in names if "evaluator_optimizer_v2" not in n}
    unparsed = [n for n, p in parsed.items() if p is None and "qwen3_" in n]
    assert not unparsed


# --------------------------------------------------------------------------- discovery


def _adapter(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{}")
    (path / "adapter_model.safetensors").write_bytes(b"")


@pytest.fixture
def ckpt_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "ckpts"
    ip = root / "voting-qwen3_0.6b-multi_lora-math"
    for step in (10, 20):
        for role in ("generator", "aggregator"):
            _adapter(ip / f"global_step_{step}" / "actor" / f"lora_adapter_{role}")
    _adapter(ip / "global_step_30" / "actor" / "lora_adapter_generator")  # incomplete: skipped
    (ip / "global_step_40" / "actor").mkdir(parents=True)  # no adapter: skipped
    (ip / "latest_checkpointed_iteration.txt").write_text("20")
    sp = root / "evaluator_optimizer-qwen3_1.7b-share_policy-math"
    _adapter(sp / "global_step_50" / "actor" / "lora_adapter")
    legacy = root / "orchestrator_workers_propose-qwen3_4b-multi_lora-math"
    for role in ("orchestrator", "worker", "synthesizer"):
        _adapter(legacy / "global_step_100" / f"lora_adapter_{role}")  # no actor/ level
    s300 = root / "single_agent-qwen3_1.7b_s300-multi_lora-math"
    _adapter(s300 / "global_step_10" / "actor" / "lora_adapter_generator")
    (s300 / "training_metadata.json").write_text(json.dumps({"global_steps": 10, "base_model": "qwen3_1.7b_s300"}))
    init = tmp_path / "init_weight"
    (init / "math" / "qwen3_1.7b_s300").mkdir(parents=True)
    monkeypatch.setenv("MARL_INIT_WEIGHT_DIR", str(init))
    _adapter(
        root / "voting-qwen3_0.6b-multi_lora-deepcoder_primeintellect" / "global_step_10" / "actor" / "lora_adapter"
    )
    (root / "notes.txt").write_text("not an experiment")
    return root


def test_discover_checkpoints(ckpt_tree: Path) -> None:
    cps = discover_checkpoints(ckpt_tree, task_type="math")
    got = {(c.experiment_name, c.checkpoint_step): c for c in cps}
    assert set(got) == {
        ("voting-qwen3_0.6b-multi_lora-math", 10),
        ("voting-qwen3_0.6b-multi_lora-math", 20),
        ("evaluator_optimizer-qwen3_1.7b-share_policy-math", 50),
        ("orchestrator_workers_propose-qwen3_4b-multi_lora-math", 100),
        ("single_agent-qwen3_1.7b_s300-multi_lora-math", 10),
    }
    ip = got[("voting-qwen3_0.6b-multi_lora-math", 10)]
    assert ip.share_policy is False and ip.actor_path.endswith("global_step_10/actor")
    assert got[("evaluator_optimizer-qwen3_1.7b-share_policy-math", 50)].share_policy is True
    legacy = got[("orchestrator_workers_propose-qwen3_4b-multi_lora-math", 100)]
    assert legacy.actor_path.endswith("global_step_100") and legacy.base_model == "Qwen/Qwen3-4B"
    s300 = got[("single_agent-qwen3_1.7b_s300-multi_lora-math", 10)]
    assert s300.model_name == "qwen3_1.7b_s300"
    assert s300.base_model == os.path.join(os.environ["MARL_INIT_WEIGHT_DIR"], "math", "qwen3_1.7b_s300")

    code = discover_checkpoints(ckpt_tree, task_type="deepcoder")
    assert [(c.experiment_name, c.share_policy) for c in code] == [
        ("voting-qwen3_0.6b-multi_lora-deepcoder_primeintellect", True)
    ]
    filtered = discover_checkpoints(ckpt_tree, experiment_filter="^voting", step_filter=[20], task_type="math")
    assert [(c.experiment_name, c.checkpoint_step) for c in filtered] == [("voting-qwen3_0.6b-multi_lora-math", 20)]
    last = filter_last_checkpoints(cps)
    assert {(c.experiment_name, c.checkpoint_step) for c in last if c.experiment_name.startswith("voting")} == {
        ("voting-qwen3_0.6b-multi_lora-math", 20)
    }
    assert set(group_checkpoints_by_model(cps)) == {
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B",
        s300.base_model,
    }


def test_detect_adapter_layout_prefers_actor(tmp_path: Path) -> None:
    _adapter(tmp_path / "global_step_1" / "actor" / "lora_adapter")
    _adapter(tmp_path / "global_step_1" / "lora_adapter_generator")
    path, sp = detect_adapter_layout(tmp_path / "global_step_1", ["generator"])
    assert path.name == "actor" and sp is True
    assert detect_adapter_layout(tmp_path, ["generator"]) is None


def test_synthetic_checkpoints() -> None:
    cps = create_synthetic_checkpoints(
        EvalMode.BASE_MODEL, "deepcoder", "Qwen/Qwen3-1.7B", ["voting", "single_agent", "nonexistent"]
    )
    assert [c.experiment_name for c in cps] == [
        "voting-qwen3_1.7b-base_model-deepcoder_primeintellect",
        "single_agent-qwen3_1.7b-base_model-deepcoder_primeintellect",
    ]
    assert all(c.checkpoint_step == 0 and c.actor_path is None for c in cps)
    tr = create_synthetic_checkpoints(
        EvalMode.SINGLE_AGENT_GENERATOR_TRANSFER, "math", "Qwen/Qwen3-4B", ["orchestrator_workers_propose"], "/sa"
    )
    assert tr[0].experiment_name == "orchestrator_workers_propose-qwen3_4b-single_agent_generator_transfer-math"
    assert tr[0].actor_path == "/sa"


# --------------------------------------------------------------------------- adapters / server


class FakeServer:
    def __init__(self) -> None:
        self.loaded: dict[str, str] = {}

    def load_lora(self, name: str, path: str) -> None:
        self.loaded[name] = path

    def unload_all_loras(self) -> None:
        self.loaded.clear()


class FakeEngine:
    def __init__(self) -> None:
        self.lora_names: dict[str, str] = {}
        self.max_prompt_length = 20480
        self.max_response_length = 5120

    def set_lora_names(self, names: dict[str, str]) -> None:
        self.lora_names = dict(names)

    def model_for(self, agent: str | None) -> str:
        return self.lora_names.get(agent or "", self.lora_names.get("default", "base"))


def test_setup_lora_modes(ckpt_tree: Path) -> None:
    ip = next(
        c for c in discover_checkpoints(ckpt_tree, task_type="math") if c.experiment_name.startswith("orchestrator")
    )
    recipe = get_recipe("math", ip.workflow_type)
    server, engine = FakeServer(), FakeEngine()
    mapping = setup_lora_for_eval(ip, server, engine, EvalMode.TRAINED_CHECKPOINT, recipe)
    prefix = f"{ip.experiment_name}_step100"
    assert mapping == {r: f"{prefix}_{r}" for r in ("orchestrator", "worker", "synthesizer")}
    assert server.loaded[f"{prefix}_worker"].endswith("lora_adapter_worker")

    gen = setup_lora_for_eval(ip, FakeServer(), engine, EvalMode.SINGLE_AGENT_GENERATOR_TRANSFER, recipe, "/sa")
    assert gen == {"worker": f"{prefix}_gen_transfer"}
    assert engine.model_for("orchestrator") == "base"  # supervisor roles run on the base model
    allr = setup_lora_for_eval(ip, FakeServer(), engine, EvalMode.SINGLE_AGENT_TRANSFER, recipe, "/sa")
    assert set(allr) == {"orchestrator", "worker", "synthesizer"}
    assert setup_lora_for_eval(ip, FakeServer(), engine, EvalMode.BASE_MODEL, recipe) == {}
    sp = CheckpointInfo("e", "voting", "0.6b", "qwen3_0.6b", "Qwen/Qwen3-0.6B", 5, "/x/actor", True)
    shared = setup_lora_for_eval(sp, FakeServer(), engine, EvalMode.TRAINED_CHECKPOINT, get_recipe("math", "voting"))
    assert shared == {"generator": "e_step5_shared", "aggregator": "e_step5_shared"}
    with pytest.raises(ValueError):
        setup_lora_for_eval(ip, FakeServer(), engine, EvalMode.SINGLE_AGENT_TRANSFER, recipe, None)


def test_server_command_and_modules(ckpt_tree: Path) -> None:
    actor = ckpt_tree / "voting-qwen3_0.6b-multi_lora-math" / "global_step_10" / "actor"
    modules = lora_modules_for_actor(actor, ["generator", "aggregator"], share_policy=False)
    assert modules == {r: str(actor / f"lora_adapter_{r}") for r in ("generator", "aggregator")}
    with pytest.raises(FileNotFoundError):
        lora_modules_for_actor(actor, ["generator"], share_policy=True)
    cmd = VLLMServerManager("Qwen/Qwen3-0.6B", max_loras=4, max_model_len=25600, lora_modules=modules).build_command()
    assert cmd[1:3] == ["-m", "vllm.entrypoints.openai.api_server"]
    joined = " ".join(cmd)
    assert "--enable-lora --max-loras 4 --max-lora-rank 64" in joined
    assert f"--lora-modules generator={modules['generator']} aggregator={modules['aggregator']}" in joined
    assert "--max-model-len 25600" in joined
    base = VLLMServerManager("m", enable_lora=False).build_command()
    assert "--enable-lora" not in base and "--max-model-len" not in base


# --------------------------------------------------------------------------- metrics / schema


def test_compute_rollout_metrics_pass_at_k() -> None:
    runs = [[True, False, True, False], [False] * 4, [True] * 4]
    m = compute_rollout_metrics(runs, uid_offset=7)
    assert m["per_problem_n_correct"] == [2, 0, 4]
    assert m["problem_results"][0]["uid"] == "eval_7"
    assert m["mean_accuracy"] == pytest.approx(6 / 12)
    per_rollout = [2 / 3, 1 / 3, 2 / 3, 1 / 3]
    assert m["std_accuracy"] == pytest.approx(math.sqrt(sum((a - 0.5) ** 2 for a in per_rollout) / 4))
    assert m["pass_at_n"] == pytest.approx(2 / 3)
    assert set(m["pass_at_k"]) == {1, 2, 4}
    assert m["pass_at_k"][1] == pytest.approx((0.5 + 0 + 1) / 3)
    assert m["pass_at_k"][2] == pytest.approx(((1 - math.comb(2, 2) / math.comb(4, 2)) + 0 + 1) / 3)
    assert m["pass_at_k"][4] == pytest.approx(2 / 3)
    single = compute_rollout_metrics([[True], [False], [False], [True]])
    assert single["pass_at_k"] == {1: 0.5} and single["std_accuracy"] == 0.0 and single["num_correct"] == 2


def test_eval_result_schema_matches_fork_line() -> None:
    result = EvalResult(
        experiment_name="voting-qwen3_0.6b-multi_lora-math",
        checkpoint_step=300,
        dataset="dapo_math",
        workflow_type="voting",
        model_size="0.6b",
        model_name="qwen3_0.6b",
        share_policy=False,
        accuracy=1 / 3,
        num_correct=1,
        num_total=3,
        eval_duration_seconds=12.345,
        eval_mode="trained_checkpoint",
        length_mode="training",
        max_prompt_length=20480,
        max_response_length=5120,
        mean_accuracy=1 / 3,
        pass_at_n=1 / 3,
        per_problem_n_correct=[1, 0, 0],
        pass_at_k={1: 1 / 3},
    )
    d = result.to_dict()
    _assert_same_schema(d, FORK_LINE)
    assert d["accuracy"] == 0.3333 and d["eval_duration_seconds"] == 12.3 and d["pass_at_k"] == {"1": 0.3333}
    if FORK_EVAL_JSONL.exists():  # the live fork file, when this checkout can see it
        lines = [json.loads(x) for x in FORK_EVAL_JSONL.read_text().splitlines() if x.strip()]
        standard = [x for x in lines if "eval_variant" not in x and x.get("n_rollouts") == 1]
        _assert_same_schema(d, standard[-1])


def test_results_jsonl_and_dedup_keys(tmp_path: Path) -> None:
    out = tmp_path / "sub" / "eval.jsonl"
    cp = CheckpointInfo("exp", "voting", "0.6b", "qwen3_0.6b", "Qwen/Qwen3-0.6B", 10, "/a", False)
    r = EvalResult(
        "exp", 10, "dapo_math", "voting", "0.6b", "qwen3_0.6b", False, 0.5, 1, 2, 1.0, length_mode="training"
    )
    save_results_to_json([r], str(out))
    out.write_text(out.read_text() + "not json\n")
    keys = load_existing_results(str(out))
    assert get_checkpoint_key(cp, "dapo_math", "trained_checkpoint", 1, "training", 2) in keys
    # A truncated run never masks the full evaluation of the same checkpoint.
    assert get_checkpoint_key(cp, "dapo_math", "trained_checkpoint", 1, "training", 1412) not in keys


def test_lengths_and_parallelism() -> None:
    train = resolve_lengths("math", True)
    assert train["voting"] == (20480, 5120) and train["evaluator_optimizer"] == (30720, 5120)
    assert resolve_lengths("math", False)["voting"] == (25600, 8192)
    assert resolve_lengths("deepcoder", False)["single_agent"] == (4096, 5120)
    assert compute_n_parallel("L40s", "1.7b", "evaluator_optimizer") == 512
    assert compute_n_parallel(None, "0.6b", "voting") == 384
    assert compute_n_parallel("RTX8000", "4b", "voting") == 64


# --------------------------------------------------------------------------- driver end to end


class FakeWorkflow:
    """Stands in for a recipe workflow: one generator0 trajectory, correct iff the answer is even."""

    seen_models: list[str] = []

    def __init__(self, rollout_engine: Any, **kwargs: Any) -> None:
        self.engine = rollout_engine
        self.kwargs = kwargs

    async def run_with_termination_handling(self, task: dict, uid: str) -> Any:
        from rllm.agents.agent import Episode, Step, Trajectory
        from rllm.engine.rollout.rollout_engine import ModelOutput
        from rllm.workflows.workflow import TerminationReason

        FakeWorkflow.seen_models.append(self.engine.model_for("generator"))
        out = ModelOutput(text="t", content="c", reasoning="", prompt_ids=[1, 2], completion_ids=[3], logprobs=[-0.1])
        step = Step(prompt_ids=[1, 2], response_ids=[3], logprobs=[-0.1], model_output=out, model_response="c")
        traj = Trajectory(name="generator0", task=task, steps=[step])
        return Episode(
            id=uid,
            task=task,
            termination_reason=TerminationReason.ENV_DONE,
            is_correct=int(task["ground_truth"]) % 2 == 0,
            trajectories=[traj],
        )


def test_main_end_to_end(ckpt_tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import marl_workflows.eval.evaluate_checkpoints as ev
    import marl_workflows.eval.vllm_server as vs
    from marl_workflows.data.prepare_smoke import load_fixture, prepare_smoke

    prepare_smoke()  # registers smoke_math under the per-test MARL_DATA_DIR (conftest)

    class StubServer(VLLMServerManager):
        def start(self, timeout: int = 600) -> None:
            self.started = True

        def load_lora(self, lora_name: str, lora_path: str) -> None:
            assert os.path.isdir(lora_path)
            self.loaded_loras.add(lora_name)

        def unload_lora(self, lora_name: str) -> None:
            self.loaded_loras.discard(lora_name)

    def fake_load_object(path: str) -> Any:
        return FakeWorkflow if path.endswith("Workflow") else (lambda *a, **k: None)

    monkeypatch.setattr(vs, "VLLMServerManager", StubServer)
    monkeypatch.setattr(ev, "load_object", fake_load_object)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: None)
    monkeypatch.setattr(ev.signal, "signal", lambda *a, **k: None)
    FakeWorkflow.seen_models = []

    out = tmp_path / "eval.jsonl"
    traj = tmp_path / "traj"
    argv = [
        "--task-type", "math",
        "--checkpoints-dir", str(ckpt_tree),
        "--experiment-filter", "^voting-qwen3_0.6b-multi_lora-math$",
        "--step-filter", "10", "20",
        "--dataset", "smoke_math",
        "--use-training-lengths",
        "--max-samples", "5",
        "--n-parallel", "4",
        "--output-json", str(out),
        "--trajectory-output-dir", str(traj),
    ]  # fmt: skip
    results = ev.main(ev.parse_args(argv))
    assert [r.checkpoint_step for r in results] == [10, 20]

    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(lines) == 2
    _assert_same_schema(lines[0], FORK_LINE)
    rows = load_fixture()[32:37]  # smoke_math test split = fixture rows 32..39
    expected = [int(int(r["ground_truth"]) % 2 == 0) for r in rows]
    assert lines[0]["per_problem_n_correct"] == expected
    assert lines[0]["num_total"] == 5 and lines[0]["length_mode"] == "training"
    assert (lines[0]["max_prompt_length"], lines[0]["max_response_length"]) == (20480, 5120)
    assert lines[0]["experiment_name"] == "voting-qwen3_0.6b-multi_lora-math" and lines[0]["share_policy"] is False
    assert set(FakeWorkflow.seen_models) == {
        "voting-qwen3_0.6b-multi_lora-math_step10_generator",
        "voting-qwen3_0.6b-multi_lora-math_step20_generator",
    }

    step_dir = traj / "voting-qwen3_0.6b-multi_lora-math" / "smoke_math" / "step_10"
    assert sorted(p.name for p in step_dir.iterdir()) == [f"eval_{i}.json" for i in range(5)]
    dump = json.loads((step_dir / "eval_0.json").read_text())
    assert dump["id"] == "eval_0" and dump["trajectories"][0]["name"] == "generator0"
    s = dump["trajectories"][0]["steps"][0]
    assert "prompt_ids" not in s and "response_ids" not in s and "logprobs" not in s
    # Key layout of a real fork dump (evaluation_trajectories/voting-qwen3_0.6b-multi_lora-math/
    # dapo_math_full/step_*/eval_0.json).
    assert list(dump) == ["id", "task", "termination_reason", "is_correct", "trajectories", "metrics", "info"]
    assert list(dump["trajectories"][0]) == ["uid", "name", "task", "steps", "reward", "info"]
    assert list(s) == [
        "chat_completions", "observation", "thought", "action", "model_response",
        "model_output", "info", "reward", "done", "mc_return",
    ]  # fmt: skip
    assert list(s["model_output"]) == [
        "text", "content", "reasoning", "tool_calls", "multi_modal_inputs", "logprobs",
        "prompt_logprobs", "prompt_length", "completion_length", "finish_reason",
    ]  # fmt: skip

    # Re-running skips both checkpoints (same key incl. the problem count).
    assert ev.main(ev.parse_args(argv)) == []


def test_error_episodes_abort(ckpt_tree: Path) -> None:
    import asyncio

    import marl_workflows.eval.evaluate_checkpoints as ev

    class ErrorWorkflow(FakeWorkflow):
        async def run_with_termination_handling(self, task: dict, uid: str) -> Any:
            from rllm.workflows.workflow import TerminationReason

            ep = await super().run_with_termination_handling(task, uid)
            ep.termination_reason = TerminationReason.ERROR
            ep.info = {"error": {"error_message": "connection refused"}}
            return ep

    cp = CheckpointInfo("exp", "voting", "0.6b", "qwen3_0.6b", "Qwen/Qwen3-0.6B", 1, None, True)
    with pytest.raises(RuntimeError, match="connection refused"):
        asyncio.run(
            ev.evaluate_checkpoint(
                cp,
                FakeServer(),
                FakeEngine(),
                [{"ground_truth": "2"}],
                task="math",
                eval_mode=EvalMode.BASE_MODEL,
                workflow_cls=ErrorWorkflow,
                workflow_kwargs={},
            )
        )
