#!/usr/bin/env python
"""Build ``configs/paper_runs.yaml`` from the research fork's wandb cache.

One-off script; its output is committed. For every canonical cell (``analysis/_lib/paper_table.csv``
in the fork) and every control run, it reads ``results/wandb_cache/<run_id>/config.json`` and
``history.parquet`` and writes one entry with the launcher inputs that reproduce the run:
task, workflow, policy, scale, control, GPU count, token budget, step count and the learning-rate
warmup. Values that the launcher would otherwise set differently are added as explicit
``overrides`` so that ``scripts/launch.py --paper-run <id> --print`` equals the logged config on
every key it sets.

The warmup is read from the logged LR series, not from the config: the wandb config holds the
values of the run's last resume segment, which can report a warmup the run never took. When a
run's history does not start at step 1, the predecessor named in the checkpoint's
``training_metadata.json`` (``resumed_from_run_id``) is read instead.

Usage::

    python scripts/export_paper_runs.py [--fork /path/to/fork] [--out configs/paper_runs.yaml] [--markdown]

If the fork or its wandb cache is absent the script says so and exits with status 1.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from marl_workflows import launch  # noqa: E402

DEFAULT_FORK = Path("/nfs/hpc/share/zengyif/workspace/rllm_0.2.1")
CKPT_PROJECT = "rllm-workflow-MARL-v2"
PEAK_LR = 2e-5
REL_TOL = 1e-3

GRID_WORKFLOWS = ("voting", "evaluator_optimizer", "orchestrator_workers_propose", "single_agent")
DISPLAY = {
    "voting": "Voting",
    "evaluator_optimizer": "Eval-Opt",
    "orchestrator_workers_propose": "Orch-Workers",
    "single_agent": "SA-RL",
}
TASK_DISPLAY = {"math": "Math", "deepcoder": "Code"}

# Control runs: run id -> (control, task, workflow, share_policy, scale).
CONTROL_RUNS: dict[str, tuple[str, str, str, bool, str]] = {
    "lottt3pm": ("per_voter_reward", "math", "voting", False, "1.7B"),
    "nsn0gxle": ("evaluator_accuracy_reward", "math", "evaluator_optimizer", False, "1.7B"),
    "5u3lopo4": ("prompt_het", "math", "voting_prompt_het", False, "1.7B"),
    "2nxiq4fy": ("single_role", "math", "controlled_sa_hga", False, "1.7B"),
    "uftb3s45": ("voter_adapters_rubric", "math", "voting_v2", False, "1.7B"),
    "d1fweqim": ("voter_adapters_broadcast_sainit", "math", "voting_v2", False, "1.7B"),
    "s4hmrfca": ("t1_length_weight", "math", "voting", True, "1.7B"),
    "c0zlpk56": ("t2_merged_minibatch", "math", "voting", False, "1.7B"),
    "akyv7jo2": ("kl_anchor_evaluator", "math", "evaluator_optimizer", False, "1.7B"),
    "6p4i9pm1": ("cap_8192", "math", "evaluator_optimizer", False, "1.7B"),
}
# Canonical cells whose run is itself a control variant.
CANONICAL_CONTROL = {"nrg6etac": "sp_extension"}
# Paper-table rows that are not reproduced (not in the paper; no cached config).
SKIPPED_ROWS = {"wz5ld9nz": "per-voter reward run initialized from SA-RL weights; not cited in the paper"}
# Resume lineage that is not recorded in a checkpoint's training_metadata.json.
EXTRA_LINEAGE = {
    "6p4i9pm1": {
        "run": "iohxa6tq",
        "step": 320,
        "path": f"checkpoints/{CKPT_PROJECT}/evaluator_optimizer-qwen3_1.7b-multi_lora-math/global_step_320",
    }
}

# Hydra keys compared with the yaml defaults (marl_workflows/configs/multi_agent_ppo_trainer.yaml);
# a logged value that differs becomes an explicit override. "+" marks leaves the yaml does not declare.
TRACKED_DEFAULTS: dict[str, tuple[str, Any]] = {
    "data.train_batch_size": ("", 64),
    "actor_rollout_ref.actor.ppo_mini_batch_size": ("", 64),
    "actor_rollout_ref.actor.optim.lr": ("", 2e-5),
    "actor_rollout_ref.actor.use_kl_loss": ("", False),
    "actor_rollout_ref.actor.kl_loss_coef": ("", 0.001),
    "actor_rollout_ref.actor.kl_loss_type": ("", "low_var_kl"),
    "actor_rollout_ref.rollout.n": ("", 8),
    "actor_rollout_ref.rollout.temperature": ("", 0.7),
    "trainer.save_freq": ("", 5),
    "trainer.test_freq": ("", 10),
    "trainer.wallclock_stop.enable": ("+", None),
    "trainer.loss_weight_per_role.mode": ("+", None),
    "trainer.kl_loss_coef_per_role.evaluator": ("+", None),
    "trainer.kl_loss_coef_per_role.generator": ("+", None),
    "trainer.kl_loss_coef_per_role.aggregator": ("+", None),
    "actor_rollout_ref.rollout.engine_kwargs.vllm.logprobs_mode": ("+", None),
    "rllm.workflow.use_rubric_reward": ("+", None),
}
# Keys the launcher sets that are not compared with the logged config.
NOT_COMPARED = {
    "marl.task",
    "marl.workflow",
    "trainer.base_model_tag",  # new key; the fork wrote the tag into training_metadata.json
    "trainer.experiment_name",  # canonical names; old runs logged other names (recorded separately)
    "trainer.resume_mode",  # the logged config holds the last segment's resume settings
    "trainer.resume_from_path",
    "actor_rollout_ref.actor.optim.lr_warmup_steps",  # read from the logged LR series
}


_ABSENT = object()


def get_path(cfg: dict[str, Any], dotted: str, default: Any = KeyError) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            if default is KeyError:
                raise KeyError(dotted)
            return default
        cur = cur[part]
    return cur


def hydra_value(value: Any) -> str:
    """Format a Python value as a Hydra override value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list | tuple):
        return "[" + ",".join(hydra_value(v) for v in value) + "]"
    return str(value)


def parse_hydra_value(text: str | None) -> Any:
    """Inverse of :func:`hydra_value` for the simple values the launcher emits."""
    if text is None:
        return None
    t = text.strip()
    if t.lower() in ("true", "false"):
        return t.lower() == "true"
    if t.lower() in ("null", "none"):
        return None
    if t.startswith("[") and t.endswith("]"):
        inner = t[1:-1].strip()
        return [parse_hydra_value(x) for x in inner.split(",")] if inner else []
    for cast in (int, float):
        try:
            return cast(t)
        except ValueError:
            pass
    return t.strip("'\"")


def values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b or (isinstance(a, bool) and isinstance(b, bool) and a == b)
    if isinstance(a, int | float) and isinstance(b, int | float):
        return abs(float(a) - float(b)) <= 1e-12 * max(1.0, abs(float(a)))
    return a == b


# --------------------------------------------------------------------------------------
# Learning-rate warmup from the logged series
# --------------------------------------------------------------------------------------


def lr_series(history_path: Path) -> tuple[int | None, list[float]]:
    """(first logged training step, all LR values at that and the next steps in order)."""
    import pandas as pd

    df = pd.read_parquet(history_path)
    step_col = "training/global_step" if "training/global_step" in df.columns else "_step"
    cols = [c for c in df.columns if c == "actor/lr" or (c.startswith("actor/") and c.endswith("/lr"))]
    if not cols:
        return None, []
    sub = df[[step_col, *cols]].dropna(subset=cols, how="all").sort_values(step_col)
    if sub.empty:
        return None, []
    first = int(sub[step_col].iloc[0])
    values = [float(v) for c in cols for v in sub[c].dropna().tolist()]
    return first, values


def infer_warmup(first_step: int | None, values: list[float]) -> int | None:
    """15-style linear warmup length from the LR values, 0 for a flat LR, None if unknown.

    The scheduler (constant with linear warmup) steps once per adapter update, so consecutive
    LR values differ by ``peak / warmup``; the smallest positive gap between distinct values
    below the peak recovers ``warmup``.
    """
    if first_step is None or first_step > 1 or not values:
        return None
    below = sorted({round(v, 15) for v in values if v < PEAK_LR * (1 - REL_TOL)})
    if not below:
        return 0
    points = [*below, PEAK_LR]
    gaps = [b - a for a, b in zip(points, points[1:], strict=False) if b - a > PEAK_LR * 1e-6]
    return round(PEAK_LR / min(gaps))


# --------------------------------------------------------------------------------------
# Entries
# --------------------------------------------------------------------------------------


def checkpoint_metadata(fork: Path, exp_name: str) -> dict[str, Any]:
    path = fork / "checkpoints" / CKPT_PROJECT / exp_name / "training_metadata.json"
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def scale_from_tag(tag: str) -> str:
    return launch.normalize_scale(tag)


def canonical_rows(fork: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Canonical grid rows of the fork's paper table and the list of grid cells without a run id."""
    rows, missing = [], []
    with open(fork / "analysis" / "_lib" / "paper_table.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["workflow"] not in GRID_WORKFLOWS or r["policy"] not in ("multi_lora", "share_policy"):
                continue
            task = "deepcoder" if r["task"].startswith("deepcoder") else "math"
            expected = f"{r['workflow']}-{r['scale']}-{r['policy']}-{r['task']}"
            if r["cell"] != expected:
                continue  # control rows carry a suffix and are listed in CONTROL_RUNS
            if r["run_id"].startswith("MISSING"):
                missing.append(r["cell"])
                continue
            rows.append(
                {
                    "run_id": r["run_id"],
                    "cell_name": r["cell"],
                    "task": task,
                    "workflow": r["workflow"],
                    "share_policy": r["policy"] == "share_policy",
                    "scale": scale_from_tag(r["scale"]),
                }
            )
    return rows, missing


def cell_label(task: str, workflow: str, share_policy: bool, scale: str) -> str:
    if workflow == "single_agent":
        return f"SA-RL-{scale}-{TASK_DISPLAY[task]}"
    return f"{DISPLAY[workflow]}-{'SP' if share_policy else 'IP'}-{scale}-{TASK_DISPLAY[task]}"


def build_entry(fork: Path, run_id: str, base: dict[str, Any], control: str | None) -> dict[str, Any]:
    cache = fork / "results" / "wandb_cache" / run_id
    cfg = json.loads((cache / "config.json").read_text())
    logged_name = get_path(cfg, "trainer.experiment_name")
    meta = checkpoint_metadata(fork, logged_name)
    notes: list[str] = []

    resumed_from: dict[str, Any] | None = None
    if meta.get("resumed_from_run_id"):
        resumed_from = {"run": meta["resumed_from_run_id"], "step": meta.get("resumed_from_step")}
    elif run_id in EXTRA_LINEAGE:
        resumed_from = dict(EXTRA_LINEAGE[run_id])

    first, values = lr_series(cache / "history.parquet")
    warmup = infer_warmup(first, values)
    if warmup is None and resumed_from:
        pred = fork / "results" / "wandb_cache" / resumed_from["run"] / "history.parquet"
        if pred.is_file():
            warmup = infer_warmup(*lr_series(pred))
            notes.append(f"warmup read from the LR series of {resumed_from['run']} (resumed from it)")
    if warmup is None:
        warmup = int(get_path(cfg, "actor_rollout_ref.actor.optim.lr_warmup_steps", 0))
        notes.append("warmup taken from the logged config (LR series unavailable)")
    if resumed_from and control == "cap_8192":
        notes.append(f"continues the checkpoint of {resumed_from['run']} at step {resumed_from['step']}")
    elif resumed_from:
        notes.append(
            f"resumed from {resumed_from['run']} at step {resumed_from['step']}; "
            "a fresh launch trains the whole run in one lineage"
        )

    workers = get_path(cfg, "rllm.workflow.code_executor_workers", None)
    mode3 = int(get_path(cfg, "rllm.workflow.max_concurrent_code_execs", 0) or 0)
    if base["task"] == "deepcoder" and mode3 > 0:
        notes.append(
            f"trained with the per-test code executor (max_concurrent_code_execs={mode3}), which this package "
            "does not provide; code_executor_workers=0 selects the process pool sized to the allocated CPUs"
        )

    tag = meta.get("base_model")
    model_path = str(get_path(cfg, "actor_rollout_ref.model.path"))
    if not tag and "init_weight" in model_path:
        tag = Path(model_path).name

    entry: dict[str, Any] = {
        "cell": cell_label(base["task"], base["workflow"], base["share_policy"], base["scale"])
        if control is None or control in CANONICAL_CONTROL.values()
        else f"control:{control}",
        "experiment_name": base.get("cell_name") or logged_name,
        "task": base["task"],
        "workflow": base["workflow"],
        "share_policy": bool(base["share_policy"]),
        "scale": base["scale"],
        "base_model_tag": tag,
        "control": control,
        "n_gpus": int(get_path(cfg, "trainer.n_gpus_per_node")),
        "ppo_max_token_len_per_gpu": int(get_path(cfg, "actor_rollout_ref.actor.ppo_max_token_len_per_gpu")),
        "code_executor_workers": int(workers) if base["task"] == "deepcoder" and workers is not None else None,
        "total_training_steps": int(get_path(cfg, "trainer.total_training_steps")),
        "lr_warmup_steps": int(warmup),
        "resumed_from": resumed_from,
        "overrides": [],
    }
    if control is not None and control not in CANONICAL_CONTROL.values():
        entry["experiment_name"] = logged_name
    if entry["experiment_name"] != logged_name:
        entry["logged_experiment_name"] = logged_name
        notes.append(f"logged under the older name {logged_name}")

    entry["overrides"] = reconcile_overrides(entry, run_id, cfg)
    entry["notes"] = notes
    return entry


def reconcile_overrides(entry: dict[str, Any], run_id: str, cfg: dict[str, Any]) -> list[str]:
    """Explicit overrides that make the launcher output equal the logged config."""
    extra: list[str] = []
    for _ in range(3):  # converges in one pass; the loop guards against interacting keys
        spec = launch.spec_from_paper_run(run_id, {run_id: {**entry, "overrides": extra}})
        overrides = launch.build_overrides(spec)
        added = False
        set_keys = set()
        for ov in overrides:
            prefix, key, value = launch.parse_override(ov)
            set_keys.add(key)
            if key in NOT_COMPARED:
                continue
            logged = get_path(cfg, key, _ABSENT)
            if logged is _ABSENT:
                continue  # key not logged by the fork (reported by tests/test_paper_runs.py)
            if not values_equal(parse_hydra_value(value), logged):
                extra = launch.merge_overrides(extra, [f"{prefix}{key}={hydra_value(logged)}"])
                added = True
        for key, (prefix, default) in TRACKED_DEFAULTS.items():
            if key in set_keys:
                continue
            logged = get_path(cfg, key, None)
            if logged is not None and not values_equal(logged, default):
                extra = launch.merge_overrides(extra, [f"{prefix}{key}={hydra_value(logged)}"])
                added = True
        if not added:
            return extra
    raise RuntimeError(f"{run_id}: overrides did not converge: {extra}")


def export(fork: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    rows, missing = canonical_rows(fork)
    cache = fork / "results" / "wandb_cache"
    runs: dict[str, Any] = {}
    skipped: list[str] = []
    order = {w: i for i, w in enumerate(GRID_WORKFLOWS)}
    scales = list(launch.SCALES)
    rows.sort(key=lambda r: (r["task"] != "math", order[r["workflow"]], scales.index(r["scale"]), r["share_policy"]))
    for r in rows:
        if not (cache / r["run_id"] / "config.json").is_file():
            skipped.append(f"{r['run_id']} ({r['cell_name']}): not in the wandb cache")
            continue
        runs[r["run_id"]] = build_entry(fork, r["run_id"], r, CANONICAL_CONTROL.get(r["run_id"]))
    for run_id, (control, task, workflow, sp, scale) in CONTROL_RUNS.items():
        if not (cache / run_id / "config.json").is_file():
            skipped.append(f"{run_id} (control {control}): not in the wandb cache")
            continue
        base = {"task": task, "workflow": workflow, "share_policy": sp, "scale": scale}
        runs[run_id] = build_entry(fork, run_id, base, control)
    for run_id, reason in SKIPPED_ROWS.items():
        skipped.append(f"{run_id}: {reason}")
    return runs, missing, skipped


HEADER = """\
# Generated by scripts/export_paper_runs.py from the research fork's wandb cache; do not edit by hand.
# One entry per paper run: the launcher inputs that reproduce it (scripts/launch.py --paper-run <id>).
#   lr_warmup_steps   read from the logged learning-rate series (15: linear warmup, 0: flat)
#   overrides         run-specific Hydra overrides beyond the launcher defaults and the control
#   resumed_from      the run this one continued from (its early steps were trained there)
"""


def to_markdown(runs: dict[str, Any]) -> str:
    lines = [
        "| Cell / control | Run | Steps | Warmup | GPUs | Command |",
        "|---|---|---|---|---|---|",
    ]
    for run_id, e in runs.items():
        lines.append(
            f"| {e['cell']} | `{run_id}` | {e['total_training_steps']} | {e['lr_warmup_steps']} | {e['n_gpus']} "
            f"| `python scripts/launch.py --paper-run {run_id} --print` |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--fork", type=Path, default=Path(os.environ.get("MARL_FORK_ROOT", DEFAULT_FORK)))
    p.add_argument("--out", type=Path, default=REPO / "configs" / "paper_runs.yaml")
    p.add_argument("--markdown", action="store_true", help="also print a markdown table of the runs")
    args = p.parse_args(argv)

    table = args.fork / "analysis" / "_lib" / "paper_table.csv"
    cache = args.fork / "results" / "wandb_cache"
    if not table.is_file() or not cache.is_dir():
        print(
            f"export_paper_runs: the research fork's paper table or wandb cache is absent "
            f"({table} exists: {table.is_file()}, {cache} exists: {cache.is_dir()}); "
            "configs/paper_runs.yaml is left unchanged.",
            file=sys.stderr,
        )
        return 1

    os.environ.pop("MARL_INIT_WEIGHT_DIR", None)  # model paths are written relative, as the fork logged them
    import yaml

    runs, missing, skipped = export(args.fork)
    body = yaml.safe_dump(
        {"project": launch.PAPER_PROJECT, "runs": runs}, sort_keys=False, default_flow_style=False, width=120
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(HEADER + body)
    print(f"wrote {len(runs)} runs to {args.out}")
    if missing:
        print("grid cells without a run id: " + ", ".join(missing))
    for s in skipped:
        print(f"skipped {s}")
    if args.markdown:
        print(to_markdown(runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
