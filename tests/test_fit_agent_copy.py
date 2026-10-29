"""Copy-drift guard for the bounded copies of upstream code.

Each copied method must equal its upstream source except inside ``# >>> MARL(<id>)`` /
``# <<< MARL(<id>)`` regions. Comparison is line by line after dropping blank and
comment-only lines (indentation is kept). Upstream lines may be removed only where a region
starts or ends. The sources are parsed with ``ast``, so the training stack is not imported.
"""

from __future__ import annotations

import ast
import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
UPSTREAM_TRAINER = REPO / "third_party/rllm/rllm/trainer/verl/agent_workflow_trainer.py"
UPSTREAM_TASK_RUNNER = REPO / "third_party/rllm/rllm/trainer/verl/train_agent_ppo.py"
OUR_TRAINER = REPO / "marl_workflows/trainer/trainer.py"
OUR_TASK_RUNNER = REPO / "marl_workflows/trainer/task_runner.py"

BEGIN = re.compile(r"^\s*# >>> MARL\((\w+)\)")
END = re.compile(r"^\s*# <<< MARL\((\w+)\)")

# (our file, our class, upstream file, upstream class, method, region ids in order)
COPIES = [
    (
        OUR_TRAINER,
        "MultiAgentWorkflowTrainer",
        UPSTREAM_TRAINER,
        "AgentWorkflowPPOTrainer",
        "init_workers",
        ["W1", "W2", "W3"],
    ),
    (
        OUR_TRAINER,
        "MultiAgentWorkflowTrainer",
        UPSTREAM_TRAINER,
        "AgentWorkflowPPOTrainer",
        "fit_agent",
        ["R1", "R2", "R3", "R4", "R5", "R6", "R7"],
    ),
    (
        OUR_TRAINER,
        "MultiAgentWorkflowTrainer",
        UPSTREAM_TRAINER,
        "AgentWorkflowPPOTrainer",
        "_pad_dataproto_to_world_size",
        ["P1"],
    ),
    (
        OUR_TASK_RUNNER,
        "MultiAgentTaskRunner",
        UPSTREAM_TASK_RUNNER,
        "TaskRunner",
        "run",
        ["T4", "T1", "T2", "T2", "T3", "T3", "T4"],
    ),
]


def _method_source(path: Path, class_name: str, method: str) -> list[str]:
    """Source lines of ``class_name.method``, extended over adjacent marker/comment lines.

    ``ast`` ranges stop at the last statement and start at ``def``; a region may open on the
    line above ``def`` or close after the last statement, so those comment lines are included.
    """
    text = path.read_text()
    lines = text.splitlines()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method:
                    start, end = item.lineno - 1, item.end_lineno
                    indent = item.col_offset
                    while start > 0 and BEGIN.match(lines[start - 1]):
                        start -= 1
                    while end < len(lines):
                        nxt = lines[end]
                        if nxt.strip() and not (nxt.lstrip().startswith("#") and len(nxt) - len(nxt.lstrip()) > indent):
                            break
                        end += 1
                    return lines[start:end]
    raise AssertionError(f"{class_name}.{method} not found in {path}")


@dataclass
class _Normalized:
    lines: list[str] = field(default_factory=list)
    in_region: list[bool] = field(default_factory=list)
    boundaries: set[int] = field(default_factory=set)
    regions: list[str] = field(default_factory=list)


def _normalize(lines: list[str]) -> _Normalized:
    out = _Normalized()
    current: str | None = None
    for line in lines:
        if m := BEGIN.match(line):
            assert current is None, f"region {m.group(1)} opens inside region {current}"
            current = m.group(1)
            out.regions.append(current)
            out.boundaries.add(len(out.lines))
            continue
        if m := END.match(line):
            assert current == m.group(1), f"region end {m.group(1)} does not close {current}"
            current = None
            out.boundaries.add(len(out.lines))
            continue
        stripped = line.rstrip()
        if not stripped.strip() or stripped.lstrip().startswith("#"):
            continue
        out.lines.append(stripped)
        out.in_region.append(current is not None)
    assert current is None, f"region {current} is not closed"
    return out


def _drift(upstream: list[str], ours: _Normalized) -> list[str]:
    """Lines that differ from upstream outside every MARL region."""
    problems: list[str] = []
    matcher = difflib.SequenceMatcher(a=upstream, b=ours.lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        problems += [f"+ {ours.lines[j]}" for j in range(j1, j2) if not ours.in_region[j]]
        if j1 == j2:  # upstream lines removed: only where a region starts or ends
            touches_region = (
                j1 in ours.boundaries
                or (j1 > 0 and ours.in_region[j1 - 1])
                or (j1 < len(ours.lines) and ours.in_region[j1])
            )
            if not touches_region:
                problems += [f"- {line}" for line in upstream[i1:i2]]
    return problems


@pytest.mark.parametrize("ours_path, ours_cls, up_path, up_cls, method, regions", COPIES, ids=[c[4] for c in COPIES])
def test_copy_differs_only_inside_marl_regions(ours_path, ours_cls, up_path, up_cls, method, regions) -> None:
    upstream = _normalize(_method_source(up_path, up_cls, method))
    assert not upstream.regions
    ours = _normalize(_method_source(ours_path, ours_cls, method))
    assert ours.regions == regions
    problems = _drift(upstream.lines, ours)
    assert not problems, f"{method} drifted from upstream outside MARL regions:\n" + "\n".join(problems)


def test_guard_detects_drift() -> None:
    upstream = _normalize(_method_source(UPSTREAM_TRAINER, "AgentWorkflowPPOTrainer", "fit_agent")).lines
    ours = _normalize(_method_source(OUR_TRAINER, "MultiAgentWorkflowTrainer", "fit_agent"))
    edited = list(ours.lines)
    k = next(i for i, line in enumerate(edited) if "compute_advantage(" in line and not ours.in_region[i])
    edited[k] = edited[k].replace("compute_advantage(", "compute_advantage_v2(")
    mutated = _Normalized(edited, ours.in_region, ours.boundaries, ours.regions)
    assert _drift(upstream, mutated) == [f"+ {edited[k]}"]

    # Removing an upstream line away from any region is reported too.
    k = next(i for i, line in enumerate(ours.lines) if "solve_partial += 1" in line)
    removed = _Normalized(
        ours.lines[:k] + ours.lines[k + 1 :],
        ours.in_region[:k] + ours.in_region[k + 1 :],
        {b if b <= k else b - 1 for b in ours.boundaries},
        ours.regions,
    )
    assert any("solve_partial += 1" in p for p in _drift(upstream, removed))


def test_final_step_flag_is_not_shadowed() -> None:
    """The fork's ``is_last_step`` flag was overwritten by the broadcast-advantage branches."""
    source = "\n".join(_method_source(OUR_TRAINER, "MultiAgentWorkflowTrainer", "fit_agent"))
    assert "is_final_step = self.global_steps >= self.total_training_steps" in source
    assert source.count("is_final_step =") == 1
