"""Stop a training segment cleanly before the SLURM job's wall-clock limit.

Enabled with ``+trainer.wallclock_stop.enable=true`` (optional ``.deadline_epoch`` and
``.margin_s``). The trainer records each step's duration; when the next step would not
finish ``margin_s`` before the deadline, it saves the last completed step (if not saved yet)
and exits, so an ``afterany`` chain of jobs resumes from that checkpoint. Same semantics as
the research fork (``rllm/trainer/verl/agent_workflow_trainer.py:806-882``).
"""

from __future__ import annotations

import datetime
import os
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["WallclockStop"]


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping) or hasattr(cfg, "get"):
        value = cfg.get(key, default)
        return default if value is None else value
    return getattr(cfg, key, default)


def _squeue_end_time(job_id: str) -> float | None:
    """The job's EndTime from ``squeue -o %e`` (ISO, local time), or ``None``."""
    try:
        out = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%e"], capture_output=True, text=True, timeout=30
        ).stdout.strip()
        lines = out.splitlines()
        if lines:
            return datetime.datetime.fromisoformat(lines[0].strip()).timestamp()
    except Exception as e:  # squeue missing, timeout, or an unparsable value such as "N/A"
        print(f"WARNING: wallclock_stop could not read the job EndTime from squeue: {e}", flush=True)
    return None


@dataclass
class WallclockStop:
    """Deadline plus an EMA of the per-step wall time."""

    deadline: float
    margin_s: float = 900.0
    alpha: float = 0.3
    ema: float | None = None
    last_iter: float | None = None

    @classmethod
    def from_config(cls, cfg: Any, env: Mapping[str, str] | None = None) -> WallclockStop | None:
        """Build from ``trainer.wallclock_stop``; ``None`` when disabled or no deadline is known.

        The deadline is ``deadline_epoch`` if given, else ``$SLURM_JOB_END_TIME``, else the
        job's EndTime from ``squeue`` when ``$SLURM_JOB_ID`` is set.
        """
        env = os.environ if env is None else env
        if not bool(_cfg_get(cfg, "enable", False)):
            return None
        deadline = _cfg_get(cfg, "deadline_epoch")
        if deadline is None:
            raw = env.get("SLURM_JOB_END_TIME")
            try:
                deadline = float(raw) if raw not in (None, "") else None
            except ValueError:
                deadline = None
        if deadline is None and env.get("SLURM_JOB_ID"):
            deadline = _squeue_end_time(env["SLURM_JOB_ID"])
        if deadline is None:
            print(
                "WARNING: wallclock_stop enabled but no deadline_epoch and no SLURM_JOB_END_TIME; disabled",
                flush=True,
            )
            return None
        stop = cls(deadline=float(deadline), margin_s=float(_cfg_get(cfg, "margin_s", 900.0)))
        print(
            f"[wallclock_stop] deadline_epoch={stop.deadline:.0f} margin_s={stop.margin_s:.0f} "
            f"remaining_s={stop.deadline - time.time():.0f}",
            flush=True,
        )
        return stop

    @property
    def has_completed_step(self) -> bool:
        """True once a step of this segment has been recorded."""
        return self.last_iter is not None

    def record(self, step_time: float | None, iter_time: float) -> None:
        """Fold one step into the estimate; the sample is the larger of the timed step and the loop iteration."""
        sample = max(float(step_time or 0.0), float(iter_time))
        self.ema = sample if self.ema is None else self.alpha * sample + (1 - self.alpha) * self.ema
        self.last_iter = float(iter_time)

    def should_stop(self, now: float | None = None) -> bool:
        """True if one more step (with a 25 % allowance) plus the margin would pass the deadline."""
        now = time.time() if now is None else now
        estimate = max(self.ema or 0.0, self.last_iter or 0.0)
        return now + 1.25 * estimate + self.margin_s > self.deadline
