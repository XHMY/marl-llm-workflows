"""Experiment tracking that resumes the same wandb run across job segments.

Replaces the research fork's edits to ``rllm/utils/tracking.py`` (lines 83-150, 234-267)
with a subclass. ``wandb.init`` honours ``WANDB_RUN_ID`` and ``WANDB_RESUME``, so the
upstream ``Tracking.__init__`` is reused unchanged: the subclass only sets those variables
around the ``super().__init__`` call and records the run id afterwards.
"""

from __future__ import annotations

import os
from typing import Any

from rllm.utils.tracking import Tracking

__all__ = ["ResumableTracking"]

_WANDB_BACKENDS = ("wandb", "tracking")
_ENV_RUN_ID = "WANDB_RUN_ID"
_ENV_RESUME = "WANDB_RESUME"


class ResumableTracking(Tracking):
    """``rllm.utils.tracking.Tracking`` with wandb resume (rllm@186d0bbb ``rllm/utils/tracking.py:59``).

    ``resume_run_id`` + ``resume_mode``: resume that run. ``resume_mode="allow"`` without an
    id: resume the most recent run with the same name in the project, if any (checkpoints
    written before the run id was recorded). Otherwise a new run is created.
    """

    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        default_backend: str | list[str] = "console",
        config: dict | None = None,
        resume_run_id: str | None = None,
        resume_mode: str | None = None,
    ) -> None:
        backends = [default_backend] if isinstance(default_backend, str) else list(default_backend)
        uses_wandb = any(b in backends for b in _WANDB_BACKENDS)
        run_id: str | None = None
        if uses_wandb:
            if resume_run_id and resume_mode:
                run_id = resume_run_id
                print(f"Attempting to resume wandb run with ID: {run_id}")
            elif resume_mode == "allow":
                run_id = self.find_run_by_name(project_name, experiment_name)
                if run_id:
                    resume_mode = "allow"
                    print(f"Found existing wandb run by name, resuming: {run_id}")
                else:
                    print("No existing wandb run found, creating new run")

        saved = {k: os.environ.get(k) for k in (_ENV_RUN_ID, _ENV_RESUME)}
        if run_id:
            os.environ[_ENV_RUN_ID] = run_id
            os.environ[_ENV_RESUME] = str(resume_mode)
        try:
            super().__init__(project_name, experiment_name, default_backend=default_backend, config=config)
        finally:
            if run_id:
                # Restore the caller's environment so a later wandb.init in this process starts fresh.
                for key, value in saved.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

        self._wandb_run_id: str | None = None
        if uses_wandb:
            import wandb

            self._wandb_run_id = wandb.run.id if wandb.run else None

    @property
    def wandb_run_id(self) -> str | None:
        """The wandb run id (``None`` without the wandb backend)."""
        return self._wandb_run_id

    @staticmethod
    def find_run_by_name(project_name: str, experiment_name: str) -> str | None:
        """Id of the most recently created run named ``experiment_name`` in ``project_name``."""
        try:
            import wandb

            runs: Any = wandb.Api().runs(
                path=project_name, filters={"display_name": experiment_name}, order="-created_at"
            )
            for run in runs:
                return run.id
        except Exception as e:  # network or auth failure must not stop training
            print(f"Warning: Failed to find wandb run by name: {e}")
        return None
