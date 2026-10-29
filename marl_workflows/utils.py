"""Small environment helpers with no heavy imports."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from marl_workflows.contracts import ENV_DATA_DIR, ENV_INIT_WEIGHT_DIR

__all__ = ["allocated_cpu_count", "init_weight_dir", "marl_data_dir", "repo_root"]

_DEFAULT_DATA_DIR = os.path.join("~", ".cache", "marl-llm-workflows")


def marl_data_dir(env: Mapping[str, str] | None = None) -> str:
    """Dataset registry root: ``$MARL_DATA_DIR`` or ``~/.cache/marl-llm-workflows``.

    Read on every call (not cached) so that tests and launchers can redirect it at runtime.
    """
    env = os.environ if env is None else env
    value = env.get(ENV_DATA_DIR) or _DEFAULT_DATA_DIR
    return os.path.abspath(os.path.expanduser(value))


def init_weight_dir(env: Mapping[str, str] | None = None) -> str:
    """Root of merged ``*_s300`` init weights: ``$MARL_INIT_WEIGHT_DIR`` or ``checkpoints/init_weight``."""
    env = os.environ if env is None else env
    return env.get(ENV_INIT_WEIGHT_DIR) or os.path.join("checkpoints", "init_weight")


def repo_root() -> Path:
    """The repository checkout containing this package (meaningful for editable / PYTHONPATH use)."""
    return Path(__file__).resolve().parent.parent


def allocated_cpu_count(env: Mapping[str, str] | None = None) -> int:
    """CPUs this job may use.

    The paper launcher set the code-executor pool to cpus_per_gpu * n_gpus, which is what
    SLURM exports as ``SLURM_CPUS_ON_NODE`` for a one-node job; outside SLURM the process
    affinity mask is the allocation (it honours cgroups / taskset, unlike ``os.cpu_count``).
    """
    env = os.environ if env is None else env
    slurm = env.get("SLURM_CPUS_ON_NODE")
    if slurm:
        try:
            n = int(slurm)
            if n > 0:
                return n
        except ValueError:
            pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)
