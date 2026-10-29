"""Dataset registry stored under ``$MARL_DATA_DIR`` instead of inside the rllm package.

Upstream ``DatasetRegistry`` writes its registry and parquet files next to its own source
(rllm@186d0bbb rllm/data/dataset.py:165-167), which would dirty the submodule. This subclass
relocates both. ``Dataset.get_data_path``/``get_verl_data_path`` stay hard-wired to the base
registry (dataset.py:105), so callers must use :meth:`MarlDatasetRegistry.data_path` and
:meth:`MarlDatasetRegistry.verl_path` instead.
"""

from __future__ import annotations

import os

from rllm.data.dataset import DatasetRegistry

from marl_workflows.contracts import ENV_DATA_DIR
from marl_workflows.utils import marl_data_dir

__all__ = ["PREPARE_MODULES", "MarlDatasetRegistry"]

# Which prepare module registers each dataset (used in error messages).
PREPARE_MODULES: dict[str, str] = {
    "dapo_math": "marl_workflows.data.prepare_dapo_math",
    "aime2025": "marl_workflows.data.prepare_aime",
    "aime2024": "marl_workflows.data.prepare_aime",
    "math500": "marl_workflows.data.prepare_math500",
    "deepcoder_primeintellect": "marl_workflows.data.prepare_deepcoder",
    "smoke_math": "marl_workflows.data.prepare_smoke",
}


class _DataDirPath:
    """Class attribute resolved against ``marl_data_dir()`` on every access.

    The base class reads ``cls._REGISTRY_DIR`` etc. inside its classmethods; resolving
    lazily lets ``MARL_DATA_DIR`` be set after import (launchers, tests).
    """

    def __init__(self, *parts: str) -> None:
        self._parts = parts

    def __get__(self, obj: object, owner: type | None = None) -> str:
        return os.path.join(marl_data_dir(), *self._parts)


class MarlDatasetRegistry(DatasetRegistry):
    """``DatasetRegistry`` rooted at ``$MARL_DATA_DIR`` (default ``~/.cache/marl-llm-workflows``)."""

    _REGISTRY_DIR = _DataDirPath("registry")
    _REGISTRY_FILE = _DataDirPath("registry", "dataset_registry.json")
    _DATASET_DIR = _DataDirPath("datasets")

    @classmethod
    def data_path(cls, name: str, split: str, *, verl: bool = False) -> str | None:
        """Path of the registered parquet (or its ``_verl`` sibling), ``None`` if absent."""
        registry = cls._load_registry()
        path = registry.get(name, {}).get(split)
        if path is None:
            return None
        if verl:
            path = path.replace(".parquet", "_verl.parquet")
        return path if os.path.exists(path) else None

    @classmethod
    def verl_path(cls, name: str, split: str) -> str:
        """Path of the verl-processed parquet; ``FileNotFoundError`` names the prepare module."""
        path = cls.data_path(name, split, verl=True)
        if path is None:
            module = PREPARE_MODULES.get(name, "the matching marl_workflows.data.prepare_* module")
            raise FileNotFoundError(
                f"Dataset {name!r} split {split!r} is not prepared under {marl_data_dir()}. "
                f"Run: python -m {module}  (set {ENV_DATA_DIR} to change the location)"
            )
        return path
