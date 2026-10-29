"""Tests for MarlDatasetRegistry: storage under $MARL_DATA_DIR, never inside the rllm package."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.stack

ROWS = [
    {"question": "1+1?", "ground_truth": "2", "data_source": "unit"},
    {"question": "2+2?", "ground_truth": "4", "data_source": "unit"},
]


def _snapshot(path: str) -> tuple[bool, str | None]:
    if not os.path.exists(path):
        return False, None
    if os.path.isdir(path):
        return True, ",".join(sorted(os.listdir(path)))
    return True, hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_registry_writes_only_under_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from rllm.data.dataset import DatasetRegistry

    from marl_workflows.data.registry import MarlDatasetRegistry

    monkeypatch.setenv("MARL_DATA_DIR", str(tmp_path))
    base_paths = [DatasetRegistry._REGISTRY_FILE, DatasetRegistry._REGISTRY_DIR, DatasetRegistry._DATASET_DIR]
    before = [_snapshot(p) for p in base_paths]

    assert MarlDatasetRegistry._REGISTRY_DIR == str(tmp_path / "registry")
    assert MarlDatasetRegistry._REGISTRY_FILE == str(tmp_path / "registry" / "dataset_registry.json")
    assert MarlDatasetRegistry._DATASET_DIR == str(tmp_path / "datasets")

    MarlDatasetRegistry.register_dataset("unit_ds", ROWS, split="train")

    raw = MarlDatasetRegistry.data_path("unit_ds", "train")
    verl = MarlDatasetRegistry.verl_path("unit_ds", "train")
    assert raw == str(tmp_path / "datasets" / "unit_ds" / "train.parquet")
    assert verl == str(tmp_path / "datasets" / "unit_ds" / "train_verl.parquet")
    assert MarlDatasetRegistry.data_path("unit_ds", "train", verl=True) == verl
    assert os.path.exists(tmp_path / "registry" / "dataset_registry.json")

    loaded = MarlDatasetRegistry.load_dataset("unit_ds", "train")
    assert loaded is not None and len(loaded) == len(ROWS)
    assert MarlDatasetRegistry.dataset_exists("unit_ds", "train")

    # Nothing changed inside the rllm package, and the base registry never learned the name.
    assert [_snapshot(p) for p in base_paths] == before
    for p in base_paths:
        assert not os.path.realpath(p).startswith(str(tmp_path))


def test_data_dir_follows_env_at_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from marl_workflows.data.registry import MarlDatasetRegistry

    a, b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv("MARL_DATA_DIR", str(a))
    MarlDatasetRegistry.register_dataset("unit_ds", ROWS, split="test")
    assert MarlDatasetRegistry.data_path("unit_ds", "test") is not None
    monkeypatch.setenv("MARL_DATA_DIR", str(b))
    assert MarlDatasetRegistry.data_path("unit_ds", "test") is None


def test_verl_path_missing_names_prepare_module() -> None:
    from marl_workflows.data.registry import MarlDatasetRegistry

    with pytest.raises(FileNotFoundError, match="marl_workflows.data.prepare_dapo_math"):
        MarlDatasetRegistry.verl_path("dapo_math", "train")
    with pytest.raises(FileNotFoundError, match="prepare_"):
        MarlDatasetRegistry.verl_path("no_such_dataset", "test")
    assert MarlDatasetRegistry.data_path("no_such_dataset", "test") is None
