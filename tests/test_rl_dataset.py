"""PolarsRLHFDataset reads rLLM-style verl parquet files exactly like verl's stock reader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.stack

# Heterogeneous nested extra_info, as rLLM's apply_verl_postprocessing writes it: each row's
# original record (different keys, nested dicts, lists, nulls) becomes one struct column.
ROWS = [
    {"question": "q0", "final_answer": "1", "extra_info": {"index": "a"}, "tags": ["x"]},
    {"question": "q1", "ground_truth": '[{"input": "1"}]', "metadata": {"func_name": None}},
    {"question": "q2", "final_answer": "3", "reward_model": {"style": "rule", "ground_truth": "3"}, "level": 5},
]


def _write_verl_parquet(path: Path) -> None:
    from rllm.data.dataset import DatasetRegistry

    verl_rows = DatasetRegistry.apply_verl_postprocessing(ROWS)
    pd.DataFrame(verl_rows).to_parquet(path)


def _config(**overrides: object):
    from omegaconf import OmegaConf

    cfg = {"filter_overlong_prompts": False, "cache_dir": None, "max_prompt_length": 64}
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def test_matches_stock_reader(tmp_path: Path) -> None:
    from verl.utils.dataset.rl_dataset import RLHFDataset

    from marl_workflows.data.rl_dataset import PolarsRLHFDataset

    path = tmp_path / "test_verl.parquet"
    _write_verl_parquet(path)
    cfg = _config(cache_dir=str(tmp_path / "cache"))
    ours = PolarsRLHFDataset(str(path), tokenizer=None, config=cfg)
    stock = RLHFDataset(str(path), tokenizer=None, config=cfg)

    assert len(ours.dataframe) == len(stock.dataframe) == len(ROWS)
    assert ours.dataframe.column_names == stock.dataframe.column_names
    for i in range(len(ROWS)):
        assert ours.dataframe[i] == stock.dataframe[i]
    # Each original record survives inside extra_info (absent keys come back as None).
    for i, row in enumerate(ROWS):
        extra = ours.dataframe[i]["extra_info"]
        for key, value in row.items():
            assert extra[key] == value


def test_max_samples_path_unchanged(tmp_path: Path) -> None:
    from marl_workflows.data.rl_dataset import PolarsRLHFDataset

    path = tmp_path / "train_verl.parquet"
    _write_verl_parquet(path)
    ds = PolarsRLHFDataset(str(path), tokenizer=None, config=_config(cache_dir=str(tmp_path / "c")), max_samples=2)
    assert [r["extra_info"]["question"] for r in ds.dataframe] == ["q0", "q1"]


def test_loadable_through_verl_custom_cls_hook() -> None:
    from verl.utils.import_utils import load_extern_type

    from marl_workflows.data.rl_dataset import PolarsRLHFDataset

    assert load_extern_type("pkg://marl_workflows.data.rl_dataset", "PolarsRLHFDataset") is PolarsRLHFDataset
