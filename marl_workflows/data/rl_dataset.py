"""``PolarsRLHFDataset``: verl's ``RLHFDataset`` with a polars parquet reader (``data.custom_cls``).

The research fork edited verl's ``RLHFDataset._read_files_and_tokenize`` in place to read
parquet with polars and wrap the Arrow table in a HuggingFace ``Dataset`` (zero copy). It is
faster for the 674 MB DeepCoder train file and tolerant of large nested ``extra_info``
structs. Here the same change is a subclass selected through verl's ``data.custom_cls``
hook (verl@v0.6.1 ``verl/trainer/main_ppo.py`` ``create_rl_dataset``)::

    data.custom_cls.path=pkg://marl_workflows.data.rl_dataset  data.custom_cls.name=PolarsRLHFDataset

``__getitem__`` is not overridden: rLLM writes ``extra_info`` as a dict (struct column), so the
fork's JSON-string ``extra_info`` branch is never taken for our data.
"""

from __future__ import annotations

import datasets
import numpy as np
from verl.utils.dataset.rl_dataset import RLHFDataset

__all__ = ["PolarsRLHFDataset"]


class PolarsRLHFDataset(RLHFDataset):
    """``RLHFDataset`` reading its parquet files with polars instead of ``datasets.load_dataset``."""

    # COPIED FROM verl@v0.6.1 verl/utils/dataset/rl_dataset.py:152-170 (RLHFDataset._read_files_and_tokenize);
    # diff regions marked MARL(polars).
    def _read_files_and_tokenize(self) -> None:
        import polars as pl

        dataframes = []
        for parquet_file in self.data_files:
            # >>> MARL(polars): fork verl/utils/dataset/rl_dataset.py:157 (polars reader)
            dataframe = pl.read_parquet(parquet_file)
            # <<< MARL(polars)
            dataframes.append(dataframe)
        # >>> MARL(polars): fork rl_dataset.py:159-160 (Arrow -> HF Dataset, zero copy)
        self.dataframe: datasets.Dataset = datasets.Dataset(pl.concat(dataframes).to_arrow())
        # <<< MARL(polars)

        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)
