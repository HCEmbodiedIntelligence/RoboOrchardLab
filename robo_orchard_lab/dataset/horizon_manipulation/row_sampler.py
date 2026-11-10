# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import numpy as np
from datasets import Dataset as HFDataset
from robo_orchard_core.utils.config import ClassType

from robo_orchard_lab.dataset.robot.row_sampler import (
    MultiRowSampler,
    MultiRowSamplerConfig,
)

__all__ = [
    "EpisodeChunkSampler",
    "EpisodeChunkSamplerConfig",
]


class EpisodeChunkSampler(MultiRowSampler):
    """Sample history and prediction chunks by packed dataset row index.

    With ``drop_incomplete_chunks=True``, ROMultiRowDataset excludes anchors
    lacking a full future chunk. Missing history still repeats the first row.
    """

    def __init__(self, cfg: "EpisodeChunkSamplerConfig") -> None:
        self.cfg = cfg
        self.hist_steps = cfg.hist_steps
        self.pred_steps = cfg.pred_steps
        self.pred_interval = cfg.pred_interval
        assert (
            isinstance(self.pred_interval, int) and self.pred_interval >= 1
        ), "pred_interval must be an integer >= 1"
        self.chunk_size = self.hist_steps + self.pred_steps

    def get_sample_indices(self, index_dataset: HFDataset) -> list[int] | None:
        """Keep anchors whose last future offset stays in the same episode.

        Episodes must occupy contiguous packed rows. The required future span
        is ``pred_steps * pred_interval``; history does not affect eligibility.
        Return None when dropping is disabled, preserving the identity mapping.
        """
        if not self.cfg.drop_incomplete_chunks:
            return None
        episodes = np.asarray(index_dataset["episode_index"])
        boundaries = np.concatenate(
            (
                [0],
                np.flatnonzero(episodes[1:] != episodes[:-1]) + 1,
                [len(episodes)],
            )
        )
        span = self.pred_steps * self.pred_interval
        return [
            index
            for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
            for index in range(int(start), int(end) - span)
        ]

    @property
    def column_rows_keys(self) -> dict[str, list[str]]:
        ret = {}
        for column in self.cfg.target_columns:
            ret[column] = [f"chunk_row_{i}" for i in range(self.chunk_size)]
        return ret

    def sample_row_idx(
        self,
        index_dataset: HFDataset,
        index: int,
    ) -> dict[str, list[int | None]]:
        """Return history and future source indices for each target column.

        Raises:
            ValueError: Dropping is enabled and the requested anchor lacks
                enough future rows. ROMultiRowDataset avoids these anchors
                using get_sample_indices before data loading.
        """
        cur_row = index_dataset[index]
        cur_episode_idx = cur_row["episode_index"]
        dataset_len = len(index_dataset)

        raw_hist_start_idx = index - self.hist_steps + 1
        hist_indices = np.arange(raw_hist_start_idx, index + 1)

        if self.pred_steps > 0:
            future_offsets = np.arange(1, self.pred_steps + 1)
            future_indices = index + future_offsets * self.pred_interval
            raw_end_idx = int(future_indices[-1])
        else:
            future_indices = np.array([], dtype=np.int64)
            raw_end_idx = index

        start_idx = max(raw_hist_start_idx, 0)
        while start_idx < dataset_len:
            start_row = index_dataset[start_idx]
            if start_row["episode_index"] == cur_episode_idx:
                break
            start_idx += 1

        end_idx = min(raw_end_idx, dataset_len - 1)
        while end_idx >= 0:
            end_row = index_dataset[end_idx]
            if end_row["episode_index"] == cur_episode_idx:
                break
            end_idx -= 1

        if self.cfg.drop_incomplete_chunks and raw_end_idx > end_idx:
            raise ValueError(
                f"Row {index} has an incomplete future chunk. "
                "Use get_sample_indices() to select eligible anchors."
            )

        raw_indices = np.concatenate([hist_indices, future_indices])
        padded_indices = np.clip(raw_indices, start_idx, end_idx)

        chunk_indices: list[int] = padded_indices.tolist()

        ret = {}
        for column in self.cfg.target_columns:
            ret[column] = chunk_indices

        return ret


class EpisodeChunkSamplerConfig(MultiRowSamplerConfig[EpisodeChunkSampler]):
    """Configuration for the EpisodeChunkSampler."""

    class_type: ClassType[EpisodeChunkSampler] = EpisodeChunkSampler

    target_columns: list[str]

    hist_steps: int = 1
    pred_steps: int = 64
    # Future sampling stride in packed kept rows.
    pred_interval: int = 1

    drop_incomplete_chunks: bool = False
    """False repeats the episode's last row for missing future steps.

    True excludes these sample anchors in ROMultiRowDataset, retaining source
    frames for other chunks. History padding is unchanged.
    """
