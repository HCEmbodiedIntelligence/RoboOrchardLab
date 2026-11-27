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

from bisect import bisect_right
from typing import Annotated, Literal

import numpy as np
from datasets import Dataset as HFDataset
from pydantic import Field
from robo_orchard_core.utils.config import ClassType

from robo_orchard_lab.dataset.robot.row_sampler import (
    CachedIndexDataset,
    MultiRowSampler,
    MultiRowSamplerConfig,
)

__all__ = [
    "EpisodeChunkSampler",
    "EpisodeChunkSamplerConfig",
    "ProcessEpisodeChunkSampler",
    "ProcessEpisodeChunkSamplerConfig",
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


class ProcessEpisodeChunkSampler(EpisodeChunkSampler):
    """Sample a fixed-size chunk with a stride based on its anchor's progress.

    History is contiguous and includes the anchor. Every future step uses
    the stride selected for that anchor. Episodes occupy contiguous packed
    rows, and progress is the zero-based row offset divided by episode length.
    """

    def __init__(self, cfg: "ProcessEpisodeChunkSamplerConfig") -> None:
        super().__init__(cfg)
        schedule = sorted((cfg.pred_interval_dict or {}).items())
        self._thresholds = tuple(threshold for threshold, _ in schedule)
        self._strides = (*(stride for _, stride in schedule), 1)
        self._episode_bounds_cache: tuple[HFDataset, list[int]] | None = None

    def get_sample_indices(self, index_dataset: HFDataset) -> list[int] | None:
        """Return anchors with complete future chunks, or None for padding."""
        # NOTE: 这一步直接将超出的anchor过滤掉了，导致后续sample_row_idx中无法获取到超出的anchor的索引
        bounds = self._episode_bounds(index_dataset)
        if not self.cfg.drop_incomplete_chunks:
            return None
        return [
            index
            for start, stop in zip(bounds[:-1], bounds[1:], strict=True)
            for index in range(start, stop - self.pred_steps)
            if index + self.pred_steps * self._stride(index, start, stop)
            < stop
        ]

    def sample_row_idx(
        self,
        index_dataset: HFDataset | CachedIndexDataset,
        index: int,
    ) -> dict[str, list[int | None]]:
        """Return source indices for each target column in chunk order.

        Pad missing history with the first episode row. Pad missing future
        steps with the last episode row when dropping is disabled.

        Raises:
            IndexError: The anchor is outside the dataset.
            ValueError: Dropping is enabled and the anchor is incomplete.
        """
        if isinstance(index_dataset, CachedIndexDataset):
            index_dataset = index_dataset.source_dataset
        if index < 0:
            index += len(index_dataset)
        if not 0 <= index < len(index_dataset):
            raise IndexError(f"Anchor row {index} is outside the dataset.")

        bounds = self._episode_bounds(index_dataset)
        episode = bisect_right(bounds, index) - 1
        start, stop = bounds[episode], bounds[episode + 1]
        stride = self._stride(index, start, stop)
        if self.cfg.drop_incomplete_chunks and (
            index + self.pred_steps * stride >= stop
        ):
            raise ValueError(
                f"Row {index} has an incomplete future chunk. "
                "Use get_sample_indices() to select eligible anchors."
            )

        chunk: list[int | None] = [
            max(start, row)
            for row in range(index - self.hist_steps + 1, index + 1)
        ]
        chunk.extend(
            min(index + step * stride, stop - 1)
            for step in range(1, self.pred_steps + 1)
        )
        return {column: chunk for column in self.cfg.target_columns}

    def _episode_bounds(self, index_dataset: HFDataset) -> list[int]:
        """Cache run boundaries for the current source index dataset."""
        cached = self._episode_bounds_cache
        if cached is not None and cached[0] is index_dataset:
            return cached[1]
        episode_rows = index_dataset.select_columns(["episode_index"])
        episodes = episode_rows.with_format("arrow")[:][
            "episode_index"
        ].to_numpy()
        changes = np.flatnonzero(episodes[1:] != episodes[:-1]) + 1
        bounds = [0, *changes.tolist(), len(episodes)]
        self._episode_bounds_cache = (index_dataset, bounds)
        return bounds

    def _stride(self, index: int, start: int, stop: int) -> int:
        progress = (index - start) / (stop - start)
        return self._strides[bisect_right(self._thresholds, progress)]


class ProcessEpisodeChunkSamplerConfig(EpisodeChunkSamplerConfig):
    """Configure future chunk strides by exclusive episode-progress bounds."""

    class_type: ClassType[ProcessEpisodeChunkSampler] = (
        ProcessEpisodeChunkSampler
    )
    pred_interval_dict: (
        dict[
            Annotated[float, Field(gt=0, le=1)],
            Annotated[int, Field(strict=True, ge=1)],
        ]
        | None
    ) = None
    """Map progress bounds in (0, 1] to positive strides in packed rows.

    For example, {0.1: 3, 0.3: 2} uses stride 3 below 10%, stride 2 from
    10% up to 30%, and stride 1 from 30% onward. None uses stride 1 throughout.
    """
