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

import pytest
from datasets import Dataset
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from robo_orchard_lab.dataset.hc_tj import HcTjRODataset, hc_tj_ro_dataset
from robo_orchard_lab.dataset.horizon_manipulation.row_sampler import (
    EpisodeChunkSampler,
    EpisodeChunkSamplerConfig,
    ProcessEpisodeChunkSampler,
    ProcessEpisodeChunkSamplerConfig,
)
from robo_orchard_lab.dataset.robot.dataset_db_engine import create_tables
from robo_orchard_lab.dataset.robot.db_orm import Episode


class TestProcessEpisodeChunkSampler:
    def test_show_source_and_sampled_chunks(self):
        source = Dataset.from_dict(
            {
                "episode_index": [0] * 20,
                "frame_index": list(range(20)),
                "obs": [f"frame_{i:02d}" for i in range(20)],
            }
        )
        cfg = ProcessEpisodeChunkSamplerConfig(
            target_columns=["obs"],
            hist_steps=2,
            pred_steps=4,
            pred_interval_dict={0.3: 2, 0.1: 3},
            drop_incomplete_chunks=True,
        )
        sampler = ProcessEpisodeChunkSamplerConfig.model_validate_json(
            cfg.model_dump_json()
        )()
        assert isinstance(sampler, ProcessEpisodeChunkSampler)
        # assert sampler.get_sample_indices(source) is not None
        assert sampler.column_rows_keys["obs"] == [
            f"chunk_row_{i}" for i in range(6)
        ]

        print("\nSource (episode frame index, data):")
        print(list(zip(source["frame_index"], source["obs"], strict=True)))
        expected = {
            0: [0, 0, 3, 6, 9, 12],
            1: [0, 1, 4, 7, 10, 13],
            2: [1, 2, 4, 6, 8, 10],
            5: [4, 5, 7, 9, 11, 13],
            6: [5, 6, 7, 8, 9, 10],
            15: [14,15,16,17,18,19],
        }
        for anchor, expected_indices in expected.items():
            indices = sampler.sample_row_idx(source, anchor)["obs"]
            sampled = source[indices]
            print(
                f"Anchor {anchor:2d} -> episode frame indices "
                f"{sampled['frame_index']} -> data {sampled['obs']}"
            )
            assert sampled["frame_index"] == expected_indices

    @pytest.mark.parametrize("thresholds", [None, {}])
    def test_default_stride_matches_original(self, thresholds):
        source = Dataset.from_dict({"episode_index": [0] * 5 + [1] * 5})
        original = EpisodeChunkSamplerConfig(
            target_columns=["obs"], hist_steps=2, pred_steps=3
        )()
        process = ProcessEpisodeChunkSamplerConfig(
            target_columns=["obs"],
            hist_steps=2,
            pred_steps=3,
            pred_interval_dict=thresholds,
        )()
        for anchor in range(len(source)):
            assert process.sample_row_idx(
                source, anchor
            ) == original.sample_row_idx(source, anchor)

    def test_drop_incomplete_anchors_in_each_episode(self):
        source = Dataset.from_dict({"episode_index": [0] * 6 + [1] + [2] * 6})
        sampler = ProcessEpisodeChunkSamplerConfig(
            target_columns=["obs"],
            hist_steps=3,
            pred_steps=2,
            pred_interval_dict={0.1: 3, 0.3: 2},
            drop_incomplete_chunks=True,
        )()
        assert sampler.get_sample_indices(source) == [1, 2, 3, 8, 9, 10]
        assert sampler.sample_row_idx(source, 1)["obs"] == [0, 0, 1, 3, 5]
        assert sampler.sample_row_idx(source, 8)["obs"] == [7, 7, 8, 10, 12]
        for anchor in [0, 4, 5, 6, 7, 11, 12]:
            with pytest.raises(ValueError, match="incomplete future chunk"):
                sampler.sample_row_idx(source, anchor)

    @pytest.mark.parametrize(
        "thresholds", [{0: 1}, {1.1: 1}, {0.1: 0}, {0.1: 1.5}]
    )
    def test_invalid_config(self, thresholds):
        with pytest.raises(ValidationError, match="pred_interval_dict"):
            ProcessEpisodeChunkSamplerConfig(
                target_columns=["obs"], pred_interval_dict=thresholds
            )


@pytest.fixture
def hc_tj_path(tmp_path, monkeypatch):
    # Joint/action values encode source frame indices for easy inspection.
    lengths = (6, 1, 8)
    frames = [frame for length in lengths for frame in range(length)]
    n_frames = len(frames)
    Dataset.from_dict(
        {
            "index": list(range(n_frames)),
            "episode_index": [
                episode
                for episode, length in enumerate(lengths)
                for _ in range(length)
            ],
            "frame_index": frames,
            "joints": frames,
            "actions": frames,
            "task_index": [None] * n_frames,
            "robot_index": [None] * n_frames,
            "instruction_index": [None] * n_frames,
            "timestamp_min": list(range(n_frames)),
            "timestamp_max": list(range(1, n_frames + 1)),
        }
    ).save_to_disk(str(tmp_path))
    engine = create_engine(f"sqlite:///{tmp_path / 'meta_db.sqlite'}")
    try:
        create_tables(engine)
        with Session(engine) as session:
            begin = 0
            for ep, length in enumerate(lengths):
                session.add(
                    Episode(
                        index=ep,
                        dataset_begin_index=begin,
                        frame_num=length,
                    )
                )
                begin += length
            session.commit()
    finally:
        engine.dispose()
    # Exercise loading, metadata and chunk access with synthetic feature data.
    monkeypatch.setattr(
        hc_tj_ro_dataset._HcTjArrowDataParse, "__call__", lambda self, row: row
    )
    return str(tmp_path)


@pytest.mark.parametrize(
    "thresholds,sampler_type,ranges,first_chunk",
    [
        (None, EpisodeChunkSampler, [(0, 4), (4, 4), (4, 10)], [0, 0, 1, 2]),
        (
            {0.1: 3, 0.3: 2},
            ProcessEpisodeChunkSampler,
            [(0, 3), (3, 3), (3, 9)],
            [0, 1, 3, 5],
        ),
    ],
)
def test_hc_tj_dataset_supports_both_samplers(
    hc_tj_path, thresholds, sampler_type, ranges, first_chunk
):
    dataset = HcTjRODataset(
        paths=[hc_tj_path, hc_tj_path],
        hist_steps=2,
        pred_steps=2,
        drop_incomplete_chunks=True,
        pred_interval_dict=thresholds,
    )
    assert type(dataset._concat.datasets[0].row_sampler) is sampler_type
    n_samples = ranges[-1][1]
    assert len(dataset) == 2 * n_samples
    assert dataset.num_episode == 6
    assert [dataset.get_episode_range(i) for i in range(6)] == [
        *ranges,
        *((start + n_samples, stop + n_samples) for start, stop in ranges),
    ]
    assert dataset[0]["joints"] == first_chunk
    assert [
        row["actions"] for row in dataset.__getitems__([0, n_samples])
    ] == [first_chunk, first_chunk]
