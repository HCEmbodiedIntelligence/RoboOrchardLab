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

from __future__ import annotations
import sys
from pathlib import Path

import pytest

CONFIG_DIR = Path(__file__).resolve().parents[4] / "projects/holobrain/configs"
sys.path.insert(0, str(CONFIG_DIR))

from config_hc_tj_ro_dataset import (  # noqa: E402
    _select_episodes,
    build_transforms,
    dataset_config,
)


class _Dataset:
    ranges = [(0, 2), (2, 5), (5, 6)]

    @property
    def num_episode(self):
        return len(self.ranges)

    def get_episode_range(self, episode_index):
        return self.ranges[episode_index]

    def __len__(self):
        return self.ranges[-1][1]

    def __getitem__(self, index):
        return index


def test_select_episodes_keeps_complete_requested_ranges():
    subset = _select_episodes(_Dataset(), [1, 2])

    assert subset.indices == [2, 3, 4, 5]
    assert list(subset) == [2, 3, 4, 5]


@pytest.mark.parametrize("episode_indices", [[], [1, 1], [-1], [3]])
def test_select_episodes_rejects_invalid_indices(episode_indices):
    with pytest.raises(ValueError):
        _select_episodes(_Dataset(), episode_indices)


@pytest.mark.parametrize("episode_indices", [[True], [1.0], ["1"]])
def test_select_episodes_rejects_non_integer_indices(episode_indices):
    with pytest.raises(TypeError):
        _select_episodes(_Dataset(), episode_indices)


def test_hc_tj_transforms_use_per_arm_gripper_layout():
    data_config = dataset_config["hc_tj"]
    transforms = build_transforms(
        {"hist_steps": 1, "pred_steps": 2, "patch_size": 1},
        "training",
        data_config,
    )
    by_type = {
        item["type"].__name__: item
        for item in transforms
        if isinstance(item, dict) and "type" in item
    }

    assert len(data_config["joint_scale_shift"]) == 16
    assert data_config["joint_scale_shift"][7] == [0.5, 0.5]
    assert data_config["joint_scale_shift"][15] == [0.5, 0.5]
    assert by_type["AddItems"]["joint_mask"] == ([True] * 7 + [False]) * 2
    assert by_type["MultiArmKinematics"]["finger_keys"] == [
        ["R_ban_Link"],
        ["L_ban_Link"],
    ]
    assert len(by_type["JointStateNoise"]["noise_range"]) == 16
    assert by_type["JointStateNoise"]["noise_range"][7] == [0.0, 0.0]
    assert by_type["JointStateNoise"]["noise_range"][15] == [0.0, 0.0]


def test_hc_tj_config_uses_fixed_16_axis_dataset():
    data_config = dataset_config["hc_tj"]
    assert data_config["data_paths"] == [
        str(
            CONFIG_DIR.parent
            / "data"
            / "arrow_dataset"
            / "hc_tj_260818_fix"
        )
    ]
