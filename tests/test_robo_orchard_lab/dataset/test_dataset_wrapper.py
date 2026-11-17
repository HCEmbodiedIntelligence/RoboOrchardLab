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

import logging

import pytest
import torch
from torch.utils.data import TensorDataset

from robo_orchard_lab.dataset.dataset_wrapper import (
    ConcatDatasetWithFlag,
    DistributedBatchFlagSampler,
)


class TestDistributedBatchFlagSampler:
    @pytest.mark.parametrize(
        "lengths, weights, expected_ratios",
        [
            ([2, 6], None, ["25.00% [25.00%]", "75.00% [75.00%]"]),
            ([2, 6], [3, 1], ["75.00% [25.00%]", "25.00% [75.00%]"]),
            ([4], None, ["100.00% [100.00%]"]),
            ([0, 4], None, ["0.00% [0.00%]", "100.00% [100.00%]"]),
            ([0, 0], None, ["0.00% [0.00%]", "0.00% [0.00%]"]),
        ],
    )
    def test_sampling_ratios_logged(
        self, caplog, lengths, weights, expected_ratios
    ):
        datasets = []
        for i, length in enumerate(lengths):
            dataset = TensorDataset(torch.arange(length))
            dataset.dataset_name = f"dataset_{i}"
            datasets.append(dataset)

        with caplog.at_level(logging.INFO):
            sampler = DistributedBatchFlagSampler(
                ConcatDatasetWithFlag(datasets),
                batch_size=2,
                drop_last=True,
                dataset_sample_weights=weights,
            )

        assert "dataset sample weights:" in caplog.text
        for i, expected_ratio in enumerate(expected_ratios):
            assert f"dataset_{i}: {expected_ratio}" in caplog.text

        if weights is None:
            assert sampler.dataset_sample_weights is None
            indices = [index for batch in sampler for index in batch]
            assert sorted(indices) == list(range(sum(lengths)))
        else:
            assert sampler.dataset_sample_weights == pytest.approx(
                [0.75, 0.25]
            )
