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

"""Qwen HoloBrain training configuration dedicated to the hc_tj dataset."""

import copy

from config_holobrain_qwen_common import (
    build_model,
    build_optimizer,
    build_processors,
    build_training_dataset,
    build_validation_dataset,
    config as _common_config,
)

__all__ = [
    "build_model",
    "build_optimizer",
    "build_processors",
    "build_training_dataset",
    "build_validation_dataset",
    "config",
]

config = copy.deepcopy(_common_config)
config.update(
    training_datasets=["hc_tj"],
    deploy_datasets=["hc_tj"],
)
