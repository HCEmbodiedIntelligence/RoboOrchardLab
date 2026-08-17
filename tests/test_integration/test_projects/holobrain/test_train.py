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

import torch

from projects.holobrain.scripts.train import evaluate


class _Metric:
    def __init__(self):
        self.updates = []
        self.reset_called = False

    def update(self, batch, model_outputs):
        self.updates.append((batch, model_outputs))

    def compute(self, accelerator):
        assert accelerator.is_main_process
        return {"count": len(self.updates)}

    def reset(self):
        self.reset_called = True


class _Accelerator:
    is_main_process = True

    def __init__(self):
        self.wait_count = 0

    def wait_for_everyone(self):
        self.wait_count += 1


def test_evaluate_runs_forward_only_and_restores_model_mode():
    model = torch.nn.Linear(1, 1)
    model.train()
    metric = _Metric()
    accelerator = _Accelerator()
    batches = [torch.ones((1, 1)), torch.full((1, 1), 2.0)]

    result = evaluate(model, batches, metric, accelerator)

    assert result == {"count": 2}
    assert len(metric.updates) == 2
    assert all(not output.requires_grad for _, output in metric.updates)
    assert metric.reset_called
    assert accelerator.wait_count == 2
    assert model.training
