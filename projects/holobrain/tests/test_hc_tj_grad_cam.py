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

"""Focused Grad-CAM math and evaluation artifact regression tests."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cam = load_script("hc_tj_grad_cam")
evaluation = load_script("eval_hc_tj_async")


def test_grad_cam_known_spatial_target():
    features = torch.tensor([[[[[1., 2.], [3., 4.]], [[4., 3.], [2., 1.]]]]], requires_grad=True)
    score = (features[:, :, 0] * 2 - features[:, :, 1]).sum()
    gradients, = torch.autograd.grad(score, features)
    result = cam.spatial_grad_cam(features.detach(), gradients)
    torch.testing.assert_close(result, torch.tensor([[[[0., 1/7], [4/7, 1.]]]]))
    assert features.grad is None


def test_zero_and_negative_gradients_are_not_saliency():
    features = torch.ones(1, 1, 2, 3, 4)
    for gradients in (torch.zeros_like(features), -torch.ones_like(features)):
        assert not cam.spatial_grad_cam(features, gradients).any()
    with pytest.raises(ValueError, match="finite"):
        cam.spatial_grad_cam(features, torch.full_like(features, float("nan")))


def test_artifacts_preserve_rgb_and_hotspot_alignment(tmp_path):
    rgb = np.full((8, 12, 3), 80, dtype=np.uint8)
    heatmap = [[0., 0., 1.], [0., 0., 1.]]
    metadata = evaluation._save_grad_cam(
        {"imgs": [rgb]}, {"grad_cam": {"head": heatmap}}, tmp_path, 3,
    )
    np.testing.assert_array_equal(np.asarray(Image.open(metadata["original"])), rgb)
    np.testing.assert_array_equal(np.load(metadata["heatmap"]), heatmap)
    overlay = np.asarray(Image.open(metadata["overlay"]))
    np.testing.assert_array_equal(overlay[:, 0], rgb[:, 0])
    assert np.all(overlay[:, -1, 0] > rgb[:, -1, 0])
    assert not metadata["empty"]


def test_missing_server_support_is_explicit(tmp_path):
    with pytest.raises(RuntimeError, match="did not return"):
        evaluation._save_grad_cam({}, {}, tmp_path, 0)


def test_frozen_decoder_and_cached_inference_tensors(monkeypatch):
    import sys
    from types import SimpleNamespace

    # Keep this lifecycle test independent of model-loading dependencies.
    monkeypatch.setitem(
        sys.modules, "robo_orchard_lab.utils.torch",
        SimpleNamespace(to_device=lambda data, device: data),
    )
    with torch.inference_mode():
        cached = torch.ones(1)

    class Decoder(torch.nn.Module):
        def forward(self, *, feature_maps, inputs):
            value = feature_maps[0].mean() * inputs["kinematics"][0]
            return value.expand(2, 3)

        def post_process(self, raw, data, **kwargs):
            return raw

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)
            self.decoder = Decoder()
            self.data_preprocessor = None

        def _forward(self, data):
            feature = torch.ones(1, 1, 2, 3, 4) * self.weight
            result = self.decoder(feature_maps=[feature], inputs=data)
            return result, [feature], None

    model = Model().eval()
    pipeline = SimpleNamespace(
        model=model,
        _build_runtime_input=lambda *args, **kwargs: ({"kinematics": [cached]}, None),
        envelope_processor=SimpleNamespace(
            post_process=lambda action, **kwargs: SimpleNamespace(action=action)
        ),
    )
    output, explanation = cam.infer_with_grad_cam(pipeline, None, joint=1, step=0)
    assert torch.is_inference(cached)
    assert not output.action.requires_grad
    assert np.asarray(explanation["head"]).max() == 1
    assert model.weight.grad is None
    assert not model.decoder._forward_pre_hooks
    with pytest.raises(ValueError, match="step index"):
        cam.infer_with_grad_cam(pipeline, None, step=10)
    assert not model.decoder._forward_pre_hooks
