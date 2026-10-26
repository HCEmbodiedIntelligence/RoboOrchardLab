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

"""Action-conditioned Grad-CAM for the HoloBrain HC-TJ inference server."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch


def spatial_grad_cam(
    features: torch.Tensor, gradients: torch.Tensor
) -> torch.Tensor:
    """Return per-view [B, V, H, W] positive Grad-CAM, normalized to [0, 1].

    Inputs are decoder spatial features and their target gradients in
    [B, V, C, H, W] order. A zero map means no positive attribution;
    normalization never invents saliency for a constant or negative map.
    """
    if features.ndim != 5 or features.shape != gradients.shape:
        raise ValueError("Grad-CAM requires matching [B,V,C,H,W] tensors")
    features, gradients = features.float(), gradients.float()
    if not torch.isfinite(features).all() or not torch.isfinite(gradients).all():
        raise ValueError("Grad-CAM features or gradients are not finite")
    weights = gradients.mean(dim=(-2, -1), keepdim=True)
    cam = (weights * features).sum(dim=2).relu()
    return cam / cam.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)


def infer_with_grad_cam(
    pipeline: Any,
    model_input: Any,
    *,
    joint: int | None = None,
    step: int | None = None,
) -> tuple[Any, dict]:
    """Run one prediction and explain its returned action using spatial Grad-CAM.

    Caller must hold the pipeline lock. This uses HoloBrain's differentiable
    ``_forward`` because ``predict`` disables gradients. The decoder's first
    spatial feature level is treated as a leaf, including for frozen models.
    Hooks are removed on success or failure; parameter gradients are untouched.
    Attribution describes the RGB-aligned multimodal decoder features, not
    raw pixels or a causal object detector. Memory grows with denoising steps.
    """
    from robo_orchard_lab.utils.torch import to_device

    model = pipeline.model
    if hasattr(model, "module"):
        model = model.module
    if model.training or not hasattr(model, "_forward"):
        raise ValueError("Grad-CAM requires an eval-mode HoloBrain model")
    captured = []

    def capture(module, positional, keyword):
        maps = list(keyword["feature_maps"])
        maps[0] = maps[0].detach().requires_grad_(True)
        captured.append(maps[0])
        return positional, {**keyword, "feature_maps": maps}

    with torch.inference_mode(False), torch.enable_grad():
        with torch.no_grad():
            data, context = pipeline._build_runtime_input(
                model_input, is_batch=False
            )
            # Ordinary pipeline calls cache inference-mode tensors in the
            # kinematics chain. A request-local copy clones those tensors
            # outside inference mode and leaves the shared cache untouched.
            if "kinematics" in data:
                data["kinematics"] = deepcopy(data["kinematics"])
            data = to_device(data, next(model.parameters()).device)
            if model.data_preprocessor is not None:
                data = model.data_preprocessor(
                    data, next(model.parameters()).device
                )
        handle = model.decoder.register_forward_pre_hook(
            capture, with_kwargs=True
        )
        try:
            raw, _, text = model._forward(data)
            results = model.decoder.post_process(raw, data, text_dict=text)
            output = pipeline.envelope_processor.post_process(
                results, model_input=data, processor_context=context

            )
            action = output.action
            if action.ndim != 2:
                raise ValueError("Grad-CAM expects one [steps,joints] action sequence")
            if joint is not None and not 0 <= joint < action.shape[1]:
                raise ValueError("Grad-CAM joint index is outside the action sequence")
            if step is not None and not 0 <= step < action.shape[0]:
                raise ValueError("Grad-CAM step index is outside the action sequence")
            selected = action if step is None else action[step : step + 1]
            selected = selected if joint is None else selected[:, joint : joint + 1]
            score = selected.float().square().mean()
            if len(captured) != 1:
                raise ValueError("Expected exactly one decoder feature capture")
            gradients, = torch.autograd.grad(score, captured[0])
            maps = spatial_grad_cam(captured[0].detach(), gradients.detach())
            if maps.shape[:2] != (1, 1):
                raise ValueError("HC-TJ Grad-CAM requires one sample and one head view")
            explanation = {
                "method": "grad_cam",
                "layer": "decoder.feature_maps[0]",
                "target": "mean_squared_action",
                "joint": joint,
                "step": step,
                "score": float(score.detach().cpu()),
                "head": maps[0, 0].cpu().tolist(),
            }
            output.action = action.detach()
            return output, explanation
        finally:
            handle.remove()
