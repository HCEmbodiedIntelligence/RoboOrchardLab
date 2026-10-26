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

"""Evaluate the HC-TJ asynchronous endpoint against real Arrow samples."""

from __future__ import annotations
import argparse
import json
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import requests

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from projects.holobrain.scripts.hc_tj_async_core import (  # noqa: E402
    ACTION_HZ,
    HC_TJ_JOINT_NAMES,
    arm_actions,
    trim_elapsed_actions,
    validate_inference_response,
)

__all__ = ["evaluate_hc_tj_inference_server"]

DEFAULT_DATASET_PATH = (
    REPOSITORY_ROOT / "projects/holobrain/data/arrow_dataset/hc_tj_260818_fix"
)


def _upload(name: str, value: np.ndarray) -> tuple[str, bytes, str]:
    payload = BytesIO()
    np.save(payload, value, allow_pickle=False)
    return f"{name}.npy", payload.getvalue(), "application/octet-stream"


def _request(sample: dict[str, Any]) -> tuple[dict, dict]:
    arrays = {
        "head_color": np.asarray(sample["imgs"][0], dtype=np.uint8),
        "head_depth": np.asarray(sample["depths"][0], dtype=np.float32),
        "head_intrinsic": np.asarray(sample["intrinsic"][0], dtype=np.float64),
        "zhi_to_eye": np.asarray(sample["T_world2cam"][0], dtype=np.float64),
        "current_state": np.asarray(
            sample["joint_state"][0], dtype=np.float32
        ),
    }
    return (
        {name: _upload(name, value) for name, value in arrays.items()},
        {
            "joint_names": json.dumps(HC_TJ_JOINT_NAMES),
            "instruction": sample["text"],
        },
    )


def _arm_metrics(
    prediction: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    error = arm_actions(prediction) - arm_actions(target)
    return {
        "mae_rad": float(np.mean(np.abs(error))),
        "rmse_rad": float(np.sqrt(np.mean(np.square(error)))),
        "max_abs_error_rad": float(np.max(np.abs(error))),
    }


def _hand_metrics(
    prediction: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    right_error = np.clip(prediction[:, 7], 0, 1) - target[:, 7]
    left_error = np.clip(prediction[:, 15], 0, 1) - target[:, 15]
    return {
        "right_mae": float(np.mean(np.abs(right_error))),
        "left_mae": float(np.mean(np.abs(left_error))),
        "right_binary_error_rate": float(
            np.mean((prediction[:, 7] >= 0.5) != (target[:, 7] >= 0.5))
        ),
        "left_binary_error_rate": float(
            np.mean((prediction[:, 15] >= 0.5) != (target[:, 15] >= 0.5))
        ),
    }


def _save_grad_cam(sample, payload, output_dir: Path, sample_index: int) -> dict:
    """Save the returned spatial map and its overlay on the source RGB image."""
    from PIL import Image

    explanation = payload.get("grad_cam")
    if not isinstance(explanation, dict) or "head" not in explanation:
        raise RuntimeError("Server did not return Grad-CAM; restart the updated server")
    heatmap = np.asarray(explanation["head"], dtype=np.float32)
    if (heatmap.ndim != 2 or not heatmap.size
            or not np.isfinite(heatmap).all()
            or heatmap.min() < 0 or heatmap.max() > 1):
        raise ValueError("Server returned an invalid Grad-CAM heatmap")
    rgb = np.asarray(sample["imgs"][0], dtype=np.uint8)
    original = Image.fromarray(rgb)
    resized = np.asarray(
        Image.fromarray(heatmap).resize(original.size, Image.Resampling.BILINEAR)
    )
    # Red/yellow highlights with opacity proportional to positive attribution.
    color = np.zeros_like(rgb, dtype=np.float32)
    color[..., 0] = 255
    color[..., 1] = 255 * (1 - resized)
    alpha = 0.6 * resized[..., None]
    overlay = np.clip(rgb * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"sample_{sample_index:06d}_head"
    original.save(f"{stem}_rgb.png")
    Image.fromarray(overlay).save(f"{stem}_grad_cam.png")
    np.save(f"{stem}_grad_cam.npy", heatmap, allow_pickle=False)
    metadata = {key: value for key, value in explanation.items() if key != "head"}
    metadata.update(
        overlay=f"{stem}_grad_cam.png", heatmap=f"{stem}_grad_cam.npy",
        original=f"{stem}_rgb.png", empty=bool(not heatmap.any()),
    )
    Path(f"{stem}_grad_cam.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def evaluate_hc_tj_inference_server(
    server_url: str,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    sample_indices: tuple[int, ...] = (0, 300, 600),
    timeout_seconds: float = 300,
    grad_cam_dir: Path | None = None,
    grad_cam_joint: int | None = None,
    grad_cam_step: int | None = None,
) -> dict[str, Any]:
    """Call v1 and report protocol, accuracy, and latency metrics."""
    from robo_orchard_lab.dataset.hc_tj import HcTjRODataset

    if grad_cam_dir is None and (grad_cam_joint is not None or grad_cam_step is not None):
        raise ValueError("Grad-CAM selectors require grad_cam_dir")
    if not (dataset_path / "state.json").is_file():
        raise FileNotFoundError(f"converted dataset not found: {dataset_path}")
    dataset = HcTjRODataset(
        paths=[str(dataset_path)], hist_steps=1, pred_steps=64, transforms=None
    )
    reports = []
    for sample_index in sample_indices:
        sample = dataset[sample_index]
        files, form = _request(sample)
        if grad_cam_dir is not None:
            form["grad_cam"] = "true"
            if grad_cam_joint is not None:
                form["grad_cam_joint"] = str(grad_cam_joint)
            if grad_cam_step is not None:
                form["grad_cam_step"] = str(grad_cam_step)
        started = time.monotonic()
        response = requests.post(
            server_url, files=files, data=form, timeout=timeout_seconds
        )
        latency_s = time.monotonic() - started
        if not response.ok:
            raise RuntimeError(
                f"server returned HTTP {response.status_code}: {response.text}"
            )
        payload = response.json()
        prediction = validate_inference_response(payload)
        visualization = (
            _save_grad_cam(sample, payload, grad_cam_dir, sample_index)
            if grad_cam_dir is not None else None
        )
        target = np.asarray(
            sample["master_joint_state"][1 : 1 + len(prediction)],
            dtype=np.float32,
        )
        if target.shape != prediction.shape:
            raise ValueError(
                f"Arrow target shape {target.shape} does not match "
                f"{prediction.shape}"
            )
        effective = trim_elapsed_actions(prediction, latency_s, ACTION_HZ)
        reports.append(
            {
                "sample_index": sample_index,
                **({"grad_cam": visualization} if visualization is not None else {}),
                "protocol": {
                    "shape": list(prediction.shape),
                    "joint_order_ok": tuple(payload["joint_names"])
                    == HC_TJ_JOINT_NAMES,
                    "finite": bool(np.all(np.isfinite(prediction))),
                    "action_min": float(prediction.min()),
                    "action_max": float(prediction.max()),
                },
                "arm": _arm_metrics(prediction, target),
                "hands": _hand_metrics(prediction, target),
                "latency_ms": latency_s * 1000,
                "server_inference_ms": float(payload["inference_ms"]),
                "effective_action_steps": len(effective),
            }
        )
    report = {
        "server_url": server_url,
        "dataset_path": str(dataset_path),
        "action_hz": ACTION_HZ,
        "samples": reports,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", required=True)
    parser.add_argument(
        "--dataset-path", type=Path, default=DEFAULT_DATASET_PATH
    )
    parser.add_argument("--sample-index", type=int, action="append")
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--grad-cam-dir", type=Path, help="Enable Grad-CAM and save RGB/overlay/NPY files here")
    parser.add_argument("--grad-cam-joint", type=int, choices=range(16), help="Explain one joint (zero-based); default: all joints")
    parser.add_argument("--grad-cam-step", type=int, choices=range(64), help="Explain one action step (zero-based); default: all steps")
    args = parser.parse_args()
    if args.grad_cam_dir is None and (args.grad_cam_joint is not None or args.grad_cam_step is not None):
        parser.error("--grad-cam-joint/step require --grad-cam-dir")
    evaluate_hc_tj_inference_server(
        args.server_url,
        args.dataset_path,
        tuple(args.sample_index or (0, 300, 600)),
        args.timeout_seconds,
        args.grad_cam_dir,
        args.grad_cam_joint,
        args.grad_cam_step,
    )


if __name__ == "__main__":
    main()
