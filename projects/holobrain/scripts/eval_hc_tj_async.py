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


def evaluate_hc_tj_inference_server(
    server_url: str,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    sample_indices: tuple[int, ...] = (0, 300, 600),
    timeout_seconds: float = 300,
) -> dict[str, Any]:
    """Call v1 and report protocol, accuracy, and latency metrics."""
    from robo_orchard_lab.dataset.hc_tj import HcTjRODataset

    if not (dataset_path / "state.json").is_file():
        raise FileNotFoundError(f"converted dataset not found: {dataset_path}")
    dataset = HcTjRODataset(
        paths=[str(dataset_path)], hist_steps=1, pred_steps=64, transforms=None
    )
    reports = []
    for sample_index in sample_indices:
        sample = dataset[sample_index]
        files, form = _request(sample)
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
    args = parser.parse_args()
    evaluate_hc_tj_inference_server(
        args.server_url,
        args.dataset_path,
        tuple(args.sample_index or (0, 300, 600)),
        args.timeout_seconds,
    )


if __name__ == "__main__":
    main()
