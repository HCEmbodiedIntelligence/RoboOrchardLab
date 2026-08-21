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

"""ROS-independent contracts and safety primitives for HC-TJ deployment."""

from __future__ import annotations
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

__all__ = [
    "ACTION_HZ",
    "ACTION_HORIZON",
    "ARM_JOINT_NAMES",
    "ArmCommandLimiter",
    "CONTROL_JOINT_NAMES",
    "HC_TJ_JOINT_NAMES",
    "HcTjInferenceRequest",
    "LEFT_HAND_CLOSED",
    "LEFT_HAND_JOINT_NAMES",
    "LEFT_HAND_OPEN",
    "ProtocolError",
    "RIGHT_HAND_CLOSED",
    "RIGHT_HAND_JOINT_NAMES",
    "RIGHT_HAND_OPEN",
    "TimedValue",
    "arm_actions",
    "hand_command",
    "interpolate_actions",
    "load_arm_joint_limits",
    "nearest_timed_value",
    "parse_camera_calibration",
    "parse_robot_camera_extrinsics",
    "trim_elapsed_actions",
    "validate_action_chunk",
    "validate_inference_request",
    "validate_inference_response",
]

ACTION_HZ = 30
ACTION_HORIZON = 64
RIGHT_ARM_JOINT_NAMES = tuple(f"Joint{i}_R" for i in range(1, 8))
LEFT_ARM_JOINT_NAMES = tuple(f"Joint{i}_L" for i in range(1, 8))
ARM_JOINT_NAMES = RIGHT_ARM_JOINT_NAMES + LEFT_ARM_JOINT_NAMES
CONTROL_JOINT_NAMES = ARM_JOINT_NAMES + ("leg_1", "leg_2", "zhi")
HC_TJ_JOINT_NAMES = (
    *RIGHT_ARM_JOINT_NAMES,
    "R_ban",
    *LEFT_ARM_JOINT_NAMES,
    "L_ban",
)

RIGHT_HAND_JOINT_NAMES = (
    "R_thumb_roll_joint",
    "R_thumb_abad_joint",
    "R_thumb_mcp_joint",
    "R_index_abad_joint",
    "R_index_pip_joint",
    "R_middle_pip_joint",
    "R_ring_abad_joint",
    "R_ring_pip_joint",
    "R_pinky_abad_joint",
    "R_pinky_pip_joint",
)
RIGHT_HAND_OPEN = np.asarray(
    [0.4, -1.642, 0.0, -0.13, 0.0, 0.0, 0.05, 0.0, 0.05, 0.0],
    dtype=np.float64,
)
RIGHT_HAND_CLOSED = np.asarray(
    [1.121, 0.045, 1.0, -0.05, 2.5, 2.5, 0.13, 2.5, 0.13, 2.5],
    dtype=np.float64,
)
LEFT_HAND_JOINT_NAMES = (
    "L_thumb_roll_joint",
    "L_thumb_abad_joint",
    "L_thumb_mcp_joint",
    "L_index_abad_joint",
    "L_index_pip_joint",
    "L_middle_pip_joint",
    "L_ring_abad_joint",
    "L_ring_pip_joint",
    "L_pinky_abad_joint",
    "L_pinky_pip_joint",
)
LEFT_HAND_OPEN = np.asarray(
    [-0.4, -0.045, 0.0, 0.05, 0.0, 0.0, -0.13, 0.0, -0.13, 0.0],
    dtype=np.float64,
)
LEFT_HAND_CLOSED = np.asarray(
    [-1.121, 1.642, -1.0, 0.13, 2.5, 2.5, -0.05, 2.5, 0.05, 2.5],
    dtype=np.float64,
)


class ProtocolError(ValueError):
    """Report an invalid HC-TJ asynchronous request or response contract."""


@dataclass(frozen=True, slots=True)
class TimedValue:
    """Pair a value with its ROS header timestamp in nanoseconds."""

    timestamp_ns: int
    value: Any


@dataclass(frozen=True, slots=True)
class HcTjInferenceRequest:
    """Validated model-native HC-TJ request.

    ``current_state`` and every action use the exact order in
    :data:`HC_TJ_JOINT_NAMES`. Depth is measured in meters and
    ``zhi_to_eye_mat`` maps homogeneous column vectors from ``zhi_Link`` into
    the model camera frame ``eye``.
    """

    head_color: np.ndarray
    head_depth: np.ndarray
    head_intrinsic_mat: np.ndarray
    zhi_to_eye_mat: np.ndarray
    current_state: np.ndarray
    instruction: str
    remaining_actions: np.ndarray | None = None
    delay_horizon: int | None = None


def _array(
    value: Any,
    *,
    name: str,
    shape: tuple[int | None, ...],
    dtype: np.dtype | tuple[np.dtype, ...],
) -> np.ndarray:
    """Validate an ndarray without silently coercing wire dtype or shape."""
    if not isinstance(value, np.ndarray):
        raise ProtocolError(f"{name} must be a NumPy array")
    dtypes = dtype if isinstance(dtype, tuple) else (dtype,)
    if value.dtype not in dtypes:
        expected = ", ".join(str(item) for item in dtypes)
        raise ProtocolError(
            f"{name} dtype must be {expected}, got {value.dtype}"
        )
    if value.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(value.shape, shape, strict=True)
    ):
        rendered = tuple("N" if item is None else item for item in shape)
        raise ProtocolError(
            f"{name} shape must be {rendered}, got {value.shape}"
        )
    if value.size == 0 or not np.all(np.isfinite(value)):
        raise ProtocolError(f"{name} must be non-empty and finite")
    return value


def _homogeneous_matrix(value: Any, name: str) -> np.ndarray:
    matrix = _array(
        value,
        name=name,
        shape=(4, 4),
        dtype=(np.dtype("float32"), np.dtype("float64")),
    )
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ProtocolError(
            f"{name} must have homogeneous bottom row [0,0,0,1]"
        )
    return matrix


def validate_inference_request(
    *,
    head_color: Any,
    head_depth: Any,
    head_intrinsic_mat: Any,
    zhi_to_eye_mat: Any,
    current_state: Any,
    joint_names: Sequence[str],
    instruction: Any,
    remaining_actions: Any = None,
    delay_horizon: Any = None,
) -> HcTjInferenceRequest:
    """Validate and normalize one model-native asynchronous request."""
    color = _array(
        head_color,
        name="head_color",
        shape=(None, None, 3),
        dtype=np.dtype("uint8"),
    )
    depth = _array(
        head_depth,
        name="head_depth",
        shape=color.shape[:2],
        dtype=(np.dtype("float32"), np.dtype("float64")),
    )
    if np.any(depth < 0):
        raise ProtocolError("head_depth in meters must be non-negative")
    intrinsic = _homogeneous_matrix(head_intrinsic_mat, "head_intrinsic")
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ProtocolError("head_intrinsic focal lengths must be positive")
    zhi_to_eye = _homogeneous_matrix(zhi_to_eye_mat, "zhi_to_eye")
    state = _array(
        current_state,
        name="current_state",
        shape=(len(HC_TJ_JOINT_NAMES),),
        dtype=np.dtype("float32"),
    )
    if tuple(joint_names) != HC_TJ_JOINT_NAMES:
        raise ProtocolError(
            "joint_names must exactly match " + json.dumps(HC_TJ_JOINT_NAMES)
        )
    if not isinstance(instruction, str) or not instruction.strip():
        raise ProtocolError("instruction must be a non-empty string")

    remaining = None
    delay = None
    if remaining_actions is not None:
        remaining = _array(
            remaining_actions,
            name="remaining_actions",
            shape=(None, len(HC_TJ_JOINT_NAMES)),
            dtype=np.dtype("float32"),
        )
        if remaining.shape[0] > ACTION_HORIZON:
            raise ProtocolError(
                f"remaining_actions may contain at most {ACTION_HORIZON} steps"
            )
        if isinstance(delay_horizon, bool) or not isinstance(
            delay_horizon, int
        ):
            raise ProtocolError("delay_horizon must be an integer")
        if not 0 <= delay_horizon <= remaining.shape[0]:
            raise ProtocolError(
                "delay_horizon exceeds the remaining action length"
            )
        delay = delay_horizon
    elif delay_horizon not in (None, 0):
        raise ProtocolError(
            "delay_horizon is only valid when remaining_actions is provided"
        )

    return HcTjInferenceRequest(
        head_color=color,
        head_depth=depth,
        head_intrinsic_mat=intrinsic,
        zhi_to_eye_mat=zhi_to_eye,
        current_state=state,
        instruction=instruction.strip(),
        remaining_actions=remaining,
        delay_horizon=delay,
    )


def validate_inference_response(payload: Any) -> np.ndarray:
    """Validate a v1 JSON response and return its native 30 Hz actions."""
    if not isinstance(payload, dict):
        raise ProtocolError("response must be a JSON object")
    if tuple(payload.get("joint_names", ())) != HC_TJ_JOINT_NAMES:
        raise ProtocolError("response joint_names do not match HC-TJ order")
    if payload.get("action_hz") != ACTION_HZ:
        raise ProtocolError(f"response action_hz must be {ACTION_HZ}")
    try:
        actions = np.asarray(payload["actions"], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as error:
        raise ProtocolError(
            "response actions are missing or invalid"
        ) from error
    if (
        actions.ndim != 2
        or actions.shape[1] != len(HC_TJ_JOINT_NAMES)
        or not 0 < actions.shape[0] <= ACTION_HORIZON
        or not np.all(np.isfinite(actions))
    ):
        raise ProtocolError(
            f"response actions must be finite [1..{ACTION_HORIZON},16]"
        )
    if payload.get("action_horizon") != actions.shape[0]:
        raise ProtocolError("response action_horizon does not match actions")
    inference_ms = payload.get("inference_ms")
    if (
        isinstance(inference_ms, bool)
        or not isinstance(inference_ms, (int, float))
        or not math.isfinite(inference_ms)
        or inference_ms < 0
    ):
        raise ProtocolError(
            "response inference_ms must be finite and non-negative"
        )
    return actions


def trim_elapsed_actions(
    actions: np.ndarray, elapsed_seconds: float, action_hz: int = ACTION_HZ
) -> np.ndarray:
    """Drop every model step whose execution interval has already elapsed."""
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        raise ValueError("elapsed_seconds must be finite and non-negative")
    if action_hz <= 0:
        raise ValueError("action_hz must be positive")
    action_array = np.asarray(actions)
    if action_array.ndim != 2:
        raise ValueError("actions must be a two-dimensional array")
    expired_steps = int(math.floor(elapsed_seconds * action_hz + 1e-12))
    return action_array[min(expired_steps, len(action_array)) :]


def interpolate_actions(
    actions: np.ndarray,
    *,
    source_hz: int = ACTION_HZ,
    target_hz: int = 100,
) -> np.ndarray:
    """Linearly resample an action chunk while preserving its full duration."""
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("actions must be a non-empty finite 2-D array")
    if source_hz <= 0 or target_hz <= 0:
        raise ValueError("source_hz and target_hz must be positive")
    output_steps = int(math.ceil(len(values) * target_hz / source_hz))
    source_times = np.arange(len(values), dtype=np.float64) / source_hz
    target_times = np.arange(output_steps, dtype=np.float64) / target_hz
    return np.stack(
        [
            np.interp(target_times, source_times, values[:, joint])
            for joint in range(values.shape[1])
        ],
        axis=1,
    ).astype(np.float32)


def arm_actions(actions: np.ndarray) -> np.ndarray:
    """Slice 16-axis actions into right-then-left arm commands."""
    values = np.asarray(actions)
    if values.shape[-1:] != (len(HC_TJ_JOINT_NAMES),):
        raise ValueError("HC-TJ actions must have a final dimension of 16")
    return np.concatenate((values[..., :7], values[..., 8:15]), axis=-1)


def hand_command(
    side: str, closure: float
) -> tuple[tuple[str, ...], np.ndarray]:
    """Map binary closure to a complete 10-axis hand command."""
    value = float(np.clip(closure, 0.0, 1.0))
    if side == "right":
        return RIGHT_HAND_JOINT_NAMES, (
            RIGHT_HAND_OPEN + value * (RIGHT_HAND_CLOSED - RIGHT_HAND_OPEN)
        )
    if side == "left":
        return LEFT_HAND_JOINT_NAMES, (
            LEFT_HAND_OPEN + value * (LEFT_HAND_CLOSED - LEFT_HAND_OPEN)
        )
    raise ValueError("side must be 'right' or 'left'")


def load_arm_joint_limits(
    urdf_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load lower and upper limits for the canonical 14 arm joints."""
    root = ET.parse(urdf_path).getroot()
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is not None and {"lower", "upper"} <= limit.attrib.keys():
            limits[joint.attrib["name"]] = (
                float(limit.attrib["lower"]),
                float(limit.attrib["upper"]),
            )
    missing = [name for name in ARM_JOINT_NAMES if name not in limits]
    if missing:
        raise ValueError(f"HC-TJ URDF is missing arm joint limits: {missing}")
    lower, upper = zip(
        *(limits[name] for name in ARM_JOINT_NAMES), strict=True
    )
    return np.asarray(lower), np.asarray(upper)


def validate_action_chunk(
    actions: np.ndarray,
    feedback_arm: np.ndarray,
    *,
    first_point_jump_rad: float = 0.10,
) -> None:
    """Reject empty, non-finite, or discontinuous action chunks before use."""
    values = np.asarray(actions)
    feedback = np.asarray(feedback_arm)
    if values.ndim != 2 or values.shape[1] != 16 or len(values) == 0:
        raise ValueError("action chunk must have shape [N,16] with N > 0")
    if feedback.shape != (14,):
        raise ValueError("feedback_arm must have shape [14]")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(feedback)):
        raise ValueError("action chunk and feedback must be finite")
    jump = float(np.max(np.abs(arm_actions(values[0]) - feedback)))
    if jump > first_point_jump_rad:
        raise ValueError(
            f"first action jumps {jump:.6f} rad from feedback; "
            f"limit is {first_point_jump_rad:.6f} rad"
        )


@dataclass(slots=True)
class ArmCommandLimiter:
    """Apply URDF position bounds and per-cycle arm velocity limits."""

    lower: np.ndarray
    upper: np.ndarray
    max_velocity_rad_s: float = 0.25
    previous: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.lower = np.asarray(self.lower, dtype=np.float64)
        self.upper = np.asarray(self.upper, dtype=np.float64)
        if self.lower.shape != (14,) or self.upper.shape != (14,):
            raise ValueError("arm limits must each have shape [14]")
        if np.any(self.lower > self.upper):
            raise ValueError("arm lower limits must not exceed upper limits")
        if self.max_velocity_rad_s <= 0:
            raise ValueError("max_velocity_rad_s must be positive")

    def reset(self, feedback_arm: np.ndarray) -> None:
        """Start rate limiting from the latest measured arm position."""
        feedback = np.asarray(feedback_arm, dtype=np.float64)
        if feedback.shape != (14,) or not np.all(np.isfinite(feedback)):
            raise ValueError("feedback_arm must be finite with shape [14]")
        self.previous = np.clip(feedback, self.lower, self.upper)

    def step(
        self, target_arm: np.ndarray, period_seconds: float
    ) -> np.ndarray:
        """Return a bounded command and store it for the next cycle."""
        if self.previous is None:
            raise RuntimeError("reset() must be called before step()")
        target = np.asarray(target_arm, dtype=np.float64)
        if target.shape != (14,) or not np.all(np.isfinite(target)):
            raise ValueError("target_arm must be finite with shape [14]")
        if not math.isfinite(period_seconds) or period_seconds <= 0:
            raise ValueError("period_seconds must be finite and positive")
        bounded = np.clip(target, self.lower, self.upper)
        max_delta = self.max_velocity_rad_s * period_seconds
        command = self.previous + np.clip(
            bounded - self.previous, -max_delta, max_delta
        )
        self.previous = command
        return command.copy()


def nearest_timed_value(
    values: Sequence[TimedValue], target_ns: int, tolerance_ns: int
) -> TimedValue:
    """Return the nearest timestamped value, preferring the earlier on ties."""
    if not values:
        raise ValueError("cannot synchronize an empty observation stream")
    selected = min(
        values,
        key=lambda item: (
            abs(item.timestamp_ns - target_ns),
            item.timestamp_ns,
        ),
    )
    if abs(selected.timestamp_ns - target_ns) > tolerance_ns:
        raise ValueError(
            "nearest observation is outside synchronization tolerance"
        )
    return selected


def parse_camera_calibration(payload: str | dict[str, Any]) -> np.ndarray:
    """Parse ``hc_camera_calibration/v1`` into a 4x4 RGB intrinsic matrix."""
    data = json.loads(payload) if isinstance(payload, str) else payload
    try:
        if data["schema"] != "hc_camera_calibration/v1":
            raise ValueError("unsupported schema")
        if data["camera"] != "eye":
            raise ValueError("camera must be 'eye'")
        color = data["resolution"]["color"]
        depth = data["resolution"]["depth"]
        if (color["width"], color["height"]) != (
            depth["width"],
            depth["height"],
        ):
            raise ValueError("RGB and depth resolutions must match")
        rgb = data["rgb"]
        values = np.asarray(
            [rgb["fx"], rgb["fy"], rgb["cx"], rgb["cy"]],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)) or min(values[:2]) <= 0:
            raise ValueError("RGB intrinsic is invalid")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"invalid hc_camera_calibration/v1 payload: {error}"
        ) from error
    intrinsic = np.eye(4, dtype=np.float64)
    intrinsic[0, 0], intrinsic[1, 1] = values[:2]
    intrinsic[0, 2], intrinsic[1, 2] = values[2:]
    return intrinsic


def parse_robot_camera_extrinsics(
    payload: str | dict[str, Any],
) -> np.ndarray:
    """Parse the static camera pose into ``zhi_Link -> eye``.

    The JSON transform describes the model camera frame ``eye`` relative to
    ``zhi_Link`` (stored as ``eye -> zhi_Link``), so this function returns its
    inverse for the model's historical world-to-camera input.
    """
    data = json.loads(payload) if isinstance(payload, str) else payload
    try:
        if data["schema"] != "hc_robot_camera_extrinsics/v1":
            raise ValueError("unsupported schema")
        if data["common_frame"] != "zhi_Link":
            raise ValueError("common_frame must be 'zhi_Link'")
        matches = [
            item
            for item in data["transforms"]
            if item.get("parent_frame") == "zhi_Link"
            and item.get("child_frame") == "camera_head"
        ]
        if len(matches) != 1:
            raise ValueError(
                "exactly one zhi_Link -> camera_head entry is required"
            )
        translation = np.asarray(matches[0]["translation_m"], dtype=np.float64)
        quat_xyzw = np.asarray(matches[0]["rotation_xyzw"], dtype=np.float64)
        if translation.shape != (3,) or quat_xyzw.shape != (4,):
            raise ValueError("translation/quaternion shape is invalid")
        if not np.all(np.isfinite(translation)) or not np.all(
            np.isfinite(quat_xyzw)
        ):
            raise ValueError("translation/quaternion must be finite")
        norm = float(np.linalg.norm(quat_xyzw))
        if not math.isclose(norm, 1.0, abs_tol=1e-5):
            raise ValueError("camera quaternion must be normalized")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"invalid hc_robot_camera_extrinsics/v1 payload: {error}"
        ) from error
    x, y, z, w = quat_xyzw / norm
    rotation = np.asarray(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ],
            [
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ],
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
        ]
    )
    eye_to_zhi_mat = np.eye(4, dtype=np.float64)
    eye_to_zhi_mat[:3, :3] = rotation
    eye_to_zhi_mat[:3, 3] = translation
    return np.linalg.inv(eye_to_zhi_mat)
