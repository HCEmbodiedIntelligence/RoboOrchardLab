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

"""Run asynchronous HoloBrain inference against the HC-TJ ROS 2 robot."""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
import threading
import time
import uuid
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import requests

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
try:
    sys.path.remove(str(_REPOSITORY_ROOT))
except ValueError:
    pass
sys.path.insert(0, str(_REPOSITORY_ROOT))

from projects.holobrain.scripts.hc_tj_async_core import (  # noqa: E402
    ACTION_HZ,
    CONTROL_JOINT_NAMES,
    HC_TJ_JOINT_NAMES,
    ActionDiscontinuityError,
    ArmCommandLimiter,
    TimedValue,
    arm_actions,
    hand_command,
    load_arm_joint_limits,
    measured_hand_closures,
    nearest_timed_value,
    parse_camera_calibration,
    parse_robot_camera_extrinsics,
    sample_actions,
    trim_elapsed_actions,
    validate_action_chunk,
    validate_inference_response,
)
from projects.holobrain.scripts.hc_tj_control import (  # noqa: E402
    ControlRelay,
    RobotControlGate,
)
from projects.holobrain.scripts.hc_tj_motion import MotionInterface  # noqa: E402

__all__ = [
    "DEFAULT_JOINT_POSITIONS",
    "HcTjAsyncClientConfig",
    "ObservationBuffer",
    "RobotInterface",
    "main",
]


# Positions follow CONTROL_JOINT_NAMES: both 7-axis arms, then leg_1,
# leg_2, and zhi. Hands are commanded by their dedicated publishers.
DEFAULT_JOINT_POSITIONS = (
    -1.130321733036805,
    -0.7488408179558761,
    1.2421810235648036,
    -1.5615996326025259,
    0.14459746322931552,
    0.436243163681092,
    -0.03755520693488968,
    1.4708313794854644,
    -0.7665212115377759,
    -1.5988311572740517,
    -1.612405156192131,
    0.03962631818631988,
    0.13397872423862664,
    -0.010267844630530324,
    0.21072105306881997,
    0.6668054431158328,
    0.15834697039601572,
)
_RESET_POSITION_TOLERANCE_RAD = 0.02
_MAX_CONSECUTIVE_DAGGER_JUMPS = 3


class RobotInterface(str, Enum):
    """Select one command backend without changing legacy defaults."""

    LEGACY = "legacy"
    MOTION = "hc_motion"


@dataclass(frozen=True, slots=True)
class HcTjAsyncClientConfig:
    """Configure the HC-TJ observation, inference, and control boundaries.

    Legacy hands use command-derived estimates, including with DAgger;
    measured hand feedback is required only by the Motion backend.
    """

    server_url: str
    instruction: str
    urdf_path: Path
    depth_units_per_meter: float
    enable_control: bool = False
    robot_interface: RobotInterface = RobotInterface.LEGACY
    motion_identity_mapping: bool = False
    enable_dagger: bool = False
    control_token: str = ""
    ownership_topic: str = "/status/control/ownership"
    robot_info_topic: str = "/status/robot/info"
    hand_state_topic: str = "/observation/hands/joint_states"
    allow_competing_publishers: bool = False
    inference_hz: float = 2.0
    control_hz: float = 100.0
    request_timeout_s: float = 30.0
    camera_timeout_s: float = 1.0
    state_timeout_s: float = 0.25
    max_arm_velocity_rad_s: float = 0.25
    max_arm_acceleration_rad_s2: float = 1.0
    """Command acceleration bound retained across chunks, in rad/s²."""
    first_point_jump_rad: float = 0.30
    latency_ewma_alpha: float = 0.25
    color_topic: str = "/observation/cameras/head/color/compressed"
    depth_topic: str = "/observation/cameras/head/depth/compressed"
    calibration_topic: str = "/observation/cameras/head/calibration"
    extrinsics_topic: str = "/observation/cameras/head/extrinsics"
    joint_state_topic: str = "/observation/joint_states"
    joint_command_topic: str = "/hc_teleop/joint_cmd"
    right_hand_topic: str = "/hc_teleop/joint_cmd_finger_right"
    left_hand_topic: str = "/hc_teleop/joint_cmd_finger_left"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "robot_interface", RobotInterface(self.robot_interface)
        )
        version = "v2" if self.enable_dagger else "v1"
        if not self.server_url.endswith(f"/hc_tj_async/{version}"):
            raise ValueError(f"server_url must target /hc_tj_async/{version}")
        if self.enable_dagger and len(self.control_token) < 16:
            raise ValueError(
                "DAgger requires a control_token of at least 16 characters"
            )
        if (
            self.robot_interface is RobotInterface.MOTION
            and not self.motion_identity_mapping
        ):
            raise ValueError(
                "Motion requires explicit identity mapping opt-in"
            )
        if not self.instruction.strip():
            raise ValueError("instruction must not be empty")
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"HC-TJ URDF not found: {self.urdf_path}")
        if self.depth_units_per_meter <= 0:
            raise ValueError(
                "depth_units_per_meter must be explicitly positive"
            )
        if not 0 < self.inference_hz <= 30.0:
            raise ValueError("inference_hz must be in (0, 30]")
        if (
            not math.isfinite(self.control_hz)
            or self.control_hz <= 0
            or self.request_timeout_s <= 0
        ):
            raise ValueError(
                "control_hz and request_timeout_s must be positive"
            )
        if self.camera_timeout_s <= 0 or self.state_timeout_s <= 0:
            raise ValueError("camera and state timeouts must be positive")
        if self.max_arm_velocity_rad_s <= 0:
            raise ValueError("arm velocity must be positive")
        if (
            not math.isfinite(self.max_arm_acceleration_rad_s2)
            or self.max_arm_acceleration_rad_s2 <= 0
        ):
            raise ValueError("arm acceleration must be finite and positive")
        if self.first_point_jump_rad <= 0:
            raise ValueError("first_point_jump_rad must be positive")
        if not 0 < self.latency_ewma_alpha <= 1:
            raise ValueError("latency_ewma_alpha must be in (0, 1]")


class ObservationBuffer:
    """Cache bounded ROS observations and synchronize on RGB time."""

    def __init__(self, capacity: int = 120) -> None:
        self._lock = threading.Lock()
        # Only the latest RGB anchors synchronization; other streams need
        # history to find the closest sample. Capacity zero stays disabled.
        self._colors: deque[TimedValue] = deque(maxlen=min(capacity, 1))
        self._depths: deque[TimedValue] = deque(maxlen=capacity)
        self._joints: deque[TimedValue] = deque(maxlen=capacity)
        self._intrinsic_mat: np.ndarray | None = None
        self._zhi_to_eye_mat: np.ndarray | None = None

    def add_color(self, timestamp_ns: int, data: bytes) -> None:
        """Cache one JPEG payload under its message timestamp."""
        with self._lock:
            self._colors.append(
                TimedValue(timestamp_ns, bytes(data), time.monotonic())
            )

    def add_depth(self, timestamp_ns: int, data: bytes) -> None:
        """Cache one 16UC1 PNG payload under its message timestamp."""
        with self._lock:
            self._depths.append(
                TimedValue(timestamp_ns, bytes(data), time.monotonic())
            )

    def add_joints(self, timestamp_ns: int, positions: np.ndarray) -> None:
        """Cache one canonical 17-axis feedback sample."""
        values = np.asarray(positions, dtype=np.float64)
        if values.shape != (17,) or not np.all(np.isfinite(values)):
            raise ValueError("joint feedback must be finite with shape [17]")
        with self._lock:
            self._joints.append(
                TimedValue(timestamp_ns, values.copy(), time.monotonic())
            )

    def set_calibration(self, payload: str) -> None:
        """Validate and atomically replace the camera intrinsic matrix."""
        value = parse_camera_calibration(payload)
        with self._lock:
            self._intrinsic_mat = value

    def set_extrinsics(self, payload: str) -> None:
        """Validate and atomically replace the static zhi-to-eye transform."""
        value = parse_robot_camera_extrinsics(payload)
        with self._lock:
            self._zhi_to_eye_mat = value

    def synchronized(
        self,
    ) -> tuple[TimedValue, TimedValue, TimedValue, np.ndarray, np.ndarray]:
        """Return newest RGB with depth <=50 ms and joints <=20 ms."""
        with self._lock:
            if (
                not self._colors
                or self._intrinsic_mat is None
                or self._zhi_to_eye_mat is None
            ):
                raise ValueError(
                    "color, calibration, or extrinsics are unavailable"
                )
            color = self._colors[-1]
            depth = nearest_timed_value(
                self._depths,
                color.timestamp_ns,
                50_000_000,
                stream_name="depth",
            )
            joints = nearest_timed_value(
                self._joints,
                color.timestamp_ns,
                20_000_000,
                stream_name="joints",
            )
            return (
                color,
                depth,
                joints,
                self._intrinsic_mat.copy(),
                self._zhi_to_eye_mat.copy(),
            )

    def latest_joints(self) -> TimedValue:
        """Return an isolated copy of the latest joint feedback."""
        with self._lock:
            if not self._joints:
                raise ValueError("joint feedback is unavailable")
            latest = self._joints[-1]
            return TimedValue(
                latest.timestamp_ns,
                latest.value.copy(),
                latest.received_monotonic_s,
            )


class _InferenceRequestConfig(Protocol):
    instruction: str
    depth_units_per_meter: float


def _timestamp_ns(message: Any) -> int:
    stamp = message.header.stamp
    timestamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    if timestamp_ns <= 0:
        raise ValueError(
            "ROS observations must have a positive header timestamp"
        )
    return timestamp_ns


def _decode_compressed(data: bytes, *, color: bool) -> np.ndarray:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "opencv-python is required to decode HC-TJ camera messages"
        ) from error
    flags = cv2.IMREAD_COLOR if color else cv2.IMREAD_UNCHANGED
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags)
    if image is None:
        raise ValueError("OpenCV could not decode CompressedImage")
    if color and (
        image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3
    ):
        raise ValueError("head color must decode as uint8 HxWx3")
    if not color and (image.dtype != np.uint16 or image.ndim != 2):
        raise ValueError("head depth must decode as 16UC1 PNG")
    return image


def _npy_upload(name: str, value: np.ndarray) -> tuple[str, bytes, str]:
    payload = BytesIO()
    np.save(payload, value, allow_pickle=False)
    return f"{name}.npy", payload.getvalue(), "application/octet-stream"


def _build_request(
    snapshot: tuple[
        TimedValue, TimedValue, TimedValue, np.ndarray, np.ndarray
    ],
    config: _InferenceRequestConfig,
    right_hand_observation: float,
    left_hand_observation: float,
    remaining_actions: np.ndarray | None,
    delay_horizon: int,
) -> tuple[dict[str, tuple[str, bytes, str]], dict[str, str]]:
    """Encode a synchronized snapshot as multipart files and form fields.

    Inputs are not modified. Non-empty remaining actions and their delay
    horizon are included together; empty or absent actions omit both fields.
    Image decoding failures propagate to the caller.
    """
    color, depth, joints, intrinsic, zhi_to_eye = snapshot
    color_image = _decode_compressed(color.value, color=True)
    depth_m = _decode_compressed(depth.value, color=False).astype(np.float32)
    depth_m /= config.depth_units_per_meter
    arm_state = np.asarray(joints.value[:14], dtype=np.float32)
    current_state = np.concatenate(
        (
            arm_state[:7],
            np.asarray([right_hand_observation], dtype=np.float32),
            arm_state[7:],
            np.asarray([left_hand_observation], dtype=np.float32),
        )
    )
    arrays = {
        "head_color": color_image,
        "head_depth": depth_m,
        "head_intrinsic": intrinsic,
        "zhi_to_eye": zhi_to_eye,
        "current_state": current_state,
    }
    form = {
        "joint_names": json.dumps(HC_TJ_JOINT_NAMES),
        "instruction": config.instruction,
    }
    if remaining_actions is not None and len(remaining_actions):
        arrays["remaining_actions"] = remaining_actions.astype(np.float32)
        form["delay_horizon"] = str(delay_horizon)
    files = {name: _npy_upload(name, value) for name, value in arrays.items()}
    return files, form


def _post_inference(
    config: HcTjAsyncClientConfig,
    files: dict[str, tuple[str, bytes, str]],
    form: dict[str, str],
    *,
    response_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[np.ndarray | None, float]:
    """Post one request and optionally observe its validated response.

    The callback runs in the request worker before returning actions. Its
    exceptions propagate through the normal inference fault handling.
    """
    start = time.monotonic()
    response = requests.post(
        config.server_url,
        files=files,
        data=form,
        timeout=config.request_timeout_s,
        headers=(
            {"Authorization": f"Bearer {config.control_token}"}
            if config.enable_dagger
            else {}
        ),
    )
    elapsed = time.monotonic() - start
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
    payload = response.json()
    if config.enable_dagger:
        if (
            payload.get("status") != "ok"
            or payload.get("actions_allowed") is not True
        ):
            return None, elapsed
        if payload.get("control_window") != form.get(
            "control_window"
        ) or payload.get("control_server") != form.get("control_server"):
            raise ValueError(
                "inference result does not match its execution window"
            )
    actions = validate_inference_response(payload)
    if response_callback is not None:
        response_callback(payload)
        elapsed = time.monotonic() - start
    return actions, elapsed


def _state_is_stale(
    now_monotonic_s: float,
    received_monotonic_s: float,
    timeout_s: float,
) -> bool:
    """Check freshness using only the receiving host's monotonic clock."""
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if not math.isfinite(now_monotonic_s) or not math.isfinite(
        received_monotonic_s
    ):
        raise ValueError("monotonic timestamps must be finite")
    if now_monotonic_s < received_monotonic_s:
        raise ValueError("monotonic clock moved backwards")
    return now_monotonic_s - received_monotonic_s > timeout_s


def _other_publisher_names(
    publisher_infos: list[Any], own_name: str, own_namespace: str
) -> list[str]:
    """Filter this node from ROS publisher endpoint information."""
    return [
        f"{info.node_namespace}/{info.node_name}"
        for info in publisher_infos
        if not (
            info.node_name == own_name and info.node_namespace == own_namespace
        )
    ]


def _ros_node_class():
    """Load ROS only on the robot and return the concrete client node class."""
    try:
        from rclpy.node import Node
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from sensor_msgs.msg import CompressedImage, JointState
        from std_msgs.msg import String
    except ImportError as error:
        raise RuntimeError(
            "HC-TJ client requires a sourced ROS 2 environment with rclpy, "
            "sensor_msgs, and std_msgs"
        ) from error

    class HcTjAsyncNode(Node):
        """Own observations, one HTTP request, and safe publication."""

        def __init__(self, config: HcTjAsyncClientConfig) -> None:
            node_suffix = uuid.uuid4().hex[:8]
            super().__init__(f"holobrain_hc_tj_async_client_{node_suffix}")
            self.config = config
            self.observations = ObservationBuffer()
            lower, upper = load_arm_joint_limits(config.urdf_path)
            self.limiter = ArmCommandLimiter(
                lower,
                upper,
                config.max_arm_velocity_rad_s,
                max_acceleration_rad_s2=config.max_arm_acceleration_rad_s2,
            )
            self._default_joint_positions = np.asarray(
                DEFAULT_JOINT_POSITIONS, dtype=np.float64
            )
            if np.any(self._default_joint_positions[:14] < lower) or np.any(
                self._default_joint_positions[:14] > upper
            ):
                raise ValueError(
                    "default arm posture is outside the HC-TJ URDF limits"
                )
            self._lock = threading.RLock()
            # Keep invalidated work until it finishes; never reuse its result.
            self._worker = ThreadPoolExecutor(max_workers=1)
            self._future: Future | None = None
            self._future_window = ""
            self._future_started = 0.0
            self._future_observed_s = -math.inf
            self._gate = RobotControlGate()
            self._control_window = self._gate.snapshot()["window_id"]
            self._resume_stamp_ns = 0
            self._replan_cutoff_s = -math.inf
            self._auto_grant: tuple[str, str, int] | None = None
            self._jump_rejections = 0
            self._relay = None
            self._motion: MotionInterface | None = None
            self._motion_publishers = {}
            self._hand_received = -math.inf
            self._hand_stamp_ns = 0
            self._last_request = -math.inf
            self._last_observation_timestamp_ns: int | None = None
            self._latency_ewma: float | None = None
            self._native_actions: np.ndarray | None = None
            self._native_started = 0.0
            self._received_action_chunk = False
            self._right_hand_observation = 0.0
            self._left_hand_observation = 0.0
            self._fault: str | None = None
            self._reset_complete = (
                not config.enable_control
                or config.enable_dagger
                or config.robot_interface is RobotInterface.MOTION
            )
            self._diagnostic_counts = Counter()
            self._diagnostic_previous = Counter()
            self._authority_events = deque(maxlen=16)
            self._diagnostic_time = time.monotonic()
            self._last_publish_time = None
            self._publish_gap_max_ms = 0.0
            self._request_elapsed_ms = 0.0
            self._inference_wait = "startup"
            self._control_wait = "startup"

            latest_reliable = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
            )
            latest_best_effort = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
            )
            # A one-sample reliable reader loses throughput over the wired
            # DDS link under delivery bursts. Retain feedback history for
            # timestamp alignment; command publishers still keep depth=1.
            joint_feedback_qos = QoSProfile(
                depth=100,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
            )
            latched = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            for topic, callback in (
                (config.color_topic, self._on_color),
                (config.depth_topic, self._on_depth),
            ):
                self.create_subscription(
                    CompressedImage, topic, callback, latest_best_effort
                )
            for topic, callback in (
                (config.calibration_topic, self._on_calibration),
                (config.extrinsics_topic, self._on_extrinsics),
            ):
                self.create_subscription(String, topic, callback, latched)
            self.create_subscription(
                JointState,
                config.joint_state_topic,
                self._on_joints,
                joint_feedback_qos,
            )
            self.create_subscription(
                JointState,
                config.hand_state_topic,
                self._on_hands,
                latest_best_effort,
            )

            self._command_qos = latest_best_effort
            if config.robot_interface is RobotInterface.MOTION:
                self.create_subscription(
                    String,
                    config.robot_info_topic,
                    self._on_robot_info,
                    latched,
                )
            if config.enable_dagger:
                self.create_subscription(
                    String, config.ownership_topic, self._on_ownership, latched
                )
                self._relay = ControlRelay(
                    self._gate.snapshot,
                    self._gate.invalidate,
                    config.server_url.rsplit("/", 2)[0] + "/hc_tj_control/v1",
                    config.control_token,
                )
                self._relay.start()
            self._joint_publisher = None
            self._right_hand_publisher = None
            self._left_hand_publisher = None
            if (
                config.enable_control
                and config.robot_interface is RobotInterface.LEGACY
            ):
                self._joint_publisher = self.create_publisher(
                    JointState, config.joint_command_topic, latest_reliable
                )
                self._right_hand_publisher = self.create_publisher(
                    JointState,
                    config.right_hand_topic,
                    latest_best_effort,
                )
                self._left_hand_publisher = self.create_publisher(
                    JointState,
                    config.left_hand_topic,
                    latest_best_effort,
                )
            elif not config.enable_control:
                self.get_logger().warning(
                    "dry-run active: no control publishers were created"
                )
            self.create_timer(0.02, self._inference_tick)
            self.create_timer(1.0 / config.control_hz, self._control_tick)
            self.create_timer(1.0, self._log_diagnostics)
            self.get_logger().info(
                "[motion-limits] "
                + json.dumps(
                    {
                        "arm_velocity_rad_s": config.max_arm_velocity_rad_s,
                        "arm_acceleration_rad_s2": (
                            config.max_arm_acceleration_rad_s2
                        ),
                        "step_rad": config.max_arm_velocity_rad_s
                        / config.control_hz,
                        "first_point_jump_rad": config.first_point_jump_rad,
                        "control_hz": config.control_hz,
                        "inference_hz": config.inference_hz,
                        "state_timeout_s": config.state_timeout_s,
                        "joint_feedback_qos": {
                            "reliability": "RELIABLE",
                            "depth": joint_feedback_qos.depth,
                        },
                        "arm_lower": lower.tolist(),
                        "arm_upper": upper.tolist(),
                    }
                )
            )

        def _log_diagnostics(self) -> None:
            """Emit bounded summaries and red Chinese motion-limit warnings."""
            now = time.monotonic()
            elapsed = max(now - self._diagnostic_time, 1e-6)
            counts = self._diagnostic_counts.copy()
            delta = counts - self._diagnostic_previous
            state = self._gate.snapshot() if self.config.enable_dagger else {}
            if self._authority_events:
                self.get_logger().info(
                    "[motion-authority] "
                    + json.dumps(list(self._authority_events))
                )
                self._authority_events.clear()
            self.get_logger().info(
                "[motion-diag] "
                + json.dumps(
                    {
                        "日志时长": round(elapsed, 3),
                        "hz": {
                            k: round(delta[k] / elapsed, 2)
                            for k in (
                                "color",
                                "depth",
                                "joints",
                                "requests", #推理请求发起次数
                                "completed", #推理完成次数
                                "published", #client 发布动作频次
                            )
                        },
                        "counts": dict(delta),
                        "inference_wait": self._inference_wait,
                        "control_wait": self._control_wait,
                        "fault": self.fault,
                        "request_ms": round(self._request_elapsed_ms, 2),
                        "inflight_age_ms": round(
                            (now - self._future_started) * 1000, 2
                        )
                        if self._future
                        else 0,
                        "publish_gap_max_ms": round(
                            self._publish_gap_max_ms, 2
                        ),
                        "action_points_left": max(
                            0,
                            math.ceil(
                                (
                                    len(self._native_actions) / ACTION_HZ
                                    - max(0.0, now - self._native_started)
                                )
                                * self.config.control_hz
                            ),
                        )
                        if self._native_actions is not None
                        else 0,
                        "max_clip_rad": self.limiter.max_clip_rad,
                        "allowed": state.get("allowed"),
                        "gate_reason": state.get("reason"),
                        "ownership": state.get("ownership"),
                    }
                )
            )
            self._diagnostic_previous = counts
            self._diagnostic_time = now
            self._publish_gap_max_ms = 0.0
            self.limiter.max_clip_rad = 0.0

        @property
        def fault(self) -> str | None:
            """Return the latched fault reason, if any."""
            with self._lock:
                return self._fault

        def _latch_fault(self, reason: str) -> None:
            if self.config.enable_dagger:
                self._gate.invalidate("client_fault")
                self._sync_control()
            with self._lock:
                first_fault = self._fault is None
                if first_fault:
                    self._fault = reason
                    self._native_actions = None
            if first_fault:
                self.get_logger().error(
                    f"Latched fault; restart required: {reason}"
                )
            else:
                self._diagnostic_counts["repeated_faults"] += 1

        def _on_color(self, message: Any) -> None:
            self._diagnostic_counts["color"] += 1
            try:
                self.observations.add_color(
                    _timestamp_ns(message), message.data
                )
            except ValueError as error:
                self._latch_fault(str(error))

        def _on_depth(self, message: Any) -> None:
            self._diagnostic_counts["depth"] += 1
            try:
                self.observations.add_depth(
                    _timestamp_ns(message), message.data
                )
            except ValueError as error:
                self._latch_fault(str(error))

        def _on_joints(self, message: Any) -> None:
            self._diagnostic_counts["joints"] += 1
            try:
                if self.config.robot_interface is RobotInterface.MOTION:
                    if self._motion is None:
                        return
                    positions = self._motion.feedback(
                        message.name, message.position
                    )
                elif tuple(message.name) != CONTROL_JOINT_NAMES:
                    raise ValueError(
                        "joint feedback order does not match HC-TJ "
                        "17-axis order"
                    )
                else:
                    positions = np.asarray(message.position)
                self.observations.add_joints(_timestamp_ns(message), positions)
            except ValueError as error:
                self._latch_fault(str(error))

        def _on_robot_info(self, message: Any) -> None:
            try:
                motion = MotionInterface.from_info(
                    json.loads(message.data),
                    identity_mapping=self.config.motion_identity_mapping,
                )
                if 1 / self.config.control_hz >= motion.servo_lease_s:
                    raise ValueError(
                        "control_hz is too slow for the Motion servo lease"
                    )
                with self._lock:
                    if self._motion is not None:
                        if self._motion != motion:
                            raise ValueError(
                                "Motion profile changed; restart and "
                                "verify mapping"
                            )
                        return
                    self._motion = motion
                    if self.config.enable_control:
                        self._motion_publishers = {
                            topic: self.create_publisher(
                                JointState, topic, self._command_qos
                            )
                            for topic in motion.topics
                        }
            except (ValueError, TypeError) as error:
                self._latch_fault(str(error))

        def _on_hands(self, message: Any) -> None:
            try:
                stamp = _timestamp_ns(message)
                age = (self.get_clock().now().nanoseconds - stamp) * 1e-9
                if not -0.05 <= age <= self.config.state_timeout_s:
                    return
                right, left = measured_hand_closures(
                    message.name, message.position
                )
                with self._lock:
                    if stamp <= self._hand_stamp_ns:
                        return
                    (
                        self._right_hand_observation,
                        self._left_hand_observation,
                    ) = right, left
                    self._hand_received, self._hand_stamp_ns = (
                        time.monotonic(),
                        stamp,
                    )
            except ValueError as error:
                self.get_logger().warning(
                    f"Rejected measured hand feedback: {error}"
                )

        def _on_ownership(self, message: Any) -> None:
            try:
                self._gate.update_ownership(
                    json.loads(message.data),
                    self.get_clock().now().nanoseconds,
                )
            except (ValueError, TypeError, KeyError):
                self._gate.invalidate("invalid ownership state")
            self._sync_control()

        def _clear_trajectory(self) -> None:
            """Drop queued output and RTC history under the action lock."""
            self._native_actions = None
            self._native_started = 0.0
            self._received_action_chunk = False
            self._last_publish_time = None
            self._latency_ewma = None
            self._resume_stamp_ns = self.get_clock().now().nanoseconds

        def _sync_control(self) -> dict[str, Any]:
            with self._gate.lock:
                state = self._gate.snapshot()
                if state["window_id"] == self._control_window:
                    return state
                self._diagnostic_counts["authority_changes"] += 1
                self._authority_events.append(state)
                with self._lock:
                    self._clear_trajectory()
                    if self._future is not None:
                        # Keep running work until done; its old window fences
                        # both its result and any exception on resumption.
                        self._future.cancel()
                    self._replan_cutoff_s = -math.inf
                    self._last_request = -math.inf
                    self._last_observation_timestamp_ns = None
                    self._control_window = state["window_id"]
                    if state["allowed"]:
                        owner = state["ownership"]
                        grant = (
                            owner["controller_instance_id"],
                            owner["session_id"],
                            owner["epoch"],
                        )
                        if grant != self._auto_grant:
                            self._auto_grant = grant
                            self._jump_rejections = 0
            return state

        def _reject_action_jump(
            self, window: str, error: ActionDiscontinuityError
        ) -> None:
            with self._gate.lock:
                if self.config.enable_dagger:
                    state = self._gate.snapshot()
                    if not self._gate.allows(window) or not self._relay.ready(
                        window
                    ):
                        return
                    if state["ownership"]["dagger_enabled"]:
                        with self._lock:
                            self._clear_trajectory()
                            self._replan_cutoff_s = time.monotonic()
                            self._jump_rejections += 1
                            self._diagnostic_counts[
                                "jump_chunks_rejected"
                            ] += 1
                        outcome = (
                            "等待新的 HC AUTO 授权"
                            if self._jump_rejections
                            >= _MAX_CONSECUTIVE_DAGGER_JUMPS
                            else "将根据最新观测重新规划"
                        )
                        self.get_logger().warning(
                            "\033[31m[首帧跳变超限] 已拒绝 DAgger 动作块，"
                            "阈值 "
                            f"{self.config.first_point_jump_rad:.6f} 弧度，"
                            f"连续拒绝 {self._jump_rejections}/"
                            f"{_MAX_CONSECUTIVE_DAGGER_JUMPS} 次，"
                            f"{outcome}。详情：{error}\033[0m"
                        )
                        return
                self.get_logger().warning(
                    "\033[31m[首帧跳变超限] 已拒绝动作块并进入故障状态，"
                    f"阈值 {self.config.first_point_jump_rad:.6f} 弧度。"
                    f"详情：{error}\033[0m"
                )
                self._latch_fault(f"inference/protocol failure: {error}")

        def _hands_ready(self) -> bool:
            # Legacy hands keep the v1 open-loop contract in DAgger too.
            if self.config.robot_interface is RobotInterface.LEGACY:
                return True
            ready = (
                time.monotonic() - self._hand_received
                <= self.config.state_timeout_s
                and (
                    not self.config.enable_dagger
                    or self._hand_stamp_ns > self._resume_stamp_ns
                )
            )
            if not ready and self.config.enable_dagger:
                with self._gate.lock:
                    if self._received_action_chunk or (
                        self._future is not None
                        and self._future_window == self._control_window
                    ):
                        self._gate.invalidate("measured hand feedback stale")
                        self._sync_control()
            return ready

        def _on_calibration(self, message: Any) -> None:
            try:
                self.observations.set_calibration(message.data)
            except ValueError as error:
                self._latch_fault(str(error))

        def _on_extrinsics(self, message: Any) -> None:
            try:
                self.observations.set_extrinsics(message.data)
            except ValueError as error:
                self._latch_fault(str(error))

        def _remaining_actions(self, now: float) -> np.ndarray | None:
            with self._lock:
                if self._native_actions is None:
                    return None
                elapsed = max(0.0, now - self._native_started)
                duration = len(self._native_actions) / ACTION_HZ
                if elapsed >= duration:
                    return None
                times = (
                    elapsed
                    + np.arange(
                        int(math.ceil((duration - elapsed) * ACTION_HZ))
                    )
                    / ACTION_HZ
                )
                return sample_actions(self._native_actions, times)

        def _check_feedback_freshness(self, sample: TimedValue) -> None:
            """Reject stalled delivery and queued old control feedback."""
            if _state_is_stale(
                time.monotonic(),
                sample.received_monotonic_s,
                self.config.state_timeout_s,
            ):
                raise ValueError("joint feedback is stale")
            if self.config.enable_control:
                # A DDS sample can reach its callback long after publication.
                # All control backends require synchronized robot/client ROS
                # clocks, including legacy reset and limiter reseeding.
                age = (
                    self.get_clock().now().nanoseconds - sample.timestamp_ns
                ) * 1e-9
                if not -0.05 <= age <= self.config.state_timeout_s:
                    raise ValueError("joint feedback source stamp is stale")

        def _accept_prediction(
            self, actions: np.ndarray, elapsed: float, now: float, window: str
        ) -> bool:
            """Validate and install a prediction; stop the tick if rejected.

            Actions are copied on acceptance. Fresh feedback, jump checks and
            a final authority check protect the execution boundary. Validation
            exceptions are handled by the inference timer.
            """
            starting = (
                self.config.enable_dagger
                and self._gate.snapshot()["ownership"]["dagger_enabled"]
                and not self._received_action_chunk
            )
            if starting:
                # Resume at point zero, retaining only unexpired duration.
                age = max(elapsed, time.monotonic() - self._future_observed_s)
                steps = (
                    len(trim_elapsed_actions(actions, age))
                    if age < self.config.camera_timeout_s
                    else 0
                )
                remaining, response_index = actions[:steps], 0
            else:
                steps = len(trim_elapsed_actions(actions, elapsed))
                response_index = len(actions) - steps
                if steps:
                    remaining = sample_actions(
                        actions, elapsed + np.arange(steps) / ACTION_HZ
                    )
                else:
                    self._diagnostic_counts["chunk_fully_elapsed"] += 1
                    remaining, response_index = actions[-1:], len(actions) - 1
            if len(remaining) == 0:
                self._diagnostic_counts["expired_start_chunks"] += 1
                self._inference_wait = "expired_start_prediction"
                return False
            feedback_sample = self.observations.latest_joints()
            self._check_feedback_freshness(feedback_sample)
            feedback = feedback_sample.value[:14]
            try:
                validate_action_chunk(
                    remaining,
                    feedback,
                    first_point_jump_rad=self.config.first_point_jump_rad,
                )
            except ActionDiscontinuityError as error:
                # Compare the untrimmed prediction with the same
                # feedback used to reject the execution point.
                raw_jump = np.max(np.abs(arm_actions(actions[0]) - feedback))
                raise ActionDiscontinuityError(
                    f"{error}; response_index="
                    f"{response_index} "
                    f"raw_first_max_delta_rad={raw_jump:.6f} "
                    f"request_ms={elapsed * 1000:.2f}"
                ) from error
            with self._gate.lock, self._lock:
                if self.config.enable_dagger and (
                    not self._gate.allows(window)
                    or not self._relay.ready(window)
                ):
                    return False
                alpha = self.config.latency_ewma_alpha
                self._latency_ewma = (
                    elapsed
                    if self._latency_ewma is None
                    else alpha * elapsed + (1 - alpha) * self._latency_ewma
                )
                self._native_actions, self._native_started = (
                    remaining.copy() if starting else actions.copy(),
                    time.monotonic() if starting else now - elapsed,
                )
                if not self._received_action_chunk:
                    self.limiter.reset(feedback)
                    self._last_publish_time = None
                    self._diagnostic_counts["limiter_resets"] += 1
                self._received_action_chunk = True
                self._jump_rejections = 0
                self._diagnostic_counts["chunks_accepted"] += 1
            return True

        def _inference_tick(self) -> None:
            """Infer only with authority; resume with fresh observations."""
            self._inference_wait = "fault" if self.fault else "ready"
            if self.fault is not None:
                return
            state = self._sync_control() if self.config.enable_dagger else None
            window = state["window_id"] if state else ""
            if state is not None and (
                not state["allowed"] or not self._relay.ready(window)
            ):
                self._inference_wait = "authority_or_relay_closed"
                return
            if self._jump_rejections >= _MAX_CONSECUTIVE_DAGGER_JUMPS:
                self._inference_wait = "awaiting_new_auto_grant_after_jump"
                return
            with self._lock:
                if not self._reset_complete:
                    return
            now = time.monotonic()
            if self._future is not None:
                if not self._future.done():
                    self._inference_wait = "inference_inflight"
                    return
                future, self._future = self._future, None
                future_window = self._future_window
                # Obsolete completions are discarded without reading their
                # results, errors or cancellations. Only current work is used.
                if state is None or future_window == window:
                    try:
                        actions, elapsed = future.result()
                        # Include worker preparation, response decoding and
                        # the callback delay, measured from the RTC snapshot.
                        if self._future_started > 0:
                            elapsed = max(elapsed, now - self._future_started)
                        self._diagnostic_counts["completed"] += 1
                        self._request_elapsed_ms = elapsed * 1000
                        # Authority may have changed since the tick started.
                        if actions is not None and (
                            state is None
                            or (
                                self._gate.allows(future_window)
                                and self._relay.ready(future_window)
                            )
                        ):
                            if not self._accept_prediction(
                                actions, elapsed, now, future_window
                            ):
                                return
                    except ActionDiscontinuityError as error:
                        self._reject_action_jump(future_window, error)
                        return
                    except Exception as error:
                        if state is None or self._gate.allows(future_window):
                            if self.config.enable_dagger and isinstance(
                                error, requests.RequestException
                            ):
                                self._gate.invalidate("inference_unavailable")
                                self._sync_control()
                                self.get_logger().warning(
                                    f"Inference transport interrupted: {error}"
                                )
                                return
                            self._latch_fault(
                                f"inference/protocol failure: {error}"
                            )
                            return
            now = time.monotonic()
            if now - self._last_request < 1.0 / self.config.inference_hz:
                self._inference_wait = "configured_request_period"
                return
            try:
                snapshot = self.observations.synchronized()
            except ValueError as error:
                self._inference_wait = "observation_sync: " + str(error)
                return
            if self.config.enable_dagger:
                cutoff = max(self._gate.resume_started, self._replan_cutoff_s)
                if any(
                    item.timestamp_ns <= self._resume_stamp_ns
                    or item.received_monotonic_s <= cutoff
                    for item in snapshot[:3]
                ):
                    self._inference_wait = "fresh_observation_after_authority"
                    return
            if _state_is_stale(
                now,
                snapshot[0].received_monotonic_s,
                self.config.camera_timeout_s,
            ):
                self._inference_wait = "camera_stale"
                if self.config.enable_dagger:
                    self._gate.invalidate("camera_unavailable")
                    self._sync_control()
                else:
                    self._latch_fault(
                        "synchronized camera observation is stale"
                    )
                return
            if snapshot[0].timestamp_ns == self._last_observation_timestamp_ns:
                self._inference_wait = "duplicate_camera_frame"
                return
            if not self._hands_ready():
                self._inference_wait = "measured_hands_stale"
                return
            remaining = self._remaining_actions(now)
            with self._lock:
                latency = self._latency_ewma or 0.0
                right_hand, left_hand = (
                    self._right_hand_observation,
                    self._left_hand_observation,
                )
            delay = (
                min(int(math.ceil(latency * ACTION_HZ)), len(remaining))
                if remaining is not None
                else 0
            )
            form = {}
            if state is not None:
                form.update(
                    control_window=window,
                    control_server=self._relay.server_instance,
                )
            self._last_request = now
            self._last_observation_timestamp_ns = snapshot[0].timestamp_ns
            self._diagnostic_counts["requests"] += 1

            future = self._worker.submit(
                self._prepare_and_request,
                snapshot,
                right_hand,
                left_hand,
                remaining,
                delay,
                form,
            )
            self._future_started = now
            self._future_observed_s = min(
                item.received_monotonic_s for item in snapshot[:3]
            )
            self._future_window = window
            self._future = future

        def _prepare_and_request(
            self,
            snapshot: tuple[
                TimedValue, TimedValue, TimedValue, np.ndarray, np.ndarray
            ],
            right_hand: float,
            left_hand: float,
            remaining: np.ndarray | None,
            delay: int,
            metadata: dict[str, str],
        ) -> tuple[np.ndarray | None, float]:
            """Build and send a snapshot in the worker, without ROS mutation.

            Snapshot arrays are owned copies or immutable buffer entries.
            Authority metadata belongs to submission time; the ROS callback
            rejects completions from invalidated windows. Exceptions travel
            through the future into the existing fault handling.
            """
            files, form = _build_request(
                snapshot, self.config, right_hand, left_hand, remaining, delay
            )
            # Takeover can happen while the worker is preparing the request.
            if self.config.enable_dagger and not self._gate.allows(
                metadata["control_window"]
            ):
                return None, 0.0
            form.update(metadata)
            return self._request_inference(files, form)

        def _request_inference(
            self,
            files: dict[str, tuple[str, bytes, str]],
            form: dict[str, str],
        ) -> tuple[np.ndarray | None, float]:
            """Execute HTTP in the worker; subclasses may observe responses."""
            return _post_inference(self.config, files, form)

        def _control_tick(self) -> None:
            self._control_wait = "ready"
            # Permission invalidation and final arm/hand publication share this
            # lock, including the legacy initial posture branch.
            with self._gate.lock:
                if self.config.enable_dagger:
                    state = self._sync_control()
                    if not state["allowed"] or not self._relay.ready(
                        state["window_id"]
                    ):
                        self._control_wait = "authority_or_relay_closed"
                        return
                    if not self._hands_ready():
                        self._control_wait = "measured_hands_stale"
                        return
                self._control_tick_permitted()

        def _control_tick_permitted(self) -> None:
            if not self.config.enable_control or self.fault is not None:
                self._control_wait = "disabled_or_fault"
                return
            if (
                self.config.robot_interface is RobotInterface.MOTION
                and self._motion is None
            ):
                return
            with self._lock:
                if (
                    self._reset_complete
                    and self._native_actions is None
                    and not self._received_action_chunk
                ):
                    self._control_wait = "awaiting_first_action"
                    return
            if not self.config.allow_competing_publishers:
                topics = (
                    self._motion.topics
                    if self._motion is not None
                    else (
                        self.config.joint_command_topic,
                        self.config.right_hand_topic,
                        self.config.left_hand_topic,
                    )
                )
                for topic in topics:
                    publishers = _other_publisher_names(
                        self.get_publishers_info_by_topic(topic),
                        self.get_name(),
                        self.get_namespace(),
                    )
                    if publishers:
                        self._latch_fault(
                            f"competing publishers on {topic}: {publishers}"
                        )
                        return
            try:
                feedback_sample = self.observations.latest_joints()
            except ValueError:
                # A timer may fire before the first ROS feedback. Keep waiting;
                # subsequent stale or non-finite data is still a latched fault.
                return
            try:
                self._check_feedback_freshness(feedback_sample)
                now = time.monotonic()
                period = 1.0 / self.config.control_hz
                if self._last_publish_time is not None:
                    period = now - self._last_publish_time
                    if period <= 0:
                        return
                with self._lock:
                    if not self._reset_complete:
                        if self.limiter.previous_command is None:
                            self.limiter.reset(feedback_sample.value[:14])
                        target = self._default_joint_positions[:14]
                        action = None
                        reset_complete = np.all(
                            np.abs(
                                feedback_sample.value
                                - self._default_joint_positions
                            )
                            <= _RESET_POSITION_TOLERANCE_RAD
                        )
                        if reset_complete:
                            self._reset_complete = True
                    else:
                        if self._native_actions is None:
                            if not self._received_action_chunk:
                                return
                            raise ValueError("action trajectory is empty")
                        elapsed = max(0.0, now - self._native_started)
                        if elapsed >= len(self._native_actions) / ACTION_HZ:
                            self._diagnostic_counts["tail_hold_ticks"] += 1
                            self._control_wait = (
                                "trajectory_exhausted_hold_last"
                            )
                        action = sample_actions(
                            self._native_actions, np.asarray([elapsed])
                        )[0]
                        target = arm_actions(action)
                    arm = self.limiter.step(
                        target,
                        period,
                        max_step_seconds=1.0 / self.config.control_hz,
                    )
                stamp = self.get_clock().now().to_msg()
                joint_message = None
                if self._motion is None:
                    joint_message = JointState()
                    joint_message.header.stamp = stamp
                    joint_message.name = list(CONTROL_JOINT_NAMES)
                    joint_message.position = [
                        *arm.tolist(),
                        *(
                            self._default_joint_positions[14:].tolist()
                            if action is None
                            else feedback_sample.value[14:].tolist()
                        ),
                    ]
                if self.config.enable_dagger and (
                    not self._gate.allows(self._control_window)
                    or not self._relay.ready(self._control_window)
                ):
                    self._sync_control()
                    return
                if self._motion is not None:
                    for topic, names, positions in self._motion.commands(arm):
                        message = JointState()
                        message.header.stamp = stamp
                        message.name, message.position = list(names), positions
                        # 补手部动作
                        message.name.append("R_ban")
                        message.name.append("L_ban")
                        fake_r_eef_command=float(
                            np.clip(action[7], 0, 1)
                        ) if action is not None else self._right_hand_observation
                        fake_l_eef_command=float(
                            np.clip(action[15], 0, 1)
                        ) if action is not None else self._left_hand_observation

                        message.position.append(fake_r_eef_command)
                        message.position.append(fake_l_eef_command)
                        self._motion_publishers[topic].publish(message)
                else:
                    # 补手部动作
                    joint_message.name.append("R_ban")
                    joint_message.name.append("L_ban")
                    fake_r_eef_command=float(
                        np.clip(action[7], 0, 1)
                    ) if action is not None else self._right_hand_observation
                    fake_l_eef_command=float(
                        np.clip(action[15], 0, 1)
                    ) if action is not None else self._left_hand_observation

                    joint_message.position.append(fake_r_eef_command)
                    joint_message.position.append(fake_l_eef_command)
                    self._joint_publisher.publish(joint_message)
                published = time.monotonic()
                if self._last_publish_time is not None:
                    self._publish_gap_max_ms = max(
                        self._publish_gap_max_ms,
                        (published - self._last_publish_time) * 1000,
                    )
                self._last_publish_time = published
                self._diagnostic_counts["published"] += 1
                if self._motion is not None:
                    return  # Hands must never bypass HC through legacy topics.

                if action is None:
                    if reset_complete:
                        self.get_logger().info(
                            "default HC-TJ posture reached; inference enabled"
                        )
                    return

                for side, closure, publisher in (
                    ("right", action[7], self._right_hand_publisher),
                    ("left", action[15], self._left_hand_publisher),
                ):
                    names, positions = hand_command(side, float(closure))
                    hand_message = JointState()
                    hand_message.header.stamp = stamp
                    hand_message.name = list(names)
                    hand_message.position = positions.tolist()
                    if self.config.enable_dagger:
                        # Match each recorded hand's binary gripper label.
                        hand_message.name.append(
                            "R_ban" if side == "right" else "L_ban"
                        )
                        hand_message.position.append(float(closure >= 0.5))
                    publisher.publish(hand_message)
                    if closure >= 0.5:
                        self._diagnostic_counts[
                            f"{side}_hand_close_ticks"
                        ] += 1
                with self._lock:
                    # Estimate hand input from the last published command.
                    # This is not measured hand position feedback.
                    self._right_hand_observation = float(
                        np.clip(action[7], 0, 1)
                    )
                    self._left_hand_observation = float(
                        np.clip(action[15], 0, 1)
                    )
            except Exception as error:
                self._latch_fault(f"control safety failure: {error}")

        def destroy_node(self) -> bool:
            """Stop the HTTP worker before releasing ROS resources."""
            if self._relay is not None:
                self._relay.close()
            self._worker.shutdown(wait=False, cancel_futures=True)
            return super().destroy_node()

    return HcTjAsyncNode


def _parse_args() -> HcTjAsyncClientConfig:
    # Omitted options use the dataclass defaults; keep one source of truth.
    parser = argparse.ArgumentParser(
        description=__doc__, argument_default=argparse.SUPPRESS
    )
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument(
        "--urdf-path",
        type=Path,
        default=_REPOSITORY_ROOT
        / "projects/holobrain/assets/hc_tj_robot.urdf",
    )
    parser.add_argument("--depth-units-per-meter", type=float, required=True)
    parser.add_argument("--enable-control", action="store_true")
    parser.add_argument(
        "--robot-interface",
        type=RobotInterface,
        choices=list(RobotInterface),
    )
    parser.add_argument("--motion-identity-mapping", action="store_true")
    parser.add_argument("--enable-dagger", action="store_true")
    parser.add_argument(
        "--control-token", default=os.environ.get("HC_TJ_GATEWAY_TOKEN", "")
    )
    parser.add_argument("--control-hz", type=float)
    parser.add_argument(
        "--max-arm-acceleration-rad-s2",
        type=float,
        help="Command acceleration limit retained across inference chunks",
    )
    parser.add_argument("--allow-competing-publishers", action="store_true")
    parser.add_argument("--inference-hz", type=float)
    parser.add_argument("--request-timeout-s", type=float)
    parser.add_argument("--camera-timeout-s", type=float)
    parser.add_argument("--state-timeout-s", type=float)
    parser.add_argument(
        "--first-point-jump-rad",
        type=float,
        default=float(os.environ.get("HC_TJ_FIRST_POINT_JUMP_RAD", "0.30")),
    )
    for name in (
        "ownership-topic",
        "robot-info-topic",
        "hand-state-topic",
        "color-topic",
        "depth-topic",
        "calibration-topic",
        "extrinsics-topic",
        "joint-state-topic",
        "joint-command-topic",
        "right-hand-topic",
        "left-hand-topic",
    ):
        parser.add_argument(f"--{name}")
    return HcTjAsyncClientConfig(**vars(parser.parse_args()))


def main() -> None:
    """Run the ROS executor until interrupted."""
    config = _parse_args()
    try:
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
    except ImportError as error:
        raise RuntimeError(
            "source the ROS 2 environment before starting the client"
        ) from error
    node_class = _ros_node_class()
    rclpy.init()
    node = node_class(config)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        # rclpy's SIGINT handler may already have shut down the default
        # context before control reaches this block.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
