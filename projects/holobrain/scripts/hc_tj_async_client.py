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
import sys
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
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
    ARM_JOINT_NAMES,
    CONTROL_JOINT_NAMES,
    HC_TJ_JOINT_NAMES,
    ArmCommandLimiter,
    TimedValue,
    arm_actions,
    hand_command,
    interpolate_actions,
    load_arm_joint_limits,
    nearest_timed_value,
    parse_camera_calibration,
    parse_robot_camera_extrinsics,
    trim_elapsed_actions,
    validate_action_chunk,
    validate_inference_response,
)

__all__ = [
    "DEFAULT_JOINT_POSITIONS",
    "HcTjAsyncClientConfig",
    "ObservationBuffer",
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


@dataclass(frozen=True, slots=True)
class HcTjAsyncClientConfig:
    """Configure legacy direct ROS or HC's existing public Motion interface.

    ``hc_motion`` uses a generated Robot API mapping and the existing DAgger
    AUTO permission; it never resets the body or enables hardware itself.
    """

    server_url: str
    instruction: str
    urdf_path: Path
    depth_units_per_meter: float
    enable_control: bool = False
    robot_interface: str = "legacy"
    robot_api_file: Path | None = None
    allow_competing_publishers: bool = False
    inference_hz: float = 2.0
    control_hz: float = 100.0
    request_timeout_s: float = 30.0
    camera_timeout_s: float = 1.0
    state_timeout_s: float = 0.25
    max_arm_velocity_rad_s: float = 0.25
    first_point_jump_rad: float = 0.10
    latency_ewma_alpha: float = 0.25
    color_topic: str = "/hc_teleop/camera_head/color/compressed"
    depth_topic: str = "/hc_teleop/camera_head/depth/compressed"
    calibration_topic: str = "/io_teleop/camera_head/calibration"
    extrinsics_topic: str = "/io_teleop/camera_head/extrinsics"
    joint_state_topic: str = "/hc_teleop/joint_states"
    joint_command_topic: str = "/hc_teleop/joint_cmd"
    right_hand_topic: str = "/hc_teleop/joint_cmd_finger_right"
    left_hand_topic: str = "/hc_teleop/joint_cmd_finger_left"

    def __post_init__(self) -> None:
        if not self.server_url.endswith("/hc_tj_async/v1"):
            raise ValueError(
                "server_url must target the /hc_tj_async/v1 endpoint"
            )
        if not self.instruction.strip():
            raise ValueError("instruction must not be empty")
        if self.robot_interface not in ("legacy", "hc_motion"):
            raise ValueError("robot_interface must be legacy or hc_motion")
        if self.robot_interface == "hc_motion":
            if (
                self.robot_api_file is None
                or not self.robot_api_file.is_file()
            ):
                raise ValueError("hc_motion requires --robot-api-file")
            defaults = {
                "color_topic": "/observation/cameras/head/color/compressed",
                "depth_topic": "/observation/cameras/head/depth/compressed",
                "calibration_topic": "/observation/cameras/head/calibration",
                "extrinsics_topic": "/observation/cameras/head/extrinsics",
                "joint_state_topic": "/observation/joint_states",
                "right_hand_topic": "/action/right_hand/joint_states",
                "left_hand_topic": "/action/left_hand/joint_states",
            }
            for field, value in defaults.items():
                if (
                    getattr(self, field)
                    == self.__dataclass_fields__[field].default
                ):
                    object.__setattr__(self, field, value)
        elif not self.urdf_path.is_file():
            raise FileNotFoundError(f"HC-TJ URDF not found: {self.urdf_path}")
        if self.depth_units_per_meter <= 0:
            raise ValueError(
                "depth_units_per_meter must be explicitly positive"
            )
        if not 0 < self.inference_hz <= 2.0:
            raise ValueError("inference_hz must be in (0, 2]")
        if self.control_hz <= 0 or self.request_timeout_s <= 0:
            raise ValueError(
                "control_hz and request_timeout_s must be positive"
            )
        if self.camera_timeout_s <= 0 or self.state_timeout_s <= 0:
            raise ValueError("camera and state timeouts must be positive")
        if self.max_arm_velocity_rad_s <= 0:
            raise ValueError("arm velocity must be positive")
        if self.first_point_jump_rad <= 0:
            raise ValueError("first_point_jump_rad must be positive")
        if not 0 < self.latency_ewma_alpha <= 1:
            raise ValueError("latency_ewma_alpha must be in (0, 1]")


class ObservationBuffer:
    """Cache bounded ROS observations and synchronize on RGB time."""

    def __init__(self, capacity: int = 120, joint_count: int = 17) -> None:
        self._lock = threading.Lock()
        self._joint_count = joint_count
        self._colors: deque[TimedValue] = deque(maxlen=capacity)
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
        """Cache one ordered feedback sample (17 legacy axes or 14 HC arms)."""
        values = np.asarray(positions, dtype=np.float64)
        if values.shape != (self._joint_count,) or not np.all(
            np.isfinite(values)
        ):
            raise ValueError(
                f"joint feedback must have finite shape [{self._joint_count}]"
            )
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
                self._depths, color.timestamp_ns, 50_000_000
            )
            joints = nearest_timed_value(
                self._joints, color.timestamp_ns, 20_000_000
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


class _HcRobotApi:
    """Translate only the two arms using HC's existing generated SI mapping."""

    def __init__(self, path: Path) -> None:
        import yaml

        resolved = yaml.safe_load(path.read_text())
        self.info = resolved["info"]
        self.identity = {
            k: resolved[k] for k in ("schema", "profile_id", "revision")
        }
        if self.identity["schema"] != "hc-robot-api/v1" or not all(
            self.identity.values()
        ):
            raise ValueError("unsupported HC Robot API identity")
        self.validate_info(self.info)
        entries = resolved["joints"]
        by_model = {entry["model_name"]: entry for entry in entries}
        if (
            len(by_model) != len(entries)
            or not set(ARM_JOINT_NAMES) <= by_model.keys()
        ):
            raise ValueError("HC-TJ requires both complete seven-axis arms")
        arms = [by_model[name] for name in ARM_JOINT_NAMES]
        self.names = [entry["public_name"] for entry in arms]
        self.scale, self.offset, self.lower, self.upper = (
            np.asarray([entry[key] for entry in arms], dtype=float)
            for key in ("scale", "offset", "model_lower", "model_upper")
        )
        if (
            len(set(self.names)) != 14
            or np.any(self.scale == 0)
            or not all(
                np.isfinite(a).all()
                for a in (self.scale, self.offset, self.lower, self.upper)
            )
            or np.any(self.lower >= self.upper)
            or any(a["unit"] != "rad" for a in arms)
        ):
            raise ValueError("invalid HC arm names, units, mapping or limits")
        self.groups = {}
        for side, names in (
            ("right", self.names[:7]),
            ("left", self.names[7:]),
        ):
            group = self.info["groups"][side + "_arm"]
            if len(group["joint_names"]) != 7 or set(
                group["joint_names"]
            ) != set(names):
                raise ValueError("HC arm group differs from resolved mapping")
            endpoint = group["endpoints"]["servo_j"]
            if not endpoint.startswith("/control/motion/"):
                raise ValueError("HC public ServoJ endpoint is missing")
            self.groups[side] = (endpoint, list(group["joint_names"]))

    def validate_info(self, info: dict[str, Any]) -> None:
        if any(info.get(key) != value for key, value in self.identity.items()):
            raise ValueError("HC online profile differs from --robot-api-file")
        if info.get("groups") != self.info.get("groups"):
            raise ValueError(
                "HC online arm endpoints differ from generated file"
            )

    def feedback(self, names: list[str], positions: list[float]) -> np.ndarray:
        if len(names) != len(positions) or len(set(names)) != len(names):
            raise ValueError("invalid named HC joint feedback")
        values = dict(zip(names, positions, strict=True))
        if not set(self.names) <= values.keys():
            raise ValueError("HC joint feedback is missing an arm joint")
        result = (
            self.scale * [values[name] for name in self.names] + self.offset
        )
        if not np.isfinite(result).all():
            raise ValueError("nonfinite HC arm feedback")
        return result

    def targets(
        self, arm: np.ndarray
    ) -> dict[str, tuple[list[str], list[float]]]:
        values = (arm - self.offset) / self.scale
        by_name = dict(zip(self.names, values.tolist(), strict=True))
        return {
            side: (names, [by_name[name] for name in names])
            for side, (_, names) in self.groups.items()
        }


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
) -> tuple[dict[str, tuple[str, bytes, str]], dict[str, str], int]:
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
    if remaining_actions is not None and len(remaining_actions):
        arrays["remaining_actions"] = remaining_actions.astype(np.float32)
    files = {name: _npy_upload(name, value) for name, value in arrays.items()}
    form = {
        "joint_names": json.dumps(HC_TJ_JOINT_NAMES),
        "instruction": config.instruction,
    }
    if "remaining_actions" in arrays:
        form["delay_horizon"] = str(delay_horizon)
    return files, form, joints.timestamp_ns


def _post_inference(
    config: HcTjAsyncClientConfig,
    files: dict[str, tuple[str, bytes, str]],
    form: dict[str, str],
    *,
    response_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[np.ndarray, float]:
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
    )
    elapsed = time.monotonic() - start
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
    payload = response.json()
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
            self._hc_api = (
                _HcRobotApi(config.robot_api_file)
                if config.robot_interface == "hc_motion"
                else None
            )
            self.observations = ObservationBuffer(
                joint_count=14 if self._hc_api else 17
            )
            lower, upper = (
                (self._hc_api.lower, self._hc_api.upper)
                if self._hc_api
                else load_arm_joint_limits(config.urdf_path)
            )
            self.limiter = ArmCommandLimiter(
                lower, upper, config.max_arm_velocity_rad_s
            )
            self._default_joint_positions = np.asarray(
                DEFAULT_JOINT_POSITIONS, dtype=np.float64
            )
            if not self._hc_api and (
                np.any(self._default_joint_positions[:14] < lower)
                or np.any(self._default_joint_positions[:14] > upper)
            ):
                raise ValueError(
                    "default arm posture is outside the HC-TJ URDF limits"
                )
            self._lock = threading.RLock()
            self._hc_info_ready = False
            self._hc_calibration_ready = False
            self._owner = {}
            self._owner_received = 0.0
            self._generation = self._future_generation = 0
            self._resume_after_s = 0.0
            self._worker = ThreadPoolExecutor(max_workers=1)
            self._future: Future | None = None
            self._last_request = -math.inf
            self._last_observation_timestamp_ns: int | None = None
            self._latency_ewma: float | None = None
            self._native_actions: np.ndarray | None = None
            self._native_started = 0.0
            self._control_actions: np.ndarray | None = None
            self._control_index = 0
            self._received_action_chunk = False
            self._right_hand_observation = 0.0
            self._left_hand_observation = 0.0
            self._fault: str | None = None
            self._reset_complete = (
                bool(self._hc_api) or not config.enable_control
            )
            self._reset_limiter_initialized = False

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
            latched = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.create_subscription(
                CompressedImage,
                config.color_topic,
                self._on_color,
                latest_best_effort,
            )
            self.create_subscription(
                CompressedImage,
                config.depth_topic,
                self._on_depth,
                latest_best_effort,
            )
            self.create_subscription(
                JointState,
                config.joint_state_topic,
                self._on_joints,
                latest_best_effort if self._hc_api else latest_reliable,
            )
            self.create_subscription(
                String, config.calibration_topic, self._on_calibration, latched
            )
            self.create_subscription(
                String,
                config.extrinsics_topic,
                self._on_extrinsics,
                latest_reliable if self._hc_api else latched,
            )
            if self._hc_api:
                self.create_subscription(
                    String, "/status/robot/info", self._on_info, latched
                )
                self.create_subscription(
                    String,
                    "/status/control/ownership",
                    self._on_owner,
                    latched,
                )

            self._joint_publisher = None
            self._arm_publishers = {}
            self._right_hand_publisher = None
            self._left_hand_publisher = None
            if config.enable_control:
                if self._hc_api:
                    self._arm_publishers = {
                        side: self.create_publisher(
                            JointState, topic, latest_reliable
                        )
                        for side, (topic, _) in self._hc_api.groups.items()
                    }
                else:
                    self._joint_publisher = self.create_publisher(
                        JointState, config.joint_command_topic, latest_reliable
                    )
                self._right_hand_publisher = self.create_publisher(
                    JointState,
                    config.right_hand_topic,
                    latest_reliable if self._hc_api else latest_best_effort,
                )
                self._left_hand_publisher = self.create_publisher(
                    JointState,
                    config.left_hand_topic,
                    latest_reliable if self._hc_api else latest_best_effort,
                )
            else:
                self.get_logger().warning(
                    "dry-run active: no control publishers were created"
                )
            self.create_timer(0.02, self._inference_tick)
            self.create_timer(1.0 / config.control_hz, self._control_tick)

        @property
        def fault(self) -> str | None:
            """Return the latched fault reason, if any."""
            with self._lock:
                return self._fault

        def _latch_fault(self, reason: str) -> None:
            with self._lock:
                if self._fault is None:
                    self._fault = reason
                    self._control_actions = None
            self.get_logger().error(
                f"Latched fault; restart required: {reason}"
            )

        def _on_color(self, message: Any) -> None:
            try:
                self.observations.add_color(
                    _timestamp_ns(message), message.data
                )
            except ValueError as error:
                self._latch_fault(str(error))

        def _on_depth(self, message: Any) -> None:
            try:
                self.observations.add_depth(
                    _timestamp_ns(message), message.data
                )
            except ValueError as error:
                self._latch_fault(str(error))

        def _on_joints(self, message: Any) -> None:
            try:
                if self._hc_api:
                    values = self._hc_api.feedback(
                        message.name, message.position
                    )
                elif tuple(message.name) != CONTROL_JOINT_NAMES:
                    raise ValueError(
                        "joint feedback order does not match HC-TJ "
                        "17-axis order"
                    )
                else:
                    values = np.asarray(message.position)
                self.observations.add_joints(_timestamp_ns(message), values)
            except ValueError as error:
                self._latch_fault(str(error))

        def _on_info(self, message: Any) -> None:
            try:
                self._hc_api.validate_info(json.loads(message.data))
                self._hc_info_ready = True
            except (KeyError, TypeError, ValueError) as error:
                self._latch_fault(str(error))

        def _on_owner(self, message: Any) -> None:
            try:
                owner = json.loads(message.data)
                fields = (
                    "controller_instance_id",
                    "session_id",
                    "epoch",
                    "state",
                )
                identity = tuple(owner[key] for key in fields)
                with self._lock:
                    if owner["controller_instance_id"] == self._owner.get(
                        "controller_instance_id"
                    ) and owner.get("sequence", 0) < self._owner.get(
                        "sequence", 0
                    ):
                        return
                    if identity != tuple(
                        self._owner.get(key) for key in fields
                    ):
                        self._generation += 1
                        self._native_actions = self._control_actions = None
                        self._received_action_chunk = False
                        self._control_index = 0
                        self._latency_ewma = None
                        self._last_request = -math.inf
                        self._last_observation_timestamp_ns = None
                        self._resume_after_s = time.monotonic()
                        self.get_logger().info(f"HC control: {owner['state']}")
                    self._owner, self._owner_received = owner, time.monotonic()
            except (KeyError, TypeError, ValueError) as error:
                self._latch_fault(f"invalid HC ownership: {error}")

        def _hc_allowed(self) -> bool:
            if (
                self._hc_api
                and self.config.enable_control
                and self._owner.get("state") == "AUTO"
                and time.monotonic() - self._owner_received > 0.3
                and self._fault is None
            ):
                self._latch_fault("HC ownership heartbeat expired")
            return not self._hc_api or (
                self._hc_info_ready
                and (
                    not self.config.enable_control
                    or (
                        self._owner.get("state") == "AUTO"
                        and self._owner.get("source") == "motion"
                        and time.monotonic() - self._owner_received <= 0.3
                    )
                )
            )

        def _on_calibration(self, message: Any) -> None:
            try:
                if self._hc_api and not self._hc_calibration_ready:
                    document = json.loads(message.data)
                    if (
                        isinstance(document, dict)
                        and document.get("schema")
                        == "hc_camera_calibration/v1"
                        and isinstance(document.get("resolution"), dict)
                        and set(document["resolution"])
                        in ({"color"}, {"depth"})
                    ):
                        # HC sends each CameraInfo as it arrives at startup.
                        return
                self.observations.set_calibration(message.data)
                self._hc_calibration_ready = True
            except (json.JSONDecodeError, ValueError) as error:
                self._latch_fault(str(error))

        def _on_extrinsics(self, message: Any) -> None:
            try:
                self.observations.set_extrinsics(message.data)
            except (json.JSONDecodeError, ValueError) as error:
                self._latch_fault(str(error))

        def _remaining_actions(self, now: float) -> np.ndarray | None:
            with self._lock:
                if self._native_actions is None:
                    return None
                elapsed = max(0.0, now - self._native_started)
                remaining = trim_elapsed_actions(self._native_actions, elapsed)
                return remaining.copy() if len(remaining) else None

        def _inference_tick(self) -> None:
            # Serialize ownership changes with HTTP completion/submission.
            with self._lock:
                if self._hc_allowed():
                    self._advance_inference()

        def _advance_inference(self) -> None:
            if self.fault is not None:
                return
            with self._lock:
                if not self._reset_complete:
                    return
            now = time.monotonic()
            if self._future is not None:
                if not self._future.done():
                    return
                future, self._future = self._future, None
                if self._future_generation != self._generation:
                    return  # Also discard exceptions from a previous takeover.
                try:
                    actions, elapsed = future.result()
                    remaining = trim_elapsed_actions(actions, elapsed)
                    if len(remaining) == 0:
                        raise ValueError(
                            "inference latency expired the entire action chunk"
                        )
                    feedback_sample = self.observations.latest_joints()
                    if _state_is_stale(
                        time.monotonic(),
                        feedback_sample.received_monotonic_s,
                        self.config.state_timeout_s,
                    ):
                        raise ValueError("joint feedback is stale")
                    feedback = feedback_sample.value[:14]
                    validate_action_chunk(
                        remaining,
                        feedback,
                        first_point_jump_rad=self.config.first_point_jump_rad,
                    )
                    control = interpolate_actions(
                        remaining, target_hz=int(self.config.control_hz)
                    )
                    with self._lock:
                        alpha = self.config.latency_ewma_alpha
                        self._latency_ewma = (
                            elapsed
                            if self._latency_ewma is None
                            else alpha * elapsed
                            + (1 - alpha) * self._latency_ewma
                        )
                        self._native_actions = remaining.copy()
                        self._native_started = time.monotonic()
                        self._control_actions = control
                        self._control_index = 0
                        self._received_action_chunk = True
                        self.limiter.reset(feedback)
                except Exception as error:
                    self._latch_fault(f"inference/protocol failure: {error}")
                    return
            if now - self._last_request < 1.0 / self.config.inference_hz:
                return
            try:
                snapshot = self.observations.synchronized()
            except ValueError:
                return
            if _state_is_stale(
                time.monotonic(),
                snapshot[0].received_monotonic_s,
                self.config.camera_timeout_s,
            ):
                self._latch_fault("synchronized camera observation is stale")
                return
            if snapshot[0].timestamp_ns == self._last_observation_timestamp_ns:
                return
            if (
                self._hc_api
                and self.config.enable_control
                and min(sample.received_monotonic_s for sample in snapshot[:3])
                <= self._resume_after_s
            ):
                return
            remaining = self._remaining_actions(now)
            with self._lock:
                latency = self._latency_ewma or 0.0
                right_hand = self._right_hand_observation
                left_hand = self._left_hand_observation
            delay = int(math.ceil(latency * ACTION_HZ))
            if remaining is not None:
                delay = min(delay, len(remaining))
            try:
                files, form, _ = _build_request(
                    snapshot,
                    self.config,
                    right_hand,
                    left_hand,
                    remaining,
                    delay,
                )
            except Exception as error:
                self._latch_fault(f"observation/protocol failure: {error}")
                return
            self._last_request = now
            self._last_observation_timestamp_ns = snapshot[0].timestamp_ns
            self._future_generation = self._generation
            self._future = self._worker.submit(
                self._request_inference, files, form
            )

        def _request_inference(
            self,
            files: dict[str, tuple[str, bytes, str]],
            form: dict[str, str],
        ) -> tuple[np.ndarray, float]:
            """Execute HTTP in the worker; subclasses may observe responses."""
            return _post_inference(self.config, files, form)

        def _other_publishers(self, topic: str) -> list[str]:
            return _other_publisher_names(
                self.get_publishers_info_by_topic(topic),
                self.get_name(),
                self.get_namespace(),
            )

        def _control_tick(self) -> None:
            # A received takeover cannot race an in-progress publication tick.
            with self._lock:
                if self._hc_allowed():
                    self._publish_control()

        def _publish_control(self) -> None:
            if not self.config.enable_control or self.fault is not None:
                return
            if not self.config.allow_competing_publishers:
                topics = (
                    [pub.topic_name for pub in self._arm_publishers.values()]
                    if self._hc_api
                    else (
                        self.config.joint_command_topic,
                        self.config.right_hand_topic,
                        self.config.left_hand_topic,
                    )
                )
                for topic in topics:
                    if publishers := self._other_publishers(topic):
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
                if _state_is_stale(
                    time.monotonic(),
                    feedback_sample.received_monotonic_s,
                    self.config.state_timeout_s,
                ):
                    raise ValueError("joint feedback is stale")
                with self._lock:
                    if not self._reset_complete:
                        if not self._reset_limiter_initialized:
                            self.limiter.reset(feedback_sample.value[:14])
                            self._reset_limiter_initialized = True
                        arm = self.limiter.step(
                            self._default_joint_positions[:14],
                            1.0 / self.config.control_hz,
                        )
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
                        if self._control_actions is None:
                            if not self._received_action_chunk:
                                return
                            raise ValueError("action trajectory is empty")
                        if self._control_index >= len(self._control_actions):
                            raise ValueError("action trajectory is exhausted")
                        action = self._control_actions[
                            self._control_index
                        ].copy()
                        self._control_index += 1
                        arm = self.limiter.step(
                            arm_actions(action), 1.0 / self.config.control_hz
                        )
                stamp = self.get_clock().now().to_msg()
                if self._hc_api:
                    for side, (names, positions) in self._hc_api.targets(
                        arm
                    ).items():
                        joint_message = JointState(
                            name=names, position=positions
                        )
                        joint_message.header.stamp = stamp
                        self._arm_publishers[side].publish(joint_message)
                else:
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
                    self._joint_publisher.publish(joint_message)

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
                    publisher.publish(hand_message)
                with self._lock:
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
            self._worker.shutdown(wait=False, cancel_futures=True)
            return super().destroy_node()

    return HcTjAsyncNode


def _parse_args() -> HcTjAsyncClientConfig:
    repository_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument(
        "--urdf-path",
        type=Path,
        default=repository_root
        / "hc_assets/hc_tj_description/urdf/hc_tj_robot.urdf",
    )
    parser.add_argument("--depth-units-per-meter", type=float, required=True)
    parser.add_argument("--enable-control", action="store_true")
    parser.add_argument(
        "--robot-interface", choices=("legacy", "hc_motion"), default="legacy"
    )
    parser.add_argument("--robot-api-file", type=Path)
    parser.add_argument("--allow-competing-publishers", action="store_true")
    parser.add_argument("--inference-hz", type=float, default=2.0)
    parser.add_argument("--request-timeout-s", type=float, default=30.0)
    parser.add_argument("--camera-timeout-s", type=float, default=1.0)
    parser.add_argument("--state-timeout-s", type=float, default=0.25)
    for name, default in (
        ("color-topic", "/hc_teleop/camera_head/color/compressed"),
        ("depth-topic", "/hc_teleop/camera_head/depth/compressed"),
        ("calibration-topic", "/io_teleop/camera_head/calibration"),
        ("extrinsics-topic", "/io_teleop/camera_head/extrinsics"),
        ("joint-state-topic", "/hc_teleop/joint_states"),
        ("joint-command-topic", "/hc_teleop/joint_cmd"),
        ("right-hand-topic", "/hc_teleop/joint_cmd_finger_right"),
        ("left-hand-topic", "/hc_teleop/joint_cmd_finger_left"),
    ):
        parser.add_argument(f"--{name}", default=default)
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
