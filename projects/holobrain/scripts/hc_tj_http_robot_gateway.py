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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Expose HC-TJ local ROS data and commands through a small HTTP API."""

from __future__ import annotations
import argparse
import json
import os
import secrets
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
try:
    sys.path.remove(str(_REPOSITORY_ROOT))
except ValueError:
    pass
sys.path.insert(0, str(_REPOSITORY_ROOT))

from projects.holobrain.scripts.hc_tj_async_core import (  # noqa: E402
    ARM_JOINT_NAMES,
    CONTROL_JOINT_NAMES,
    ArmCommandLimiter,
    ProtocolError,
    arm_actions,
    hand_command,
    interpolate_actions,
    load_arm_joint_limits,
    validate_action_chunk,
)
from projects.holobrain.scripts.hc_tj_http_protocol import (  # noqa: E402
    ACTION_PATH,
    HEALTH_PATH,
    JOINT_FEEDBACK_CONTENT_TYPE,
    JOINT_FEEDBACK_PATH,
    OBSERVATION_CONTENT_TYPE,
    OBSERVATION_PATH,
    RobotJointFeedbackPacket,
    RobotObservationPacket,
    decode_action_chunk,
    encode_joint_feedback,
    encode_observation,
)

__all__ = [
    "GatewayConfig",
    "JointFeedback",
    "RobotObservationStore",
    "main",
]


@dataclass(frozen=True, slots=True)
class GatewayConfig:
    """Configure the independent HC-TJ robot-side HTTP gateway."""

    token: str
    host: str = "0.0.0.0"
    port: int = 8765
    enable_control: bool = False
    enable_hands: bool = False
    urdf_path: Path | None = None
    control_hz: float = 100.0
    state_timeout_s: float = 0.25
    action_timeout_s: float = 0.75
    max_arm_velocity_rad_s: float = 0.25
    first_point_jump_rad: float = 0.30
    color_topic: str = "/hc_teleop/camera_head/color/compressed"
    depth_topic: str = "/hc_teleop/camera_head/depth/compressed"
    calibration_topic: str = "/io_teleop/camera_head/calibration"
    extrinsics_topic: str = "/io_teleop/camera_head/extrinsics"
    joint_state_topic: str = "/hc_teleop/joint_states"
    joint_command_topic: str = "/hc_teleop/joint_cmd"
    right_hand_topic: str = "/hc_teleop/joint_cmd_finger_right"
    left_hand_topic: str = "/hc_teleop/joint_cmd_finger_left"

    def __post_init__(self) -> None:
        if len(self.token) < 16:
            raise ValueError("token must contain at least 16 characters")
        if not 0 < self.port <= 65535:
            raise ValueError("port must be in [1, 65535]")
        if (
            self.control_hz <= 0
            or self.state_timeout_s <= 0
            or self.action_timeout_s <= 0
        ):
            raise ValueError(
                "control rate, state timeout, and action timeout "
                "must be positive"
            )
        if self.max_arm_velocity_rad_s <= 0 or self.first_point_jump_rad <= 0:
            raise ValueError("control safety limits must be positive")
        if self.enable_control and (
            self.urdf_path is None or not self.urdf_path.is_file()
        ):
            raise ValueError("--enable-control requires a valid --urdf-path")
        if self.enable_hands and not self.enable_control:
            raise ValueError("--enable-hands requires --enable-control")


@dataclass(frozen=True, slots=True)
class _CompressedFrame:
    timestamp_ns: int
    format: str
    data: bytes


@dataclass(frozen=True, slots=True)
class JointFeedback:
    """Canonical robot feedback paired with gateway-local receive time."""

    timestamp_ns: int
    positions: np.ndarray
    received_monotonic_s: float


class RobotObservationStore:
    """Synchronize local ROS streams and serve the newest unseen RGB frame."""

    def __init__(self, capacity: int = 120) -> None:
        self._condition = threading.Condition()
        self._colors: deque[_CompressedFrame] = deque(maxlen=capacity)
        self._depths: deque[_CompressedFrame] = deque(maxlen=capacity)
        self._joints: deque[JointFeedback] = deque(maxlen=capacity)
        self._calibration_json: str | None = None
        self._extrinsics_json: str | None = None

    def add_color(
        self, timestamp_ns: int, image_format: str, data: bytes
    ) -> None:
        """Store one compressed RGB frame without decoding it."""
        self._add_frame(self._colors, timestamp_ns, image_format, data)

    def add_depth(
        self, timestamp_ns: int, image_format: str, data: bytes
    ) -> None:
        """Store one compressed depth frame without decoding it."""
        self._add_frame(self._depths, timestamp_ns, image_format, data)

    def _add_frame(
        self,
        target: deque[_CompressedFrame],
        timestamp_ns: int,
        image_format: str,
        data: bytes,
    ) -> None:
        if timestamp_ns <= 0 or not image_format or not data:
            raise ValueError(
                "compressed frame metadata and data must be valid"
            )
        with self._condition:
            target.append(
                _CompressedFrame(timestamp_ns, image_format, bytes(data))
            )
            self._condition.notify_all()

    def add_joints(
        self,
        timestamp_ns: int,
        names: tuple[str, ...],
        positions: np.ndarray,
    ) -> None:
        """Store one canonical 17-axis joint feedback sample."""
        values = np.asarray(positions, dtype=np.float64)
        if timestamp_ns <= 0:
            raise ValueError("joint feedback timestamp must be positive")
        if names != CONTROL_JOINT_NAMES:
            raise ValueError("joint feedback names do not match HC-TJ")
        if values.shape != (17,) or not np.all(np.isfinite(values)):
            raise ValueError("joint feedback must be finite with shape [17]")
        with self._condition:
            self._joints.append(
                JointFeedback(
                    timestamp_ns,
                    values.copy(),
                    time.monotonic(),
                )
            )
            self._condition.notify_all()

    def set_calibration(self, payload: str) -> None:
        """Store the camera calibration JSON after basic syntax validation."""
        json.loads(payload)
        with self._condition:
            self._calibration_json = payload
            self._condition.notify_all()

    def set_extrinsics(self, payload: str) -> None:
        """Store the robot-camera extrinsics JSON after syntax validation."""
        json.loads(payload)
        with self._condition:
            self._extrinsics_json = payload
            self._condition.notify_all()

    def latest_joints(self) -> JointFeedback:
        """Return an isolated copy of the newest feedback sample."""
        with self._condition:
            if not self._joints:
                raise ValueError("joint feedback is unavailable")
            sample = self._joints[-1]
            return JointFeedback(
                sample.timestamp_ns,
                sample.positions.copy(),
                sample.received_monotonic_s,
            )

    def status(self) -> dict[str, Any]:
        """Return observation readiness diagnostics for the health endpoint."""
        with self._condition:
            result: dict[str, Any] = {
                "color_frames": len(self._colors),
                "depth_frames": len(self._depths),
                "joint_samples": len(self._joints),
                "calibration": self._calibration_json is not None,
                "extrinsics": self._extrinsics_json is not None,
            }
            if self._colors and self._depths:
                result["newest_color_depth_delta_ms"] = (
                    abs(
                        self._colors[-1].timestamp_ns
                        - self._depths[-1].timestamp_ns
                    )
                    / 1_000_000
                )
            if self._colors and self._joints:
                result["newest_color_joint_delta_ms"] = (
                    abs(
                        self._colors[-1].timestamp_ns
                        - self._joints[-1].timestamp_ns
                    )
                    / 1_000_000
                )
            return result

    def wait_for_snapshot(
        self, after_observation_id: int, timeout_s: float
    ) -> RobotObservationPacket | None:
        """Wait for one synchronized observation newer than ``after``."""
        if after_observation_id < 0 or not 0 <= timeout_s <= 5:
            raise ValueError("invalid observation cursor or wait timeout")
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                packet = self._snapshot(after_observation_id)
                if packet is not None:
                    return packet
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

    def _snapshot(
        self, after_observation_id: int
    ) -> RobotObservationPacket | None:
        if (
            not self._colors
            or not self._depths
            or not self._joints
            or self._calibration_json is None
            or self._extrinsics_json is None
        ):
            return None
        color = self._colors[-1]
        if color.timestamp_ns <= after_observation_id:
            return None
        depth = min(
            self._depths,
            key=lambda item: abs(item.timestamp_ns - color.timestamp_ns),
        )
        joints = min(
            self._joints,
            key=lambda item: abs(item.timestamp_ns - color.timestamp_ns),
        )
        if abs(depth.timestamp_ns - color.timestamp_ns) > 50_000_000:
            return None
        if abs(joints.timestamp_ns - color.timestamp_ns) > 20_000_000:
            return None
        return RobotObservationPacket(
            observation_id=color.timestamp_ns,
            color_timestamp_ns=color.timestamp_ns,
            color_format=color.format,
            color_data=color.data,
            depth_timestamp_ns=depth.timestamp_ns,
            depth_format=depth.format,
            depth_data=depth.data,
            joint_timestamp_ns=joints.timestamp_ns,
            joint_names=CONTROL_JOINT_NAMES,
            joint_positions=tuple(joints.positions.tolist()),
            calibration_json=self._calibration_json,
            extrinsics_json=self._extrinsics_json,
        )


class _GatewayHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], gateway_node: Any) -> None:
        super().__init__(address, _GatewayRequestHandler)
        self.gateway_node = gateway_node


class _GatewayRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "HcTjRobotGateway/1"

    @property
    def gateway(self) -> Any:
        return self.server.gateway_node  # type: ignore[attr-defined]

    def _authorized(self) -> bool:
        expected = f"Bearer {self.gateway.config.token}"
        actual = self.headers.get("Authorization", "")
        return secrets.compare_digest(actual, expected)

    def _send_bytes(
        self, status: HTTPStatus, payload: bytes, content_type: str
    ) -> None:
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if payload:
            self.wfile.write(payload)

    def _send_json(self, status: HTTPStatus, data: dict[str, Any]) -> None:
        payload = json.dumps(data, separators=(",", ":")).encode("utf-8")
        self._send_bytes(status, payload, "application/json")

    def _require_authorization(self) -> bool:
        if self._authorized():
            return True
        self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
        return False

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == HEALTH_PATH:
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok" if self.gateway.fault is None else "fault",
                    "control_enabled": self.gateway.config.enable_control,
                    "hands_enabled": self.gateway.config.enable_hands,
                    "control_state": self.gateway.control_state,
                    "fault": self.gateway.fault,
                    "last_stop_reason": self.gateway.last_stop_reason,
                    "observations": self.gateway.observations.status(),
                },
            )
            return
        if parsed.path == JOINT_FEEDBACK_PATH:
            if not self._require_authorization():
                return
            try:
                feedback = self.gateway.observations.latest_joints()
                packet = RobotJointFeedbackPacket(
                    timestamp_ns=feedback.timestamp_ns,
                    joint_names=CONTROL_JOINT_NAMES,
                    joint_positions=tuple(feedback.positions.tolist()),
                )
                self._send_bytes(
                    HTTPStatus.OK,
                    encode_joint_feedback(packet),
                    JOINT_FEEDBACK_CONTENT_TYPE,
                )
            except (ProtocolError, ValueError) as error:
                self._send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(error)}
                )
            return
        if parsed.path != OBSERVATION_PATH:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        if not self._require_authorization():
            return
        try:
            query = parse_qs(parsed.query, strict_parsing=True)
            after = int(query.get("after", ["0"])[0])
            wait_s = float(query.get("wait_s", ["1.0"])[0])
            packet = self.gateway.observations.wait_for_snapshot(after, wait_s)
            if packet is None:
                self._send_bytes(HTTPStatus.NO_CONTENT, b"", "text/plain")
                return
            self._send_bytes(
                HTTPStatus.OK,
                encode_observation(packet),
                OBSERVATION_CONTENT_TYPE,
            )
        except (ProtocolError, TypeError, ValueError) as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != ACTION_PATH:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        if not self._require_authorization():
            return
        try:
            content_length = int(self.headers.get("Content-Length", "-1"))
            if not 0 < content_length <= 1_000_000:
                raise ProtocolError("invalid action Content-Length")
            sequence_id, actions = decode_action_chunk(
                self.rfile.read(content_length)
            )
            accepted_steps = self.gateway.accept_action_chunk(
                sequence_id, actions
            )
            self._send_json(
                HTTPStatus.OK,
                {
                    "accepted": True,
                    "sequence_id": sequence_id,
                    "control_steps": accepted_steps,
                },
            )
        except PermissionError as error:
            self._send_json(HTTPStatus.FORBIDDEN, {"error": str(error)})
        except (ProtocolError, TypeError, ValueError) as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except _ActionStreamPausedError as error:
            self._send_json(
                HTTPStatus.CONFLICT,
                {
                    "error": str(error),
                    "error_type": "action_stream_paused",
                    "recoverable": True,
                },
            )
        except RuntimeError as error:
            self._send_json(HTTPStatus.CONFLICT, {"error": str(error)})

    def log_message(self, message_format: str, *args: Any) -> None:
        self.gateway.get_logger().info(message_format % args)


def _timestamp_ns(message: Any) -> int:
    stamp = message.header.stamp
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    if value <= 0:
        raise ValueError("ROS message timestamp must be positive")
    return value


def _action_stream_timed_out(
    now_monotonic_s: float,
    last_action_received_monotonic_s: float | None,
    timeout_s: float,
) -> bool:
    if timeout_s <= 0:
        raise ValueError("action timeout must be positive")
    return (
        last_action_received_monotonic_s is None
        or now_monotonic_s - last_action_received_monotonic_s > timeout_s
    )


class _ActionStreamPausedError(RuntimeError):
    """Signal that only a fresh sequence-one action session may resume."""


@dataclass(slots=True)
class _ActionStreamSchedule:
    """Own the recoverable action-stream session state under the node lock."""

    actions: np.ndarray | None = None
    action_index: int = 0
    last_sequence_id: int = -1
    last_action_received_monotonic_s: float | None = None
    paused: bool = False
    last_stop_reason: str | None = None

    @property
    def control_state(self) -> str:
        if self.paused:
            return "paused"
        return "active" if self.actions is not None else "idle"

    def schedule(
        self,
        sequence_id: int,
        actions: np.ndarray,
        received_monotonic_s: float,
    ) -> bool:
        """Schedule a chunk and report whether it starts a control session.

        The first chunk and a valid sequence-one chunk after a deadman pause
        start a session. Later chunks are trajectory replacements within the
        same session and must preserve the previous command limiter state.
        """
        starts_session = self.last_sequence_id < 0 or self.paused
        if self.paused:
            if sequence_id != 1:
                raise _ActionStreamPausedError(
                    "action stream is paused; start a fresh action session "
                    "with sequence_id 1"
                )
        elif sequence_id <= self.last_sequence_id:
            raise ValueError("action sequence_id must increase")
        self.actions = actions
        self.action_index = 0
        self.last_sequence_id = sequence_id
        self.last_action_received_monotonic_s = received_monotonic_s
        self.paused = False
        self.last_stop_reason = None
        return starts_session

    def pause_if_timed_out(self, now: float, timeout_s: float) -> bool:
        """Stop an active stream after timeout while permitting safe re-arm."""
        if self.actions is None or not _action_stream_timed_out(
            now, self.last_action_received_monotonic_s, timeout_s
        ):
            return False
        self.actions = None
        self.action_index = 0
        self.last_action_received_monotonic_s = None
        self.paused = True
        self.last_stop_reason = "action stream heartbeat timed out"
        return True

    def stop(self, reason: str) -> None:
        """Clear scheduled actions for a non-recoverable gateway fault."""
        self.actions = None
        self.action_index = 0
        self.last_action_received_monotonic_s = None
        self.paused = False
        self.last_stop_reason = reason

    def next_action(self) -> np.ndarray | None:
        """Take the next scheduled action or report trajectory exhaustion."""
        if self.actions is None:
            return None
        if self.action_index >= len(self.actions):
            raise ValueError("action trajectory is exhausted")
        action = self.actions[self.action_index].copy()
        self.action_index += 1
        return action


def _arm_joint_command(
    arm_positions: np.ndarray,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    values = np.asarray(arm_positions, dtype=np.float64)
    if values.shape != (len(ARM_JOINT_NAMES),) or not np.all(
        np.isfinite(values)
    ):
        raise ValueError("arm command must contain 14 finite positions")
    return ARM_JOINT_NAMES, tuple(values.tolist())


def _ros_node_class():
    try:
        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
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
            "robot gateway requires sourced ROS 2 Humble"
        ) from error

    class HcTjHttpRobotGateway(Node):
        """Own local ROS I/O while HTTP threads exchange immutable payloads."""

        def __init__(self, config: GatewayConfig) -> None:
            super().__init__("holobrain_hc_tj_http_robot_gateway")
            self.config = config
            self.observations = RobotObservationStore()
            self._action_lock = threading.Lock()
            self._action_stream = _ActionStreamSchedule()
            self._fault: str | None = None
            self._next_publisher_check = 0.0
            self._camera_callback_group = MutuallyExclusiveCallbackGroup()
            self._state_callback_group = MutuallyExclusiveCallbackGroup()
            self._control_callback_group = MutuallyExclusiveCallbackGroup()

            reliable = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
            )
            best_effort = QoSProfile(
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
                best_effort,
                callback_group=self._camera_callback_group,
            )
            self.create_subscription(
                CompressedImage,
                config.depth_topic,
                self._on_depth,
                best_effort,
                callback_group=self._camera_callback_group,
            )
            self.create_subscription(
                JointState,
                config.joint_state_topic,
                self._on_joints,
                reliable,
                callback_group=self._state_callback_group,
            )
            self.create_subscription(
                String,
                config.calibration_topic,
                self._on_calibration,
                latched,
                callback_group=self._camera_callback_group,
            )
            self.create_subscription(
                String,
                config.extrinsics_topic,
                self._on_extrinsics,
                latched,
                callback_group=self._camera_callback_group,
            )

            self._joint_publisher = None
            self._right_hand_publisher = None
            self._left_hand_publisher = None
            self._limiter = None
            if config.enable_control:
                lower, upper = load_arm_joint_limits(config.urdf_path)
                self._limiter = ArmCommandLimiter(
                    lower,
                    upper,
                    config.max_arm_velocity_rad_s,
                )
                self._joint_publisher = self.create_publisher(
                    JointState, config.joint_command_topic, reliable
                )
                if config.enable_hands:
                    self._right_hand_publisher = self.create_publisher(
                        JointState, config.right_hand_topic, best_effort
                    )
                    self._left_hand_publisher = self.create_publisher(
                        JointState, config.left_hand_topic, best_effort
                    )
                self.create_timer(
                    1.0 / config.control_hz,
                    self._control_tick,
                    callback_group=self._control_callback_group,
                )
            else:
                self.get_logger().warning(
                    "dry-run gateway: no ROS control publishers were created"
                )

        @property
        def fault(self) -> str | None:
            with self._action_lock:
                return self._fault

        @property
        def control_state(self) -> str:
            with self._action_lock:
                if self._fault is not None:
                    return "fault"
                return self._action_stream.control_state

        @property
        def last_stop_reason(self) -> str | None:
            with self._action_lock:
                return self._action_stream.last_stop_reason

        def _latch_fault(self, reason: str) -> None:
            newly_latched = False
            with self._action_lock:
                if self._fault is None:
                    self._fault = reason
                    self._action_stream.stop(reason)
                    newly_latched = True
            if newly_latched:
                self.get_logger().error(f"Latched control fault: {reason}")

        def _on_color(self, message: Any) -> None:
            try:
                self.observations.add_color(
                    _timestamp_ns(message), message.format, message.data
                )
            except ValueError as error:
                self._latch_fault(str(error))

        def _on_depth(self, message: Any) -> None:
            try:
                self.observations.add_depth(
                    _timestamp_ns(message), message.format, message.data
                )
            except ValueError as error:
                self._latch_fault(str(error))

        def _on_joints(self, message: Any) -> None:
            try:
                self.observations.add_joints(
                    _timestamp_ns(message),
                    tuple(message.name),
                    np.asarray(message.position),
                )
            except ValueError as error:
                self._latch_fault(str(error))

        def _on_calibration(self, message: Any) -> None:
            try:
                self.observations.set_calibration(message.data)
            except (json.JSONDecodeError, ValueError) as error:
                self._latch_fault(str(error))

        def _on_extrinsics(self, message: Any) -> None:
            try:
                self.observations.set_extrinsics(message.data)
            except (json.JSONDecodeError, ValueError) as error:
                self._latch_fault(str(error))

        def accept_action_chunk(
            self, sequence_id: int, actions: np.ndarray
        ) -> int:
            """Validate and atomically schedule one native 30 Hz chunk."""
            if not self.config.enable_control:
                raise PermissionError("robot gateway control is disabled")
            if self.fault is not None:
                raise RuntimeError(f"gateway fault is latched: {self.fault}")
            feedback = self.observations.latest_joints()
            if (
                time.monotonic() - feedback.received_monotonic_s
                > self.config.state_timeout_s
            ):
                raise RuntimeError("joint feedback is stale")
            validate_action_chunk(
                actions,
                feedback.positions[:14],
                first_point_jump_rad=self.config.first_point_jump_rad,
            )
            interpolated = interpolate_actions(
                actions, target_hz=int(self.config.control_hz)
            )
            with self._action_lock:
                if self._fault is not None:
                    raise RuntimeError(
                        f"gateway fault is latched: {self._fault}"
                    )
                starts_session = self._action_stream.schedule(
                    sequence_id, interpolated, time.monotonic()
                )
                # RTC replans replace the remaining trajectory frequently.
                # Preserve the last published command across replacements so
                # the limiter cannot jump backward to delayed joint feedback.
                if starts_session:
                    self._limiter.reset(feedback.positions[:14])
            return len(interpolated)

        def _competing_publishers(self) -> list[str]:
            conflicts = []
            topics = [self.config.joint_command_topic]
            if self.config.enable_hands:
                topics.extend(
                    (self.config.right_hand_topic, self.config.left_hand_topic)
                )
            for topic in topics:
                for info in self.get_publishers_info_by_topic(topic):
                    if info.node_name != self.get_name():
                        conflicts.append(
                            f"{topic}:{info.node_namespace}/{info.node_name}"
                        )
            return conflicts

        def _control_tick(self) -> None:
            if self.fault is not None:
                return
            now = time.monotonic()
            if now >= self._next_publisher_check:
                self._next_publisher_check = now + 1.0
                if conflicts := self._competing_publishers():
                    self._latch_fault(
                        f"competing command publishers: {conflicts}"
                    )
                    return
            with self._action_lock:
                if self._action_stream.actions is None:
                    return
                paused = self._action_stream.pause_if_timed_out(
                    now, self.config.action_timeout_s
                )
            if paused:
                self.get_logger().warning(
                    "Action stream heartbeat timed out; control paused. "
                    "Awaiting a fresh sequence_id 1 action session."
                )
                return
            try:
                feedback = self.observations.latest_joints()
            except ValueError:
                self._latch_fault("joint feedback is unavailable")
                return
            try:
                if (
                    now - feedback.received_monotonic_s
                    > self.config.state_timeout_s
                ):
                    raise ValueError("joint feedback is stale")
                with self._action_lock:
                    action = self._action_stream.next_action()
                    if action is None:
                        return
                    arm = self._limiter.step(
                        arm_actions(action), 1.0 / self.config.control_hz
                    )
                command = JointState()
                command.header.stamp = self.get_clock().now().to_msg()
                # HoloBrain predicts arms and hands only. Never include the
                # three waist joints in a command frame: even echoing feedback
                # would activate the independent waist servoJ controller.
                names, positions = _arm_joint_command(arm)
                command.name = list(names)
                command.position = list(positions)
                self._joint_publisher.publish(command)
                if self.config.enable_hands:
                    for side, closure, publisher in (
                        ("right", action[7], self._right_hand_publisher),
                        ("left", action[15], self._left_hand_publisher),
                    ):
                        names, positions = hand_command(side, float(closure))
                        hand = JointState()
                        hand.header.stamp = command.header.stamp
                        hand.name = list(names)
                        hand.position = positions.tolist()
                        publisher.publish(hand)
            except (RuntimeError, ValueError) as error:
                self._latch_fault(str(error))

    return HcTjHttpRobotGateway


def _parse_args() -> GatewayConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token",
        default=os.environ.get("HC_TJ_GATEWAY_TOKEN"),
        help="Bearer token; defaults to HC_TJ_GATEWAY_TOKEN",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--enable-control", action="store_true")
    parser.add_argument("--enable-hands", action="store_true")
    parser.add_argument("--urdf-path", type=Path)
    parser.add_argument("--control-hz", type=float, default=100.0)
    parser.add_argument("--state-timeout-s", type=float, default=0.25)
    parser.add_argument("--action-timeout-s", type=float, default=0.75)
    parser.add_argument("--max-arm-velocity-rad-s", type=float, default=0.25)
    parser.add_argument("--first-point-jump-rad", type=float, default=0.30)
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
    args = parser.parse_args()
    if not args.token:
        parser.error("--token or HC_TJ_GATEWAY_TOKEN is required")
    return GatewayConfig(**vars(args))


def main() -> None:
    """Run local ROS I/O and the authenticated HTTP gateway."""
    config = _parse_args()
    try:
        import rclpy
        from rclpy.executors import (
            ExternalShutdownException,
            MultiThreadedExecutor,
        )
    except ImportError as error:
        raise RuntimeError(
            "source ROS 2 before starting the gateway"
        ) from error
    node_class = _ros_node_class()
    rclpy.init()
    node = node_class(config)
    server = _GatewayHTTPServer((config.host, config.port), node)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    node.get_logger().info(
        f"HC-TJ HTTP robot gateway listening on {config.host}:{config.port}"
    )
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
