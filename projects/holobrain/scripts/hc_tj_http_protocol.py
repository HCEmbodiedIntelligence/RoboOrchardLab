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

"""Binary observation and JSON action contracts for the HC-TJ HTTP gateway."""

from __future__ import annotations
import json
import math
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

from projects.holobrain.scripts.hc_tj_async_core import (
    ACTION_HORIZON,
    ACTION_HZ,
    CONTROL_JOINT_NAMES,
    HC_TJ_JOINT_NAMES,
    ProtocolError,
)

__all__ = [
    "ACTION_PATH",
    "HEALTH_PATH",
    "JOINT_FEEDBACK_CONTENT_TYPE",
    "JOINT_FEEDBACK_PATH",
    "OBSERVATION_CONTENT_TYPE",
    "OBSERVATION_PATH",
    "PROTOCOL_VERSION",
    "RobotJointFeedbackPacket",
    "RobotObservationPacket",
    "decode_action_chunk",
    "decode_joint_feedback",
    "decode_observation",
    "encode_action_chunk",
    "encode_joint_feedback",
    "encode_observation",
]

PROTOCOL_VERSION = "hc_tj_http_gateway/v1"
OBSERVATION_PATH = "/hc_tj/observation/v1"
JOINT_FEEDBACK_PATH = "/hc_tj/joint_feedback/v1"
ACTION_PATH = "/hc_tj/action_chunk/v1"
HEALTH_PATH = "/healthz"
OBSERVATION_CONTENT_TYPE = "application/vnd.hc-tj.observation-v1"
JOINT_FEEDBACK_CONTENT_TYPE = "application/vnd.hc-tj.joint-feedback-v1"
_HEADER_LENGTH = struct.Struct(">I")
_MAX_HEADER_BYTES = 1_000_000
_MAX_IMAGE_BYTES = 8_000_000


@dataclass(frozen=True, slots=True)
class RobotObservationPacket:
    """One synchronized compressed RGB-D and canonical joint observation."""

    observation_id: int
    color_timestamp_ns: int
    color_format: str
    color_data: bytes
    depth_timestamp_ns: int
    depth_format: str
    depth_data: bytes
    joint_timestamp_ns: int
    joint_names: tuple[str, ...]
    joint_positions: tuple[float, ...]
    calibration_json: str
    extrinsics_json: str

    def __post_init__(self) -> None:
        timestamps = (
            self.observation_id,
            self.color_timestamp_ns,
            self.depth_timestamp_ns,
            self.joint_timestamp_ns,
        )
        if any(value <= 0 for value in timestamps):
            raise ProtocolError("observation timestamps must be positive")
        if self.observation_id != self.color_timestamp_ns:
            raise ProtocolError("observation_id must equal the RGB timestamp")
        if self.joint_names != CONTROL_JOINT_NAMES:
            raise ProtocolError("joint_names must match the canonical 17 axes")
        if len(self.joint_positions) != len(CONTROL_JOINT_NAMES) or not all(
            math.isfinite(value) for value in self.joint_positions
        ):
            raise ProtocolError("joint_positions must be 17 finite values")
        for name, value in (
            ("color_format", self.color_format),
            ("depth_format", self.depth_format),
            ("calibration_json", self.calibration_json),
            ("extrinsics_json", self.extrinsics_json),
        ):
            if not isinstance(value, str) or not value:
                raise ProtocolError(f"{name} must be a non-empty string")
        for name, value in (
            ("color_data", self.color_data),
            ("depth_data", self.depth_data),
        ):
            if not isinstance(value, bytes) or not value:
                raise ProtocolError(f"{name} must be non-empty bytes")
            if len(value) > _MAX_IMAGE_BYTES:
                raise ProtocolError(f"{name} exceeds the wire size limit")


@dataclass(frozen=True, slots=True)
class RobotJointFeedbackPacket:
    """One latest canonical joint feedback sample from the robot gateway."""

    timestamp_ns: int
    joint_names: tuple[str, ...]
    joint_positions: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.timestamp_ns <= 0:
            raise ProtocolError("joint feedback timestamp must be positive")
        if self.joint_names != CONTROL_JOINT_NAMES:
            raise ProtocolError("joint_names must match the canonical 17 axes")
        if len(self.joint_positions) != len(CONTROL_JOINT_NAMES) or not all(
            math.isfinite(value) for value in self.joint_positions
        ):
            raise ProtocolError("joint_positions must be 17 finite values")


def encode_joint_feedback(packet: RobotJointFeedbackPacket) -> bytes:
    """Encode one latest canonical joint feedback sample as JSON."""
    return json.dumps(
        {
            "protocol_version": PROTOCOL_VERSION,
            "joint_timestamp_ns": packet.timestamp_ns,
            "joint_names": packet.joint_names,
            "joint_positions": packet.joint_positions,
        },
        separators=(",", ":"),
    ).encode("utf-8")


def decode_joint_feedback(payload: bytes) -> RobotJointFeedbackPacket:
    """Decode and validate one latest joint feedback response."""
    try:
        data: dict[str, Any] = json.loads(payload)
        if data["protocol_version"] != PROTOCOL_VERSION:
            raise ProtocolError("unsupported joint feedback protocol version")
        return RobotJointFeedbackPacket(
            timestamp_ns=int(data["joint_timestamp_ns"]),
            joint_names=tuple(data["joint_names"]),
            joint_positions=tuple(float(x) for x in data["joint_positions"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ProtocolError("joint feedback payload is invalid") from error


def encode_observation(packet: RobotObservationPacket) -> bytes:
    """Encode length-prefixed JSON followed by compressed image bytes."""
    header = {
        "protocol_version": PROTOCOL_VERSION,
        "observation_id": packet.observation_id,
        "color_timestamp_ns": packet.color_timestamp_ns,
        "color_format": packet.color_format,
        "color_length": len(packet.color_data),
        "depth_timestamp_ns": packet.depth_timestamp_ns,
        "depth_format": packet.depth_format,
        "depth_length": len(packet.depth_data),
        "joint_timestamp_ns": packet.joint_timestamp_ns,
        "joint_names": packet.joint_names,
        "joint_positions": packet.joint_positions,
        "calibration_json": packet.calibration_json,
        "extrinsics_json": packet.extrinsics_json,
    }
    header_bytes = json.dumps(
        header, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    if len(header_bytes) > _MAX_HEADER_BYTES:
        raise ProtocolError("observation header exceeds the wire size limit")
    return b"".join(
        (
            _HEADER_LENGTH.pack(len(header_bytes)),
            header_bytes,
            packet.color_data,
            packet.depth_data,
        )
    )


def decode_observation(payload: bytes) -> RobotObservationPacket:
    """Decode and validate one binary HTTP observation response."""
    if len(payload) < _HEADER_LENGTH.size:
        raise ProtocolError("observation payload is truncated")
    header_length = _HEADER_LENGTH.unpack_from(payload)[0]
    if not 0 < header_length <= _MAX_HEADER_BYTES:
        raise ProtocolError("observation header length is invalid")
    header_end = _HEADER_LENGTH.size + header_length
    if header_end > len(payload):
        raise ProtocolError("observation header is truncated")
    try:
        header = json.loads(payload[_HEADER_LENGTH.size : header_end])
        if header["protocol_version"] != PROTOCOL_VERSION:
            raise ProtocolError("unsupported observation protocol version")
        color_length = int(header["color_length"])
        depth_length = int(header["depth_length"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ProtocolError("observation header is invalid") from error
    if not 0 < color_length <= _MAX_IMAGE_BYTES:
        raise ProtocolError("color payload length is invalid")
    if not 0 < depth_length <= _MAX_IMAGE_BYTES:
        raise ProtocolError("depth payload length is invalid")
    color_end = header_end + color_length
    depth_end = color_end + depth_length
    if depth_end != len(payload):
        raise ProtocolError("observation image lengths do not match payload")
    try:
        return RobotObservationPacket(
            observation_id=int(header["observation_id"]),
            color_timestamp_ns=int(header["color_timestamp_ns"]),
            color_format=str(header["color_format"]),
            color_data=bytes(payload[header_end:color_end]),
            depth_timestamp_ns=int(header["depth_timestamp_ns"]),
            depth_format=str(header["depth_format"]),
            depth_data=bytes(payload[color_end:depth_end]),
            joint_timestamp_ns=int(header["joint_timestamp_ns"]),
            joint_names=tuple(header["joint_names"]),
            joint_positions=tuple(float(x) for x in header["joint_positions"]),
            calibration_json=str(header["calibration_json"]),
            extrinsics_json=str(header["extrinsics_json"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ProtocolError("observation fields are invalid") from error


def encode_action_chunk(sequence_id: int, actions: np.ndarray) -> bytes:
    """Encode one native 30 Hz action chunk for robot-side execution."""
    values = np.asarray(actions)
    if sequence_id < 0:
        raise ProtocolError("action sequence_id must be non-negative")
    if (
        values.dtype != np.float32
        or values.ndim != 2
        or values.shape[1] != len(HC_TJ_JOINT_NAMES)
        or not 0 < len(values) <= ACTION_HORIZON
        or not np.all(np.isfinite(values))
    ):
        raise ProtocolError("actions must be finite float32 [1..64,16]")
    return json.dumps(
        {
            "protocol_version": PROTOCOL_VERSION,
            "sequence_id": sequence_id,
            "joint_names": HC_TJ_JOINT_NAMES,
            "action_hz": ACTION_HZ,
            "actions": values.tolist(),
        },
        separators=(",", ":"),
    ).encode("utf-8")


def decode_action_chunk(payload: bytes) -> tuple[int, np.ndarray]:
    """Decode and validate one native action chunk request."""
    try:
        data: dict[str, Any] = json.loads(payload)
        if data["protocol_version"] != PROTOCOL_VERSION:
            raise ProtocolError("unsupported action protocol version")
        if tuple(data["joint_names"]) != HC_TJ_JOINT_NAMES:
            raise ProtocolError("action joint_names do not match HC-TJ")
        if data["action_hz"] != ACTION_HZ:
            raise ProtocolError("action_hz must be 30")
        sequence_id = int(data["sequence_id"])
        actions = np.asarray(data["actions"], dtype=np.float32)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ProtocolError("action payload is invalid") from error
    # Reuse the encoder's validation without retaining the encoded copy.
    encode_action_chunk(sequence_id, actions)
    return sequence_id, actions
