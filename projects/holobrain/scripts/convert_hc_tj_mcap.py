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

"""Convert the hc_tj ROS 2 MCAP recording to a HoloBrain RODataset."""

from __future__ import annotations
import argparse
import bisect
import hashlib
import json
import math
import struct
import tempfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Generator, Iterator, Sequence

import cv2
import datasets as hg_datasets
import numpy as np
import torch

from robo_orchard_lab.dataset.datatypes import (
    BatchCameraDataEncoded,
    BatchCameraDataEncodedFeature,
    BatchFrameTransform,
    BatchJointsState,
    BatchJointsStateFeature,
    Distortion,
)
from robo_orchard_lab.dataset.robot.packaging import (
    DataFrame,
    DatasetPackaging,
    EpisodeData,
    EpisodeMeta,
    EpisodePackaging,
    InstructionData,
    RobotData,
    RobotDescriptionFormat,
    TaskData,
)

__all__ = ["convert_hc_tj_mcap", "main"]

MCAP_MAGIC = b"\x89MCAP0\r\n"
MCAP_DATA_END = b"\x0f" + struct.pack("<Q", 4) + struct.pack("<I", 0)
MCAP_EMPTY_FOOTER = (
    b"\x02" + struct.pack("<Q", 20) + struct.pack("<QQI", 0, 0, 0)
)

COLOR_TOPIC = "/io_teleop/camera_head/color"
DEPTH_TOPIC = "/io_teleop/camera_head/depth"
JOINT_STATE_TOPIC = "io_teleop/joint_states"
JOINT_COMMAND_TOPIC = "io_teleop/joint_cmd"
RIGHT_FINGER_COMMAND_TOPIC = "/io_teleop/joint_cmd_finger_right"
LEFT_FINGER_COMMAND_TOPIC = "/io_teleop/joint_cmd_finger_left"
CALIBRATION_TOPIC = "/io_teleop/camera_head/calibration"
EXTRINSICS_TOPIC = "/io_teleop/camera_head/extrinsics"

EXPECTED_TOPIC_TYPES = {
    COLOR_TOPIC: "sensor_msgs/msg/CompressedImage",
    DEPTH_TOPIC: "sensor_msgs/msg/CompressedImage",
    JOINT_STATE_TOPIC: "sensor_msgs/msg/JointState",
    JOINT_COMMAND_TOPIC: "sensor_msgs/msg/JointState",
    RIGHT_FINGER_COMMAND_TOPIC: "sensor_msgs/msg/JointState",
    LEFT_FINGER_COMMAND_TOPIC: "sensor_msgs/msg/JointState",
    CALIBRATION_TOPIC: "std_msgs/msg/String",
    EXTRINSICS_TOPIC: "std_msgs/msg/String",
}

ARM_JOINT_NAMES = tuple(
    [f"Joint{i}_R" for i in range(1, 8)] + [f"Joint{i}_L" for i in range(1, 8)]
)
RIGHT_ARM_JOINT_NAMES = ARM_JOINT_NAMES[:7]
LEFT_ARM_JOINT_NAMES = ARM_JOINT_NAMES[7:]
RIGHT_FINGER_JOINT_NAME = "R_ban"
LEFT_FINGER_JOINT_NAME = "L_ban"
HOLOBRAIN_JOINT_NAMES = (
    RIGHT_ARM_JOINT_NAMES
    + (RIGHT_FINGER_JOINT_NAME,)
    + LEFT_ARM_JOINT_NAMES
    + (LEFT_FINGER_JOINT_NAME,)
)
EXPECTED_SOURCE_JOINT_NAMES = ARM_JOINT_NAMES + ("leg_1", "leg_2", "zhi")

DEFAULT_DEPTH_TOLERANCE_NS = 50_000_000
DEFAULT_JOINT_TOLERANCE_NS = 20_000_000
DEFAULT_FINGER_TOLERANCE_NS = 150_000_000


@dataclass(slots=True)
class _RecoveryInfo:
    recovered: bool
    source_size: int
    recovery_offset: int | None
    discarded_bytes: int
    complete_records: int
    complete_chunks: int


@dataclass(slots=True)
class _CompressedSample:
    timestamp_ns: int
    data: bytes
    format: str


@dataclass(slots=True)
class _JointSample:
    timestamp_ns: int
    position: np.ndarray
    velocity: np.ndarray
    effort: np.ndarray


@dataclass(slots=True)
class _FingerSample:
    timestamp_ns: int
    position: float


@dataclass(slots=True)
class _ParsedSource:
    colors: list[_CompressedSample]
    depths: list[_CompressedSample]
    joint_states: list[_JointSample]
    joint_commands: list[_JointSample]
    right_finger_commands: list[_FingerSample]
    left_finger_commands: list[_FingerSample]
    intrinsic_mat: np.ndarray
    distortion: np.ndarray
    image_hw: tuple[int, int]
    camera_to_zhi_xyz: np.ndarray
    camera_to_zhi_quat_wxyz: np.ndarray
    message_counts: dict[str, int]
    command_zero_tail_messages: int
    finger_trimmed_tail_messages: int


@dataclass(slots=True)
class _SynchronizedFrame:
    color: _CompressedSample
    depth: _CompressedSample
    joint_state: _JointSample
    joint_command: _JointSample
    right_finger_command: _FingerSample
    left_finger_command: _FingerSample


class _October2026Adapter:
    """Adapt observation/action recordings to the original input contract.

    All temporary layout knowledge lives here. To retire it, remove this
    class and the adapter selection/normalization hooks in ``_load_source``.
    Synchronization and dataset packaging do not depend on this layout.
    """

    COMMAND_JOINT_NAMES = ARM_JOINT_NAMES
    TOPICS = {
        COLOR_TOPIC: "/observation/cameras/head/color/compressed",
        DEPTH_TOPIC: "/observation/cameras/head/depth/compressed",
        JOINT_STATE_TOPIC: "/observation/joint_states",
        JOINT_COMMAND_TOPIC: "/action/joint_states",
        RIGHT_FINGER_COMMAND_TOPIC: "/action/right_hand/joint_states",
        LEFT_FINGER_COMMAND_TOPIC: "/action/left_hand/joint_states",
        CALIBRATION_TOPIC: "/observation/cameras/head/calibration",
        EXTRINSICS_TOPIC: "/observation/cameras/head/extrinsics",
    }

    @staticmethod
    def normalize(message: Any, topic: str) -> Any:
        """Copy projected joints; leave other decoded messages untouched.

        Reject duplicate/missing names and malformed arrays before dropping
        unused joints. Existing parsers validate the projected values and
        binary gripper semantics. The decoded input is never mutated.
        """
        if topic == JOINT_COMMAND_TOPIC:
            # Waist targets disappear after startup in these recordings.
            selected_names = _October2026Adapter.COMMAND_JOINT_NAMES
        elif topic == RIGHT_FINGER_COMMAND_TOPIC:
            selected_names = (RIGHT_FINGER_JOINT_NAME,)
        elif topic == LEFT_FINGER_COMMAND_TOPIC:
            selected_names = (LEFT_FINGER_JOINT_NAME,)
        else:
            return message

        names = list(message.name)
        if len(set(names)) != len(names) or not set(selected_names) <= set(
            names
        ):
            raise ValueError(
                f"{_October2026Adapter.TOPICS[topic]} "
                "has duplicate or missing "
                f"joint names; required {list(selected_names)}, got {names}"
            )
        indices = [names.index(name) for name in selected_names]
        fields = {}
        for field in ("position", "velocity", "effort"):
            values = getattr(message, field)
            if len(values) == 0 and field != "position":
                fields[field] = values.copy()
                continue
            if len(values) != len(names):
                raise ValueError(
                    f"{_October2026Adapter.TOPICS[topic]} {field} has "
                    f"{len(values)} values for {len(names)} joint names"
                )
            fields[field] = np.asarray(values)[indices].copy()
        return replace(message, name=list(selected_names), **fields)


def _normalize_msgtype(msgtype: str) -> str:
    """Normalize ROS 2 type aliases embedded by the source MCAP."""
    if "/msg/" in msgtype:
        return msgtype
    package, separator, type_name = msgtype.partition("/")
    if not separator or not package or not type_name:
        return msgtype
    return f"{package}/msg/{type_name}"


def _scan_complete_mcap_records(data: bytes) -> tuple[int, int, int]:
    """Return the last complete top-level record boundary and counts.

    The returned offset excludes an MCAP trailing magic sequence. A valid
    closed file is recognized by its trailing magic and does not need this
    boundary for reading.
    """
    if not data.startswith(MCAP_MAGIC):
        raise ValueError("Input is not an MCAP file: leading magic is invalid")

    offset = len(MCAP_MAGIC)
    records = 0
    chunks = 0
    while offset + 9 <= len(data):
        if data[offset : offset + len(MCAP_MAGIC)] == MCAP_MAGIC:
            break
        opcode = data[offset]
        record_size = struct.unpack_from("<Q", data, offset + 1)[0]
        record_end = offset + 9 + record_size
        if record_end > len(data):
            break
        records += 1
        chunks += opcode == 0x06
        offset = record_end
    return offset, records, chunks


def _scan_complete_mcap_file(source_path: Path) -> tuple[int, int, int, bool]:
    """Scan top-level MCAP record boundaries without loading the file."""
    source_size = source_path.stat().st_size
    with source_path.open("rb") as source:
        if source.read(len(MCAP_MAGIC)) != MCAP_MAGIC:
            raise ValueError(
                "Input is not an MCAP file: leading magic is invalid"
            )
        if source_size >= len(MCAP_MAGIC):
            source.seek(source_size - len(MCAP_MAGIC))
            normally_closed = source.read(len(MCAP_MAGIC)) == MCAP_MAGIC
        else:
            normally_closed = False

        offset = len(MCAP_MAGIC)
        records = 0
        chunks = 0
        while offset + 9 <= source_size:
            source.seek(offset)
            header = source.read(9)
            if header[: len(MCAP_MAGIC)] == MCAP_MAGIC:
                break
            record_size = struct.unpack_from("<Q", header, 1)[0]
            record_end = offset + 9 + record_size
            if record_end > source_size:
                break
            records += 1
            chunks += header[0] == 0x06
            offset = record_end
    return offset, records, chunks, normally_closed


def _copy_prefix(source_path: Path, destination: Path, size: int) -> None:
    """Copy exactly ``size`` bytes for a recovered MCAP prefix."""
    remaining = size
    with source_path.open("rb") as source, destination.open("wb") as target:
        while remaining:
            block = source.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise OSError(f"Unexpected EOF while recovering {source_path}")
            target.write(block)
            remaining -= len(block)
        target.write(MCAP_DATA_END + MCAP_EMPTY_FOOTER + MCAP_MAGIC)


@contextmanager
def _readable_mcap_path(
    source_path: Path, recover_truncated: bool
) -> Generator[tuple[Path, _RecoveryInfo], None, None]:
    """Yield a readable MCAP path without modifying the source file.

    A normally closed MCAP is yielded directly. An unclosed or truncated MCAP
    requires ``recover_truncated=True``; only complete top-level records are
    copied, followed by a minimal DataEnd/Footer/trailing-magic sequence. The
    temporary copy is deleted when the context exits, including on errors.
    """
    source_size = source_path.stat().st_size
    recovery_offset, complete_records, complete_chunks, normally_closed = (
        _scan_complete_mcap_file(source_path)
    )
    if normally_closed:
        yield (
            source_path,
            _RecoveryInfo(
                recovered=False,
                source_size=source_size,
                recovery_offset=None,
                discarded_bytes=0,
                complete_records=complete_records,
                complete_chunks=complete_chunks,
            ),
        )
        return

    if not recover_truncated:
        raise ValueError(
            "Input MCAP is missing a valid footer or contains a truncated "
            "record; pass --recover-truncated to keep complete records"
        )
    if recovery_offset <= len(MCAP_MAGIC):
        raise ValueError("Input MCAP contains no complete records to recover")

    with tempfile.TemporaryDirectory(prefix="hc_tj_mcap_recovery_") as tmp:
        recovered_path = Path(tmp) / source_path.name
        _copy_prefix(source_path, recovered_path, recovery_offset)
        yield (
            recovered_path,
            _RecoveryInfo(
                recovered=True,
                source_size=source_size,
                recovery_offset=recovery_offset,
                discarded_bytes=source_size - recovery_offset,
                complete_records=complete_records,
                complete_chunks=complete_chunks,
            ),
        )


def _deserialize_message(
    typestore: Any,
    raw: bytes | memoryview,
    msgtype: str,
    *,
    allow_zero_tail: bool,
    allow_eight_byte_tail: bool = False,
) -> tuple[Any, bool]:
    """Deserialize CDR with topic-specific eight-byte tail exceptions."""
    raw_bytes = bytes(raw)
    try:
        return typestore.deserialize_cdr(raw_bytes, msgtype), False
    except Exception as original_error:
        tail = raw_bytes[-8:]
        tail_allowed = len(tail) == 8 and (
            allow_eight_byte_tail or (allow_zero_tail and tail == bytes(8))
        )
        if not tail_allowed:
            raise ValueError(
                f"Failed to deserialize {msgtype}; unsupported trailing data "
                "or malformed CDR payload"
            ) from original_error
        try:
            return typestore.deserialize_cdr(raw_bytes[:-8], msgtype), True
        except Exception as trimmed_error:
            raise ValueError(
                f"Failed to deserialize {msgtype} after removing its allowed "
                "eight-byte zero tail"
            ) from trimmed_error


def _header_timestamp_ns(message: Any, log_time_ns: int) -> int:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return int(log_time_ns)
    timestamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return timestamp_ns if timestamp_ns > 0 else int(log_time_ns)


def _parse_joint_sample(
    message: Any,
    timestamp_ns: int,
    topic: str,
    *,
    expected_names: tuple[str, ...] = EXPECTED_SOURCE_JOINT_NAMES,
) -> _JointSample:
    """Validate the source order and select the leading 14 arm joints.

    ``expected_names`` is the original 17-joint layout by default; an input
    adapter may provide the arm-only order after validating and projecting
    its source. Missing optional velocity/effort arrays become zeros.
    """
    names = tuple(message.name)
    if names != expected_names:
        raise ValueError(
            f"{topic} joint order mismatch: expected "
            f"{list(expected_names)}, got {list(names)}"
        )

    def select(values: Sequence[float], field: str) -> np.ndarray:
        if len(values) == 0 and field != "position":
            return np.zeros(len(ARM_JOINT_NAMES), dtype=np.float32)
        if len(values) != len(names):
            raise ValueError(
                f"{topic} {field} has {len(values)} values for "
                f"{len(names)} joint names"
            )
        selected = np.asarray(values[: len(ARM_JOINT_NAMES)], dtype=np.float32)
        if not np.all(np.isfinite(selected)):
            raise ValueError(f"{topic} {field} contains non-finite values")
        return selected

    return _JointSample(
        timestamp_ns=timestamp_ns,
        position=select(message.position, "position"),
        velocity=select(message.velocity, "velocity"),
        effort=select(message.effort, "effort"),
    )


def _parse_finger_sample(
    message: Any,
    timestamp_ns: int,
    topic: str,
    expected_name: str,
) -> _FingerSample:
    """Parse one binary gripper command from a singleton JointState."""
    names = tuple(message.name)
    if names != (expected_name,):
        raise ValueError(
            f"{topic} joint order mismatch: expected {[expected_name]}, "
            f"got {list(names)}"
        )
    if len(message.position) != 1:
        raise ValueError(
            f"{topic} position must contain exactly one value, got "
            f"{len(message.position)}"
        )
    position = float(message.position[0])
    if not math.isfinite(position) or position not in (0.0, 1.0):
        raise ValueError(
            f"{topic} position must be the binary end-effector state 0 or 1"
        )
    return _FingerSample(timestamp_ns=timestamp_ns, position=position)


def _json_object(message: Any, topic: str) -> dict[str, Any]:
    try:
        value = json.loads(message.data)
    except (AttributeError, json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"{topic} does not contain valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{topic} JSON root must be an object")
    return value


def _finite_vector(value: Any, length: int, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must contain {length} finite numbers")
    return array


def _parse_calibrations(
    values: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    if not values:
        raise ValueError(f"Required topic {CALIBRATION_TOPIC} has no messages")
    parsed: list[tuple[np.ndarray, np.ndarray, tuple[int, int]]] = []
    for value in values:
        try:
            if value["schema"] != "hc_camera_calibration/v1":
                raise ValueError("unsupported schema")
            if value["camera"] != "eye":
                raise ValueError("camera must be 'eye'")
            color_resolution = value["resolution"]["color"]
            depth_resolution = value["resolution"]["depth"]
            color_wh = (
                int(color_resolution["width"]),
                int(color_resolution["height"]),
            )
            depth_wh = (
                int(depth_resolution["width"]),
                int(depth_resolution["height"]),
            )
            if color_wh != depth_wh or min(color_wh) <= 0:
                raise ValueError("RGB and depth resolutions must match")
            rgb = value["rgb"]
            intrinsic = np.array(
                [
                    [rgb["fx"], 0.0, rgb["cx"]],
                    [0.0, rgb["fy"], rgb["cy"]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            if (
                not np.all(np.isfinite(intrinsic))
                or min(float(rgb["fx"]), float(rgb["fy"])) <= 0
            ):
                raise ValueError("RGB intrinsic is invalid")
            expected_order = ["k1", "k2", "p1", "p2", "k3"]
            if rgb["distortion_order"] != expected_order:
                raise ValueError("unsupported RGB distortion order")
            distortion = _finite_vector(
                rgb["distortion_coefficients"], 5, "RGB distortion"
            ).astype(np.float32)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid {CALIBRATION_TOPIC} payload: {error}"
            ) from error
        parsed.append((intrinsic, distortion, (color_wh[1], color_wh[0])))

    first = parsed[0]
    for current in parsed[1:]:
        if (
            current[2] != first[2]
            or not np.allclose(current[0], first[0], rtol=0, atol=1e-7)
            or not np.allclose(current[1], first[1], rtol=0, atol=1e-7)
        ):
            raise ValueError("Camera calibration changes within the episode")
    return first


def _parse_extrinsics(
    values: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    if not values:
        raise ValueError(f"Required topic {EXTRINSICS_TOPIC} has no messages")
    parsed: list[tuple[np.ndarray, np.ndarray]] = []
    for value in values:
        try:
            if value["schema"] != "hc_robot_camera_extrinsics/v1":
                raise ValueError("unsupported schema")
            if value["common_frame"] != "zhi_Link":
                raise ValueError("common_frame must be 'zhi_Link'")
            camera_transforms = [
                item
                for item in value["transforms"]
                if item.get("parent_frame") == "zhi_Link"
                and item.get("child_frame") == "camera_head"
            ]
            if len(camera_transforms) != 1:
                raise ValueError(
                    "exactly one zhi_Link -> camera_head transform is required"
                )
            camera_tf = camera_transforms[0]
            xyz = _finite_vector(
                camera_tf["translation_m"], 3, "camera translation"
            ).astype(np.float32)
            quat_xyzw = _finite_vector(
                camera_tf["rotation_xyzw"], 4, "camera quaternion"
            )
            quat_norm = float(np.linalg.norm(quat_xyzw))
            if not math.isclose(quat_norm, 1.0, rel_tol=0, abs_tol=1e-5):
                raise ValueError("camera quaternion must be normalized")
            quat_xyzw /= quat_norm
            quat_wxyz = quat_xyzw[[3, 0, 1, 2]].astype(np.float32)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid {EXTRINSICS_TOPIC} payload: {error}"
            ) from error
        parsed.append((xyz, quat_wxyz))

    first = parsed[0]
    for current in parsed[1:]:
        same_quat = np.allclose(current[1], first[1], rtol=0, atol=1e-7)
        opposite_quat = np.allclose(current[1], -first[1], rtol=0, atol=1e-7)
        if not np.allclose(current[0], first[0], rtol=0, atol=1e-7) or not (
            same_quat or opposite_quat
        ):
            raise ValueError("Camera extrinsics change within the episode")
    return first


def _validate_compressed_samples(
    samples: list[_CompressedSample],
    *,
    topic: str,
    image_hw: tuple[int, int],
    depth: bool,
) -> None:
    if not samples:
        raise ValueError(f"Required topic {topic} has no messages")
    for index, sample in enumerate(samples):
        format_lower = sample.format.lower()
        if depth:
            if "png" not in format_lower:
                raise ValueError(f"{topic} message {index} is not PNG")
            read_flag = cv2.IMREAD_UNCHANGED
        else:
            if "jpeg" not in format_lower and "jpg" not in format_lower:
                raise ValueError(f"{topic} message {index} is not JPEG")
            read_flag = cv2.IMREAD_COLOR
        decoded = cv2.imdecode(np.frombuffer(sample.data, np.uint8), read_flag)
        if decoded is None:
            raise ValueError(f"{topic} message {index} cannot be decoded")
        if decoded.shape[:2] != image_hw:
            raise ValueError(
                f"{topic} message {index} shape {decoded.shape[:2]} does not "
                f"match calibration {image_hw}"
            )
        if depth and (decoded.ndim != 2 or decoded.dtype != np.uint16):
            raise ValueError(
                f"{topic} message {index} must decode as single-channel uint16"
            )


def _load_source(read_path: Path) -> _ParsedSource:
    """Read and strictly validate all hc_tj source streams once."""
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
    except ImportError as error:
        raise RuntimeError(
            "rosbags is required; install the holobrain_0 optional extra"
        ) from error

    colors: list[_CompressedSample] = []
    depths: list[_CompressedSample] = []
    joint_states: list[_JointSample] = []
    joint_commands: list[_JointSample] = []
    right_finger_commands: list[_FingerSample] = []
    left_finger_commands: list[_FingerSample] = []
    calibration_values: list[dict[str, Any]] = []
    extrinsics_values: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    command_zero_tail_messages = 0
    finger_trimmed_tail_messages = 0
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    with AnyReader([read_path]) as reader:
        source_topics = {connection.topic for connection in reader.connections}
        adapter = None
        topic_map = {topic: topic for topic in EXPECTED_TOPIC_TYPES}
        if source_topics.intersection(_October2026Adapter.TOPICS.values()):
            if source_topics.intersection(EXPECTED_TOPIC_TYPES):
                raise ValueError(
                    "Mixed hc_tj MCAP topic layouts are ambiguous"
                )
            adapter = _October2026Adapter()
            topic_map = adapter.TOPICS
        canonical_topics = {
            actual: topic for topic, actual in topic_map.items()
        }
        connections_by_topic: dict[str, list[Any]] = {
            topic: [] for topic in EXPECTED_TOPIC_TYPES
        }
        for connection in reader.connections:
            topic = canonical_topics.get(connection.topic)
            if topic is not None:
                connections_by_topic[topic].append(connection)
        for topic, expected_type in EXPECTED_TOPIC_TYPES.items():
            connections = connections_by_topic[topic]
            if not connections:
                raise ValueError(
                    f"Required MCAP topic is missing: {topic_map[topic]}"
                )
            actual_types = {
                _normalize_msgtype(connection.msgtype)
                for connection in connections
            }
            if actual_types != {expected_type}:
                raise ValueError(
                    f"{topic} type mismatch: expected {expected_type}, got "
                    f"{sorted(actual_types)}"
                )

        selected_connections = [
            connection
            for connections in connections_by_topic.values()
            for connection in connections
        ]
        for connection, log_time_ns, raw in reader.messages(
            connections=selected_connections
        ):
            topic = canonical_topics[connection.topic]
            msgtype = EXPECTED_TOPIC_TYPES[topic]
            message, trimmed = _deserialize_message(
                typestore,
                raw,
                msgtype,
                allow_zero_tail=adapter is None
                and topic == JOINT_COMMAND_TOPIC,
                allow_eight_byte_tail=adapter is None
                and topic
                in (RIGHT_FINGER_COMMAND_TOPIC, LEFT_FINGER_COMMAND_TOPIC),
            )
            if adapter is not None:
                message = adapter.normalize(message, topic)
            counts[connection.topic] += 1
            if trimmed and topic == JOINT_COMMAND_TOPIC:
                command_zero_tail_messages += 1
            elif trimmed:
                finger_trimmed_tail_messages += 1
            timestamp_ns = _header_timestamp_ns(message, log_time_ns)
            if topic == COLOR_TOPIC or topic == DEPTH_TOPIC:
                sample = _CompressedSample(
                    timestamp_ns=timestamp_ns,
                    data=bytes(message.data),
                    format=str(message.format),
                )
                (colors if topic == COLOR_TOPIC else depths).append(sample)
            elif topic == JOINT_STATE_TOPIC:
                joint_states.append(
                    _parse_joint_sample(message, timestamp_ns, topic)
                )
            elif topic == JOINT_COMMAND_TOPIC:
                joint_commands.append(
                    _parse_joint_sample(
                        message,
                        timestamp_ns,
                        topic,
                        expected_names=(
                            adapter.COMMAND_JOINT_NAMES
                            if adapter is not None
                            else EXPECTED_SOURCE_JOINT_NAMES
                        ),
                    )
                )
            elif topic == RIGHT_FINGER_COMMAND_TOPIC:
                right_finger_commands.append(
                    _parse_finger_sample(
                        message, timestamp_ns, topic, RIGHT_FINGER_JOINT_NAME
                    )
                )
            elif topic == LEFT_FINGER_COMMAND_TOPIC:
                left_finger_commands.append(
                    _parse_finger_sample(
                        message, timestamp_ns, topic, LEFT_FINGER_JOINT_NAME
                    )
                )
            elif topic == CALIBRATION_TOPIC:
                calibration_values.append(_json_object(message, topic))
            elif topic == EXTRINSICS_TOPIC:
                extrinsics_values.append(_json_object(message, topic))

    intrinsic_mat, distortion, image_hw = _parse_calibrations(
        calibration_values
    )
    camera_to_zhi_xyz, camera_to_zhi_quat_wxyz = _parse_extrinsics(
        extrinsics_values
    )
    _validate_compressed_samples(
        colors, topic=COLOR_TOPIC, image_hw=image_hw, depth=False
    )
    _validate_compressed_samples(
        depths, topic=DEPTH_TOPIC, image_hw=image_hw, depth=True
    )
    for series in (
        colors,
        depths,
        joint_states,
        joint_commands,
        right_finger_commands,
        left_finger_commands,
    ):
        series.sort(key=lambda sample: sample.timestamp_ns)
    if not joint_states:
        raise ValueError(f"Required topic {JOINT_STATE_TOPIC} has no messages")
    if not joint_commands:
        raise ValueError(
            f"Required topic {JOINT_COMMAND_TOPIC} has no messages"
        )
    if not right_finger_commands:
        raise ValueError(
            f"Required topic {RIGHT_FINGER_COMMAND_TOPIC} has no messages"
        )
    if not left_finger_commands:
        raise ValueError(
            f"Required topic {LEFT_FINGER_COMMAND_TOPIC} has no messages"
        )
    return _ParsedSource(
        colors=colors,
        depths=depths,
        joint_states=joint_states,
        joint_commands=joint_commands,
        right_finger_commands=right_finger_commands,
        left_finger_commands=left_finger_commands,
        intrinsic_mat=intrinsic_mat,
        distortion=distortion,
        image_hw=image_hw,
        camera_to_zhi_xyz=camera_to_zhi_xyz,
        camera_to_zhi_quat_wxyz=camera_to_zhi_quat_wxyz,
        message_counts=dict(sorted(counts.items())),
        command_zero_tail_messages=command_zero_tail_messages,
        finger_trimmed_tail_messages=finger_trimmed_tail_messages,
    )


def _nearest_sample(
    samples: Sequence[Any], timestamps: Sequence[int], target_ns: int
) -> tuple[Any, int]:
    index = bisect.bisect_left(timestamps, target_ns)
    candidates = []
    if index < len(samples):
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    if not candidates:
        raise ValueError("Cannot synchronize against an empty source stream")
    selected = min(
        candidates,
        key=lambda candidate: (
            abs(timestamps[candidate] - target_ns),
            timestamps[candidate],
        ),
    )
    return samples[selected], abs(timestamps[selected] - target_ns)


def _synchronize_source(
    source: _ParsedSource,
) -> tuple[list[_SynchronizedFrame], dict[str, Any]]:
    """Synchronize source streams to RGB frames and report alignment.

    Finger commands are lower-rate binary states, so they use a wider
    nearest-neighbor tolerance than the arm joint streams.
    """
    streams = {
        "depth": (source.depths, DEFAULT_DEPTH_TOLERANCE_NS),
        "joint_state": (
            source.joint_states,
            DEFAULT_JOINT_TOLERANCE_NS,
        ),
        "joint_command": (
            source.joint_commands,
            DEFAULT_JOINT_TOLERANCE_NS,
        ),
        "right_finger_command": (
            source.right_finger_commands,
            DEFAULT_FINGER_TOLERANCE_NS,
        ),
        "left_finger_command": (
            source.left_finger_commands,
            DEFAULT_FINGER_TOLERANCE_NS,
        ),
    }
    timestamp_lists = {
        name: [sample.timestamp_ns for sample in samples]
        for name, (samples, _) in streams.items()
    }
    errors: dict[str, list[int]] = {name: [] for name in streams}
    drops: Counter[str] = Counter()
    frames: list[_SynchronizedFrame] = []
    # 按照rgb 来同步
    for color in source.colors:
        selected: dict[str, Any] = {}
        failed_stream = None
        for name, (samples, tolerance_ns) in streams.items():
            sample, error_ns = _nearest_sample(
                samples, timestamp_lists[name], color.timestamp_ns
            )
            if error_ns > tolerance_ns:
                failed_stream = name
                break
            selected[name] = sample
            errors[name].append(error_ns)
        if failed_stream is not None:
            drops[f"{failed_stream}_outside_tolerance"] += 1
            continue
        frames.append(
            _SynchronizedFrame(
                color=color,
                depth=selected["depth"],
                joint_state=selected["joint_state"],
                joint_command=selected["joint_command"],
                right_finger_command=selected["right_finger_command"],
                left_finger_command=selected["left_finger_command"],
            )
        )
    if not frames:
        raise ValueError(
            "No RGB frames satisfy all synchronization tolerances"
        )

    sync_error_ms = {}
    for name, values in errors.items():
        array = np.asarray(values, dtype=np.float64) / 1_000_000
        sync_error_ms[name] = {
            "max": float(array.max()) if len(array) else None,
            "mean": float(array.mean()) if len(array) else None,
            "p95": float(np.percentile(array, 95)) if len(array) else None,
            "tolerance": streams[name][1] / 1_000_000,
        }
    return frames, {
        "candidate_rgb_frames": len(source.colors),
        "output_frames": len(frames),
        "dropped_frames": len(source.colors) - len(frames),
        "dropped_by_reason": dict(sorted(drops.items())),
        "nearest_neighbor_error_ms": sync_error_ms,
    }


class _HcTjEpisodePackaging(EpisodePackaging):
    """Adapt one validated, synchronized hc_tj episode for RODataset."""

    def __init__(
        self,
        frames: list[_SynchronizedFrame],
        source: _ParsedSource,
        urdf_content: str,
        task_name: str,
        instruction: str,
        source_sha256: str,
    ) -> None:
        self.frames = frames
        self.source = source
        self.urdf_content = urdf_content
        self.task_name = task_name
        self.instruction = instruction
        self.source_sha256 = source_sha256

    def generate_episode_meta(self) -> EpisodeMeta:
        """Return the single episode's robot, task, and truncation metadata."""
        return EpisodeMeta(
            episode=EpisodeData(
                truncated=True,
                info={"uuid": self.source_sha256},
            ),
            robot=RobotData(
                name="hc_tj",
                content=self.urdf_content,
                content_format=RobotDescriptionFormat.URDF,
            ),
            task=TaskData(
                name=self.task_name,
                description=self.instruction,
            ),
        )

    @staticmethod
    def _joint_state(
        sample: _JointSample,
        right_finger: _FingerSample,
        left_finger: _FingerSample,
    ) -> BatchJointsState:
        """Insert each binary gripper state after its seven arm joints.

        The finger topics carry position only, so their velocity and effort
        entries are always zero in the returned 16-axis state.
        """
        right_arm = len(RIGHT_ARM_JOINT_NAMES)

        def with_fingers(
            values: np.ndarray, right_value: float, left_value: float
        ) -> np.ndarray:
            return np.concatenate(
                (
                    values[:right_arm],
                    np.asarray([right_value], dtype=np.float32),
                    values[right_arm:],
                    np.asarray([left_value], dtype=np.float32),
                )
            )

        position = with_fingers(
            sample.position, right_finger.position, left_finger.position
        )
        velocity = with_fingers(sample.velocity, 0.0, 0.0)
        effort = with_fingers(sample.effort, 0.0, 0.0)
        return BatchJointsState(
            position=torch.from_numpy(position[None]),
            velocity=torch.from_numpy(velocity[None]),
            effort=torch.from_numpy(effort[None]),
            names=list(HOLOBRAIN_JOINT_NAMES),
            timestamps=[sample.timestamp_ns],
        )

    def _camera(
        self, sample: _CompressedSample, *, format_name: str
    ) -> BatchCameraDataEncoded:
        # BatchFrameTransform(child=eye, parent=zhi_Link) stores eye ->
        # zhi_Link. Its inverse is the zhi_Link -> eye extrinsic consumed by
        # the historical T_world2cam reader field.
        camera_to_zhi = BatchFrameTransform(
            parent_frame_id="zhi_Link",
            child_frame_id="eye",
            xyz=torch.from_numpy(self.source.camera_to_zhi_xyz[None]),
            quat=torch.from_numpy(self.source.camera_to_zhi_quat_wxyz[None]),
            timestamps=[sample.timestamp_ns],
        )
        return BatchCameraDataEncoded(
            topic=COLOR_TOPIC if format_name == "jpeg" else DEPTH_TOPIC,
            frame_id="eye",
            image_shape=self.source.image_hw,
            intrinsic_matrices=torch.from_numpy(
                self.source.intrinsic_mat[None]
            ),
            distortion=Distortion(
                model="plumb_bob",
                coefficients=torch.from_numpy(self.source.distortion),
            ),
            pose=camera_to_zhi,
            sensor_data=[sample.data],
            format=format_name,
            timestamps=[sample.timestamp_ns],
        )

    def generate_frames(self) -> Generator[DataFrame, None, None]:
        """Yield RGB-keyed frames with 16-axis arm and gripper states."""
        instruction = InstructionData(
            name=self.task_name,
            json_content={"description": self.instruction},
        )
        try:
            for frame in self.frames:
                timestamp_ns = frame.color.timestamp_ns
                yield DataFrame(
                    features={
                        # The source finger commands encode the current
                        # end-effector state, so the synchronized value is
                        # present in both observation and action columns.
                        "joints": self._joint_state(
                            frame.joint_state,
                            frame.right_finger_command,
                            frame.left_finger_command,
                        ),
                        "actions": self._joint_state(
                            frame.joint_command,
                            frame.right_finger_command,
                            frame.left_finger_command,
                        ),
                        "head": self._camera(frame.color, format_name="jpeg"),
                        "head_depth": self._camera(
                            frame.depth, format_name="png"
                        ),
                    },
                    instruction=instruction,
                    timestamp_ns_min=timestamp_ns,
                    timestamp_ns_max=timestamp_ns,
                )
        finally:
            # Packaging consumes one episode before requesting the next.
            # Release compressed buffers so directory conversion is bounded
            # by one MCAP rather than all source episodes.
            self.frames.clear()
            self.source.colors.clear()
            self.source.depths.clear()
            self.source.joint_states.clear()
            self.source.joint_commands.clear()
            self.source.right_finger_commands.clear()
            self.source.left_finger_commands.clear()


def _dataset_features() -> hg_datasets.Features:
    return hg_datasets.Features(
        {
            "joints": BatchJointsStateFeature(dtype="float32"),
            "actions": BatchJointsStateFeature(dtype="float32"),
            "head": BatchCameraDataEncodedFeature(dtype="float32"),
            "head_depth": BatchCameraDataEncodedFeature(dtype="float32"),
        }
    )


def _resolve_input_paths(
    input_path: str | Path | Sequence[str | Path],
) -> list[Path]:
    """Resolve files and directories into unique, ordered MCAP paths."""
    raw_inputs = (
        [input_path]
        if isinstance(input_path, (str, Path))
        else list(input_path)
    )
    if not raw_inputs:
        raise ValueError("At least one input MCAP or directory is required")

    resolved: set[Path] = set()
    for raw_input in raw_inputs:
        path = Path(raw_input).expanduser().resolve()
        if path.is_dir():
            resolved.update(item.resolve() for item in path.rglob("*.mcap"))
        elif path.is_file():
            if path.suffix != ".mcap":
                raise ValueError(
                    f"Input file must use the .mcap suffix: {path}"
                )
            resolved.add(path)
        else:
            raise FileNotFoundError(f"Input path does not exist: {path}")
    if not resolved:
        raise FileNotFoundError(
            "No .mcap files were found in the provided input paths"
        )
    return sorted(resolved, key=str)


def _sha256_file(source_path: Path) -> str:
    """Hash a source MCAP without retaining its contents in memory."""
    digest = hashlib.sha256()
    with source_path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_report(
    source_path: Path,
    source_sha256: str,
    recovery: _RecoveryInfo,
    parsed: _ParsedSource,
    frame_report: dict[str, Any],
) -> dict[str, Any]:
    """Build the audit report for one converted source episode."""
    return {
        "source": {
            "path": str(source_path),
            "size_bytes": recovery.source_size,
            "sha256": source_sha256,
        },
        "recovery": {
            "recovered": recovery.recovered,
            "complete_records": recovery.complete_records,
            "complete_chunks": recovery.complete_chunks,
            "recovery_offset": recovery.recovery_offset,
            "discarded_bytes": recovery.discarded_bytes,
        },
        "messages": parsed.message_counts,
        "joint_command_zero_tail_messages": (
            parsed.command_zero_tail_messages
        ),
        "finger_command_trimmed_tail_messages": (
            parsed.finger_trimmed_tail_messages
        ),
        "frames": frame_report,
    }


class _HcTjEpisodeCollection:
    """Lazily parse ordered MCAP paths into pickleable episode sources."""

    def __init__(
        self,
        *,
        source_paths: list[Path],
        urdf_content: str,
        task_name: str,
        instruction: str,
        recover_truncated: bool,
    ) -> None:
        self.source_paths = source_paths
        self.urdf_content = urdf_content
        self.task_name = task_name
        self.instruction = instruction
        self.recover_truncated = recover_truncated
        self.reports: list[dict[str, Any]] = []

    def __iter__(self) -> Iterator[_HcTjEpisodePackaging]:
        self.reports.clear()
        for source_path in self.source_paths:
            source_sha256 = _sha256_file(source_path)
            with _readable_mcap_path(source_path, self.recover_truncated) as (
                read_path,
                recovery,
            ):
                parsed = _load_source(read_path)
                frames, frame_report = _synchronize_source(parsed)
            self.reports.append(
                _episode_report(
                    source_path,
                    source_sha256,
                    recovery,
                    parsed,
                    frame_report,
                )
            )
            yield _HcTjEpisodePackaging(
                frames=frames,
                source=parsed,
                urdf_content=self.urdf_content,
                task_name=self.task_name,
                instruction=self.instruction,
                source_sha256=source_sha256,
            )


def convert_hc_tj_mcap(
    *,
    input_path: str | Path | Sequence[str | Path],
    output_path: str | Path,
    urdf_path: str | Path,
    task_name: str,
    instruction: str,
    recover_truncated: bool = False,
    force_overwrite: bool = False,
) -> dict[str, Any]:
    """Convert hc_tj MCAP files into one multi-episode HoloBrain RODataset.

    The ``joints`` and ``actions`` columns use a 16-axis per-arm layout:
    ``Joint1_R..Joint7_R, R_ban, Joint1_L..Joint7_L, L_ban``. The two
    ``*_ban`` positions are synchronized binary end-effector states.
    Original ``io_teleop`` and October 2026 ``observation/action`` layouts
    are detected per file; mixing their recognized topics in one file is
    rejected. The latter uses issued action joints and extracts ``*_ban``
    from hand actions, ignoring extra head/hand joints. Reports retain the
    original source topic names. Both layouts use the same synchronization
    tolerances and camera calibration/extrinsics contract.

    Args:
        input_path: A source MCAP, a recursively searched directory, or a
            sequence of either. Inputs are read-only and never rewritten.
        output_path: Local destination directory for the RODataset.
        urdf_path: hc_tj URDF stored with the episode metadata.
        task_name: Stable task identifier.
        instruction: Natural-language instruction used for training.
        recover_truncated: Keep complete top-level MCAP records and ignore the
            damaged suffix when the file is not normally closed.
        force_overwrite: Allow ``DatasetPackaging`` to replace the destination.

    Returns:
        The JSON-compatible conversion report also written to the output.

    Raises:
        ValueError: If source topics, payloads, images, joints, calibration,
            extrinsics, recovery, or synchronization violate the hc_tj
            contract.
        FileExistsError: If the destination exists without force overwrite.
    """
    source_paths = _resolve_input_paths(input_path)
    destination = Path(output_path).expanduser().resolve()
    robot_path = Path(urdf_path).expanduser().resolve()
    if not robot_path.is_file():
        raise FileNotFoundError(f"URDF does not exist: {robot_path}")
    if not task_name.strip():
        raise ValueError("task_name must not be blank")
    if not instruction.strip():
        raise ValueError("instruction must not be blank")

    urdf_content = robot_path.read_text(encoding="utf-8")
    episodes = _HcTjEpisodeCollection(
        source_paths=source_paths,
        urdf_content=urdf_content,
        task_name=task_name.strip(),
        instruction=instruction.strip(),
        recover_truncated=recover_truncated,
    )

    if destination.exists() and not force_overwrite:
        raise FileExistsError(
            f"The dataset path '{destination}' already exists. "
            "Use force_overwrite=True to replace it."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.staging-", dir=destination.parent
    ) as temporary_parent:
        staging = Path(temporary_parent) / "dataset"
        DatasetPackaging(features=_dataset_features()).packaging(
            episodes=episodes,
            dataset_path=staging,
            fail_fast=True,
        )

        episode_reports = episodes.reports
        report: dict[str, Any] = {
            "episode_count": len(episode_reports),
            "total_frames": sum(
                episode_report["frames"]["output_frames"]
                for episode_report in episode_reports
            ),
            "episodes": episode_reports,
        }
        if len(episode_reports) == 1:
            # Preserve the original single-file report contract.
            report.update(episode_reports[0])

        report_path = staging / "conversion_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        backup = Path(temporary_parent) / "previous_dataset"
        if destination.exists():
            destination.rename(backup)
        try:
            staging.rename(destination)
        except Exception:
            if backup.exists():
                backup.rename(destination)
            raise
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert an hc_tj MCAP to HoloBrain RODataset Arrow data."
    )
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        dest="input_path",
        help=(
            "One or more MCAP files or directories "
            "(directories are recursive)."
        ),
    )
    parser.add_argument("--output", required=True, dest="output_path")
    parser.add_argument("--urdf", required=True, dest="urdf_path")
    parser.add_argument("--task-name", default="hc_tj")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--recover-truncated", action="store_true")
    parser.add_argument("--force-overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the hc_tj conversion command-line interface."""
    args = _build_parser().parse_args(argv)
    report = convert_hc_tj_mcap(**vars(args))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
