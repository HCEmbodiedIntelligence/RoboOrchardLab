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

from __future__ import annotations
import json
import struct
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("rosbags")

from projects.holobrain.scripts.convert_hc_tj_mcap import (
    CALIBRATION_TOPIC,
    COLOR_TOPIC,
    DEPTH_TOPIC,
    EXPECTED_SOURCE_JOINT_NAMES,
    EXTRINSICS_TOPIC,
    HOLOBRAIN_JOINT_NAMES,
    JOINT_COMMAND_TOPIC,
    JOINT_STATE_TOPIC,
    LEFT_FINGER_COMMAND_TOPIC,
    MCAP_MAGIC,
    RIGHT_FINGER_COMMAND_TOPIC,
    _deserialize_message,
    _load_source,
    _October2026Adapter,
    _parse_finger_sample,
    _readable_mcap_path,
    convert_hc_tj_mcap,
)
from projects.holobrain.scripts.replay_hc_tj_mcap_urdf import (
    FINGER_MAX_POSITION_RAD,
    FINGER_MIN_POSITION_RAD,
    TORSO_JOINT_CONFIG,
    _load_joint_samples,
    _source_joint_config,
)
from robo_orchard_lab.dataset.robot import RODataset
from robo_orchard_lab.dataset.robot.db_orm import Episode


def _encode_image(image: np.ndarray, suffix: str) -> bytes:
    success, encoded = cv2.imencode(suffix, image)
    assert success
    return encoded.tobytes()


def _calibration() -> dict:
    return {
        "schema": "hc_camera_calibration/v1",
        "camera": "eye",
        "resolution": {
            "color": {"width": 6, "height": 4},
            "depth": {"width": 6, "height": 4},
        },
        "rgb": {
            "fx": 4.0,
            "fy": 4.1,
            "cx": 3.0,
            "cy": 2.0,
            "distortion_order": ["k1", "k2", "p1", "p2", "k3"],
            "distortion_coefficients": [0.0] * 5,
        },
    }


def _extrinsics() -> dict:
    return {
        "schema": "hc_robot_camera_extrinsics/v1",
        "common_frame": "zhi_Link",
        "transforms": [
            {
                "parent_frame": "zhi_Link",
                "child_frame": "camera_head",
                "translation_m": [0.1, 0.2, 0.3],
                "rotation_xyzw": [0.1, 0.2, 0.3, 0.9273618495],
            }
        ],
    }


def _write_synthetic_mcap(path: Path, *, october: bool = False) -> Path:
    from rosbags.rosbag2 import StoragePlugin, Writer
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    types = typestore.types
    time_type = types["builtin_interfaces/msg/Time"]
    header_type = types["std_msgs/msg/Header"]
    string_type = types["std_msgs/msg/String"]
    compressed_type = types["sensor_msgs/msg/CompressedImage"]
    joint_type = types["sensor_msgs/msg/JointState"]
    rgb = _encode_image(np.full((4, 6, 3), 31, np.uint8), ".jpg")
    depth = _encode_image(np.arange(24, dtype=np.uint16).reshape(4, 6), ".png")

    with Writer(path, version=9, storage_plugin=StoragePlugin.MCAP) as writer:
        topic_map = {
            COLOR_TOPIC: "/observation/cameras/head/color/compressed",
            DEPTH_TOPIC: "/observation/cameras/head/depth/compressed",
            JOINT_STATE_TOPIC: "/observation/joint_states",
            JOINT_COMMAND_TOPIC: "/action/joint_states",
            RIGHT_FINGER_COMMAND_TOPIC: "/action/right_hand/joint_states",
            LEFT_FINGER_COMMAND_TOPIC: "/action/left_hand/joint_states",
            CALIBRATION_TOPIC: "/observation/cameras/head/calibration",
            EXTRINSICS_TOPIC: "/observation/cameras/head/extrinsics",
        }
        connections = {
            topic: writer.add_connection(
                topic_map[topic] if october else topic,
                msgtype,
                typestore=typestore,
            )
            for topic, msgtype in {
                COLOR_TOPIC: "sensor_msgs/msg/CompressedImage",
                DEPTH_TOPIC: "sensor_msgs/msg/CompressedImage",
                JOINT_STATE_TOPIC: "sensor_msgs/msg/JointState",
                JOINT_COMMAND_TOPIC: "sensor_msgs/msg/JointState",
                RIGHT_FINGER_COMMAND_TOPIC: "sensor_msgs/msg/JointState",
                LEFT_FINGER_COMMAND_TOPIC: "sensor_msgs/msg/JointState",
                CALIBRATION_TOPIC: "std_msgs/msg/String",
                EXTRINSICS_TOPIC: "std_msgs/msg/String",
            }.items()
        }

        def write(topic: str, message, timestamp_ns: int, zero_tail=False):
            if october and topic == JOINT_COMMAND_TOPIC:
                # Startup includes waist targets; later commands omit them.
                size = 17 if timestamp_ns < 1_040_000_000 else 14
                message = replace(
                    message,
                    name=message.name[:size]
                    + ["head_1", "head_2", "R_ban", "L_ban"],
                    position=np.concatenate(
                        [message.position[:size], [9.0] * 4]
                    ),
                    velocity=np.array([], dtype=np.float64),
                    effort=np.array([], dtype=np.float64),
                )
                # The adapter must select by name, not positional prefixes.
                message = replace(
                    message,
                    name=message.name[::-1],
                    position=message.position[::-1].copy(),
                )
            elif october and topic in (
                RIGHT_FINGER_COMMAND_TOPIC,
                LEFT_FINGER_COMMAND_TOPIC,
            ):
                prefix = message.name[0][0]
                message = replace(
                    message,
                    name=[f"{prefix}_finger_{i}" for i in range(10)]
                    + message.name,
                    position=np.concatenate([[0.25] * 10, message.position]),
                )
            raw = bytes(typestore.serialize_cdr(message, message.__msgtype__))
            writer.write(
                connections[topic],
                timestamp_ns,
                raw + (bytes(8) if zero_tail and not october else b""),
            )

        write(
            CALIBRATION_TOPIC,
            string_type(data=json.dumps(_calibration())),
            900_000_000,
        )
        write(
            EXTRINSICS_TOPIC,
            string_type(data=json.dumps(_extrinsics())),
            900_000_001,
        )
        for index, timestamp_ns in enumerate((1_000_000_000, 1_040_000_000)):
            stamp = time_type(
                sec=timestamp_ns // 1_000_000_000,
                nanosec=timestamp_ns % 1_000_000_000,
            )
            header = header_type(stamp=stamp, frame_id="eye")
            write(
                COLOR_TOPIC,
                compressed_type(
                    header=header,
                    format="jpeg",
                    data=np.frombuffer(rgb, dtype=np.uint8),
                ),
                timestamp_ns,
            )
            write(
                DEPTH_TOPIC,
                compressed_type(
                    header=header,
                    format="16UC1; compressedDepth png",
                    data=np.frombuffer(depth, dtype=np.uint8),
                ),
                timestamp_ns + 1_000_000,
            )
            position = np.arange(17, dtype=np.float64) / 100 + index
            joint = joint_type(
                header=header,
                name=list(EXPECTED_SOURCE_JOINT_NAMES),
                position=position,
                velocity=np.zeros(17, dtype=np.float64),
                effort=np.zeros(17, dtype=np.float64),
            )
            write(JOINT_STATE_TOPIC, joint, timestamp_ns + 2_000_000)
            write(
                JOINT_COMMAND_TOPIC,
                joint,
                timestamp_ns + 3_000_000,
                zero_tail=True,
            )
            write(
                RIGHT_FINGER_COMMAND_TOPIC,
                joint_type(
                    header=header,
                    name=["R_ban"],
                    position=np.asarray([index % 2], dtype=np.float64),
                    velocity=np.asarray([], dtype=np.float64),
                    effort=np.asarray([], dtype=np.float64),
                ),
                timestamp_ns + 4_000_000,
            )
            write(
                LEFT_FINGER_COMMAND_TOPIC,
                joint_type(
                    header=header,
                    name=["L_ban"],
                    position=np.asarray([1 - index % 2], dtype=np.float64),
                    velocity=np.asarray([], dtype=np.float64),
                    effort=np.asarray([], dtype=np.float64),
                ),
                timestamp_ns + 4_000_000,
            )
    return path / f"{path.name}.mcap"


def test_recovery_is_explicit_and_temporary(tmp_path: Path) -> None:
    complete = b"\x01" + struct.pack("<Q", 3) + b"abc"
    partial = b"\x06" + struct.pack("<Q", 10) + b"bad"
    source = tmp_path / "truncated.mcap"
    source.write_bytes(MCAP_MAGIC + complete + partial)

    with pytest.raises(ValueError, match="recover-truncated"):
        with _readable_mcap_path(source, recover_truncated=False):
            pass

    with _readable_mcap_path(source, recover_truncated=True) as (
        recovered,
        info,
    ):
        temporary_parent = recovered.parent
        assert recovered != source
        assert recovered.exists()
        assert recovered.read_bytes().startswith(MCAP_MAGIC + complete)
        assert recovered.read_bytes().endswith(MCAP_MAGIC)
        assert info.recovery_offset == len(MCAP_MAGIC + complete)
        assert info.discarded_bytes == len(partial)
    assert not temporary_parent.exists()
    assert source.read_bytes() == MCAP_MAGIC + complete + partial


def test_zero_tail_compatibility_is_exact() -> None:
    class FakeTypestore:
        def deserialize_cdr(self, raw: bytes, msgtype: str):
            if raw == b"payload":
                return msgtype
            raise AssertionError

    store = FakeTypestore()
    message, trimmed = _deserialize_message(
        store,
        b"payload" + bytes(8),
        "sensor_msgs/msg/JointState",
        allow_zero_tail=True,
    )
    assert message == "sensor_msgs/msg/JointState"
    assert trimmed is True
    message, trimmed = _deserialize_message(
        store,
        b"payloadtaildata",
        "sensor_msgs/msg/JointState",
        allow_zero_tail=False,
        allow_eight_byte_tail=True,
    )
    assert message == "sensor_msgs/msg/JointState"
    assert trimmed is True
    with pytest.raises(ValueError, match="unsupported trailing data"):
        _deserialize_message(
            store,
            b"payload" + b"\x00" * 7 + b"\x01",
            "sensor_msgs/msg/JointState",
            allow_zero_tail=True,
        )
    with pytest.raises(ValueError, match="after removing"):
        _deserialize_message(
            store,
            b"bad" + bytes(16),
            "sensor_msgs/msg/JointState",
            allow_zero_tail=True,
        )


@pytest.mark.parametrize("october", [False, True])
def test_synthetic_mcap_converts_to_ro_dataset(
    tmp_path: Path, october: bool
) -> None:
    mcap_path = _write_synthetic_mcap(tmp_path / "bag", october=october)
    output = tmp_path / "dataset"
    urdf = (
        Path(__file__).resolve().parents[4]
        / "hc_assets"
        / "hc_tj_description"
        / "urdf"
        / "hc_tj_robot.urdf"
    )
    source_digest = mcap_path.read_bytes()

    report = convert_hc_tj_mcap(
        input_path=mcap_path,
        output_path=output,
        urdf_path=urdf,
        task_name="synthetic_hc_tj",
        instruction="Move both arms.",
    )

    assert report["recovery"]["recovered"] is False
    assert report["episode_count"] == 1
    assert report["total_frames"] == 2
    assert len(report["episodes"]) == 1
    assert report["frames"]["candidate_rgb_frames"] == 2
    assert report["frames"]["output_frames"] == 2
    assert report["joint_command_zero_tail_messages"] == (0 if october else 2)
    assert report["finger_command_trimmed_tail_messages"] == 0
    assert mcap_path.read_bytes() == source_digest
    assert {
        "state.json",
        "dataset_info.json",
        "meta_db.duckdb",
        "conversion_report.json",
    }.issubset(path.name for path in output.iterdir())

    dataset = RODataset(str(output))
    assert len(dataset) == 2
    frame = dataset[0]
    assert frame["joints"].names == list(HOLOBRAIN_JOINT_NAMES)
    assert tuple(frame["joints"].position.shape) == (1, 16)
    np.testing.assert_array_equal(
        frame["joints"].position.numpy()[0, [7, 15]], [0, 1]
    )
    np.testing.assert_array_equal(
        frame["actions"].position.numpy()[0, [7, 15]], [0, 1]
    )
    np.testing.assert_array_equal(
        frame["joints"].velocity.numpy()[0, [7, 15]], [0, 0]
    )
    replay_actions, replay_joints, episode_count = _load_joint_samples(
        output, episode_index=0
    )
    assert episode_count == 1
    assert len(replay_actions) == len(replay_joints) == 2
    replay_config = _source_joint_config(replay_actions[0])
    assert replay_config["R_ban"] == FINGER_MAX_POSITION_RAD
    assert replay_config["L_ban"] == FINGER_MIN_POSITION_RAD
    assert all(
        replay_config[name] == value
        for name, value in TORSO_JOINT_CONFIG.items()
    )
    right_topic = (
        "/action/right_hand/joint_states"
        if october
        else RIGHT_FINGER_COMMAND_TOPIC
    )
    left_topic = (
        "/action/left_hand/joint_states"
        if october
        else LEFT_FINGER_COMMAND_TOPIC
    )
    assert report["messages"][right_topic] == 2
    assert report["messages"][left_topic] == 2
    np.testing.assert_allclose(
        frame["actions"].position.numpy()[0, :7], np.arange(7) / 100
    )
    np.testing.assert_allclose(
        dataset[1]["actions"].position.numpy()[0, 8:15],
        np.arange(7, 14) / 100 + 1,
    )
    assert (
        report["frames"]["nearest_neighbor_error_ms"]["right_finger_command"][
            "tolerance"
        ]
        == 150.0
    )
    assert frame["head"].frame_id == "eye"
    assert frame["head"].pose.parent_frame_id == "zhi_Link"
    assert frame["head"].pose.child_frame_id == "eye"
    np.testing.assert_allclose(
        frame["head"].pose.quat.numpy()[0],
        [0.92736185, 0.1, 0.2, 0.3],
        atol=1e-6,
    )
    decoded_depth = cv2.imdecode(
        np.frombuffer(frame["head_depth"].sensor_data[0], np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    assert decoded_depth.dtype == np.uint16
    assert decoded_depth.shape == (4, 6)


def test_october_projection_validates_and_does_not_mutate() -> None:
    from rosbags.typesys import Stores, get_typestore

    joint_type = get_typestore(Stores.ROS2_HUMBLE).types[
        "sensor_msgs/msg/JointState"
    ]
    message = joint_type(
        header=None,
        name=["unused", "R_ban"],
        position=np.array([0.25, 1.0]),
        velocity=np.array([]),
        effort=np.array([]),
    )
    normalized = _October2026Adapter.normalize(
        message, RIGHT_FINGER_COMMAND_TOPIC
    )
    assert normalized.name == ["R_ban"]
    assert (
        _parse_finger_sample(
            normalized, 1, RIGHT_FINGER_COMMAND_TOPIC, "R_ban"
        ).position
        == 1.0
    )
    normalized.position[0] = 0
    assert message.position.tolist() == [0.25, 1.0]
    assert message.name == ["unused", "R_ban"]
    for invalid in (
        replace(message, name=["R_ban", "R_ban"]),
        replace(message, name=["unused", "missing"]),
        replace(message, position=np.array([1.0])),
        replace(message, velocity=np.array([1.0])),
    ):
        with pytest.raises(ValueError, match="joint names|values"):
            _October2026Adapter.normalize(invalid, RIGHT_FINGER_COMMAND_TOPIC)
    for value in (0.5, np.nan):
        invalid = replace(message, position=np.array([0.25, value]))
        normalized = _October2026Adapter.normalize(
            invalid, RIGHT_FINGER_COMMAND_TOPIC
        )
        with pytest.raises(ValueError, match="binary"):
            _parse_finger_sample(
                normalized, 1, RIGHT_FINGER_COMMAND_TOPIC, "R_ban"
            )


@pytest.mark.parametrize("mixed", [False, True])
def test_incomplete_or_mixed_layout_rejected(
    tmp_path: Path, mixed: bool
) -> None:
    from rosbags.rosbag2 import StoragePlugin, Writer
    from rosbags.typesys import Stores, get_typestore

    path = tmp_path / "bad"
    store = get_typestore(Stores.ROS2_HUMBLE)
    with Writer(path, version=9, storage_plugin=StoragePlugin.MCAP) as writer:
        writer.add_connection(
            "/observation/cameras/head/color/compressed",
            "sensor_msgs/msg/CompressedImage",
            typestore=store,
        )
        if mixed:
            writer.add_connection(
                COLOR_TOPIC, "sensor_msgs/msg/CompressedImage", typestore=store
            )
    with pytest.raises(ValueError, match="Mixed" if mixed else "missing"):
        _load_source(path / "bad.mcap")


def test_mcap_directory_converts_to_multi_episode_dataset(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "sources"
    nested = source_root / "nested"
    nested.mkdir(parents=True)
    first = _write_synthetic_mcap(source_root / "episode_b")
    second = _write_synthetic_mcap(nested / "episode_a", october=True)
    output = tmp_path / "dataset"
    urdf = (
        Path(__file__).resolve().parents[4]
        / "hc_assets"
        / "hc_tj_description"
        / "urdf"
        / "hc_tj_robot.urdf"
    )

    report = convert_hc_tj_mcap(
        input_path=source_root,
        output_path=output,
        urdf_path=urdf,
        task_name="synthetic_hc_tj",
        instruction="Move both arms.",
    )

    assert report["episode_count"] == 2
    assert report["total_frames"] == 4
    assert [item["source"]["path"] for item in report["episodes"]] == sorted(
        [str(first.resolve()), str(second.resolve())]
    )
    assert all(
        item["frames"]["output_frames"] == 2 for item in report["episodes"]
    )
    dataset = RODataset(str(output))
    assert len(dataset) == 4
    assert len(list(dataset.iterate_meta(Episode))) == 2
    assert [dataset[index]["frame_index"] for index in range(4)] == [
        0,
        1,
        0,
        1,
    ]


def test_empty_mcap_directory_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "empty"
    source.mkdir()
    with pytest.raises(FileNotFoundError, match="No .mcap files"):
        convert_hc_tj_mcap(
            input_path=source,
            output_path=tmp_path / "dataset",
            urdf_path=tmp_path / "unused.urdf",
            task_name="hc_tj",
            instruction="Move both arms.",
        )


def test_failed_force_overwrite_preserves_existing_dataset(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sources"
    source.mkdir()
    _write_synthetic_mcap(source / "episode_a")
    (source / "episode_z.mcap").write_bytes(b"not an MCAP")
    output = tmp_path / "dataset"
    output.mkdir()
    marker = output / "existing.txt"
    marker.write_text("keep me", encoding="utf-8")
    urdf = (
        Path(__file__).resolve().parents[4]
        / "hc_assets"
        / "hc_tj_description"
        / "urdf"
        / "hc_tj_robot.urdf"
    )

    with pytest.raises(ValueError, match="leading magic is invalid"):
        convert_hc_tj_mcap(
            input_path=source,
            output_path=output,
            urdf_path=urdf,
            task_name="synthetic_hc_tj",
            instruction="Move both arms.",
            force_overwrite=True,
        )

    assert marker.read_text(encoding="utf-8") == "keep me"
    assert list(output.iterdir()) == [marker]
