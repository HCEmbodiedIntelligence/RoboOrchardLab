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
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("rosbags")

from projects.holobrain.scripts.convert_hc_tj_mcap import (
    ARM_JOINT_NAMES,
    CALIBRATION_TOPIC,
    COLOR_TOPIC,
    DEPTH_TOPIC,
    EXPECTED_SOURCE_JOINT_NAMES,
    EXTRINSICS_TOPIC,
    JOINT_COMMAND_TOPIC,
    JOINT_STATE_TOPIC,
    MCAP_MAGIC,
    _deserialize_message,
    _readable_mcap_path,
    convert_hc_tj_mcap,
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


def _write_synthetic_mcap(path: Path) -> Path:
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
        connections = {
            topic: writer.add_connection(topic, msgtype, typestore=typestore)
            for topic, msgtype in {
                COLOR_TOPIC: "sensor_msgs/msg/CompressedImage",
                DEPTH_TOPIC: "sensor_msgs/msg/CompressedImage",
                JOINT_STATE_TOPIC: "sensor_msgs/msg/JointState",
                JOINT_COMMAND_TOPIC: "sensor_msgs/msg/JointState",
                CALIBRATION_TOPIC: "std_msgs/msg/String",
                EXTRINSICS_TOPIC: "std_msgs/msg/String",
            }.items()
        }

        def write(topic: str, message, timestamp_ns: int, zero_tail=False):
            raw = bytes(typestore.serialize_cdr(message, message.__msgtype__))
            writer.write(
                connections[topic],
                timestamp_ns,
                raw + (bytes(8) if zero_tail else b""),
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


def test_synthetic_mcap_converts_to_ro_dataset(tmp_path: Path) -> None:
    mcap_path = _write_synthetic_mcap(tmp_path / "bag")
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
    assert report["joint_command_zero_tail_messages"] == 2
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
    assert frame["joints"].names == list(ARM_JOINT_NAMES)
    assert tuple(frame["joints"].position.shape) == (1, 14)
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


def test_mcap_directory_converts_to_multi_episode_dataset(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "sources"
    nested = source_root / "nested"
    nested.mkdir(parents=True)
    first = _write_synthetic_mcap(source_root / "episode_b")
    second = _write_synthetic_mcap(nested / "episode_a")
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
