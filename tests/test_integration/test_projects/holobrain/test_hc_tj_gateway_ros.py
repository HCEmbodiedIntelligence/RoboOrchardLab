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

"""Real ROS boundary checks in an isolated domain; no model or robot needed."""

import json
import os
import time
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
from rclpy.node import Node  # noqa: E402
from rclpy.qos import (  # noqa: E402
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import JointState  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from projects.holobrain.scripts.hc_tj_async_core import (  # noqa: E402
    CONTROL_JOINT_NAMES,
    hand_command,
    parse_camera_calibration,
    parse_robot_camera_extrinsics,
)
from projects.holobrain.scripts.hc_tj_http_protocol import (  # noqa: E402
    decode_observation,
    encode_observation,
)
from projects.holobrain.scripts.hc_tj_http_robot_gateway import (  # noqa: E402
    GatewayConfig,
    _ActionStreamPausedError,
    _ros_node_class,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("ROS_DOMAIN_ID") != "231"
    or os.environ.get("ROS_LOCALHOST_ONLY") != "1"
    or not os.environ.get("HC_TJ_TEST_URDF"),
    reason="requires isolated ROS domain 231, localhost and test URDF",
)


@pytest.fixture
def node():
    rclpy.init()
    gateway = _ros_node_class()(
        GatewayConfig(
            token="isolated-ros-test-token",
            enable_dagger=True,
            enable_control=True,
            enable_hands=True,
            urdf_path=Path(os.environ["HC_TJ_TEST_URDF"]),
        )
    )
    try:
        yield gateway
    finally:
        gateway.destroy_node()
        rclpy.shutdown()


def ownership(node, state, sequence):
    node._on_ownership(
        String(
            data=json.dumps(
                {
                    "controller_instance_id": "isolated-test",
                    "session_id": "session",
                    "epoch": sequence,
                    "sequence": sequence,
                    "published_stamp_ns": node.get_clock().now().nanoseconds,
                    "state": state,
                    "phase": "grant" if state == "AUTO" else "hold",
                    "source": "motion" if state == "AUTO" else "none",
                }
            )
        )
    )
    return node.control_snapshot()


def measured_hands(node, stamp=None):
    message = JointState()
    ns = stamp or node.get_clock().now().nanoseconds
    message.header.stamp.sec, message.header.stamp.nanosec = divmod(
        ns, 1_000_000_000
    )
    for side in ("right", "left"):
        names, values = hand_command(side, 0)
        message.name.extend(names)
        message.position.extend(values.tolist())
    node._on_hands(message)


def measured_joints(node):
    message = JointState()
    message.header.stamp = node.get_clock().now().to_msg()
    message.name = list(CONTROL_JOINT_NAMES)
    message.position = [0.0] * 17
    node._on_joints(message)


def test_pause_clears_schedule_and_rejects_old_window(node):
    state = ownership(node, "AUTO", 1)
    measured_hands(node)
    measured_joints(node)
    actions = np.zeros((3, 16))
    node.accept_action_chunk(1, actions, control_window=state["window_id"])
    assert node._action_stream.actions is not None
    ownership(node, "HOLD", 2)
    assert node._action_stream.actions is None
    with pytest.raises(_ActionStreamPausedError):
        node.accept_action_chunk(2, actions, control_window=state["window_id"])
    assert node.fault is None


def test_resume_rejects_pre_resume_source_timestamps(node):
    ownership(node, "HOLD", 1)
    before = node.get_clock().now().nanoseconds
    state = ownership(node, "AUTO", 2)
    assert all(bound >= before for bound in state["observation_floor"])
    measured_hands(node, before)
    assert not node.control_snapshot()["hand_fresh"]
    measured_hands(node)
    assert node.control_snapshot()["hand_fresh"]


@pytest.mark.parametrize("had_feedback", [False, True])
def test_missing_hand_feedback_does_not_block_auto_publication(
    node, had_feedback
):
    state = ownership(node, "AUTO", 1)
    if had_feedback:
        measured_hands(node)
    measured_joints(node)
    node.accept_action_chunk(
        1, np.zeros((3, 16)), control_window=state["window_id"]
    )
    node._hand_received = time.monotonic() - 1
    node._joint_publisher = Mock()
    node._right_hand_publisher = Mock()
    node._left_hand_publisher = Mock()
    node._control_tick()
    assert node.control_snapshot()["allowed"]
    assert not node.control_snapshot()["hand_fresh"]
    assert node._action_stream.action_index == 1
    node._joint_publisher.publish.assert_called_once()
    node._right_hand_publisher.publish.assert_called_once()
    node._left_hand_publisher.publish.assert_called_once()
    assert node.fault is None


def test_missing_arm_feedback_still_blocks_actions(node):
    state = ownership(node, "AUTO", 1)
    with pytest.raises(ValueError, match="joint feedback"):
        node.accept_action_chunk(
            1, np.zeros((3, 16)), control_window=state["window_id"]
        )


@pytest.mark.parametrize(
    ("offered", "late_join"),
    [
        ("volatile", False),
        ("transient_local", False),
        ("transient_local", True),
    ],
)
def test_metadata_dds_completes_observation_without_control(
    offered, late_join
):
    rclpy.init()
    driver = Node("isolated_camera_metadata_source")
    gateway = None
    config = GatewayConfig(
        token="isolated-metadata-test-token",
        calibration_topic="/observation/cameras/head/calibration",
        extrinsics_topic="/observation/cameras/head/extrinsics",
        **({} if late_join else {"extrinsics_durability": "volatile"}),
    )
    latched = QoSProfile(
        depth=1, reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    extrinsics_qos = QoSProfile(
        depth=1, reliability=ReliabilityPolicy.RELIABLE,
        durability=(
            DurabilityPolicy.VOLATILE
            if offered == "volatile" else DurabilityPolicy.TRANSIENT_LOCAL
        ),
    )
    calibration_pub = driver.create_publisher(
        String, config.calibration_topic, latched
    )
    extrinsics_pub = driver.create_publisher(
        String, config.extrinsics_topic, extrinsics_qos
    )
    calibration = String(data=json.dumps({
        "schema": "hc_camera_calibration/v1", "camera": "eye",
        "resolution": {
            "color": {"width": 640, "height": 400},
            "depth": {"width": 640, "height": 400},
        },
        "rgb": {"fx": 500, "fy": 500, "cx": 320, "cy": 200},
    }))
    extrinsics = String(data=json.dumps({
        "schema": "hc_robot_camera_extrinsics/v1",
        "common_frame": "zhi_Link",
        "transforms": [{
            "parent_frame": "zhi_Link", "child_frame": "camera_head",
            "translation_m": [0, 0, 0], "rotation_xyzw": [0, 0, 0, 1],
        }],
    }))
    try:
        if late_join:
            calibration_pub.publish(calibration)
            extrinsics_pub.publish(extrinsics)
        gateway = _ros_node_class()(config)
        stamp = gateway.get_clock().now().nanoseconds
        gateway.observations.add_color(stamp, "jpeg", b"color")
        gateway.observations.add_depth(stamp, "16UC1; png", b"depth")
        measured_joints(gateway)
        assert gateway.observations.wait_for_snapshot(0, 0) is None
        deadline = time.monotonic() + 4
        packet = None
        while time.monotonic() < deadline:
            if not late_join:
                calibration_pub.publish(calibration)
                extrinsics_pub.publish(extrinsics)
            rclpy.spin_once(driver, timeout_sec=.01)
            rclpy.spin_once(gateway, timeout_sec=.01)
            packet = gateway.observations.wait_for_snapshot(0, 0)
            if packet is not None:
                break
        assert packet is not None, gateway.observations.status()
        restored = decode_observation(encode_observation(packet))
        intrinsic = parse_camera_calibration(restored.calibration_json)
        extrinsic = parse_robot_camera_extrinsics(restored.extrinsics_json)
        assert intrinsic.shape == extrinsic.shape == (4, 4)
        assert gateway._joint_publisher is None
        assert gateway._left_hand_publisher is None
        assert gateway._right_hand_publisher is None
    finally:
        if gateway is not None:
            gateway.destroy_node()
        driver.destroy_node()
        rclpy.shutdown()
