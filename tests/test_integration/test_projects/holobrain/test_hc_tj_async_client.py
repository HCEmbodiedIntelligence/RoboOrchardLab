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

"""Exercise ROS boundary wiring without connecting to robot hardware."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from projects.holobrain.scripts import hc_tj_async_client as client
from projects.holobrain.scripts.hc_tj_async_core import CONTROL_JOINT_NAMES

URDF = Path(__file__).resolve().parents[4] / (
    "hc_assets/hc_tj_description/urdf/hc_tj_robot.urdf"
)


@pytest.fixture
def hand_config(tmp_path):
    data = {"grippers": {}}
    for side, marker in (("right", "R_ban"), ("left", "L_ban")):
        data["grippers"][side] = {
            "finger_joint_names": [f"{side}_finger_{i}" for i in range(10)],
            "finger_open": [0.1] * 10,
            "finger_closed": [0.9] * 10,
            "binary_state_joint": marker,
        }
    path = tmp_path / "arm_teleop.yaml"
    path.write_text(json.dumps(data))
    return path


def config(**kwargs):
    return client.HcTjAsyncClientConfig(
        server_url="http://localhost/hc_tj_async/v1",
        instruction="pick",
        urdf_path=URDF,
        depth_units_per_meter=1000,
        **kwargs,
    )


def motion_config(hand_config, **kwargs):
    return config(
        robot_interface=client.RobotInterface.MOTION,
        hand_config_path=hand_config,
        calibration_topic="/test/calibration",
        extrinsics_topic="/test/extrinsics",
        **kwargs,
    )


@pytest.fixture
def ros(monkeypatch):
    class Message:
        def __init__(self):
            self.header = SimpleNamespace(
                stamp=SimpleNamespace(sec=1, nanosec=0)
            )
            self.name = []
            self.position = []

    class Publisher:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    class Node:
        def __init__(self, name):
            self.name = name
            self.subscriptions = {}
            self.publishers = {}
            self.timers = []
            self.competitors = {}
            self.now_ns = 1_000_000_000

        def create_subscription(self, kind, topic, callback, qos):
            self.subscriptions[topic] = (kind, callback, qos)

        def create_publisher(self, kind, topic, qos):
            pub = Publisher()
            self.publishers[topic] = pub
            return pub

        def create_timer(self, period, callback):
            self.timers.append((period, callback))

        def get_clock(self):
            return SimpleNamespace(
                now=lambda: SimpleNamespace(
                    nanoseconds=self.now_ns,
                    to_msg=lambda: SimpleNamespace(
                        sec=self.now_ns // 1_000_000_000,
                        nanosec=self.now_ns % 1_000_000_000,
                    ),
                )
            )

        def get_logger(self):
            return SimpleNamespace(
                warning=lambda text: None,
                error=lambda text: None,
                info=lambda text: None,
            )

        def get_publishers_info_by_topic(self, topic):
            return self.competitors.get(topic, [])

        def get_name(self):
            return self.name

        def get_namespace(self):
            return "/"

        def destroy_node(self):
            return True

    sensor_qos = object()
    modules = {
        "rclpy.node": SimpleNamespace(Node=Node),
        "rclpy.qos": SimpleNamespace(
            QoSProfile=lambda **kwargs: SimpleNamespace(**kwargs),
            DurabilityPolicy=SimpleNamespace(TRANSIENT_LOCAL="latched"),
            ReliabilityPolicy=SimpleNamespace(RELIABLE="reliable"),
            qos_profile_sensor_data=sensor_qos,
        ),
        "sensor_msgs.msg": SimpleNamespace(
            JointState=Message, CompressedImage=Message
        ),
        "std_msgs.msg": SimpleNamespace(String=Message),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    nodes = []

    def create(cfg):
        node = client._ros_node_class()(cfg)
        nodes.append(node)
        return node

    yield SimpleNamespace(
        create=create, Message=Message, sensor_qos=sensor_qos
    )
    for node in nodes:
        node.destroy_node()


def feedback(ros, names, positions):
    message = ros.Message()
    message.name = list(names)
    message.position = list(positions)
    return message


class TestInterfaceConfig:
    def test_legacy_defaults_and_motion_overrides(self, hand_config):
        old = config()
        assert old.control_hz == 100
        assert old.color_topic == "/io_teleop/camera_head/color"
        assert old.calibration_topic == "/io_teleop/camera_head/calibration"
        new = motion_config(hand_config, color_topic="/custom/color")
        assert new.control_hz == 50
        assert new.color_topic == "/custom/color"
        assert new.joint_state_topic == "/observation/joint_states"

    def test_motion_requires_explicit_calibration(self):
        with pytest.raises(ValueError, match="calibration"):
            config(robot_interface=client.RobotInterface.MOTION)

    @pytest.mark.parametrize("hz", [10, 0, 12.5, float("nan")])
    def test_rejects_rates_incompatible_with_servo(self, hand_config, hz):
        with pytest.raises(ValueError, match="control_hz"):
            motion_config(hand_config, control_hz=hz)

    def test_cli_selects_motion_defaults(self, monkeypatch, hand_config):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "client",
                "--server-url",
                "http://localhost/hc_tj_async/v1",
                "--instruction",
                "pick",
                "--depth-units-per-meter",
                "1000",
                "--robot-interface",
                "motion",
                "--hand-config-path",
                str(hand_config),
                "--calibration-topic",
                "/cal",
                "--extrinsics-topic",
                "/ext",
            ],
        )
        cfg = client._parse_args()
        assert cfg.control_hz == 50
        assert cfg.right_hand_topic == "/action/right_hand/joint_states"


class TestMotionBoundary:
    def test_file_calibration_populates_snapshot_without_subscriptions(
        self, ros, hand_config, tmp_path
    ):
        calibration = tmp_path / "calibration.json"
        calibration.write_text(
            json.dumps(
                {
                    "schema": "hc_camera_calibration/v1",
                    "camera": "eye",
                    "resolution": {
                        "color": {"width": 6, "height": 4},
                        "depth": {"width": 6, "height": 4},
                    },
                    "rgb": {"fx": 4, "fy": 5, "cx": 3, "cy": 2},
                }
            )
        )
        extrinsics = tmp_path / "extrinsics.json"
        extrinsics.write_text(
            json.dumps(
                {
                    "schema": "hc_robot_camera_extrinsics/v1",
                    "common_frame": "zhi_Link",
                    "transforms": [
                        {
                            "parent_frame": "zhi_Link",
                            "child_frame": "camera_head",
                            "translation_m": [1, 2, 3],
                            "rotation_xyzw": [0, 0, 0, 1],
                        }
                    ],
                }
            )
        )
        node = ros.create(
            config(
                robot_interface=client.RobotInterface.MOTION,
                hand_config_path=hand_config,
                calibration_path=calibration,
                extrinsics_path=extrinsics,
            )
        )
        assert len(node.subscriptions) == 3
        node.observations.add_color(1_000_000_000, b"jpeg")
        node.observations.add_depth(1_000_000_000, b"png")
        node._on_joints(
            feedback(
                ros,
                CONTROL_JOINT_NAMES[:14],
                client.DEFAULT_JOINT_POSITIONS[:14],
            )
        )
        snapshot = node.observations.synchronized()
        assert snapshot[3][0, 0] == 4
        np.testing.assert_allclose(snapshot[4][:3, 3], [-1, -2, -3])
        with pytest.raises(ValueError, match="choose either"):
            motion_config(hand_config, calibration_path=calibration)

    @pytest.mark.parametrize("motion_names", [False, True])
    def test_maps_shuffled_named_arm_feedback(self, ros, motion_names):
        names = (
            [
                f"{side}_joint_{i}"
                for side in ("right", "left")
                for i in range(1, 8)
            ]
            if motion_names
            else list(CONTROL_JOINT_NAMES[:14])
        )
        msg = feedback(
            ros, names[::-1] + ["extra"], list(range(14))[::-1] + [99]
        )
        assert client._motion_joint_positions(msg)[:14].tolist() == list(
            range(14)
        )

    @pytest.mark.parametrize(
        "problem", ["missing", "duplicate", "alias", "nan", "length"]
    )
    def test_rejects_invalid_feedback(self, ros, problem):
        msg = feedback(ros, CONTROL_JOINT_NAMES[:14], range(14))
        if problem == "missing":
            msg.name.pop()
            msg.position.pop()
        elif problem == "duplicate":
            msg.name[-1] = msg.name[0]
        elif problem == "alias":
            msg.name.append("right_joint_1")
            msg.position.append(0)
        elif problem == "nan":
            msg.position[0] = np.nan
        else:
            msg.position.pop()
        with pytest.raises(ValueError):
            client._motion_joint_positions(msg)

    def test_hand_endpoints_and_invalid_profile(self, hand_config):
        hands = client._load_motion_hands(hand_config)
        assert hands["right"][0][-1] == "R_ban"
        assert hands["left"][1][:, -1].tolist() == [0, 1]
        data = json.loads(hand_config.read_text())
        data["grippers"]["left"]["binary_state_joint"] = "wrong"
        hand_config.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="X1 left"):
            client._load_motion_hands(hand_config)

    @pytest.mark.parametrize("motion", [False, True])
    def test_dry_run_creates_no_publishers(self, ros, hand_config, motion):
        cfg = motion_config(hand_config) if motion else config()
        node = ros.create(cfg)
        assert not node.publishers
        qos = node.subscriptions[cfg.color_topic][2]
        if motion:
            assert qos is ros.sensor_qos
        else:
            assert qos.reliability == "reliable"

    @pytest.mark.parametrize("motion", [False, True])
    def test_reset_and_action_publication(self, ros, hand_config, motion):
        cfg = (
            motion_config(hand_config, enable_control=True)
            if motion
            else config(enable_control=True)
        )
        node = ros.create(cfg)
        positions = np.asarray(client.DEFAULT_JOINT_POSITIONS)
        msg = feedback(ros, CONTROL_JOINT_NAMES, positions)
        node._on_joints(msg)
        node._control_tick()
        assert node._reset_complete
        assert not node.publishers[cfg.right_hand_topic].messages
        action = np.r_[positions[:7], 0.49, positions[7:14], 0.5]
        node._control_actions = action[None, :]
        node._received_action_chunk = True
        node._control_tick()
        assert node.fault is None
        right = node.publishers[cfg.right_hand_topic].messages[-1]
        left = node.publishers[cfg.left_hand_topic].messages[-1]
        if motion:
            assert cfg.joint_command_topic not in node.publishers
            for side, offset, topic in (
                ("right", 0, cfg.right_arm_topic),
                ("left", 7, cfg.left_arm_topic),
            ):
                command = node.publishers[topic].messages[-1]
                assert command.name == [
                    f"{side}_joint_{i}" for i in range(1, 8)
                ]
                np.testing.assert_allclose(
                    command.position, positions[offset : offset + 7]
                )
                assert command.header.stamp.sec == 1
            assert right.position == [0.1] * 10 + [0.0]
            assert left.position == [0.9] * 10 + [1.0]
            assert node._right_hand_observation == 0
            assert node._left_hand_observation == 1
        else:
            assert (
                len(
                    node.publishers[cfg.joint_command_topic]
                    .messages[-1]
                    .position
                )
                == 17
            )
            assert len(right.position) == len(left.position) == 10
            assert node._right_hand_observation == 0.49

    @pytest.mark.parametrize(
        "topic_attr",
        [
            "right_arm_topic",
            "left_arm_topic",
            "right_hand_topic",
            "left_hand_topic",
        ],
    )
    def test_motion_competing_publisher_blocks_output(
        self, ros, hand_config, topic_attr
    ):
        cfg = motion_config(hand_config, enable_control=True)
        node = ros.create(cfg)
        node.competitors[getattr(cfg, topic_attr)] = [
            SimpleNamespace(node_name="other", node_namespace="/")
        ]
        node._control_tick()
        assert "competing publishers" in node.fault
        assert all(not pub.messages for pub in node.publishers.values())

    def test_stale_feedback_blocks_motion(self, ros, hand_config):
        node = ros.create(motion_config(hand_config, enable_control=True))
        node._on_joints(
            feedback(ros, CONTROL_JOINT_NAMES, client.DEFAULT_JOINT_POSITIONS)
        )
        node.now_ns += 500_000_000
        node._control_tick()
        assert "stale" in node.fault
        assert all(not pub.messages for pub in node.publishers.values())
