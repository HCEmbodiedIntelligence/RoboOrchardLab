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

"""Exercise actual direct-client contracts with fake ROS and no hardware."""

import json
import sys
import time
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

from projects.holobrain.scripts import hc_tj_async_client as client
from projects.holobrain.scripts.hc_tj_async_core import (
    CONTROL_JOINT_NAMES,
    TimedValue,
)
from projects.holobrain.scripts.hc_tj_motion import MotionInterface


@pytest.fixture
def config(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        "<robot>"
        + "".join(
            f'<joint name="{name}"><limit lower="-4" upper="4"/></joint>'
            for name in CONTROL_JOINT_NAMES[:14]
        )
        + "</robot>"
    )

    def build(**kwargs):
        dagger = kwargs.get("enable_dagger", False)
        return client.HcTjAsyncClientConfig(
            server_url="http://localhost/holobrain/hc_tj_async/"
            + ("v2" if dagger else "v1"),
            instruction="pick",
            urdf_path=urdf,
            depth_units_per_meter=1000,
            **kwargs,
        )

    return build


@pytest.fixture
def ros(monkeypatch):
    class Message:
        def __init__(self):
            self.header = SimpleNamespace(
                stamp=SimpleNamespace(sec=1, nanosec=0)
            )
            self.name, self.position, self.data = [], [], ""

    class Publisher:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    class Node:
        def __init__(self, name):
            self.name = name
            self.subscriptions, self.publishers, self.competitors = {}, {}, {}
            self.timers = []
            self.now_ns = 1_000_000_000

        def create_subscription(self, kind, topic, callback, qos):
            self.subscriptions[topic] = (kind, callback, qos)

        def create_publisher(self, kind, topic, qos):
            pub = Publisher()
            pub.qos = qos
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
                warning=lambda _: None,
                error=lambda _: None,
                info=lambda _: None,
            )

        def get_publishers_info_by_topic(self, topic):
            return self.competitors.get(topic, [])

        def get_name(self):
            return self.name

        def get_namespace(self):
            return "/"

        def destroy_node(self):
            return True

    class Relay:
        def __init__(self, *args):
            self.server_instance = "server"
            self.connected = True

        def start(self):
            pass

        def close(self):
            pass

        def ready(self, window):
            return self.connected

    for name, module in {
        "rclpy.node": SimpleNamespace(Node=Node),
        "rclpy.qos": SimpleNamespace(
            QoSProfile=lambda **kwargs: SimpleNamespace(**kwargs),
            DurabilityPolicy=SimpleNamespace(TRANSIENT_LOCAL="latched"),
            HistoryPolicy=SimpleNamespace(KEEP_LAST="latest"),
            ReliabilityPolicy=SimpleNamespace(
                RELIABLE="reliable", BEST_EFFORT="best_effort"
            ),
        ),
        "sensor_msgs.msg": SimpleNamespace(
            JointState=Message, CompressedImage=Message
        ),
        "std_msgs.msg": SimpleNamespace(String=Message),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(client, "ControlRelay", Relay)
    nodes = []

    def create(cfg):
        node = client._ros_node_class()(cfg)
        nodes.append(node)
        return node

    yield SimpleNamespace(create=create, Message=Message)
    for node in nodes:
        node.destroy_node()


def motion_info():
    groups = {}
    for group, prefix, count in (
        ("right_arm", "right_joint_", 7),
        ("left_arm", "left_joint_", 7),
        ("waist", "waist_joint_", 3),
    ):
        groups[group] = dict(
            joint_names=[f"{prefix}{i}" for i in range(1, count + 1)],
            units=["rad"] * count,
            limits=[dict(lower=-4, upper=4)] * count,
            endpoints=(
                {
                    kind: f"/control/motion/{group}/{kind}"
                    for kind in (
                        "servo_j",
                        "servo_p",
                        "move_j",
                        "move_l",
                        "move_p",
                    )
                }
                if group != "waist"
                else {}
            ),
        )
    return dict(
        schema="hc-robot-api/v1",
        profile_id="x1",
        revision="profile-a",
        groups=groups,
        servo=dict(servo_lease_ms=100),
    )


def send_state(node, ros, state="AUTO", epoch=1):
    msg = ros.Message()
    msg.data = json.dumps(
        dict(
            controller_instance_id="hc",
            session_id="session",
            epoch=epoch,
            sequence=epoch,
            state=state,
            phase="grant" if state == "AUTO" else "revoke",
            source="motion",
            published_stamp_ns=node.now_ns,
        )
    )
    node._on_ownership(msg)


def feedback(node, ros, names=CONTROL_JOINT_NAMES):
    msg = ros.Message()
    msg.name, msg.position = list(names), list(client.DEFAULT_JOINT_POSITIONS)
    msg.header.stamp = node.get_clock().now().to_msg()
    node._on_joints(msg)


class TestInterfaceConfig:
    def test_joint_history_does_not_queue_commands(self, ros, config):
        cfg = config(enable_control=True)
        node = ros.create(cfg)
        joint_qos = node.subscriptions[cfg.joint_state_topic][2]
        assert joint_qos.reliability == "reliable"
        assert joint_qos.depth == 100
        assert node.publishers[cfg.joint_command_topic].qos.depth == 1
        assert node.subscriptions[cfg.color_topic][2].depth == 1

    def test_observations_use_standard_robot_api_topics(self, config):
        cfg = config()
        assert cfg.color_topic == "/observation/cameras/head/color/compressed"
        assert cfg.depth_topic == "/observation/cameras/head/depth/compressed"
        assert cfg.calibration_topic == "/observation/cameras/head/calibration"
        assert cfg.extrinsics_topic == "/observation/cameras/head/extrinsics"
        assert cfg.joint_state_topic == "/observation/joint_states"
        assert cfg.joint_command_topic == "/hc_teleop/joint_cmd"
        assert cfg.right_hand_topic == "/hc_teleop/joint_cmd_finger_right"
        assert cfg.control_hz == 100

    def test_motion_requires_explicit_identity_mapping(self, config):
        with pytest.raises(ValueError, match="identity mapping"):
            config(robot_interface="hc_motion")
        cfg = config(robot_interface="hc_motion", motion_identity_mapping=True)
        assert cfg.joint_state_topic == "/observation/joint_states"
        assert cfg.color_topic == config().color_topic

    def test_dagger_requires_token_and_v2(self, config):
        with pytest.raises(ValueError, match="control_token"):
            config(enable_dagger=True)


class TestMotionBoundary:
    def test_named_mapping_preserves_right_left_waist(self):
        motion = MotionInterface.from_info(
            motion_info(), identity_mapping=True
        )
        names = list(motion.joint_names)
        assert motion.feedback(
            names[::-1], list(range(17))[::-1]
        ).tolist() == list(range(17))
        with pytest.raises(ValueError, match="waist"):
            motion.feedback(names[:14], [0] * 14)
        commands = motion.commands(np.arange(14) / 10)
        assert commands[0][1][0] == "right_joint_1"
        assert commands[1][2][0] == 0.7
        assert "move_l" in motion.capabilities["left_arm"]

    @pytest.mark.parametrize("problem", ["duplicate", "nan", "length"])
    def test_invalid_named_feedback_is_rejected(self, problem):
        motion = MotionInterface.from_info(
            motion_info(), identity_mapping=True
        )
        names, positions = list(motion.joint_names), [0.0] * 17
        if problem == "duplicate":
            names[-1] = names[0]
        elif problem == "nan":
            positions[0] = np.nan
        else:
            positions.pop()
        with pytest.raises(ValueError):
            motion.feedback(names, positions)

    def test_rejects_unknown_profile_and_bad_units(self):
        info = motion_info()
        info["profile_id"] = "other"
        with pytest.raises(ValueError):
            MotionInterface.from_info(info, identity_mapping=True)
        info = motion_info()
        info["groups"]["left_arm"]["units"][0] = "deg"
        with pytest.raises(ValueError):
            MotionInterface.from_info(info, identity_mapping=True)

    def test_motion_creates_only_discovered_servoj_publishers(
        self, config, ros
    ):
        node = ros.create(
            config(
                robot_interface="hc_motion",
                motion_identity_mapping=True,
                enable_control=True,
            )
        )
        assert not node.publishers
        msg = ros.Message()
        msg.data = json.dumps(motion_info())
        node._on_robot_info(msg)
        assert set(node.publishers) == set(node._motion.topics)
        feedback(node, ros, node._motion.joint_names)
        q = np.asarray(client.DEFAULT_JOINT_POSITIONS)
        node.limiter.reset(q[:14])
        node._control_actions = np.r_[q[:7], 0, q[7:14], 0][None, :]
        node._control_tick()
        assert node.fault is None
        assert all(len(pub.messages) == 1 for pub in node.publishers.values())
        assert all(
            len(pub.messages[0].name) == 7 for pub in node.publishers.values()
        )

    def test_dry_run_has_no_publishers(self, config, ros):
        for motion in (False, True):
            node = ros.create(
                config(
                    robot_interface="hc_motion" if motion else "legacy",
                    motion_identity_mapping=motion,
                )
            )
            if motion:
                msg = ros.Message()
                msg.data = json.dumps(motion_info())
                node._on_robot_info(msg)
            assert not node.publishers

    def test_competing_motion_publisher_blocks_arm_output(self, config, ros):
        node = ros.create(
            config(
                robot_interface="hc_motion",
                motion_identity_mapping=True,
                enable_control=True,
            )
        )
        msg = ros.Message()
        msg.data = json.dumps(motion_info())
        node._on_robot_info(msg)
        node._control_actions = np.zeros((1, 16))
        node.competitors[node._motion.topics[0]] = [
            SimpleNamespace(node_name="other", node_namespace="/")
        ]
        node._control_tick()
        assert "competing publishers" in node.fault
        assert all(not pub.messages for pub in node.publishers.values())


def test_legacy_reset_and_hand_output_are_preserved(config, ros):
    cfg = config(enable_control=True)
    node = ros.create(cfg)
    feedback(node, ros)
    node._control_tick()
    assert node._reset_complete
    assert len(node.publishers[cfg.joint_command_topic].messages) == 1
    assert not node.publishers[cfg.right_hand_topic].messages
    q = np.asarray(client.DEFAULT_JOINT_POSITIONS)
    node._control_actions = np.r_[q[:7], 0.49, q[7:14], 0.5][None, :]
    node._received_action_chunk = True
    node._control_tick()
    assert node.fault is None
    assert (
        len(node.publishers[cfg.right_hand_topic].messages[0].position) == 10
    )
    assert node._right_hand_observation == 0.49


@pytest.mark.parametrize("reset_complete", [False, True])
def test_queued_feedback_cannot_authorize_legacy_output(
    config, ros, reset_complete
):
    node = ros.create(config(enable_control=True, state_timeout_s=0.2))
    # The callback just ran, but the source sample sat in DDS for 500 ms.
    feedback(node, ros)
    node.now_ns += 500_000_000
    node._reset_complete = reset_complete
    if reset_complete:
        node._control_actions = np.zeros((1, 16))
        node._received_action_chunk = True
    node._control_tick()
    assert "joint feedback source stamp is stale" in node.fault
    assert all(not pub.messages for pub in node.publishers.values())


def test_queued_feedback_cannot_reseed_limiter(config, ros):
    node = ros.create(config(enable_control=True, state_timeout_s=0.2))
    feedback(node, ros)
    node.now_ns += 500_000_000
    node._reset_complete = True
    node._future = Future()
    node._future.set_result((np.zeros((8, 16)), 0.01))
    node._inference_tick()
    assert "joint feedback source stamp is stale" in node.fault
    assert node.limiter.previous is None
    assert not node._received_action_chunk
    assert node._control_actions is None


def test_trimmed_jump_reports_joint_and_raw_prediction(config, ros):
    node = ros.create(config(enable_control=True, first_point_jump_rad=0.3))
    feedback(node, ros)
    q = np.asarray(client.DEFAULT_JOINT_POSITIONS)
    actions = np.tile(np.r_[q[:7], 0, q[7:14], 0], (12, 1))
    actions[7:, 12] += 0.324  # Fifth left arm joint, after right gripper.
    node._reset_complete = True
    node._future = Future()
    node._future.set_result((actions, 0.235))
    node._inference_tick()
    assert "first action jumps 0.324000" in node.fault
    assert "joint=Joint5_L" in node.fault
    assert f"target_rad={q[11] + 0.324:.6f}" in node.fault
    assert f"feedback_rad={q[11]:.6f}" in node.fault
    assert "response_index=7 raw_first_max_delta_rad=0.000000" in node.fault
    assert node._control_actions is None
    assert all(not pub.messages for pub in node.publishers.values())


class TestDaggerBoundary:
    def test_pause_clears_both_blocks_and_no_fault_or_reset(self, config, ros):
        node = ros.create(
            config(
                enable_dagger=True, control_token="x" * 16, enable_control=True
            )
        )
        send_state(node, ros)
        node._native_actions = node._control_actions = np.ones((4, 16))
        node._received_action_chunk = True
        node._control_index = 2
        node.now_ns += 1
        send_state(node, ros, "HOLD", 2)
        node._control_tick()
        assert node._native_actions is node._control_actions is None
        assert not node._received_action_chunk and node._control_index == 0
        assert node._reset_complete and node.fault is None
        assert all(not pub.messages for pub in node.publishers.values())

    def test_old_future_cannot_refill_after_resume(self, config, ros):
        node = ros.create(config(enable_dagger=True, control_token="x" * 16))
        send_state(node, ros)
        old = node._control_window
        node._future = Future()
        node._future_window = old
        node.now_ns += 1
        send_state(node, ros, "HOLD", 2)
        node.now_ns += 1
        send_state(node, ros, "AUTO", 3)
        node._future.set_result((np.zeros((64, 16)), 0.01))
        node._inference_tick()
        assert node._native_actions is node._control_actions is None
        assert node.fault is None

    def test_suppressed_observations_continue_while_old_call_runs(
        self, config, ros, monkeypatch
    ):
        node = ros.create(config(enable_dagger=True, control_token="x" * 16))
        send_state(node, ros, "HOLD", 1)
        node._future = Future()
        snapshot = tuple(
            TimedValue(1, b"image", time.monotonic()) for _ in range(3)
        ) + (np.eye(4), np.eye(4))
        monkeypatch.setattr(
            node.observations, "synchronized", lambda: snapshot
        )
        monkeypatch.setattr(
            client, "_build_request", lambda *args: ({}, {}, 1)
        )
        calls = []
        monkeypatch.setattr(
            node,
            "_request_inference",
            lambda files, form: calls.append(form) or (None, 0.01),
        )
        node._inference_tick()
        node._discard_future.result(timeout=1)
        assert calls[0]["control_window"] == node._control_window
        assert not node._future.done()
        node._future.set_result((None, 0.01))

    def test_old_joint_sample_cannot_form_resume_first_request(
        self, config, ros, monkeypatch
    ):
        node = ros.create(config(enable_dagger=True, control_token="x" * 16))
        send_state(node, ros)
        recent = time.monotonic()
        snapshot = (
            TimedValue(node.now_ns + 1, b"rgb", recent),
            TimedValue(node.now_ns + 1, b"depth", recent),
            TimedValue(node.now_ns - 1, np.zeros(17), recent),
            np.eye(4),
            np.eye(4),
        )
        monkeypatch.setattr(
            node.observations, "synchronized", lambda: snapshot
        )
        monkeypatch.setattr(
            client,
            "_build_request",
            lambda *args: pytest.fail("old sample submitted"),
        )
        node._inference_tick()
        assert node._future is None

    def test_closed_gate_prevents_initial_reset(self, config, ros):
        node = ros.create(
            config(
                enable_dagger=True, control_token="x" * 16, enable_control=True
            )
        )
        node._reset_complete = False
        feedback(node, ros)
        node._control_tick()
        assert all(not pub.messages for pub in node.publishers.values())

    def test_inference_disconnect_rotates_window_without_reset(
        self, config, ros
    ):
        node = ros.create(config(enable_dagger=True, control_token="x" * 16))
        send_state(node, ros)
        old = node._control_window
        node._native_actions = node._control_actions = np.zeros((64, 16))
        node._received_action_chunk = True
        node._future_window = old
        node._future = Future()
        node._future.set_exception(client.requests.ConnectionError("offline"))
        node._inference_tick()
        assert node.fault is None
        assert node._control_window != old
        assert node._native_actions is node._control_actions is None
        assert node._reset_complete and not node._received_action_chunk
        node.now_ns += 1
        send_state(node, ros, "AUTO", 2)
        assert node._gate.snapshot()["allowed"]

    def test_resume_waits_for_new_camera_without_fault(
        self, config, ros, monkeypatch
    ):
        node = ros.create(config(enable_dagger=True, control_token="x" * 16))
        send_state(node, ros)
        snapshot = tuple(
            TimedValue(node.now_ns - 1, b"image", time.monotonic() - 5)
            for _ in range(3)
        ) + (np.eye(4), np.eye(4))
        monkeypatch.setattr(
            node.observations, "synchronized", lambda: snapshot
        )
        node._inference_tick()
        assert node.fault is None and node._future is None

    def test_first_resume_request_has_no_old_rtc_or_delay(
        self, config, ros, monkeypatch
    ):
        node = ros.create(config(enable_dagger=True, control_token="x" * 16))
        send_state(node, ros)
        node._native_actions = node._control_actions = np.zeros((64, 16))
        node._latency_ewma = 1
        node.now_ns += 1
        send_state(node, ros, "HOLD", 2)
        node.now_ns += 1
        send_state(node, ros, "AUTO", 3)
        node._gate.resume_started -= 0.001
        snapshot = tuple(
            TimedValue(node.now_ns + 1, b"image", time.monotonic())
            for _ in range(3)
        ) + (np.eye(4), np.eye(4))
        monkeypatch.setattr(
            node.observations, "synchronized", lambda: snapshot
        )
        built = []
        monkeypatch.setattr(
            client,
            "_build_request",
            lambda *args: built.append(args) or ({}, {}, 1),
        )
        monkeypatch.setattr(
            node, "_request_inference", lambda *args: (None, 0.01)
        )
        node._inference_tick()
        node._future.result(timeout=1)
        assert built[0][-2:] == (None, 0)
        assert built[0][2:4] == (0.0, 0.0)

    def test_permission_is_rechecked_at_final_publication(
        self, config, ros, monkeypatch
    ):
        node = ros.create(
            config(
                enable_dagger=True, control_token="x" * 16, enable_control=True
            )
        )
        send_state(node, ros)
        feedback(node, ros)
        q = np.asarray(client.DEFAULT_JOINT_POSITIONS)
        node.limiter.reset(q[:14])
        node._control_actions = np.r_[q[:7], 0, q[7:14], 0][None, :]
        node._received_action_chunk = True

        def revoke_while_preparing(*args):
            node._gate.invalidate("pause")
            return q[:14]

        monkeypatch.setattr(
            client.ArmCommandLimiter, "step", revoke_while_preparing
        )
        node._control_tick()
        assert node.fault is None
        assert node._native_actions is node._control_actions is None
        assert all(not pub.messages for pub in node.publishers.values())

    def test_legacy_hands_publish_and_feed_inference_without_feedback(
        self, config, ros, monkeypatch
    ):
        cfg = config(
            enable_dagger=True, control_token="x" * 16, enable_control=True
        )
        node = ros.create(cfg)
        assert cfg.hand_state_topic not in node.subscriptions
        send_state(node, ros)
        node.now_ns += 1
        feedback(node, ros)
        q = np.asarray(client.DEFAULT_JOINT_POSITIONS)
        node.limiter.reset(q[:14])
        node._control_actions = np.r_[q[:7], 0.4, q[7:14], 0.6][None, :]
        node._received_action_chunk = True
        window = node._control_window
        node._control_tick()
        assert all(len(pub.messages) == 1 for pub in node.publishers.values())
        assert node._gate.allows(window)
        assert node.fault is None
        node._gate.resume_started -= 0.001
        snapshot = tuple(
            TimedValue(node.now_ns, b"image", time.monotonic())
            for _ in range(3)
        ) + (np.eye(4), np.eye(4))
        monkeypatch.setattr(
            node.observations, "synchronized", lambda: snapshot
        )
        built = []
        monkeypatch.setattr(
            client,
            "_build_request",
            lambda *args: built.append(args) or ({}, {}, 1),
        )
        monkeypatch.setattr(
            node, "_request_inference", lambda *args: (None, 0.01)
        )
        node._inference_tick()
        node._future.result(timeout=1)
        assert built[0][2:4] == (0.4, 0.6)
        assert node._gate.allows(window)
        assert node.fault is None

    def test_motion_hand_loss_still_revokes_window(self, config, ros):
        node = ros.create(
            config(
                enable_dagger=True,
                control_token="x" * 16,
                robot_interface="hc_motion",
                motion_identity_mapping=True,
            )
        )
        assert node.config.hand_state_topic in node.subscriptions
        send_state(node, ros)
        old = node._control_window
        node._native_actions = node._control_actions = np.zeros((64, 16))
        node._received_action_chunk = True
        assert not node._hands_ready()
        assert node._control_window != old
        assert node._native_actions is node._control_actions is None
