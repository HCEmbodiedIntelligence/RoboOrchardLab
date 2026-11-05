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
import threading
import time
from concurrent.futures import Future
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest

from projects.holobrain.scripts import hc_tj_async_client as client
from projects.holobrain.scripts.hc_tj_async_core import (
    CONTROL_JOINT_NAMES,
    TimedValue,
)
from projects.holobrain.scripts.hc_tj_motion import MotionInterface


class TestContinuousExecution:
    def test_post_stall_acceleration_uses_realized_command_velocity(
        self, config, ros, monkeypatch
    ):
        clock = [100.0]
        monkeypatch.setattr(client.time, "monotonic", lambda: clock[0])
        node = ros.create(config(enable_control=True))
        q = np.asarray(client.DEFAULT_JOINT_POSITIONS)
        node._reset_complete = node._received_action_chunk = True
        node.limiter.reset(q[:14])
        actions = np.tile(np.r_[q[:7], 0, q[7:14], 0], (64, 1))
        actions[:, 0] += 1
        node._native_actions, node._native_started = actions, clock[0]
        for _ in range(40):
            clock[0] += 0.01
            feedback(node, ros)
            node._control_tick()
        output = node.publishers[node.config.joint_command_topic].messages
        before = output[-1].position[0]
        clock[0] += 0.02
        feedback(node, ros)
        node._control_tick()
        stalled = output[-1].position[0]
        stalled_velocity = (stalled - before) / 0.02
        assert stalled - before == pytest.approx(0.0025)
        assert node.limiter.previous_command_velocity[0] == pytest.approx(
            stalled_velocity
        )
        clock[0] += 0.01
        feedback(node, ros)
        node._control_tick()
        next_velocity = (output[-1].position[0] - stalled) / 0.01
        assert next_velocity == pytest.approx(stalled_velocity + 0.01)

    @pytest.mark.parametrize("inference_hz", [2, 6, 8, 15])
    def test_chunks_preserve_command_with_lagging_feedback(
        self, config, ros, monkeypatch, inference_hz
    ):
        clock = [100.0]
        monkeypatch.setattr(client.time, "monotonic", lambda: clock[0])
        node = ros.create(
            config(enable_control=True, inference_hz=inference_hz)
        )
        q = np.asarray(client.DEFAULT_JOINT_POSITIONS)
        feedback(node, ros)
        node._reset_complete = True
        node.limiter.reset(q[:14])
        node._received_action_chunk = True
        node._native_started = clock[0]
        actions = np.tile(np.r_[q[:7], 0, q[7:14], 0], (64, 1))
        actions[:, 0] += 0.15
        node._native_actions = actions
        node._last_request = clock[0]
        output = node.publishers[node.config.joint_command_topic].messages
        for tick in range(101):
            clock[0] = 100 + tick * 0.01
            # Deliberately keep measured feedback behind the commands.
            feedback(node, ros)
            if tick and tick % max(1, round(100 / inference_hz)) == 0:
                node._future = Future()
                node._future_started = clock[0] - 0.025
                node._future.set_result((actions, 0.01))
                node._inference_tick()
            node._control_tick()
        positions = np.asarray([msg.position[0] for msg in output])
        velocities = np.diff(positions) / 0.01
        assert node.fault is None
        assert len(output) == 101
        assert np.min(velocities) >= -1e-8
        assert np.max(velocities) <= 0.25 + 1e-8
        assert np.max(np.abs(np.diff(velocities))) <= 0.01 + 1e-8
        assert node._diagnostic_counts["limiter_resets"] == 0
        assert positions[-1] > q[0] + 0.05

    def test_rtc_and_control_share_fractional_clock_after_stall(
        self, config, ros, monkeypatch
    ):
        clock = [100.0]
        monkeypatch.setattr(client.time, "monotonic", lambda: clock[0])
        node = ros.create(config(enable_control=True))
        feedback(node, ros)
        q = np.asarray(client.DEFAULT_JOINT_POSITIONS)
        node._reset_complete = node._received_action_chunk = True
        node.limiter.reset(q[:14])
        actions = np.tile(np.r_[q[:7], 0, q[7:14], 0], (64, 1))
        actions[:, 0] += np.arange(64) / 30 * 0.1
        node._native_actions, node._native_started = actions, clock[0]
        node._control_tick()
        clock[0] += 0.125
        remaining = node._remaining_actions(clock[0])
        captured = []
        original = client.ArmCommandLimiter.step

        def record(limiter, target, period, **kwargs):
            captured.append((target.copy(), period))
            return original(limiter, target, period, **kwargs)

        monkeypatch.setattr(client.ArmCommandLimiter, "step", record)
        feedback(node, ros)
        node._control_tick()
        assert captured[0][0][0] == pytest.approx(q[0] + 0.0125)
        assert captured[0][0][0] == pytest.approx(remaining[0, 0])
        assert captured[0][1] == pytest.approx(0.125)
        # A delayed tick advances trajectory time, not just one array index.
        output = node.publishers[node.config.joint_command_topic].messages
        assert output[-1].position[0] - output[-2].position[0] <= 0.0025

    def test_full_response_age_preserves_phase(self, config, ros, monkeypatch):
        clock = [100.0]
        monkeypatch.setattr(client.time, "monotonic", lambda: clock[0])
        node = ros.create(config())
        feedback(node, ros)
        q = np.asarray(client.DEFAULT_JOINT_POSITIONS)
        actions = np.tile(np.r_[q[:7], 0, q[7:14], 0], (64, 1))
        actions[:, 0] += np.arange(64) / 30 * 0.1
        node._future = Future()
        node._future_started = 99.875
        node._future.set_result((actions, 0.05))
        node._inference_tick()
        assert node._request_elapsed_ms == pytest.approx(125)
        assert node._remaining_actions(clock[0])[0, 0] == pytest.approx(
            q[0] + 0.0125
        )
        assert node._native_started == pytest.approx(99.875)

    def test_background_preparation_does_not_block_control_and_is_bounded(
        self, config, ros, monkeypatch
    ):
        node = ros.create(config(enable_control=True, inference_hz=30))
        feedback(node, ros)
        q = np.asarray(client.DEFAULT_JOINT_POSITIONS)
        node._reset_complete = node._received_action_chunk = True
        node.limiter.reset(q[:14])
        node._native_actions = np.r_[q[:7], 0, q[7:14], 0][None, :]
        node._native_started = time.monotonic()
        snapshot = tuple(
            TimedValue(node.now_ns, None, time.monotonic()) for _ in range(3)
        ) + (np.eye(4), np.eye(4))
        monkeypatch.setattr(
            node.observations, "synchronized", lambda: snapshot
        )
        entered, release = threading.Event(), threading.Event()
        workers = []

        def prepare(*args):
            workers.append(threading.get_ident())
            entered.set()
            if not release.wait(2):
                raise RuntimeError("test did not release preprocessing")
            return {}, {}

        monkeypatch.setattr(client, "_build_request", prepare)
        monkeypatch.setattr(
            node, "_request_inference", lambda *args: (None, 0.1)
        )
        try:
            node._inference_tick()
            pending = node._future
            assert entered.wait(1)
            node._control_tick()
            node._inference_tick()
            assert node._future is pending
            assert (
                len(node.publishers[node.config.joint_command_topic].messages)
                == 1
            )
            assert len(workers) == 1
            assert workers[0] != threading.get_ident()
        finally:
            release.set()
            node._future.result(timeout=1)

    def test_new_authority_reseeds_position_and_velocity(
        self, config, ros, monkeypatch
    ):
        node = ros.create(config(enable_dagger=True, control_token="x" * 16))
        send_state(node, ros)
        feedback(node, ros)
        q = np.asarray(client.DEFAULT_JOINT_POSITIONS)
        node.limiter.reset(q[:14] + 0.1)
        node.limiter.step(q[:14] + 1, 0.01)
        node._received_action_chunk = True
        node.now_ns += 1
        send_state(node, ros, "HOLD", 2)
        node.now_ns += 1
        send_state(node, ros, "AUTO", 3)
        feedback(node, ros)
        node._future = Future()
        node._future_window = node._control_window
        node._future_observed_s = time.monotonic()
        actions = np.tile(np.r_[q[:7], 0, q[7:14], 0], (64, 1))
        node._future.set_result((actions, 0.01))
        node._inference_tick()
        assert node.fault is None
        np.testing.assert_allclose(node.limiter.previous_command, q[:14])
        np.testing.assert_array_equal(
            node.limiter.previous_command_velocity, 0
        )
        assert node._diagnostic_counts["limiter_resets"] == 1

    def test_authority_revoked_during_preparation_discards_completion(
        self, config, ros, monkeypatch
    ):
        node = ros.create(
            config(
                enable_control=True, enable_dagger=True, control_token="x" * 16
            )
        )
        send_state(node, ros)
        node._gate.resume_started -= 0.001
        feedback(node, ros)
        q = np.asarray(client.DEFAULT_JOINT_POSITIONS)
        snapshot = tuple(
            TimedValue(node.now_ns + 1, None, time.monotonic())
            for _ in range(3)
        ) + (np.eye(4), np.eye(4))
        monkeypatch.setattr(
            node.observations, "synchronized", lambda: snapshot
        )
        entered, release = threading.Event(), threading.Event()

        def prepare(*args):
            entered.set()
            if not release.wait(2):
                raise RuntimeError("test did not release preprocessing")
            return {}, {}

        actions = np.tile(np.r_[q[:7], 0, q[7:14], 0], (64, 1))
        monkeypatch.setattr(client, "_build_request", prepare)
        monkeypatch.setattr(
            node, "_request_inference", lambda *args: (actions, 0.01)
        )
        try:
            node._inference_tick()
            assert entered.wait(1)
            node.now_ns += 1
            send_state(node, ros, "HOLD", 2)
        finally:
            release.set()
            node._future.result(timeout=1)
        node._inference_tick()
        node._control_tick()
        assert node.fault is None
        assert node._native_actions is None
        assert node.limiter.previous_command is None
        assert all(not pub.messages for pub in node.publishers.values())


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


def send_state(node, ros, state="AUTO", epoch=1, dagger_enabled=True):
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
            dagger_enabled=dagger_enabled,
            published_stamp_ns=node.now_ns,
        )
    )
    node._on_ownership(msg)


def feedback(node, ros, names=CONTROL_JOINT_NAMES):
    msg = ros.Message()
    msg.name, msg.position = list(names), list(client.DEFAULT_JOINT_POSITIONS)
    msg.header.stamp = node.get_clock().now().to_msg()
    node._on_joints(msg)


class TestInferenceRequest:
    @pytest.mark.parametrize("remaining_steps", [None, 0, 2])
    def test_encodes_observation_and_optional_rtc(
        self, config, remaining_steps
    ):
        cv2 = pytest.importorskip("cv2")
        color = np.zeros((4, 6, 3), dtype=np.uint8)
        depth = np.full((4, 6), 1500, dtype=np.uint16)
        color_ok, color_bytes = cv2.imencode(".jpg", color)
        depth_ok, depth_bytes = cv2.imencode(".png", depth)
        assert color_ok and depth_ok
        joints = np.arange(17, dtype=np.float64)
        snapshot = (
            TimedValue(1, color_bytes.tobytes()),
            TimedValue(1, depth_bytes.tobytes()),
            TimedValue(1, joints),
            np.eye(4),
            np.eye(4),
        )
        remaining = (
            None
            if remaining_steps is None
            else np.arange(remaining_steps * 16, dtype=np.float64).reshape(
                -1, 16
            )
        )
        files, form = client._build_request(
            snapshot, config(), 0.4, 0.6, remaining, 1
        )
        arrays = {
            name: np.load(BytesIO(payload), allow_pickle=False)
            for name, (_, payload, _) in files.items()
        }
        np.testing.assert_array_equal(arrays["head_color"], color)
        np.testing.assert_array_equal(arrays["head_depth"], depth / 1000)
        assert arrays["head_depth"].dtype == np.float32
        np.testing.assert_allclose(
            arrays["current_state"], np.r_[joints[:7], 0.4, joints[7:14], 0.6]
        )
        np.testing.assert_array_equal(arrays["head_intrinsic"], np.eye(4))
        np.testing.assert_array_equal(arrays["zhi_to_eye"], np.eye(4))
        assert json.loads(form["joint_names"]) == list(
            client.HC_TJ_JOINT_NAMES
        )
        assert form["instruction"] == "pick"
        if remaining_steps:
            np.testing.assert_array_equal(
                arrays["remaining_actions"], remaining
            )
            assert arrays["remaining_actions"].dtype == np.float32
            assert form["delay_horizon"] == "1"
        else:
            assert "remaining_actions" not in arrays
            assert "delay_horizon" not in form


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
        node._native_actions = np.r_[q[:7], 0, q[7:14], 0][None, :]
        node._control_tick()
        assert node.fault is None
        assert all(len(pub.messages) == 1 for pub in node.publishers.values())
        assert all(
            len(pub.messages[0].name) == 7 for pub in node.publishers.values()
        )
        for offset, topic in zip((0, 7), node._motion.topics, strict=True):
            command = node.publishers[topic].messages[0]
            assert command.name == list(
                node._motion.joint_names[offset : offset + 7]
            )
            np.testing.assert_allclose(
                command.position, q[offset : offset + 7]
            )
            assert command.header.stamp.sec == 1
            assert command.header.stamp.nanosec == 0

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
        node._native_actions = np.zeros((1, 16))
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
    initial = node.publishers[cfg.joint_command_topic].messages[0]
    assert initial.name == list(CONTROL_JOINT_NAMES)
    assert initial.position == list(client.DEFAULT_JOINT_POSITIONS)
    assert not node.publishers[cfg.right_hand_topic].messages
    q = np.asarray(client.DEFAULT_JOINT_POSITIONS)
    node._native_actions = np.r_[q[:7], 0.49, q[7:14], 0.5][None, :]
    node._received_action_chunk = True
    node._control_tick()
    assert node.fault is None
    assert (
        len(node.publishers[cfg.right_hand_topic].messages[0].position) == 11
    )
    assert node.publishers[cfg.right_hand_topic].messages[0].position[-1] == 0
    assert node.publishers[cfg.left_hand_topic].messages[0].position[-1] == 1
    assert node._right_hand_observation == 0.49


class TestStandardGripperFields:
    @pytest.mark.parametrize("dagger", [False, True])
    @pytest.mark.parametrize("closure", [0.0, 0.1, 0.49, 0.5, 1.0, 1.1])
    def test_model_fields_preserve_arm_waist_and_finger_targets(
        self, config, ros, monkeypatch, dagger, closure
    ):
        clock = [100.0]
        monkeypatch.setattr(client.time, "monotonic", lambda: clock[0])
        cfg = config(
            enable_control=True,
            enable_dagger=dagger,
            control_token="x" * 16 if dagger else "",
        )
        node = ros.create(cfg)
        if dagger:
            send_state(node, ros)
        q = np.asarray(client.DEFAULT_JOINT_POSITIONS)
        q[14:] = [0.13, 0.27, 0.39]
        node._reset_complete = node._received_action_chunk = True
        node.limiter.reset(q[:14])
        action = np.r_[q[:7], closure, q[7:14], 0.6]
        node._native_actions = action[None, :].copy()
        node._native_started = clock[0]
        for tick in range(1, 11):
            clock[0] += 0.01
            node.now_ns += 10_000_000
            state = ros.Message()
            state.name, state.position = list(CONTROL_JOINT_NAMES), q.tolist()
            state.header.stamp = node.get_clock().now().to_msg()
            node._on_joints(state)
            node._control_tick()
            assert node.fault is None
            combined = node.publishers[cfg.joint_command_topic].messages[-1]
            assert combined.name == list(CONTROL_JOINT_NAMES) + [
                "R_ban",
                "L_ban",
            ]
            # Preserve all existing arm and measured-waist command values.
            np.testing.assert_array_equal(combined.position[:17], q)
            expected = np.minimum(
                np.clip([closure, 0.6], 0, 1) * 1.5707, tick * 0.035
            )
            np.testing.assert_allclose(combined.position[17:], expected)
        np.testing.assert_array_equal(node._native_actions[0], action)

    @pytest.mark.parametrize("dagger", [True, False])
    def test_dagger_hand_commands_append_their_own_binary_label(
        self, config, ros, dagger
    ):
        cfg = config(
            enable_control=True,
            enable_dagger=dagger,
            control_token="x" * 16 if dagger else "",
        )
        node = ros.create(cfg)
        if dagger:
            send_state(node, ros)
        node._reset_complete = node._received_action_chunk = True
        q = np.asarray(client.DEFAULT_JOINT_POSITIONS)
        node.limiter.reset(q[:14])
        for tick, (right, left) in enumerate([(0, 1), (1, 0), (0.49, 0.5)], 1):
            node.now_ns += 10_000_000
            feedback(node, ros)
            node._native_actions = np.r_[q[:7], right, q[7:14], left][None, :]
            node._native_started = time.monotonic()
            node._control_tick()
            assert node.fault is None
            for side, value, topic, label in (
                ("right", right, cfg.right_hand_topic, "R_ban"),
                ("left", left, cfg.left_hand_topic, "L_ban"),
            ):
                names, positions = client.hand_command(side, value)
                messages = node.publishers[topic].messages
                assert len(messages) == tick
                hand = messages[-1]
                assert hand.name == list(names) + ([label] if dagger else [])
                expected = positions.tolist()
                if dagger:
                    expected.append(float(value >= 0.5))
                np.testing.assert_array_equal(hand.position, expected)
                assert hand.header.stamp == node.get_clock().now().to_msg()
        assert node._diagnostic_counts["right_hand_close_ticks"] == 1
        assert node._diagnostic_counts["left_hand_close_ticks"] == 2

    def test_missing_model_action_does_not_invent_gripper_fields(
        self, config, ros
    ):
        node = ros.create(config(enable_control=True))
        node._reset_complete = True
        feedback(node, ros)
        node._control_tick()
        assert all(not pub.messages for pub in node.publishers.values())


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
        node._native_actions = np.zeros((1, 16))
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
    assert node.limiter.previous_command is None
    assert not node._received_action_chunk
    assert node._native_actions is None


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
    assert node._native_actions is None
    assert all(not pub.messages for pub in node.publishers.values())


class TestDaggerBoundary:
    @pytest.mark.parametrize(
        ("latency", "camera_timeout", "accepted"),
        [(0.24498, 1.0, True), (0.45, 1.0, False), (0.24498, 0.2, False)],
    )
    def test_first_dagger_chunk_starts_from_feedback(
        self, config, ros, monkeypatch, latency, camera_timeout, accepted
    ):
        node = ros.create(
            config(
                enable_dagger=True,
                control_token="x" * 16,
                enable_control=True,
                first_point_jump_rad=0.6,
                camera_timeout_s=camera_timeout,
            )
        )
        send_state(node, ros)
        node.now_ns += 1
        send_state(node, ros, "HOLD", 2)
        node.now_ns += 1
        send_state(node, ros, "AUTO", 3)
        clock = time.monotonic() + 0.001
        monkeypatch.setattr(client.time, "monotonic", lambda: clock)
        node.now_ns += 1
        joints = np.array(client.DEFAULT_JOINT_POSITIONS)
        joints[3] = -0.717218
        msg = ros.Message()
        msg.name, msg.position = list(CONTROL_JOINT_NAMES), joints.tolist()
        msg.header.stamp = node.get_clock().now().to_msg()
        node._on_joints(msg)
        snapshot = tuple(
            TimedValue(node.now_ns, None, clock) for _ in range(3)
        )
        monkeypatch.setattr(
            node.observations,
            "synchronized",
            lambda: snapshot + (np.eye(4), np.eye(4)),
        )
        monkeypatch.setattr(client, "_build_request", lambda *args: ({}, {}))
        actions = np.tile(np.r_[joints[:7], 0, joints[7:14], 0], (12, 1))
        actions[:7, 3] -= 0.440730
        actions[7:, 3] = -1.337241
        monkeypatch.setattr(
            node, "_request_inference", lambda *args: (actions, latency)
        )
        node._inference_tick()
        node._future.result(timeout=1)
        node._control_tick()
        assert all(not pub.messages for pub in node.publishers.values())
        clock += latency
        node.now_ns += round(latency * 1e9)
        msg.header.stamp = node.get_clock().now().to_msg()
        node._on_joints(msg)
        node._inference_tick()
        node._control_tick()
        assert node.fault is None
        if accepted:
            # Seven elapsed intervals shorten the tail. Point zero is first.
            np.testing.assert_array_equal(node._native_actions, actions[:5])
            output = node.publishers[node.config.joint_command_topic].messages
            step = (
                node.config.max_arm_acceleration_rad_s2
                / node.config.control_hz**2
            )
            assert output[0].position[3] == pytest.approx(joints[3] - step)
        else:
            assert node._native_actions is None
            assert all(not pub.messages for pub in node.publishers.values())

    def test_resume_jump_replans_from_fresh_frame_before_publishing(
        self, config, ros, monkeypatch
    ):
        node = ros.create(
            config(
                enable_dagger=True,
                control_token="x" * 16,
                enable_control=True,
                first_point_jump_rad=0.6,
            )
        )
        send_state(node, ros, "HOLD")
        node.now_ns += 1
        send_state(node, ros, "AUTO", 2)
        joints = np.array(client.DEFAULT_JOINT_POSITIONS)
        joints[3] = -0.717218
        msg = ros.Message()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.name, msg.position = list(CONTROL_JOINT_NAMES), joints.tolist()
        node._on_joints(msg)
        safe = np.tile(np.r_[joints[:7], 0, joints[7:14], 0], (12, 1))
        rejected = safe.copy()
        rejected[:7, 3] -= 0.440730
        rejected[7:, 3] = -1.337241
        # A replan may fail even after earlier chunks in this grant executed.
        node._received_action_chunk = True
        node._native_actions = safe.copy()
        node._latency_ewma = 0.245
        node._future_window = node._control_window
        node._future = Future()
        node._future.set_result((rejected, 0.24498))
        clock = time.monotonic()
        monkeypatch.setattr(client.time, "monotonic", lambda: clock)
        before_rejection = clock
        node._inference_tick()
        node._control_tick()
        assert node.fault is None and node._jump_rejections == 1
        assert node._native_actions is None
        assert all(not pub.messages for pub in node.publishers.values())

        built = []
        monkeypatch.setattr(
            client,
            "_build_request",
            lambda *args: built.append(args) or ({}, {}),
        )
        monkeypatch.setattr(
            node, "_request_inference", lambda *args: (safe, 0.01)
        )
        # A sensor may stamp a queued image slightly ahead of the ROS clock.
        stamp = node.now_ns + 1
        received = before_rejection
        monkeypatch.setattr(
            node.observations,
            "synchronized",
            lambda: (
                tuple(TimedValue(stamp, None, received) for _ in range(3))
                + (np.eye(4), np.eye(4))
            ),
        )
        node._inference_tick()
        assert not built
        clock += 0.01
        received = clock
        stamp = node.now_ns
        node._inference_tick()
        assert not built
        stamp += 1
        node._inference_tick()
        node._future.result(timeout=1)
        assert built[0][-2:] == (None, 0)
        node._inference_tick()
        node._control_tick()
        assert node.fault is None and node._jump_rejections == 0
        assert node.publishers[node.config.joint_command_topic].messages

    def test_repeated_jumps_wait_for_a_new_hc_grant(self, config, ros):
        node = ros.create(
            config(
                enable_dagger=True, control_token="x" * 16, enable_control=True
            )
        )
        send_state(node, ros)
        feedback(node, ros)
        q = np.array(client.DEFAULT_JOINT_POSITIONS)
        safe = np.tile(np.r_[q[:7], 0, q[7:14], 0], (12, 1))
        rejected = safe.copy()
        rejected[:, 3] += 1
        for _ in range(3):
            node._future_window = node._control_window
            node._future_observed_s = time.monotonic()
            node._future = Future()
            node._future.set_result((rejected, 0.01))
            node._inference_tick()
        node._gate.invalidate("inference_unavailable")
        node.now_ns += 1
        send_state(node, ros)  # New transport window, same HC grant.
        node._inference_tick()
        node._control_tick()
        assert node._inference_wait == "awaiting_new_auto_grant_after_jump"
        assert node.fault is None and node._future is None
        assert all(not pub.messages for pub in node.publishers.values())

        node.now_ns += 1
        send_state(node, ros, "HOLD", 2)
        node.now_ns += 1
        send_state(node, ros, "AUTO", 3)
        node._future_window = node._control_window
        node._future_observed_s = time.monotonic()
        node._future = Future()
        node._future.set_result((safe, 0.01))
        node._inference_tick()
        node._control_tick()
        assert node.fault is None and node._jump_rejections == 0
        assert node.publishers[node.config.joint_command_topic].messages

    @pytest.mark.parametrize(
        "dagger_enabled,target,reason",
        [
            (False, 1.0, "first action jumps"),
            (True, np.nan, "must be finite"),
        ],
    )
    def test_other_faults_still_latch(
        self, config, ros, dagger_enabled, target, reason
    ):
        node = ros.create(
            config(
                enable_dagger=True, control_token="x" * 16, enable_control=True
            )
        )
        send_state(node, ros, dagger_enabled=dagger_enabled)
        feedback(node, ros)
        q = np.array(client.DEFAULT_JOINT_POSITIONS)
        chunk = np.tile(np.r_[q[:7], 0, q[7:14], 0], (12, 1))
        chunk[:, 3] += target
        node._future_window = node._control_window
        node._future_observed_s = time.monotonic()
        node._future = Future()
        node._future.set_result((chunk, 0.01))
        node._inference_tick()
        assert reason in node.fault
        assert not node._received_action_chunk
        assert all(not pub.messages for pub in node.publishers.values())

    def test_pause_clears_both_blocks_and_no_fault_or_reset(self, config, ros):
        node = ros.create(
            config(
                enable_dagger=True, control_token="x" * 16, enable_control=True
            )
        )
        send_state(node, ros)
        node._native_actions = np.ones((4, 16))
        node._received_action_chunk = True
        node.now_ns += 1
        send_state(node, ros, "HOLD", 2)
        node._control_tick()
        assert node._native_actions is None
        assert not node._received_action_chunk
        assert node._reset_complete and node.fault is None
        assert all(not pub.messages for pub in node.publishers.values())

    def test_old_future_cannot_refill_after_resume(self, config, ros):
        node = ros.create(config(enable_dagger=True, control_token="x" * 16))
        send_state(node, ros)
        old = node._control_window
        node._future = Future()
        node._future_window = old
        old_future = node._future
        old_future.set_running_or_notify_cancel()
        node.now_ns += 1
        send_state(node, ros, "HOLD", 2)
        node.now_ns += 1
        send_state(node, ros, "AUTO", 3)
        old_future.set_result((np.zeros((64, 16)), 0.01))
        node._inference_tick()
        assert node._native_actions is None
        assert node.fault is None

    @pytest.mark.parametrize("outcome", ["result", "error", "cancelled"])
    def test_takeover_stops_inference_and_resume_starts_fresh(
        self, config, ros, monkeypatch, outcome
    ):
        node = ros.create(config(enable_dagger=True, control_token="x" * 16))
        send_state(node, ros)
        old_future = Future()
        if outcome != "cancelled":
            old_future.set_running_or_notify_cancel()
        node._future = old_future
        node._future_window = node._control_window
        node._native_actions = np.ones((64, 16))
        node._received_action_chunk = True
        node._latency_ewma = 1.0
        node.now_ns += 1
        send_state(node, ros, "HOLD", 2)
        assert node._future is old_future
        assert node._native_actions is None
        assert node._latency_ewma is None
        assert not node._received_action_chunk
        monkeypatch.setattr(
            node.observations,
            "synchronized",
            lambda: pytest.fail("observation accessed during takeover"),
        )
        node._inference_tick()
        assert node._inference_wait == "authority_or_relay_closed"
        node.now_ns += 1
        send_state(node, ros, "AUTO", 3)
        if outcome != "cancelled":
            node._inference_tick()
            assert node._inference_wait == "inference_inflight"
        else:
            assert old_future.cancelled()
        if outcome == "error":
            old_future.set_exception(RuntimeError("obsolete failure"))
        elif outcome == "result":
            old_future.set_result((np.ones((64, 16)), 0.01))
        node._gate.resume_started -= 0.001
        snapshot = tuple(
            TimedValue(node.now_ns + 1, None, time.monotonic())
            for _ in range(3)
        ) + (np.eye(4), np.eye(4))
        monkeypatch.setattr(
            node.observations, "synchronized", lambda: snapshot
        )
        built = []
        monkeypatch.setattr(
            client,
            "_build_request",
            lambda *args: built.append(args) or ({}, {}),
        )
        monkeypatch.setattr(
            node, "_request_inference", lambda *args: (None, 0.01)
        )
        node._inference_tick()
        node._future.result(timeout=1)
        assert built[0][-2:] == (None, 0)
        assert node._native_actions is None
        assert node.fault is None

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
        node._native_actions = np.zeros((64, 16))
        node._received_action_chunk = True
        node._future_window = old
        node._future = Future()
        node._future.set_exception(client.requests.ConnectionError("offline"))
        node._inference_tick()
        assert node.fault is None
        assert node._control_window != old
        assert node._native_actions is None
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
        node._native_actions = np.zeros((64, 16))
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
            lambda *args: built.append(args) or ({}, {}),
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
        node._native_actions = np.r_[q[:7], 0, q[7:14], 0][None, :]
        node._received_action_chunk = True

        def revoke_while_preparing(*args, **kwargs):
            node._gate.invalidate("pause")
            return q[:14]

        monkeypatch.setattr(
            client.ArmCommandLimiter, "step", revoke_while_preparing
        )
        node._control_tick()
        assert node.fault is None
        assert node._native_actions is None
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
        node._native_actions = np.r_[q[:7], 0.4, q[7:14], 0.6][None, :]
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
            lambda *args: built.append(args) or ({}, {}),
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
        node._native_actions = np.zeros((64, 16))
        node._received_action_chunk = True
        assert not node._hands_ready()
        assert node._control_window != old
        assert node._native_actions is None


class TestCommandLine:
    @pytest.mark.parametrize(
        "overrides",
        [
            {},
            {
                "enable_control": True,
                "enable_dagger": True,
                "control_token": "x" * 16,
                "robot_interface": "hc_motion",
                "motion_identity_mapping": True,
                "allow_competing_publishers": True,
                "inference_hz": 6.0,
                "control_hz": 50.0,
                "request_timeout_s": 10.0,
                "camera_timeout_s": 2.0,
                "state_timeout_s": 0.5,
                "max_arm_acceleration_rad_s2": 0.5,
                "first_point_jump_rad": 0.4,
                "color_topic": "/custom/color",
                "ownership_topic": "/custom/owner",
            },
        ],
    )
    def test_cli_matches_config(self, config, monkeypatch, overrides):
        expected = config(**overrides)
        monkeypatch.delenv("HC_TJ_GATEWAY_TOKEN", raising=False)
        monkeypatch.delenv("HC_TJ_FIRST_POINT_JUMP_RAD", raising=False)
        argv = [
            "client",
            "--server-url",
            expected.server_url,
            "--instruction",
            expected.instruction,
            "--urdf-path",
            str(expected.urdf_path),
            "--depth-units-per-meter",
            str(expected.depth_units_per_meter),
        ]
        for name, value in overrides.items():
            argv.append("--" + name.replace("_", "-"))
            if value is not True:
                argv.append(str(value))
        monkeypatch.setattr(sys, "argv", argv)
        assert client._parse_args() == expected

    def test_environment_defaults_and_cli_precedence(
        self, config, monkeypatch
    ):
        expected = config(
            enable_dagger=True,
            control_token="e" * 16,
            first_point_jump_rad=0.6,
        )
        monkeypatch.setenv("HC_TJ_GATEWAY_TOKEN", expected.control_token)
        monkeypatch.setenv("HC_TJ_FIRST_POINT_JUMP_RAD", "0.6")
        argv = [
            "client",
            "--server-url",
            expected.server_url,
            "--instruction",
            expected.instruction,
            "--urdf-path",
            str(expected.urdf_path),
            "--depth-units-per-meter",
            str(expected.depth_units_per_meter),
            "--enable-dagger",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        assert client._parse_args() == expected
        argv.extend(
            ["--control-token", "c" * 16, "--first-point-jump-rad", "0.8"]
        )
        assert client._parse_args() == config(
            enable_dagger=True,
            control_token="c" * 16,
            first_point_jump_rad=0.8,
        )
