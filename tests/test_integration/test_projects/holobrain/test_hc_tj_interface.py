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

"""Client/server transport and unmodified HC Motion integration checks."""

from __future__ import annotations
import copy
import importlib.util
import json
import logging
import os
import sys
import threading
import time
import types
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml

from projects.holobrain.scripts.hc_tj_async_client import (
    HcTjAsyncClientConfig,
    _HcRobotApi,
    _ros_node_class,
)
from projects.holobrain.scripts.hc_tj_async_core import (
    ARM_JOINT_NAMES,
    HC_TJ_JOINT_NAMES,
    hand_command,
    parse_robot_camera_extrinsics,
)

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def server_module():
    """Run actual Flask routes with a test model and CLI configuration."""
    cli = types.ModuleType("robo_orchard_core.utils.cli")
    cli.SettingConfig = object
    cli.pydantic_from_argparse = lambda *_: types.SimpleNamespace(
        model_dir="unused",
        inference_prefix="inference",
        port=0,
        server_name="holobrain",
        num_joints=7,
        valid_action_step=64,
    )
    logs = types.ModuleType("robo_orchard_core.utils.logging")
    logs.LoggerManager = lambda: types.SimpleNamespace(
        get_child=logging.getLogger
    )
    plugin_type = type(
        "RTCInferencePlugin",
        (),
        {"__module__": "robo_orchard_lab.models.rtc_plugin.rtc_plugin"},
    )

    class Backend:
        def __init__(self):
            self.model = types.SimpleNamespace(
                decoder=types.SimpleNamespace(
                    async_inference_plugin=plugin_type()
                ),
                eval=lambda: None,
            )
            self.inputs = []
            self.target = None
            self.delay_next = False
            self.entered, self.release = threading.Event(), threading.Event()

        def __call__(self, value):
            self.inputs.append(value)
            if self.delay_next:
                self.delay_next = False
                self.entered.set()
                if not self.release.wait(10):
                    raise RuntimeError("test response was not released")
                raise RuntimeError("delayed failure from the previous AUTO")
            target = self.target if self.target is not None else np.zeros(16)
            return types.SimpleNamespace(
                action=torch.tensor(np.tile(target, (64, 1)))
            )

    backend = Backend()
    pipeline = types.ModuleType("robo_orchard_lab.models.holobrain.pipeline")
    pipeline.HoloBrainInferencePipeline = types.SimpleNamespace(
        load_pipeline=lambda **_: backend
    )
    processor = types.ModuleType("robo_orchard_lab.models.holobrain.processor")
    processor.MultiArmManipulationInput = types.SimpleNamespace
    gevent = types.ModuleType("gevent.pywsgi")
    gevent.WSGIServer = object
    replacements = {
        cli.__name__: cli,
        logs.__name__: logs,
        pipeline.__name__: pipeline,
        processor.__name__: processor,
        gevent.__name__: gevent,
    }
    spec = importlib.util.spec_from_file_location(
        "hc_interface_server_test",
        ROOT / "projects/holobrain/scripts/inference_server.py",
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, replacements):
        spec.loader.exec_module(module)
    yield module, backend
    backend.release.set()


def wire_request():
    arrays = dict(
        head_color=np.zeros((4, 6, 3), np.uint8),
        head_depth=np.ones((4, 6), np.float32),
        head_intrinsic=np.eye(4, dtype=np.float64),
        zhi_to_eye=np.eye(4, dtype=np.float64),
        current_state=np.zeros(16, np.float32),
    )
    result = dict(
        joint_names=json.dumps(HC_TJ_JOINT_NAMES), instruction="test"
    )
    for name, array in arrays.items():
        data = BytesIO()
        np.save(data, array, allow_pickle=False)
        data.seek(0)
        result[name] = (data, name + ".npy")
    return result


def test_actual_flask_hc_protocol_and_unchanged_legacy_route(server_module):
    module, backend = server_module
    client = module.app.test_client()
    response = client.post("/holobrain/hc_tj_async/v1", data=wire_request())
    assert response.status_code == 200
    result = response.get_json()
    assert np.asarray(result["actions"]).shape == (64, 16)
    assert result["action_hz"] == 30
    assert result["joint_names"] == list(HC_TJ_JOINT_NAMES)
    assert backend.inputs[-1].history_joint_state[0].shape == (16,)
    assert backend.inputs[-1].depth["head"][0][0, 0] == 1.0
    malformed = wire_request()
    malformed["joint_names"] = json.dumps(list(reversed(HC_TJ_JOINT_NAMES)))
    assert (
        client.post("/holobrain/hc_tj_async/v1", data=malformed).status_code
        == 400
    )
    assert client.post("/holobrain", data={}).status_code != 404


@pytest.fixture
def hc_mapping_file(tmp_path):
    entries = [
        dict(
            model_name=name,
            public_name=f"axis_{i}",
            scale=-2.0,
            offset=0.1,
            model_lower=-3.0,
            model_upper=3.0,
            unit="rad",
        )
        for i, name in enumerate(ARM_JOINT_NAMES)
    ]
    identity = dict(
        schema="hc-robot-api/v1", profile_id="x1", revision="fixture"
    )
    groups = {
        side + "_arm": dict(
            joint_names=list(
                reversed(
                    [e["public_name"] for e in entries[start : start + 7]]
                )
            ),
            endpoints=dict(servo_j=f"/control/motion/{side}_arm/servo_j"),
        )
        for side, start in (("right", 0), ("left", 7))
    }
    resolved = dict(
        identity, joints=entries, info=dict(identity, groups=groups)
    )
    path = tmp_path / "robot_api.resolved.yaml"
    path.write_text(yaml.safe_dump(resolved))
    return path


def test_public_mapping_handles_order_scale_offset_and_extra_body(
    tmp_path, hc_mapping_file
):
    path = hc_mapping_file
    api = _HcRobotApi(path)
    names = ["head"] + list(reversed(api.names))
    feedback = api.feedback(names, [1.0] + [0.2] * 14)
    np.testing.assert_allclose(feedback, [-0.3] * 14)
    assert all(
        values == pytest.approx([0.2] * 7)
        for _, values in api.targets(feedback).values()
    )
    with pytest.raises(ValueError, match="missing"):
        api.feedback(names[:-1], [0.2] * 14)
    with pytest.raises(ValueError, match="profile"):
        api.validate_info(dict(api.info, revision="other"))
    config = HcTjAsyncClientConfig(
        server_url="http://unused/holobrain/hc_tj_async/v1",
        instruction="test",
        urdf_path=tmp_path / "not-required",
        depth_units_per_meter=1000,
        robot_interface="hc_motion",
        robot_api_file=path,
    )
    assert config.joint_state_topic == "/observation/joint_states"
    assert config.right_hand_topic == "/action/right_hand/joint_states"
    assert config.left_hand_topic == "/action/left_hand/joint_states"


def test_openarmx_mapping_uses_measured_grippers_and_profile_camera_base(
    hc_mapping_file,
):
    document = yaml.safe_load(hc_mapping_file.read_text())
    document["profile_id"] = document["info"]["profile_id"] = "openarmx"
    for index, entry in enumerate(document["joints"]):
        side = "right" if index < 7 else "left"
        entry["model_name"] = f"openarmx_{side}_joint{index % 7 + 1}"
    for side in ("right", "left"):
        document["info"]["groups"][side + "_arm"]["base_frame"] = (
            "openarmx_body_link0"
        )
        document["joints"].append(
            dict(
                model_name=f"openarmx_{side}_finger_joint1",
                public_name=f"{side}_gripper_joint_1",
                unit="m",
                scale=-2.0,
                offset=0.044,
                model_lower=0.0,
                model_upper=0.044,
            )
        )
    hc_mapping_file.write_text(yaml.safe_dump(document))
    api = _HcRobotApi(hc_mapping_file)
    names = [*api.names, "right_gripper_joint_1", "left_gripper_joint_1"]
    values = [0.2] * 14 + [0.0055, 0.0165]
    measured = api.feedback(list(reversed(names)), list(reversed(values)))
    np.testing.assert_allclose(measured[:14], [-0.3] * 14)
    np.testing.assert_allclose(measured[14:], [0.25, 0.75])
    targets, positions = api.gripper_targets((0.0, 1.0))
    assert targets == names[-2:]
    np.testing.assert_allclose(positions, [0.0, 0.022])
    with pytest.raises(ValueError, match="missing a gripper"):
        api.feedback(names[:-1], values[:-1])
    with pytest.raises(ValueError, match="nonfinite"):
        api.gripper_targets((float("nan"), 0.0))
    camera = dict(
        schema="hc_robot_camera_extrinsics/v1",
        common_frame="openarmx_body_link0",
        transforms=[
            dict(
                parent_frame="openarmx_body_link0",
                child_frame="camera_head",
                translation_m=[0.1, 0.2, 0.3],
                rotation_xyzw=[0.0, 0.0, 0.0, 1.0],
            )
        ],
    )
    transform = parse_robot_camera_extrinsics(
        camera, base_frame=api.camera_base_frame
    )
    np.testing.assert_allclose(transform[:3, 3], [-0.1, -0.2, -0.3])
    with pytest.raises(ValueError, match="common_frame"):
        parse_robot_camera_extrinsics(camera)
    document["joints"][-1]["unit"] = "rad"
    hc_mapping_file.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="gripper mapping"):
        _HcRobotApi(hc_mapping_file)


@pytest.mark.parametrize("source_clock_offset_s", [-3600, 3600])
def test_auto_requires_new_receipts_independent_of_source_clock(
    tmp_path,
    monkeypatch,
    server_module,
    hc_mapping_file,
    source_clock_offset_s,
):
    rclpy = pytest.importorskip("rclpy")
    import cv2
    from sensor_msgs.msg import CompressedImage, JointState
    from std_msgs.msg import String
    from werkzeug.serving import make_server

    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path / "ros"))
    server, backend = server_module
    http = make_server("127.0.0.1", 0, server.app)
    worker = threading.Thread(target=http.serve_forever, daemon=True)
    worker.start()
    rclpy.init(args=[], domain_id=231)
    node = None
    try:
        node = _ros_node_class()(
            HcTjAsyncClientConfig(
                server_url=f"http://127.0.0.1:{http.server_port}/holobrain/hc_tj_async/v1",
                instruction="test",
                urdf_path=tmp_path / "unused",
                depth_units_per_meter=1000,
                robot_interface="hc_motion",
                robot_api_file=hc_mapping_file,
                enable_control=True,
            )
        )
        node._on_info(String(data=json.dumps(node._hc_api.info)))
        calibration = dict(
            schema="hc_camera_calibration/v1",
            camera="eye",
            rgb=dict(fx=4, fy=4, cx=3, cy=2),
            resolution=dict(
                color=dict(width=6, height=4), depth=dict(width=6, height=4)
            ),
        )
        for stream in ("color", "depth"):
            partial = dict(
                calibration,
                resolution={stream: calibration["resolution"][stream]},
            )
            if stream == "depth":
                partial["depth"] = partial.pop("rgb")
            node._on_calibration(String(data=json.dumps(partial)))
            assert node.fault is None
        node._on_calibration(String(data=json.dumps(calibration)))
        node.observations.set_extrinsics(
            json.dumps(
                dict(
                    schema="hc_robot_camera_extrinsics/v1",
                    common_frame="zhi_Link",
                    transforms=[
                        dict(
                            parent_frame="zhi_Link",
                            child_frame="camera_head",
                            translation_m=[0, 0, 0],
                            rotation_xyzw=[0, 0, 0, 1],
                        )
                    ],
                )
            )
        )
        stamp = time.time_ns() + source_clock_offset_s * 10**9

        def observations():
            nonlocal stamp
            stamp += 1_000_000
            for callback, extension, array in (
                (node._on_color, ".jpg", np.zeros((4, 6, 3), np.uint8)),
                (node._on_depth, ".png", np.full((4, 6), 1000, np.uint16)),
            ):
                msg = CompressedImage(
                    data=cv2.imencode(extension, array)[1].tobytes()
                )
                msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(
                    stamp, 10**9
                )
                callback(msg)
            joints = JointState(name=node._hc_api.names, position=[0.05] * 14)
            joints.header.stamp.sec, joints.header.stamp.nanosec = divmod(
                stamp, 10**9
            )
            node._on_joints(joints)

        observations()
        node._on_owner(
            String(
                data=json.dumps(
                    dict(
                        controller_instance_id="controller",
                        session_id="session",
                        epoch=1,
                        sequence=1,
                        state="AUTO",
                        source="motion",
                    )
                )
            )
        )
        node._inference_tick()
        # Cached frames predate AUTO even if their source clock is ahead.
        assert node._future is None
        observations()
        node._inference_tick()
        # New receipts work even if their source time is behind.
        assert node._future is not None
        node._future.result(timeout=3)
        assert len(backend.inputs) == 1
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
        http.shutdown()
        http.server_close()
        worker.join(3)


@pytest.mark.parametrize("profile_name", ["x1", "openarmx"])
def test_client_through_existing_hc_motion_and_fake_robot(
    tmp_path, monkeypatch, server_module, profile_name
):
    """Run HTTP/DDS/HC/Motion/driver with fake cameras, model and plant."""
    baseline = os.environ.get("HC_INTERFACE_TEST_HC_ROOT")
    if not baseline:
        pytest.skip(
            "set HC_INTERFACE_TEST_HC_ROOT to an unchanged HC checkout "
            "with built ROS packages"
        )
    pytest.importorskip("rclpy")
    import cv2
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from sensor_msgs.msg import CompressedImage, JointState
    from std_msgs.msg import Header, String
    from werkzeug.serving import make_server

    baseline = Path(baseline)
    openarmx = profile_name == "openarmx"
    adapter_source = None
    if openarmx:
        adapter_root = os.environ.get("HC_INTERFACE_TEST_OPENARMX_ROOT")
        if not adapter_root:
            pytest.skip(
                "set HC_INTERFACE_TEST_OPENARMX_ROOT to the source checkout"
            )
        import openarmx_hc_robot_adapter.node as adapter

        adapter_source = Path(adapter.__file__).resolve()
        assert (
            adapter_source
            == (
                Path(adapter_root)
                / "src/openarmx_hc_robot_adapter"
                / "openarmx_hc_robot_adapter/node.py"
            ).resolve()
        )
    monkeypatch.syspath_prepend(str(baseline))
    spec = importlib.util.spec_from_file_location(
        "hc_interface_rig", baseline / "tests/robot_api_fake_rig.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    server, backend = server_module
    http = make_server("127.0.0.1", 0, server.app)
    worker = threading.Thread(target=http.serve_forever, daemon=True)
    worker.start()
    node = None
    try:
        with module.FakeRig(
            baseline / "adapters/robots" / profile_name,
            tmp_path / "rig",
            with_teleop=openarmx,
        ) as rig:
            config = HcTjAsyncClientConfig(
                server_url=f"http://127.0.0.1:{http.server_port}/holobrain/hc_tj_async/v1",
                instruction="test",
                urdf_path=tmp_path / "unused",
                depth_units_per_meter=1000,
                robot_interface="hc_motion",
                robot_api_file=rig.runtime / "robot_api.resolved.yaml",
                enable_control=True,
            )
            node = _ros_node_class()(config)
            rig.executor.add_node(node)
            gripped, sequence, key, phase = False, 0, "", ""
            peer = "interface-test-teleop"
            control = rig.bridge.ownership
            latched = QoSProfile(
                depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL
            )
            calibration = rig.node.create_publisher(
                String, config.calibration_topic, latched
            )
            extrinsics = rig.node.create_publisher(
                String, config.extrinsics_topic, 10
            )
            color = rig.node.create_publisher(
                CompressedImage, config.color_topic, 1
            )
            depth = rig.node.create_publisher(
                CompressedImage, config.depth_topic, 1
            )
            ack = rig.node.create_publisher(String, "/status/control/ack", 100)
            clutch = rig.node.create_publisher(
                String, "/events/teleop/clutch", 100
            )
            observed = {side: [] for side in ("right", "left")}
            candidates = []
            gripper_candidates = []
            if openarmx:
                gripper_names = {
                    entry["public_name"] for entry in node._hc_api.grippers
                }
                rig.node.create_subscription(
                    JointState,
                    "/internal/control/joint_targets",
                    lambda msg: gripper_candidates.append(msg)
                    if set(msg.name) == gripper_names
                    and ":motion:" in msg.header.frame_id
                    else None,
                    10,
                )
            vr = (
                rig.node.create_publisher(String, "/vrdata", 1)
                if openarmx
                else None
            )
            vr_frame = dict(
                protocol_version=3,
                sequence=0,
                vr_timestamp=0.0,
                tracking=dict(head=True, left=True, right=True),
                poses={
                    "head": dict(
                        position=[0.0, 1.5, 0.0],
                        quaternion=[0.0, 0.0, 0.0, 1.0],
                    ),
                    "left": dict(
                        position=[-0.2, 1.3, -0.4],
                        quaternion=[0.0, 0.0, 0.0, 1.0],
                    ),
                    "right": dict(
                        position=[0.2, 1.3, -0.4],
                        quaternion=[0.0, 0.0, 0.0, 1.0],
                    ),
                },
                inputs=dict(
                    left=dict(grip=0.0, trigger=0.5),
                    right=dict(grip=0.0, trigger=0.5),
                ),
            )
            for side, topic in (
                ("right", config.right_hand_topic),
                ("left", config.left_hand_topic),
            ):
                rig.node.create_subscription(
                    JointState,
                    topic,
                    lambda msg, s=side: observed[s].append(msg),
                    10,
                )
            for topic, _ in node._hc_api.groups.values():
                rig.node.create_subscription(
                    JointState, topic, candidates.append, 10
                )

            def acknowledge():
                ack.publish(
                    String(
                        data=json.dumps(
                            dict(
                                participant="teleop",
                                instance=peer,
                                token=key,
                                phase=phase,
                                stamp_ns=rig.node.get_clock()
                                .now()
                                .nanoseconds,
                                success=True,
                            )
                        )
                    )
                )

            def permit(message):
                nonlocal key, phase
                if message.frame_id.startswith("hc1:"):
                    identity, stage, _, target = message.frame_id.split("#")
                    if target == peer:
                        key, phase = identity, stage
                        acknowledge()

            rig.node.create_subscription(
                Header, "/control/motion/permit", permit, 1
            )
            jpeg = cv2.imencode(".jpg", np.full((4, 6, 3), 100, np.uint8))[
                1
            ].tobytes()
            png = cv2.imencode(".png", np.full((4, 6), 1000, np.uint16))[
                1
            ].tobytes()
            calibration.publish(
                String(
                    data=json.dumps(
                        dict(
                            schema="hc_camera_calibration/v1",
                            camera="eye",
                            resolution=dict(
                                color=dict(width=6, height=4),
                                depth=dict(width=6, height=4),
                            ),
                            rgb=dict(fx=4.0, fy=4.0, cx=3.0, cy=2.0),
                        )
                    )
                )
            )
            initial = dict(rig.positions)
            arm_initial = [initial[n] for n in node._hc_api.model_names]
            target = np.asarray([*arm_initial[:7], 0.0, *arm_initial[7:], 0.0])
            target[3] += 0.015
            target[11] += 0.015
            target[7], target[15] = 0.2, 0.8
            backend.target = target

            def feed():
                nonlocal sequence
                sequence += 1
                now = rig.node.get_clock().now()
                if openarmx:
                    vr_frame.update(
                        sequence=sequence,
                        vr_timestamp=time.monotonic(),
                        hc_received_ns=time.time_ns(),
                    )
                    vr_frame["inputs"]["right"]["grip"] = float(gripped)
                    vr.publish(String(data=json.dumps(vr_frame)))
                else:
                    clutch.publish(
                        String(
                            data=json.dumps(
                                dict(
                                    device_instance=peer,
                                    sequence=sequence,
                                    stamp_ns=now.nanoseconds,
                                    valid=True,
                                    gripped=gripped,
                                )
                            )
                        )
                    )
                    acknowledge()
                for publisher, data, fmt in (
                    (color, jpeg, "jpeg"),
                    (depth, png, "png"),
                ):
                    message = CompressedImage(format=fmt, data=data)
                    message.header.stamp = now.to_msg()
                    publisher.publish(message)
                extrinsics.publish(
                    String(
                        data=json.dumps(
                            dict(
                                schema="hc_robot_camera_extrinsics/v1",
                                common_frame=node._hc_api.camera_base_frame,
                                transforms=[
                                    dict(
                                        parent_frame=node._hc_api.camera_base_frame,
                                        child_frame="camera_head",
                                        translation_m=[0.0, 0.0, 0.0],
                                        rotation_xyzw=[0.0, 0.0, 0.0, 1.0],
                                    )
                                ],
                            )
                        )
                    )
                )

            rig.node.create_timer(0.02, feed)
            control.enable()
            rig.until(
                lambda: control.input_armed
                and len(control.peers) == 2
                and rig.bridge.control_ready()
            )
            rig.pump(0.1)
            assert not backend.inputs and not candidates
            control.start_auto(recording_ready=True, control_ready=True)
            rig.until(
                lambda: all(
                    rig.positions[n] > initial[n] + 0.002
                    for n in (
                        node._hc_api.model_names[3],
                        node._hc_api.model_names[10],
                    )
                ),
                8,
            )
            assert all(observed.values()) and node.fault is None
            for side, value in (("right", 0.2), ("left", 0.8)):
                if openarmx:
                    names = [f"openarmx_{side}_finger_joint1"]
                    positions = [0.044 * (1 - value)]
                    rig.until(
                        lambda n=names[0], p=positions[0]: abs(
                            rig.positions[n] - p
                        )
                        < 0.0005
                    )
                    assert any(
                        topic
                        == f"/{side}_forward_position_controller/commands"
                        and abs(values[-1] - positions[0]) < 0.0005
                        for topic, values in rig.controller_commands
                    )
                else:
                    names, positions = hand_command(side, value)
                assert observed[side][-1].name == list(names)
                np.testing.assert_allclose(
                    observed[side][-1].position, positions
                )
            assert all(
                len(msg.name) == 7
                and all("joint_" in name for name in msg.name)
                for msg in candidates
            )
            assert backend.inputs[-1].history_joint_state[0].shape == (16,)
            if openarmx:
                np.testing.assert_allclose(
                    backend.inputs[0].history_joint_state[0][[7, 15]],
                    [
                        1 - initial[f"openarmx_{side}_finger_joint1"] / 0.044
                        for side in ("right", "left")
                    ],
                )
                stale_grippers = copy.deepcopy(gripper_candidates[-1])
                delayed = rig.node.create_publisher(
                    JointState, "/internal/control/joint_targets", 10
                )
                auto_grippers = {
                    s: rig.positions[f"openarmx_{s}_finger_joint1"]
                    for s in ("right", "left")
                }

            backend.delay_next = True
            rig.until(backend.entered.is_set, 4)
            gripped = True
            rig.until(lambda: node._owner.get("state") == "MANUAL", 4)
            rig.pump(0.1)
            counts = (
                len(candidates),
                len(gripper_candidates)
                if openarmx
                else len(observed["right"]),
                0 if openarmx else len(observed["left"]),
            )
            rig.pump(0.15)
            assert counts == (
                len(candidates),
                len(gripper_candidates)
                if openarmx
                else len(observed["right"]),
                0 if openarmx else len(observed["left"]),
            )
            if openarmx:
                manual_before = {
                    s: rig.positions[f"openarmx_{s}_finger_joint1"]
                    for s in ("right", "left")
                }
                vr_frame["inputs"]["right"]["trigger"] = 0.1
                vr_frame["inputs"]["left"]["trigger"] = 0.9
                rig.until(
                    lambda: all(
                        abs(
                            rig.positions[f"openarmx_{s}_finger_joint1"]
                            - manual_before[s]
                        )
                        > 0.003
                        for s in ("right", "left")
                    )
                )
                manual_after = {
                    s: rig.positions[f"openarmx_{s}_finger_joint1"]
                    for s in ("right", "left")
                }
                stale_grippers.header.stamp = (
                    rig.node.get_clock().now().to_msg()
                )
                delayed.publish(stale_grippers)
                rig.pump(0.1)
                assert not any(
                    msg.header.stamp == stale_grippers.header.stamp
                    for messages in observed.values()
                    for msg in messages
                )
            requests_before_release = len(backend.inputs)
            measured_before_release = (
                [
                    1 - rig.positions[f"openarmx_{side}_finger_joint1"] / 0.044
                    for side in ("right", "left")
                ]
                if openarmx
                else None
            )
            gripped = False
            rig.until(lambda: node._owner.get("state") == "AUTO", 4)
            backend.release.set()
            rig.until(lambda: len(candidates) > counts[0], 5)
            assert node.fault is None
            if openarmx:
                np.testing.assert_allclose(
                    backend.inputs[
                        requests_before_release
                    ].history_joint_state[0][[7, 15]],
                    measured_before_release,
                    atol=0.03,
                )
                rig.until(
                    lambda: all(
                        abs(
                            rig.positions[f"openarmx_{s}_finger_joint1"] - goal
                        )
                        < 0.0005
                        for s, goal in (("right", 0.0352), ("left", 0.0088))
                    )
                )
                stale_grippers.header.stamp = (
                    rig.node.get_clock().now().to_msg()
                )
                delayed.publish(stale_grippers)
                rig.pump(0.1)
                assert not any(
                    msg.header.stamp == stale_grippers.header.stamp
                    for messages in observed.values()
                    for msg in messages
                )
                (tmp_path / "openarmx-evidence.json").write_text(
                    json.dumps(
                        dict(
                            adapter_source=str(adapter_source),
                            ros_domain=rig.domain,
                            model="deterministic test backend",
                            hardware="fake plant",
                            auto_grippers_m=auto_grippers,
                            manual_before_m=manual_before,
                            manual_after_m=manual_after,
                            resumed_grippers_m={
                                s: rig.positions[f"openarmx_{s}_finger_joint1"]
                                for s in ("right", "left")
                            },
                            manual_program_output_stopped=True,
                            old_tokens_rejected_in_manual_and_auto=True,
                            http_requests=len(backend.inputs),
                            real_vr_adapter=True,
                        ),
                        indent=2,
                    )
                    + "\n"
                )
            control.hold()
            rig.until(lambda: node._owner.get("state") == "HOLD")
            rig.executor.remove_node(node)
            node.destroy_node()
            node = None
    finally:
        backend.release.set()
        if node is not None:
            node.destroy_node()
        http.shutdown()
        http.server_close()
        worker.join(3)
