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

"""Replay one Arrow sample through the real client on isolated ROS topics."""

from __future__ import annotations
import argparse
import json
import math
import sys
import threading
import time
from io import BytesIO
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from projects.holobrain.scripts.hc_tj_async_client import (  # noqa: E402
    HcTjAsyncClientConfig,
    _ros_node_class,
)
from projects.holobrain.scripts.hc_tj_async_core import (  # noqa: E402
    ACTION_HZ,
    CONTROL_JOINT_NAMES,
    HC_TJ_JOINT_NAMES,
    LEFT_HAND_JOINT_NAMES,
    RIGHT_HAND_JOINT_NAMES,
)

DEFAULT_DATASET_PATH = (
    REPOSITORY_ROOT / "projects/holobrain/data/arrow_dataset/hc_tj_260818_fix"
)
TOPIC_PREFIX = "/holobrain_hc_tj_replay"


def _fake_server(port: int):
    from flask import Flask, jsonify, request
    from werkzeug.serving import make_server

    app = Flask(__name__)

    @app.post("/holobrain/hc_tj_async/v1")
    def infer():
        state = np.load(BytesIO(request.files["current_state"].read()))
        actions = np.repeat(state[None].astype(np.float32), 64, axis=0)
        # Small arm motion proves 30 -> 100 Hz interpolation and rate limiting.
        actions[:, 0] += np.linspace(0, 0.05, 64, dtype=np.float32)
        actions[:, 7] = 1.0
        actions[:, 15] = 0.0
        return jsonify(
            actions=actions.tolist(),
            joint_names=list(HC_TJ_JOINT_NAMES),
            action_hz=ACTION_HZ,
            action_horizon=64,
            inference_ms=1.0,
        )

    server = make_server("127.0.0.1", port, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _extrinsics_json(zhi_to_eye_mat: np.ndarray) -> str:
    eye_to_zhi_mat = np.linalg.inv(zhi_to_eye_mat)
    rotation = eye_to_zhi_mat[:3, :3]
    trace = float(np.trace(rotation))
    w = np.sqrt(max(0.0, 1.0 + trace)) / 2
    if w < 1e-8:
        raise ValueError("replay transform needs a non-degenerate quaternion")
    xyz = np.asarray(
        [
            (rotation[2, 1] - rotation[1, 2]) / (4 * w),
            (rotation[0, 2] - rotation[2, 0]) / (4 * w),
            (rotation[1, 0] - rotation[0, 1]) / (4 * w),
        ]
    )
    return json.dumps(
        {
            "schema": "hc_robot_camera_extrinsics/v1",
            "common_frame": "zhi_Link",
            "transforms": [
                {
                    "parent_frame": "zhi_Link",
                    "child_frame": "camera_head",
                    "translation_m": eye_to_zhi_mat[:3, 3].tolist(),
                    "rotation_xyzw": [*xyz.tolist(), w],
                }
            ],
        }
    )


def run_replay(dataset_path: Path, port: int) -> None:
    """Run an isolated replay and verify deployment invariants."""
    import cv2
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CompressedImage, JointState
    from std_msgs.msg import String

    from robo_orchard_lab.dataset.hc_tj import HcTjRODataset

    dataset = HcTjRODataset(
        paths=[str(dataset_path)], hist_steps=1, pred_steps=64, transforms=None
    )
    sample = dataset[0]
    server, server_thread = _fake_server(port)
    rclpy.init()
    observer = Node("holobrain_hc_tj_replay_driver")
    reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
    latched = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    topics = {
        "color_topic": f"{TOPIC_PREFIX}/color",
        "depth_topic": f"{TOPIC_PREFIX}/depth",
        "calibration_topic": f"{TOPIC_PREFIX}/calibration",
        "extrinsics_topic": f"{TOPIC_PREFIX}/extrinsics",
        "joint_state_topic": f"{TOPIC_PREFIX}/joint_states",
        "joint_command_topic": f"{TOPIC_PREFIX}/joint_cmd",
        "right_hand_topic": f"{TOPIC_PREFIX}/right_hand",
        "left_hand_topic": f"{TOPIC_PREFIX}/left_hand",
    }
    config = HcTjAsyncClientConfig(
        server_url=f"http://127.0.0.1:{port}/holobrain/hc_tj_async/v1",
        instruction=sample["text"],
        urdf_path=(
            REPOSITORY_ROOT
            / "hc_assets/hc_tj_description/urdf/hc_tj_robot.urdf"
        ),
        depth_units_per_meter=1000,
        enable_control=True,
        **topics,
    )
    client = _ros_node_class()(config)
    color_pub = observer.create_publisher(
        CompressedImage, topics["color_topic"], reliable
    )
    depth_pub = observer.create_publisher(
        CompressedImage, topics["depth_topic"], reliable
    )
    joints_pub = observer.create_publisher(
        JointState, topics["joint_state_topic"], reliable
    )
    calibration_pub = observer.create_publisher(
        String, topics["calibration_topic"], latched
    )
    extrinsics_pub = observer.create_publisher(
        String, topics["extrinsics_topic"], latched
    )
    received: dict[str, list] = {"joint": [], "right": [], "left": []}
    observer.create_subscription(
        JointState,
        topics["joint_command_topic"],
        lambda message: received["joint"].append((time.monotonic(), message)),
        reliable,
    )
    observer.create_subscription(
        JointState,
        topics["right_hand_topic"],
        lambda message: received["right"].append(message),
        reliable,
    )
    observer.create_subscription(
        JointState,
        topics["left_hand_topic"],
        lambda message: received["left"].append(message),
        reliable,
    )
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(observer)
    executor.add_node(client)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    try:
        color_ok, color_data = cv2.imencode(".jpg", sample["imgs"][0])
        depth_ok, depth_data = cv2.imencode(
            ".png", np.rint(sample["depths"][0] * 1000).astype(np.uint16)
        )
        if not color_ok or not depth_ok:
            raise RuntimeError("failed to encode replay images")
        intrinsic = sample["intrinsic"][0]
        height, width = sample["imgs"][0].shape[:2]
        calibration = String()
        calibration.data = json.dumps(
            {
                "schema": "hc_camera_calibration/v1",
                "camera": "eye",
                "resolution": {
                    "color": {"width": width, "height": height},
                    "depth": {"width": width, "height": height},
                },
                "rgb": {
                    "fx": float(intrinsic[0, 0]),
                    "fy": float(intrinsic[1, 1]),
                    "cx": float(intrinsic[0, 2]),
                    "cy": float(intrinsic[1, 2]),
                },
            }
        )
        extrinsics = String()
        extrinsics.data = _extrinsics_json(sample["T_world2cam"][0])
        calibration_pub.publish(calibration)
        extrinsics_pub.publish(extrinsics)

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and len(received["joint"]) < 80:
            now = observer.get_clock().now().to_msg()
            color = CompressedImage()
            color.header.stamp = now
            color.format = "jpeg"
            color.data = color_data.tobytes()
            depth = CompressedImage()
            depth.header.stamp = now
            depth.format = "16UC1; png"
            depth.data = depth_data.tobytes()
            state = JointState()
            state.header.stamp = now
            state.name = list(CONTROL_JOINT_NAMES)
            arms = np.concatenate(
                (sample["joint_state"][0, :7], sample["joint_state"][0, 8:15])
            )
            state.position = [*arms.tolist(), 0.1, 0.2, 0.3]
            color_pub.publish(color)
            depth_pub.publish(depth)
            joints_pub.publish(state)
            time.sleep(1 / 30)

        if len(received["joint"]) < 50:
            raise AssertionError(
                f"expected >=50 arm commands, got {len(received['joint'])}"
            )
        command = received["joint"][-1][1]
        if tuple(command.name) != CONTROL_JOINT_NAMES or not np.allclose(
            command.position[-3:], [0.1, 0.2, 0.3]
        ):
            raise AssertionError("17-axis command order or waist hold failed")
        if tuple(received["right"][-1].name) != RIGHT_HAND_JOINT_NAMES:
            raise AssertionError(
                "right hand command is not the complete 10-axis map"
            )
        if tuple(received["left"][-1].name) != LEFT_HAND_JOINT_NAMES:
            raise AssertionError(
                "left hand command is not the complete 10-axis map"
            )
        times = np.asarray([item[0] for item in received["joint"][-50:]])
        measured_hz = 49 / (times[-1] - times[0])
        if not 85 <= measured_hz <= 115:
            raise AssertionError(
                f"control rate is {measured_hz:.1f} Hz, expected 100 Hz"
            )

        bad_state = JointState()
        bad_state.header.stamp = observer.get_clock().now().to_msg()
        bad_state.name = list(CONTROL_JOINT_NAMES)
        bad_state.position = [math.nan] * 17
        joints_pub.publish(bad_state)
        time.sleep(0.2)
        count_after_fault = len(received["joint"])
        time.sleep(0.2)
        if len(received["joint"]) != count_after_fault:
            raise AssertionError(
                "latched fault did not stop command publication"
            )
        print(
            json.dumps(
                {
                    "joint_commands": len(received["joint"]),
                    "measured_control_hz": measured_hz,
                    "waist_hold": True,
                    "complete_hand_maps": True,
                    "fault_stop": True,
                    "topics": topics,
                },
                indent=2,
            )
        )
    finally:
        executor.shutdown()
        executor_thread.join(timeout=2)
        client.destroy_node()
        observer.destroy_node()
        rclpy.shutdown()
        server.shutdown()
        server_thread.join(timeout=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path", type=Path, default=DEFAULT_DATASET_PATH
    )
    parser.add_argument("--port", type=int, default=28765)
    args = parser.parse_args()
    run_replay(args.dataset_path, args.port)


if __name__ == "__main__":
    main()
