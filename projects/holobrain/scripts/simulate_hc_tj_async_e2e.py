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

"""Run the real HC-TJ async client/server path against a simulated robot.

This script connects to an externally started ``inference_server.py``,
publishes one Arrow dataset sample as isolated ROS camera/calibration/joint
feedback, and runs the real
``hc_tj_async_client.py`` node. Client commands are applied immediately to an
in-memory robot and fed back in the next ROS observation. The GUI displays the
commanded URDF pose together with the source RGB and depth images.
使用URDF机器人来模拟真实的HC-TJ机器人，使用训练数据集反方动作
Example::

    python projects/holobrain/own_test/simulate_hc_tj_async_e2e.py

    python projects/holobrain/own_test/simulate_hc_tj_async_e2e.py \
        --grad-cam --grad-cam-joint 0 --grad-cam-step 0

Grad-CAM holds the latest completed request's RGB-D image until the next
response arrives, so an old heatmap is never overlaid on a newer frame.
"""

from __future__ import annotations
import argparse
import json
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from projects.holobrain.scripts.hc_tj_async_client import (  # noqa: E402
    DEFAULT_JOINT_POSITIONS,
    HcTjAsyncClientConfig,
    _post_inference,
    _ros_node_class,
)
from projects.holobrain.scripts.hc_tj_async_core import (  # noqa: E402
    CONTROL_JOINT_NAMES,
    LEFT_HAND_CLOSED,
    LEFT_HAND_JOINT_NAMES,
    LEFT_HAND_OPEN,
    RIGHT_HAND_CLOSED,
    RIGHT_HAND_JOINT_NAMES,
    RIGHT_HAND_OPEN,
)
from projects.holobrain.scripts.replay_hc_tj_async import (  # noqa: E402
    _extrinsics_json,
)
from projects.holobrain.scripts.replay_hc_tj_mcap_urdf import (  # noqa: E402
    FINGER_MAX_POSITION_RAD,
    FINGER_MIN_POSITION_RAD,
)

__all__ = ["main", "run_simulation"]

DEFAULT_DATASET_PATH = (
    REPOSITORY_ROOT / "projects/holobrain/data/arrow_dataset/hc_tj_261026"
)
DEFAULT_URDF_PATH = (
    REPOSITORY_ROOT / "hc_assets/hc_tj_description/urdf/hc_tj_robot.urdf"
)
TOPIC_PREFIX = "/holobrain_hc_tj_e2e"
ACTIONS_PER_INFERENCE = 10
DATASET_FRAME_STRIDE = 10
SIMULATION_CONTROL_HZ = 20.0


@dataclass(slots=True)
class SimulationState:
    """Store the instantaneous simulated robot state across ROS callbacks."""

    joint_positions: np.ndarray
    right_hand_closure: float = 0.0
    left_hand_closure: float = 0.0
    joint_command_count: int = 0
    right_hand_command_count: int = 0
    left_hand_command_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def apply_joint_command(self, message: Any) -> None:
        """Apply one canonical 17-axis client command instantaneously."""
        positions = np.asarray(message.position, dtype=np.float64)
        if tuple(message.name) != CONTROL_JOINT_NAMES:
            raise ValueError("joint command does not use HC-TJ 17-axis order")
        if positions.shape != (17,) or not np.all(np.isfinite(positions)):
            raise ValueError("joint command must be finite with shape [17]")
        with self.lock:
            self.joint_positions = positions.copy()
            self.joint_command_count += 1

    def apply_hand_command(self, side: str, message: Any) -> None:
        """Recover the model closure value from one 10-axis hand command."""
        positions = np.asarray(message.position, dtype=np.float64)
        if side == "right":
            expected_names = RIGHT_HAND_JOINT_NAMES
            opened, closed = RIGHT_HAND_OPEN, RIGHT_HAND_CLOSED
        elif side == "left":
            expected_names = LEFT_HAND_JOINT_NAMES
            opened, closed = LEFT_HAND_OPEN, LEFT_HAND_CLOSED
        else:
            raise ValueError("side must be 'right' or 'left'")
        if tuple(message.name) != expected_names or positions.shape != (10,):
            raise ValueError(
                f"{side} hand command does not use its 10-axis map"
            )
        delta = closed - opened
        closure = float(
            np.clip(
                np.dot(positions - opened, delta) / np.dot(delta, delta), 0, 1
            )
        )
        with self.lock:
            if side == "right":
                self.right_hand_closure = closure
                self.right_hand_command_count += 1
            else:
                self.left_hand_closure = closure
                self.left_hand_command_count += 1

    def feedback(self) -> np.ndarray:
        """Return an isolated 17-axis feedback snapshot."""
        with self.lock:
            return self.joint_positions.copy()

    def urdf_config(self) -> dict[str, float]:
        """Return the current main-joint and simplified gripper URDF config."""
        with self.lock:
            config = dict(
                zip(CONTROL_JOINT_NAMES, self.joint_positions, strict=True)
            )
            finger_range = FINGER_MAX_POSITION_RAD - FINGER_MIN_POSITION_RAD
            config["R_ban"] = FINGER_MAX_POSITION_RAD - (
                self.right_hand_closure * finger_range
            )
            config["L_ban"] = FINGER_MAX_POSITION_RAD - (
                self.left_hand_closure * finger_range
            )
            return config

    def summary(self) -> dict[str, int]:
        """Return command counters for the final JSON report."""
        with self.lock:
            return {
                "joint_commands": self.joint_command_count,
                "right_hand_commands": self.right_hand_command_count,
                "left_hand_commands": self.left_hand_command_count,
            }


def _wait_for_inference_server(host: str, port: int, timeout_s: float) -> None:
    """Wait until the externally managed inference server accepts TCP."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(
        f"external inference server {host}:{port} was unavailable for "
        f"{timeout_s}s"
    )


def _camera_messages(
    sample: dict, compressed_image_type: type
) -> tuple[Any, Any]:
    """Encode one dataset RGB-D pair into reusable ROS compressed messages."""
    color_ok, color_data = cv2.imencode(".jpg", sample["imgs"][0])
    depth_mm = np.rint(sample["depths"][0] * 1000).astype(np.uint16)
    depth_ok, depth_data = cv2.imencode(".png", depth_mm)
    if not color_ok or not depth_ok:
        raise RuntimeError("failed to encode the dataset RGB-D sample")
    color = compressed_image_type()
    color.format = "jpeg"
    color.data = color_data.tobytes()
    depth = compressed_image_type()
    depth.format = "16UC1; png"
    depth.data = depth_data.tobytes()
    return color, depth


def _calibration_messages(sample: dict, string_type: type) -> tuple[Any, Any]:
    """Build the latched intrinsic and zhi-to-eye ROS messages."""
    intrinsic = sample["intrinsic"][0]
    height, width = sample["imgs"][0].shape[:2]
    calibration = string_type()
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
    extrinsics = string_type()
    extrinsics.data = _extrinsics_json(sample["T_world2cam"][0])
    return calibration, extrinsics


def _rgb_depth_preview(
    sample: dict,
    color: np.ndarray | None = None,
    label: str = "waiting for RGB",
) -> np.ndarray:
    """Compose the latest published BGR frame and metric-depth preview."""
    if color is None:
        color = np.zeros_like(sample["imgs"][0], dtype=np.uint8)
    else:
        color = np.asarray(color, dtype=np.uint8).copy()
    depth = np.asarray(sample["depths"][0], dtype=np.float32)
    positive = depth[np.isfinite(depth) & (depth > 0)]
    depth_max = float(np.percentile(positive, 99)) if len(positive) else 1.0
    normalized = np.clip(depth / max(depth_max, 1e-6), 0, 1)
    depth_color = cv2.applyColorMap(
        np.rint(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    if depth_color.shape[:2] != color.shape[:2]:
        depth_color = cv2.resize(depth_color, (color.shape[1], color.shape[0]))
    cv2.putText(
        color,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        depth_color,
        f"depth 0..{depth_max:.2f} m",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    return np.concatenate((color, depth_color), axis=1)


def _grad_cam_preview(
    files: dict[str, tuple[str, bytes, str]],
    payload: dict[str, Any],
    label: str,
) -> np.ndarray:
    """Overlay a response's positive CAM on its exact submitted BGR frame."""
    explanation = payload.get("grad_cam")
    if not isinstance(explanation, dict) or "head" not in explanation:
        raise ValueError(
            "server did not return Grad-CAM; restart the updated server"
        )
    heatmap = np.asarray(explanation["head"], dtype=np.float32)
    if (
        heatmap.ndim != 2
        or not heatmap.size
        or not np.isfinite(heatmap).all()
        or heatmap.min() < 0
        or heatmap.max() > 1
    ):
        raise ValueError("server returned an invalid Grad-CAM heatmap")
    color = np.load(BytesIO(files["head_color"][1]), allow_pickle=False)
    depth = np.load(BytesIO(files["head_depth"][1]), allow_pickle=False)
    resized = cv2.resize(heatmap, (color.shape[1], color.shape[0]))
    # OpenCV uses BGR. Zero attribution leaves the original pixel untouched.
    tint = np.zeros_like(color, dtype=np.float32)
    tint[..., 2] = 255
    tint[..., 1] = 255 * (1 - resized)
    alpha = 0.6 * resized[..., None]
    overlay = np.clip(color * (1 - alpha) + tint * alpha, 0, 255).astype(
        np.uint8
    )
    if not heatmap.any():
        label += " (no positive response)"
    return _rgb_depth_preview({"depths": [depth]}, overlay, label)


def _show_gui(
    urdf_path: Path,
    state: SimulationState,
    preview_getter: Callable[[], np.ndarray],
    fault_getter: Callable[[], str | None],
) -> None:
    """Show the commanded URDF and source RGB-D until the viewer closes."""
    try:
        from yourdfpy import URDF
    except ImportError as error:
        raise RuntimeError(
            "yourdfpy is required for URDF display; use the "
            "holobrain environment"
        ) from error

    robot = URDF.load(str(urdf_path), build_scene_graph=True)
    missing = set((*CONTROL_JOINT_NAMES, "R_ban", "L_ban")) - set(
        robot.actuated_joint_names
    )
    if missing:
        raise ValueError(
            f"URDF is missing simulated joints: {sorted(missing)}"
        )
    cv2.namedWindow("HC-TJ async RGB-D", cv2.WINDOW_NORMAL)
    cv2.imshow("HC-TJ async RGB-D", preview_getter())
    print("URDF and RGB-D viewers opened; close the URDF viewer to finish.")

    def update_scene(_: Any) -> None:
        fault = fault_getter()
        if fault is not None:
            raise RuntimeError(fault)
        robot.update_cfg(state.urdf_config())
        cv2.imshow("HC-TJ async RGB-D", preview_getter())
        cv2.waitKey(1)

    try:
        robot.show(callback=update_scene)
    except ModuleNotFoundError as error:
        if error.name == "pyglet":
            raise RuntimeError(
                "yourdfpy rendering requires pyglet; install 'pyglet<2'"
            ) from error
        raise
    finally:
        cv2.destroyAllWindows()


def run_simulation(
    *,
    dataset_path: Path,
    urdf_path: Path,
    sample_index: int,
    instruction: str | None,
    server_host: str,
    server_name: str,
    port: int,
    server_wait_timeout_s: float,
    state_timeout_s: float,
    no_gui: bool,
    duration_s: float,
    grad_cam: bool = False,
    grad_cam_joint: int | None = None,
    grad_cam_step: int | None = None,
) -> None:
    """Run dataset-to-server-to-client simulation with optional Grad-CAM.

    ``grad_cam`` requests and renders attribution on each request's own RGB.
    Joint (0..15) and step (0..63) optionally restrict the action target;
    both default to all and require ``grad_cam=True``. GUI display holds
    the last completed attribution, while ROS observations keep advancing.
    Attribution errors stop the client through its inference fault path.
    """
    if not grad_cam and (
        grad_cam_joint is not None or grad_cam_step is not None
    ):
        raise ValueError("Grad-CAM joint/step require grad_cam=True")
    for name, value, limit in (
        ("joint", grad_cam_joint, 16),
        ("step", grad_cam_step, 64),
    ):
        if value is not None and not 0 <= value < limit:
            raise ValueError(f"Grad-CAM {name} must be in [0, {limit})")

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
    if not 0 <= sample_index < len(dataset):
        raise IndexError(
            f"sample index {sample_index} is outside [0, {len(dataset)})"
        )
    sample = dataset[sample_index]
    state = SimulationState(
        joint_positions=np.asarray(DEFAULT_JOINT_POSITIONS, dtype=np.float64)
    )
    preview = _rgb_depth_preview(sample)
    preview_lock = threading.Lock()
    preview_frame_count = 0
    grad_cam_image = None
    grad_cam_response_count = 0
    active_sample_index = sample_index
    published_sample_index = sample_index

    def update_preview(
        source_sample: dict, jpeg_data: bytes, timestamp_ns: int
    ) -> None:
        """Record the exact RGB payload most recently published over ROS."""
        nonlocal preview, preview_frame_count
        color_image = cv2.imdecode(
            np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if color_image is None:
            raise RuntimeError("failed to decode the published RGB message")
        with preview_lock:
            preview_frame_count += 1
            preview = _rgb_depth_preview(
                source_sample,
                color_image,
                f"published RGB #{preview_frame_count} "
                f"t={timestamp_ns / 1e9:.3f}s"
                + (" | waiting for Grad-CAM" if grad_cam else ""),
            )

    def preview_snapshot() -> np.ndarray:
        """Copy the last attribution, or the live preview while waiting."""
        with preview_lock:
            if grad_cam_image is not None:
                return grad_cam_image.copy()
            return preview.copy()

    executor = None
    executor_thread = None
    publisher_thread = None
    driver = None
    client = None
    publisher_stop = threading.Event()
    callback_error: list[Exception] = []

    try:
        _wait_for_inference_server(server_host, port, server_wait_timeout_s)
        rclpy.init()
        driver = Node("holobrain_hc_tj_e2e_simulator")
        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        # Hand publishers use BEST_EFFORT. Receiving these commands also
        # advances the action count that gates subsequent inference requests.
        hand_qos = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.BEST_EFFORT
        )
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
            server_url=(
                f"http://{server_host}:{port}/{server_name}/hc_tj_async/v1"
            ),
            instruction=instruction or sample["text"],
            urdf_path=urdf_path,
            depth_units_per_meter=1000,
            enable_control=True,
            inference_hz=2.0,
            control_hz=SIMULATION_CONTROL_HZ,
            first_point_jump_rad=1.0,
            state_timeout_s=state_timeout_s,
            **topics,
        )
        base_client_class = _ros_node_class()

        class TenActionInferenceClient(base_client_class):
            """Gate each request after the first on ten executed actions."""

            def __init__(self, client_config: HcTjAsyncClientConfig) -> None:
                super().__init__(client_config)
                self._next_request_action_count = ACTIONS_PER_INFERENCE

            def _request_inference(
                self,
                files: dict[str, tuple[str, bytes, str]],
                form: dict[str, str],
            ) -> tuple[np.ndarray, float]:
                if not grad_cam:
                    return super()._request_inference(files, form)
                form = {**form, "grad_cam": "true"}
                if grad_cam_joint is not None:
                    form["grad_cam_joint"] = str(grad_cam_joint)
                if grad_cam_step is not None:
                    form["grad_cam_step"] = str(grad_cam_step)

                def show_response(payload: dict[str, Any]) -> None:
                    nonlocal grad_cam_image, grad_cam_response_count
                    joint_label = (
                        "all" if grad_cam_joint is None else grad_cam_joint
                    )
                    step_label = (
                        "all" if grad_cam_step is None else grad_cam_step
                    )
                    label = (
                        f"Grad-CAM request #{grad_cam_response_count + 1} "
                        f"joint={joint_label} "
                        f"step={step_label}"
                    )
                    rendered = _grad_cam_preview(files, payload, label)
                    with preview_lock:
                        grad_cam_image = rendered
                        grad_cam_response_count += 1

                return _post_inference(
                    self.config, files, form, response_callback=show_response
                )

            def _inference_tick(self) -> None:
                if self._future is not None:
                    if self._future.done():
                        # Let the base class consume this response without
                        # immediately chaining another asynchronous request.
                        self._last_request = time.monotonic()
                    super()._inference_tick()
                    return
                with self._lock:
                    received_action_chunk = self._received_action_chunk
                if not received_action_chunk:
                    super()._inference_tick()
                    return
                summary = state.summary()
                executed_actions = min(
                    summary["right_hand_commands"],
                    summary["left_hand_commands"],
                )
                required_actions = self._next_request_action_count
                required_frame = (
                    sample_index
                    + (required_actions // ACTIONS_PER_INFERENCE)
                    * DATASET_FRAME_STRIDE
                )
                if (
                    executed_actions < required_actions
                    or published_sample_index < required_frame
                ):
                    return
                self._last_request = float("-inf")
                super()._inference_tick()
                if self._future is not None:
                    self._next_request_action_count += ACTIONS_PER_INFERENCE

        # 用来中转一道，把observation 发到inference server和接受动作数据
        client = TenActionInferenceClient(config)

        color_pub = driver.create_publisher(
            CompressedImage, topics["color_topic"], reliable
        )
        depth_pub = driver.create_publisher(
            CompressedImage, topics["depth_topic"], reliable
        )
        joints_pub = driver.create_publisher(
            JointState, topics["joint_state_topic"], reliable
        )
        calibration_pub = driver.create_publisher(
            String, topics["calibration_topic"], latched
        )
        extrinsics_pub = driver.create_publisher(
            String, topics["extrinsics_topic"], latched
        )

        def guarded(callback: Callable[[Any], None]) -> Callable[[Any], None]:
            def invoke(message: Any) -> None:
                try:
                    callback(message)
                except Exception as error:
                    callback_error.append(error)
                    publisher_stop.set()

            return invoke

        driver.create_subscription(
            JointState,
            topics["joint_command_topic"],
            guarded(state.apply_joint_command),
            reliable,
        )
        driver.create_subscription(
            JointState,
            topics["right_hand_topic"],
            guarded(
                lambda message: state.apply_hand_command("right", message)
            ),
            hand_qos,
        )
        driver.create_subscription(
            JointState,
            topics["left_hand_topic"],
            guarded(lambda message: state.apply_hand_command("left", message)),
            hand_qos,
        )
        color, depth = _camera_messages(sample, CompressedImage)
        active_sample = sample
        completed_action_batches = 0
        calibration, extrinsics = _calibration_messages(sample, String)
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(driver)
        executor.add_node(client)
        executor_thread = threading.Thread(target=executor.spin, daemon=True)
        executor_thread.start()
        calibration_pub.publish(calibration)
        extrinsics_pub.publish(extrinsics)

        def advance_dataset_frame() -> bool:
            nonlocal active_sample, active_sample_index
            nonlocal calibration, color, completed_action_batches
            nonlocal depth, extrinsics
            summary = state.summary()
            executed_actions = min(
                summary["right_hand_commands"],
                summary["left_hand_commands"],
            )
            action_batches = executed_actions // ACTIONS_PER_INFERENCE
            if action_batches <= completed_action_batches:
                return True
            next_index = sample_index + (action_batches * DATASET_FRAME_STRIDE)
            if next_index >= len(dataset):
                callback_error.append(
                    IndexError(
                        f"next dataset frame {next_index} is outside "
                        f"[0, {len(dataset)})"
                    )
                )
                publisher_stop.set()
                return False
            active_sample_index = next_index
            active_sample = dataset[active_sample_index]
            color, depth = _camera_messages(active_sample, CompressedImage)
            calibration, extrinsics = _calibration_messages(
                active_sample, String
            )
            calibration_pub.publish(calibration)
            extrinsics_pub.publish(extrinsics)
            completed_action_batches = action_batches
            print(
                f"executed {executed_actions} actions; switched dataset "
                f"frame to {active_sample_index}"
            )
            return True

        def publish_observations() -> None:
            nonlocal published_sample_index
            period_s = 1.0 / 30.0
            while not publisher_stop.is_set():
                if not advance_dataset_frame():
                    return
                started = time.monotonic()
                stamp = driver.get_clock().now().to_msg()
                color.header.stamp = stamp
                depth.header.stamp = stamp
                joint_state = JointState()
                joint_state.header.stamp = stamp
                joint_state.name = list(CONTROL_JOINT_NAMES)
                joint_state.position = state.feedback().tolist()
                color_pub.publish(color)
                update_preview(
                    active_sample,
                    color.data,
                    stamp.sec * 1_000_000_000 + stamp.nanosec,
                )
                depth_pub.publish(depth)
                joints_pub.publish(joint_state)
                published_sample_index = active_sample_index
                publisher_stop.wait(
                    max(0.0, period_s - (time.monotonic() - started))
                )

        publisher_thread = threading.Thread(
            target=publish_observations, daemon=True
        )
        publisher_thread.start()
        print(
            json.dumps(
                {
                    "dataset": str(dataset_path),
                    "sample_index": sample_index,
                    "instruction": config.instruction,
                    "server_url": config.server_url,
                    "topics": topics,
                    "grad_cam": grad_cam,
                    "grad_cam_joint": grad_cam_joint,
                    "grad_cam_step": grad_cam_step,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        if no_gui:
            deadline = time.monotonic() + duration_s
            while time.monotonic() < deadline and not publisher_stop.is_set():
                if client.fault is not None:
                    raise RuntimeError(f"HC-TJ client fault: {client.fault}")
                time.sleep(0.05)
        else:
            _show_gui(
                urdf_path,
                state,
                preview_snapshot,
                lambda: client.fault,
            )
        publisher_stop.set()
        publisher_thread.join(timeout=2)
        if callback_error:
            raise RuntimeError(
                f"simulator callback failed: {callback_error[0]}"
            )
        if client.fault is not None:
            raise RuntimeError(f"HC-TJ client fault: {client.fault}")
        summary = state.summary()
        if (
            summary["right_hand_commands"] == 0
            or summary["left_hand_commands"] == 0
        ):
            raise RuntimeError(
                "no hand commands were received; inference/control did not "
                "complete"
            )
        print(
            json.dumps(
                {
                    "result": "passed",
                    **summary,
                    "grad_cam_responses": grad_cam_response_count,
                },
                indent=2,
            )
        )
    finally:
        publisher_stop.set()
        if publisher_thread is not None:
            publisher_thread.join(timeout=2)
        if executor is not None:
            executor.shutdown()
        if executor_thread is not None:
            executor_thread.join(timeout=2)
        if client is not None:
            client.destroy_node()
        if driver is not None:
            driver.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _parse_args() -> argparse.Namespace:
    """Parse and validate the standalone simulation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path", type=Path, default=DEFAULT_DATASET_PATH
    )
    parser.add_argument("--urdf-path", type=Path, default=DEFAULT_URDF_PATH)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--instruction")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-name", default="holobrain")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--server-wait-timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--state-timeout-s",
        type=float,
        default=2.0,
        help="simulated ROS feedback staleness limit (default: 2.0 s)",
    )
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument(
        "--grad-cam",
        action="store_true",
        help="overlay Grad-CAM on the RGB of the latest completed request",
    )
    parser.add_argument("--grad-cam-joint", type=int, choices=range(16))
    parser.add_argument("--grad-cam-step", type=int, choices=range(64))
    args = parser.parse_args()
    if not args.grad_cam and (
        args.grad_cam_joint is not None or args.grad_cam_step is not None
    ):
        parser.error("--grad-cam-joint/step require --grad-cam")
    for path_name in ("dataset_path", "urdf_path"):
        path = getattr(args, path_name)
        if not path.exists():
            parser.error(
                f"--{path_name.replace('_', '-')} does not exist: {path}"
            )
    if (
        args.server_wait_timeout_s <= 0
        or args.state_timeout_s <= 0
        or args.duration_s <= 0
    ):
        parser.error("timeouts and duration must be positive")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    return args


def main() -> None:
    """Run the standalone real-model HC-TJ ROS simulation."""
    args = _parse_args()
    run_simulation(
        dataset_path=args.dataset_path,
        urdf_path=args.urdf_path,
        sample_index=args.sample_index,
        instruction=args.instruction,
        server_host=args.server_host,
        server_name=args.server_name,
        port=args.port,
        server_wait_timeout_s=args.server_wait_timeout_s,
        state_timeout_s=args.state_timeout_s,
        no_gui=args.no_gui,
        duration_s=args.duration_s,
        grad_cam=args.grad_cam,
        grad_cam_joint=args.grad_cam_joint,
        grad_cam_step=args.grad_cam_step,
    )


if __name__ == "__main__":
    main()
