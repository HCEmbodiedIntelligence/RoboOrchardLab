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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run GPU-host HoloBrain inference through an HC-TJ HTTP robot gateway."""

from __future__ import annotations
import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
try:
    sys.path.remove(str(_REPOSITORY_ROOT))
except ValueError:
    pass
sys.path.insert(0, str(_REPOSITORY_ROOT))

from projects.holobrain.scripts.hc_tj_async_client import (  # noqa: E402
    _build_request,
)
from projects.holobrain.scripts.hc_tj_async_core import (  # noqa: E402
    ACTION_HZ,
    TimedValue,
    parse_camera_calibration,
    parse_robot_camera_extrinsics,
    trim_elapsed_actions,
    validate_action_chunk,
    validate_inference_response,
)
from projects.holobrain.scripts.hc_tj_http_protocol import (  # noqa: E402
    ACTION_PATH,
    HEALTH_PATH,
    JOINT_FEEDBACK_CONTENT_TYPE,
    JOINT_FEEDBACK_PATH,
    OBSERVATION_CONTENT_TYPE,
    OBSERVATION_PATH,
    RobotJointFeedbackPacket,
    RobotObservationPacket,
    decode_joint_feedback,
    decode_observation,
    encode_action_chunk,
)

__all__ = [
    "HcTjHttpAsyncClientConfig",
    "main",
    "packet_to_snapshot",
    "run_client",
]


@dataclass(frozen=True, slots=True)
class HcTjHttpAsyncClientConfig:
    """Configure the GPU-host side of the HTTP inference loop."""

    robot_url: str
    robot_token: str
    server_url: str
    instruction: str
    depth_units_per_meter: float
    enable_control: bool = False
    inference_hz: float = 2.0
    request_timeout_s: float = 30.0
    observation_wait_s: float = 2.0
    first_point_jump_rad: float = 0.30
    latency_ewma_alpha: float = 0.25

    def __post_init__(self) -> None:
        if not self.robot_url.startswith(("http://", "https://")):
            raise ValueError("robot_url must use HTTP or HTTPS")
        if self.robot_url.endswith("/"):
            raise ValueError("robot_url must not end with a slash")
        if len(self.robot_token) < 16:
            raise ValueError("robot_token must contain at least 16 characters")
        if not self.server_url.endswith("/hc_tj_async/v1"):
            raise ValueError(
                "server_url must target the /hc_tj_async/v1 endpoint"
            )
        if not self.instruction.strip():
            raise ValueError("instruction must not be empty")
        if self.depth_units_per_meter <= 0:
            raise ValueError("depth_units_per_meter must be positive")
        if not 0 < self.inference_hz <= 50.0:
            raise ValueError("inference_hz must be in (0, 50]")
        if self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive")
        if not 0 <= self.observation_wait_s <= 5:
            raise ValueError("observation_wait_s must be in [0, 5]")
        if self.first_point_jump_rad <= 0:
            raise ValueError("first_point_jump_rad must be positive")
        if not 0 < self.latency_ewma_alpha <= 1:
            raise ValueError("latency_ewma_alpha must be in (0, 1]")


class _ActionStreamPausedError(RuntimeError):
    """Request a fresh inference session after the robot deadman pauses."""


def packet_to_snapshot(
    packet: RobotObservationPacket,
) -> tuple[TimedValue, TimedValue, TimedValue, np.ndarray, np.ndarray]:
    """Convert the HTTP wire packet into the existing inference boundary."""
    received = time.monotonic()
    return (
        TimedValue(packet.color_timestamp_ns, packet.color_data, received),
        TimedValue(packet.depth_timestamp_ns, packet.depth_data, received),
        TimedValue(
            packet.joint_timestamp_ns,
            np.asarray(packet.joint_positions, dtype=np.float64),
            received,
        ),
        parse_camera_calibration(packet.calibration_json),
        parse_robot_camera_extrinsics(packet.extrinsics_json),
    )


def _request_observation(
    session: requests.Session,
    config: HcTjHttpAsyncClientConfig,
    after: int,
) -> RobotObservationPacket | None:
    response = session.get(
        config.robot_url + OBSERVATION_PATH,
        params={"after": after, "wait_s": config.observation_wait_s},
        headers={"Authorization": f"Bearer {config.robot_token}"},
        timeout=config.observation_wait_s + 2.0,
    )
    if response.status_code == 204:
        return None
    if not response.ok:
        raise RuntimeError(
            f"robot observation HTTP {response.status_code}: {response.text}"
        )
    content_type = response.headers.get("Content-Type", "").split(";", 1)[0]
    if content_type != OBSERVATION_CONTENT_TYPE:
        raise RuntimeError(
            f"unexpected robot observation Content-Type: {content_type}"
        )
    return decode_observation(response.content)


def _request_inference(
    session: requests.Session,
    config: HcTjHttpAsyncClientConfig,
    files: dict[str, tuple[str, bytes, str]],
    form: dict[str, str],
) -> tuple[np.ndarray, float]:
    started = time.monotonic()
    response = session.post(
        config.server_url,
        files=files,
        data=form,
        timeout=config.request_timeout_s,
    )
    elapsed = time.monotonic() - started
    if not response.ok:
        raise RuntimeError(
            f"inference HTTP {response.status_code}: {response.text}"
        )
    return validate_inference_response(response.json()), elapsed


def _request_joint_feedback(
    session: requests.Session,
    config: HcTjHttpAsyncClientConfig,
) -> RobotJointFeedbackPacket:
    """Read the gateway's latest feedback after inference completes."""
    response = session.get(
        config.robot_url + JOINT_FEEDBACK_PATH,
        headers={"Authorization": f"Bearer {config.robot_token}"},
        timeout=3.0,
    )
    if not response.ok:
        raise RuntimeError(
            f"robot joint feedback HTTP {response.status_code}: "
            f"{response.text}"
        )
    content_type = response.headers.get("Content-Type", "").split(";", 1)[0]
    if content_type != JOINT_FEEDBACK_CONTENT_TYPE:
        raise RuntimeError(
            f"unexpected robot joint feedback Content-Type: {content_type}"
        )
    return decode_joint_feedback(response.content)


def _send_action_chunk(
    session: requests.Session,
    config: HcTjHttpAsyncClientConfig,
    sequence_id: int,
    actions: np.ndarray,
) -> dict[str, Any]:
    response = session.post(
        config.robot_url + ACTION_PATH,
        data=encode_action_chunk(sequence_id, actions.astype(np.float32)),
        headers={
            "Authorization": f"Bearer {config.robot_token}",
            "Content-Type": "application/json",
        },
        timeout=5.0,
    )
    if not response.ok:
        try:
            error_data = response.json()
        except requests.exceptions.JSONDecodeError:
            error_data = {}
        error_message = str(error_data.get("error", response.text))
        if response.status_code == 409 and (
            error_data.get("error_type") == "action_stream_paused"
            or "action stream is paused" in error_message
        ):
            raise _ActionStreamPausedError(error_message)
        raise RuntimeError(
            f"robot action HTTP {response.status_code}: {response.text}"
        )
    data: dict[str, Any] = response.json()
    if not data.get("accepted") or data.get("sequence_id") != sequence_id:
        raise RuntimeError("robot gateway returned an invalid action receipt")
    return data


def _check_gateway(
    session: requests.Session, config: HcTjHttpAsyncClientConfig
) -> dict[str, Any]:
    response = session.get(config.robot_url + HEALTH_PATH, timeout=3.0)
    if not response.ok:
        raise RuntimeError(f"robot gateway health HTTP {response.status_code}")
    status = response.json()
    if status.get("status") != "ok":
        raise RuntimeError(f"robot gateway is faulted: {status.get('fault')}")
    if config.enable_control and not status.get("control_enabled"):
        raise RuntimeError(
            "local control requested but robot gateway control is disabled"
        )
    if config.enable_control and status.get("control_state") == "paused":
        print(
            "Robot gateway is paused; this new client session will re-arm "
            "only after all action safety checks pass.",
            flush=True,
        )
    return status


def run_client(config: HcTjHttpAsyncClientConfig) -> None:
    """Continuously fetch observations, infer, and optionally send actions."""
    robot_session = requests.Session()
    inference_session = requests.Session()
    gateway_status = _check_gateway(robot_session, config)
    hands_enabled = bool(gateway_status.get("hands_enabled"))
    last_observation_id = 0
    last_inference_started = -math.inf
    native_actions: np.ndarray | None = None
    native_started = 0.0
    latency_ewma: float | None = None
    sequence_id = 0
    right_hand_observation = 0.0
    left_hand_observation = 0.0
    mode = "CONTROL" if config.enable_control else "DRY-RUN"
    print(f"HC-TJ HTTP client started in {mode} mode", flush=True)

    while True:
        wait = 1.0 / config.inference_hz - (
            time.monotonic() - last_inference_started
        )
        if wait > 0:
            time.sleep(wait)
        packet = _request_observation(
            robot_session, config, last_observation_id
        )
        if packet is None:
            continue
        last_observation_id = packet.observation_id
        snapshot = packet_to_snapshot(packet)
        now = time.monotonic()
        remaining = None
        if native_actions is not None:
            remaining = trim_elapsed_actions(
                native_actions, max(0.0, now - native_started)
            )
            if not len(remaining):
                remaining = None
            elif hands_enabled:
                right_hand_observation = float(np.clip(remaining[0, 7], 0, 1))
                left_hand_observation = float(np.clip(remaining[0, 15], 0, 1))
        delay_horizon = int(math.ceil((latency_ewma or 0.0) * ACTION_HZ))
        if remaining is not None:
            delay_horizon = min(delay_horizon, len(remaining))
        files, form, _ = _build_request(
            snapshot,
            config,
            right_hand_observation,
            left_hand_observation,
            remaining,
            delay_horizon,
        )
        last_inference_started = time.monotonic()
        actions, elapsed = _request_inference(
            inference_session, config, files, form
        )
        feedback_request_started = time.monotonic()
        fresh_feedback = _request_joint_feedback(robot_session, config)
        action_delay = elapsed + (time.monotonic() - feedback_request_started)
        executable = trim_elapsed_actions(actions, action_delay)
        if not len(executable):
            raise RuntimeError(
                "inference latency expired the entire action chunk"
            )
        feedback = np.asarray(
            fresh_feedback.joint_positions[:14], dtype=np.float64
        )
        validate_action_chunk(
            executable,
            feedback,
            first_point_jump_rad=config.first_point_jump_rad,
        )
        receipt = None
        if config.enable_control:
            sequence_id += 1
            try:
                receipt = _send_action_chunk(
                    robot_session,
                    config,
                    sequence_id,
                    executable,
                )
            except _ActionStreamPausedError:
                print(
                    "Robot deadman paused control; discarded the stale "
                    "action chunk and starting a fresh inference session.",
                    flush=True,
                )
                sequence_id = 0
                native_actions = None
                native_started = 0.0
                latency_ewma = None
                continue
        alpha = config.latency_ewma_alpha
        latency_ewma = (
            elapsed
            if latency_ewma is None
            else alpha * elapsed + (1 - alpha) * latency_ewma
        )
        native_actions = executable.copy()
        native_started = time.monotonic()
        suffix = (
            f", gateway_steps={receipt['control_steps']}"
            if receipt is not None
            else ""
        )
        print(
            f"observation={packet.observation_id}, "
            f"inference={elapsed * 1000:.1f} ms, "
            f"actions={len(executable)}{suffix}",
            flush=True,
        )


def _parse_args() -> HcTjHttpAsyncClientConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-url", required=True)
    parser.add_argument(
        "--robot-token",
        default=os.environ.get("HC_TJ_GATEWAY_TOKEN"),
        help="Bearer token; defaults to HC_TJ_GATEWAY_TOKEN",
    )
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--depth-units-per-meter", type=float, required=True)
    parser.add_argument("--enable-control", action="store_true")
    parser.add_argument("--inference-hz", type=float, default=2.0)
    parser.add_argument("--request-timeout-s", type=float, default=30.0)
    parser.add_argument("--observation-wait-s", type=float, default=2.0)
    parser.add_argument("--first-point-jump-rad", type=float, default=0.30)
    parser.add_argument("--latency-ewma-alpha", type=float, default=0.25)
    args = parser.parse_args()
    if not args.robot_token:
        parser.error("--robot-token or HC_TJ_GATEWAY_TOKEN is required")
    return HcTjHttpAsyncClientConfig(**vars(args))


def main() -> None:
    """Run until interrupted or a protocol/safety fault occurs."""
    try:
        run_client(_parse_args())
    except KeyboardInterrupt:
        print("HC-TJ HTTP client stopped", flush=True)


if __name__ == "__main__":
    main()
