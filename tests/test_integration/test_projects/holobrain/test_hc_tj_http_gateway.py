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

import json
import sys

import numpy as np
import pytest

from projects.holobrain.scripts import hc_tj_http_async_client as http_client
from projects.holobrain.scripts.hc_tj_async_core import (
    ARM_JOINT_NAMES,
    CONTROL_JOINT_NAMES,
    ProtocolError,
)
from projects.holobrain.scripts.hc_tj_http_async_client import (
    HcTjHttpAsyncClientConfig,
    _parse_args,
    packet_to_snapshot,
)
from projects.holobrain.scripts.hc_tj_http_protocol import (
    RobotJointFeedbackPacket,
    RobotObservationPacket,
    decode_action_chunk,
    decode_joint_feedback,
    decode_observation,
    encode_action_chunk,
    encode_joint_feedback,
    encode_observation,
)
from projects.holobrain.scripts.hc_tj_http_robot_gateway import (
    GatewayConfig,
    RobotObservationStore,
    _action_stream_timed_out,
    _ActionStreamPausedError,
    _ActionStreamSchedule,
    _arm_joint_command,
)


def _calibration_json() -> str:
    return json.dumps(
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


def _extrinsics_json() -> str:
    return json.dumps(
        {
            "schema": "hc_robot_camera_extrinsics/v1",
            "common_frame": "zhi_Link",
            "transforms": [
                {
                    "parent_frame": "zhi_Link",
                    "child_frame": "camera_head",
                    "translation_m": [0, 0, 0],
                    "rotation_xyzw": [0, 0, 0, 1],
                }
            ],
        }
    )


def _packet() -> RobotObservationPacket:
    return RobotObservationPacket(
        observation_id=1_000_000_000,
        color_timestamp_ns=1_000_000_000,
        color_format="jpeg",
        color_data=b"jpeg-data",
        depth_timestamp_ns=1_010_000_000,
        depth_format="16UC1; png",
        depth_data=b"png-data",
        joint_timestamp_ns=1_005_000_000,
        joint_names=CONTROL_JOINT_NAMES,
        joint_positions=tuple(np.arange(17, dtype=np.float64)),
        calibration_json=_calibration_json(),
        extrinsics_json=_extrinsics_json(),
    )


class TestHttpProtocol:
    def test_joint_feedback_round_trip(self):
        packet = RobotJointFeedbackPacket(
            timestamp_ns=1_000_000_000,
            joint_names=CONTROL_JOINT_NAMES,
            joint_positions=tuple(np.arange(17, dtype=np.float64)),
        )

        assert decode_joint_feedback(encode_joint_feedback(packet)) == packet

    def test_observation_binary_round_trip(self):
        packet = _packet()

        decoded = decode_observation(encode_observation(packet))

        assert decoded == packet

    def test_observation_rejects_trailing_bytes(self):
        with pytest.raises(ProtocolError, match="lengths"):
            decode_observation(encode_observation(_packet()) + b"extra")

    def test_action_chunk_round_trip_preserves_float32(self):
        actions = np.arange(48, dtype=np.float32).reshape(3, 16)

        sequence_id, decoded = decode_action_chunk(
            encode_action_chunk(7, actions)
        )

        assert sequence_id == 7
        assert decoded.dtype == np.float32
        np.testing.assert_array_equal(decoded, actions)

    def test_action_chunk_requires_native_shape_and_dtype(self):
        with pytest.raises(ProtocolError, match="float32"):
            encode_action_chunk(0, np.zeros((3, 16), dtype=np.float64))


class TestObservationStore:
    def test_selects_synchronized_newest_frame_once(self):
        store = RobotObservationStore()
        store.set_calibration(_calibration_json())
        store.set_extrinsics(_extrinsics_json())
        store.add_depth(1_010_000_000, "16UC1; png", b"png")
        store.add_joints(
            1_005_000_000,
            CONTROL_JOINT_NAMES,
            np.zeros(17),
        )
        store.add_color(1_000_000_000, "jpeg", b"jpeg")

        packet = store.wait_for_snapshot(0, 0)

        assert packet is not None
        assert packet.observation_id == 1_000_000_000
        assert store.wait_for_snapshot(packet.observation_id, 0) is None
        assert store.status() == {
            "color_frames": 1,
            "depth_frames": 1,
            "joint_samples": 1,
            "calibration": True,
            "extrinsics": True,
            "newest_color_depth_delta_ms": 10.0,
            "newest_color_joint_delta_ms": 5.0,
        }

    def test_rejects_out_of_window_depth(self):
        store = RobotObservationStore()
        store.set_calibration(_calibration_json())
        store.set_extrinsics(_extrinsics_json())
        store.add_depth(1_051_000_000, "16UC1; png", b"png")
        store.add_joints(
            1_005_000_000,
            CONTROL_JOINT_NAMES,
            np.zeros(17),
        )
        store.add_color(1_000_000_000, "jpeg", b"jpeg")

        assert store.wait_for_snapshot(0, 0) is None


class TestHttpClientBoundary:
    def test_validates_actions_against_post_inference_feedback(
        self, monkeypatch
    ):
        config = HcTjHttpAsyncClientConfig(
            robot_url="http://robot:8765",
            robot_token="0123456789abcdef",
            server_url="http://gpu:2000/holobrain/hc_tj_async/v1",
            instruction="test",
            depth_units_per_meter=1000,
            enable_control=True,
            first_point_jump_rad=0.3,
        )
        observation_calls = 0
        sent_actions: list[np.ndarray] = []

        def request_observation(*_args, **_kwargs):
            nonlocal observation_calls
            observation_calls += 1
            if observation_calls > 1:
                raise StopIteration
            return _packet()

        monkeypatch.setattr(http_client.requests, "Session", object)
        monkeypatch.setattr(
            http_client,
            "_check_gateway",
            lambda *_args: {"control_enabled": True},
        )
        monkeypatch.setattr(
            http_client, "_request_observation", request_observation
        )
        monkeypatch.setattr(
            http_client,
            "_build_request",
            lambda *_args: ({}, {}, None),
        )
        monkeypatch.setattr(
            http_client,
            "_request_inference",
            lambda *_args: (np.zeros((64, 16)), 0.0),
        )
        monkeypatch.setattr(
            http_client,
            "_request_joint_feedback",
            lambda *_args: RobotJointFeedbackPacket(
                timestamp_ns=1_100_000_000,
                joint_names=CONTROL_JOINT_NAMES,
                joint_positions=tuple(np.zeros(17)),
            ),
        )
        monkeypatch.setattr(
            http_client,
            "_send_action_chunk",
            lambda _session, _config, _sequence_id, actions: (
                sent_actions.append(actions.copy()) or {"control_steps": 1}
            ),
        )
        monkeypatch.setattr(http_client.time, "sleep", lambda _seconds: None)

        with pytest.raises(StopIteration):
            http_client.run_client(config)

        assert len(sent_actions) == 1

    def test_client_automatically_starts_fresh_session_after_pause(
        self, monkeypatch
    ):
        config = HcTjHttpAsyncClientConfig(
            robot_url="http://robot:8765",
            robot_token="0123456789abcdef",
            server_url="http://gpu:2000/holobrain/hc_tj_async/v1",
            instruction="test",
            depth_units_per_meter=1000,
            enable_control=True,
            first_point_jump_rad=0.3,
        )
        observation_calls = 0
        sequence_ids: list[int] = []

        def request_observation(*_args, **_kwargs):
            nonlocal observation_calls
            observation_calls += 1
            if observation_calls > 2:
                raise StopIteration
            return _packet()

        def send_action(_session, _config, sequence_id, _actions):
            sequence_ids.append(sequence_id)
            if len(sequence_ids) == 1:
                raise http_client._ActionStreamPausedError("paused")
            return {"control_steps": 1}

        monkeypatch.setattr(http_client.requests, "Session", object)
        monkeypatch.setattr(
            http_client,
            "_check_gateway",
            lambda *_args: {"control_enabled": True},
        )
        monkeypatch.setattr(
            http_client, "_request_observation", request_observation
        )
        monkeypatch.setattr(
            http_client,
            "_build_request",
            lambda *_args: ({}, {}, None),
        )
        monkeypatch.setattr(
            http_client,
            "_request_inference",
            lambda *_args: (np.zeros((64, 16)), 0.0),
        )
        monkeypatch.setattr(
            http_client,
            "_request_joint_feedback",
            lambda *_args: RobotJointFeedbackPacket(
                timestamp_ns=1_100_000_000,
                joint_names=CONTROL_JOINT_NAMES,
                joint_positions=tuple(np.zeros(17)),
            ),
        )
        monkeypatch.setattr(http_client, "_send_action_chunk", send_action)
        monkeypatch.setattr(http_client.time, "sleep", lambda _seconds: None)

        with pytest.raises(StopIteration):
            http_client.run_client(config)

        assert sequence_ids == [1, 1]

    def test_parses_latency_ewma_alpha(self, monkeypatch):
        monkeypatch.setenv("HC_TJ_GATEWAY_TOKEN", "0123456789abcdef")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "hc_tj_http_async_client.py",
                "--robot-url",
                "http://robot:8765",
                "--server-url",
                "http://gpu:2000/holobrain/hc_tj_async/v1",
                "--instruction",
                "test",
                "--depth-units-per-meter",
                "1000",
                "--latency-ewma-alpha",
                "0.8",
            ],
        )

        assert _parse_args().latency_ewma_alpha == 0.8

    def test_accepts_eight_hz_inference_rate(self):
        client = HcTjHttpAsyncClientConfig(
            robot_url="http://robot:8765",
            robot_token="0123456789abcdef",
            server_url="http://gpu:2000/holobrain/hc_tj_async/v1",
            instruction="test",
            depth_units_per_meter=1000,
            inference_hz=8.0,
        )

        assert client.inference_hz == 8.0

    def test_rejects_inference_rate_above_fifty_hz(self):
        with pytest.raises(
            ValueError, match=r"inference_hz must be in \(0, 50\]"
        ):
            HcTjHttpAsyncClientConfig(
                robot_url="http://robot:8765",
                robot_token="0123456789abcdef",
                server_url="http://gpu:2000/holobrain/hc_tj_async/v1",
                instruction="test",
                depth_units_per_meter=1000,
                inference_hz=50.1,
            )

    def test_arm_command_never_contains_waist_axes(self):
        names, positions = _arm_joint_command(np.arange(14))

        assert names == ARM_JOINT_NAMES
        assert len(positions) == 14
        assert not {"leg_1", "leg_2", "zhi"} & set(names)

    def test_action_stream_deadman_boundary(self):
        assert not _action_stream_timed_out(10.75, 10.0, 0.75)
        assert _action_stream_timed_out(10.750001, 10.0, 0.75)
        assert _action_stream_timed_out(10.0, None, 0.75)

    def test_action_stream_timeout_pauses_until_new_client_session(self):
        stream = _ActionStreamSchedule()
        actions = np.zeros((3, 16))
        assert not stream.pause_if_timed_out(100.0, 0.75)

        stream.schedule(7, actions, 10.0)

        assert stream.pause_if_timed_out(10.750001, 0.75)
        assert stream.control_state == "paused"
        assert stream.actions is None
        assert stream.last_stop_reason == "action stream heartbeat timed out"
        with pytest.raises(_ActionStreamPausedError, match="fresh action"):
            stream.schedule(8, actions, 11.0)

        stream.schedule(1, actions, 11.0)

        assert stream.control_state == "active"
        assert stream.last_sequence_id == 1
        assert stream.last_stop_reason is None

    def test_action_stream_resets_limiter_only_at_session_boundaries(self):
        stream = _ActionStreamSchedule()
        actions = np.zeros((3, 16))

        assert stream.schedule(1, actions, 10.0) is True
        assert stream.schedule(2, actions, 10.5) is False
        assert stream.pause_if_timed_out(11.250001, 0.75)
        assert stream.schedule(1, actions, 12.0) is True

    def test_packet_maps_to_existing_inference_snapshot(self):
        snapshot = packet_to_snapshot(_packet())

        assert len(snapshot) == 5
        assert snapshot[0].value == b"jpeg-data"
        assert snapshot[1].value == b"png-data"
        np.testing.assert_array_equal(snapshot[2].value, np.arange(17))
        assert snapshot[3].shape == (4, 4)
        assert snapshot[4].shape == (4, 4)

    def test_both_sides_default_to_dry_run(self):
        gateway = GatewayConfig(token="0123456789abcdef")
        client = HcTjHttpAsyncClientConfig(
            robot_url="http://robot:8765",
            robot_token="0123456789abcdef",
            server_url="http://gpu:2000/holobrain/hc_tj_async/v1",
            instruction="test",
            depth_units_per_meter=1000,
        )

        assert gateway.enable_control is False
        assert gateway.enable_hands is False
        assert gateway.action_timeout_s == 0.75
        assert gateway.first_point_jump_rad == 0.3
        assert client.enable_control is False
        assert client.inference_hz == 2.0
        assert client.latency_ewma_alpha == 0.25
        assert client.first_point_jump_rad == 0.3
