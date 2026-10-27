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

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from projects.holobrain.scripts.hc_tj_async_client import (
    DEFAULT_JOINT_POSITIONS,
    HcTjAsyncClientConfig,
    ObservationBuffer,
    _other_publisher_names,
    _state_is_stale,
)
from projects.holobrain.scripts.hc_tj_async_core import (
    ACTION_HZ,
    ARM_JOINT_NAMES,
    CONTROL_JOINT_NAMES,
    HC_TJ_JOINT_NAMES,
    LEFT_HAND_CLOSED,
    LEFT_HAND_OPEN,
    RIGHT_HAND_CLOSED,
    RIGHT_HAND_OPEN,
    ArmCommandLimiter,
    ProtocolError,
    TimedValue,
    arm_actions,
    hand_command,
    interpolate_actions,
    load_arm_joint_limits,
    nearest_timed_value,
    parse_camera_calibration,
    parse_robot_camera_extrinsics,
    trim_elapsed_actions,
    validate_action_chunk,
    validate_inference_request,
    validate_inference_response,
)


@pytest.fixture
def urdf_path(tmp_path: Path) -> Path:
    path = tmp_path / "arms.urdf"
    joints = "".join(
        f'<link name="{name}"/><joint name="{name}" type="revolute">'
        f'<parent link="base"/><child link="{name}"/>'
        '<limit lower="-3.14" upper="3.14" effort="10" velocity="1"/>'
        "</joint>"
        for name in ARM_JOINT_NAMES
    )
    path.write_text(
        '<robot name="test"><link name="base"/>' + joints + "</robot>"
    )
    return path


def _request_kwargs() -> dict:
    return {
        "head_color": np.zeros((4, 6, 3), dtype=np.uint8),
        "head_depth": np.ones((4, 6), dtype=np.float32),
        "head_intrinsic_mat": np.eye(4, dtype=np.float64),
        "zhi_to_eye_mat": np.eye(4, dtype=np.float64),
        "current_state": np.zeros(16, dtype=np.float32),
        "joint_names": HC_TJ_JOINT_NAMES,
        "instruction": "pick the block",
    }


class TestProtocol:
    def test_validates_16_axis_request_and_rtc(self):
        kwargs = _request_kwargs()
        kwargs["remaining_actions"] = np.zeros((5, 16), dtype=np.float32)
        kwargs["delay_horizon"] = 2

        value = validate_inference_request(**kwargs)

        assert value.current_state.shape == (16,)
        assert value.remaining_actions.shape == (5, 16)
        assert value.delay_horizon == 2

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("head_color", np.zeros((4, 6, 3), dtype=np.float32)),
            ("head_depth", np.zeros((4, 7), dtype=np.float32)),
            ("current_state", np.zeros(14, dtype=np.float32)),
            ("joint_names", tuple(reversed(HC_TJ_JOINT_NAMES))),
        ],
    )
    def test_rejects_wrong_wire_contract(self, field, value):
        kwargs = _request_kwargs()
        kwargs[field] = value
        with pytest.raises(ProtocolError):
            validate_inference_request(**kwargs)

    def test_rejects_remaining_actions_without_valid_delay(self):
        kwargs = _request_kwargs()
        kwargs["remaining_actions"] = np.zeros((5, 16), dtype=np.float32)
        kwargs["delay_horizon"] = -1
        with pytest.raises(ProtocolError, match="delay_horizon"):
            validate_inference_request(**kwargs)

    def test_validates_response_metadata_and_finite_actions(self):
        actions = np.zeros((64, 16), dtype=np.float32)
        payload = {
            "actions": actions.tolist(),
            "joint_names": list(HC_TJ_JOINT_NAMES),
            "action_hz": ACTION_HZ,
            "action_horizon": 64,
            "inference_ms": 12.5,
        }
        assert validate_inference_response(payload).shape == (64, 16)
        payload["actions"][0][0] = float("nan")
        with pytest.raises(ProtocolError, match="finite"):
            validate_inference_response(payload)


class TestTrajectoryAndSafety:
    def test_slices_right_arm_then_left_arm(self):
        action = np.arange(16)
        assert arm_actions(action).tolist() == [*range(7), *range(8, 15)]

    def test_trims_elapsed_prefix_with_ceiling(self):
        actions = np.arange(80).reshape(5, 16)
        assert np.array_equal(trim_elapsed_actions(actions, 0), actions)
        assert np.array_equal(
            trim_elapsed_actions(actions, 0.034), actions[1:]
        )
        assert np.array_equal(
            trim_elapsed_actions(actions, 0.067), actions[2:]
        )

    def test_interpolates_30_to_100_hz_for_full_duration(self):
        actions = np.stack((np.zeros(16), np.ones(16)))
        output = interpolate_actions(actions)
        assert output.shape == (7, 16)
        assert output[0, 0] == pytest.approx(0)
        assert output[3, 0] == pytest.approx(0.9)
        assert output[-1, 0] == pytest.approx(1.0)

    def test_loads_urdf_limits_and_rate_limits(self, urdf_path):
        lower, upper = load_arm_joint_limits(urdf_path)
        assert lower.shape == upper.shape == (len(ARM_JOINT_NAMES),)
        limiter = ArmCommandLimiter(lower, upper, max_velocity_rad_s=0.25)
        limiter.reset(np.zeros(14))
        command = limiter.step(np.full(14, 100.0), 0.01)
        assert np.max(np.abs(command)) == pytest.approx(0.0025)
        assert np.all(command <= upper)

    def test_rejects_first_point_jump_and_nonfinite_action(self):
        actions = np.zeros((3, 16), dtype=np.float32)
        actions[0, 0] = 0.101
        with pytest.raises(ValueError, match="jumps"):
            validate_action_chunk(actions, np.zeros(14))
        actions[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            validate_action_chunk(actions, np.zeros(14))

    @pytest.mark.parametrize(
        ("side", "opened", "closed"),
        [
            ("right", RIGHT_HAND_OPEN, RIGHT_HAND_CLOSED),
            ("left", LEFT_HAND_OPEN, LEFT_HAND_CLOSED),
        ],
    )
    def test_omnihand_endpoint_mapping(self, side, opened, closed):
        names, open_command = hand_command(side, 0)
        _, closed_command = hand_command(side, 1)
        assert len(names) == 10
        assert np.allclose(open_command, opened)
        assert np.allclose(closed_command, closed)
        _, clipped = hand_command(side, 2)
        assert np.allclose(clipped, closed)

    @pytest.mark.parametrize(
        ("side", "input_ranges", "output_ranges"),
        [
            (
                "right",
                [
                    (0.4, 1.121),
                    (-1.642, 0.045),
                    (0.0, 1.0),
                    (-0.13, -0.05),
                    (0.0, 2.5),
                    (0.0, 2.5),
                    (0.05, 0.13),
                    (0.0, 2.5),
                    (0.05, 0.13),
                    (0.0, 2.5),
                ],
                [
                    (4096, 800),
                    (4096, 0),
                    (4096, 0),
                    (4096, 0),
                    (4096, 0),
                    (4096, 0),
                    (0, 4096),
                    (4096, 0),
                    (4096, 0),
                    (4096, 0),
                ],
            ),
            (
                "left",
                [
                    (-0.4, -1.121),
                    (-0.045, 1.642),
                    (0.0, -1.0),
                    (0.05, 0.13),
                    (0.0, 2.5),
                    (0.0, 2.5),
                    (-0.13, -0.05),
                    (0.0, 2.5),
                    (-0.13, 0.05),
                    (0.0, 2.5),
                ],
                [
                    (4096, 800),
                    (4096, 0),
                    (4096, 0),
                    (4096, 0),
                    (4096, 0),
                    (4096, 0),
                    (0, 4096),
                    (4096, 0),
                    (4096, 0),
                    (4096, 0),
                ],
            ),
        ],
    )
    def test_omnihand_commands_match_driver_calibration(
        self, side, input_ranges, output_ranges
    ):
        expected_open = np.asarray(
            [3800, 4096, 4096, 1899, 4096, 4096, 4096, 4096, 0, 4096]
        )
        expected_fist = np.asarray(
            [3800, 4096, 66, 1899, 2300, 2300, 1932, 604, 2094, 408]
        )

        def driver_motor_positions(command):
            positions = []
            for value, input_range, output_range in zip(
                command, input_ranges, output_ranges, strict=True
            ):
                fraction = np.clip(
                    (value - input_range[0])
                    / (input_range[1] - input_range[0]),
                    0.0,
                    1.0,
                )
                positions.append(
                    output_range[0]
                    + fraction * (output_range[1] - output_range[0])
                )
            return np.asarray(positions)

        _, open_command = hand_command(side, 0.0)
        _, fist_command = hand_command(side, 1.0)
        assert driver_motor_positions(open_command) == pytest.approx(
            expected_open
        )
        assert driver_motor_positions(fist_command) == pytest.approx(
            expected_fist
        )

    def test_default_posture_matches_the_17_axis_control_order(self):
        assert len(DEFAULT_JOINT_POSITIONS) == len(CONTROL_JOINT_NAMES)
        assert CONTROL_JOINT_NAMES == (
            "Joint1_R",
            "Joint2_R",
            "Joint3_R",
            "Joint4_R",
            "Joint5_R",
            "Joint6_R",
            "Joint7_R",
            "Joint1_L",
            "Joint2_L",
            "Joint3_L",
            "Joint4_L",
            "Joint5_L",
            "Joint6_L",
            "Joint7_L",
            "leg_1",
            "leg_2",
            "zhi",
        )
        assert DEFAULT_JOINT_POSITIONS == pytest.approx(
            (
                -1.130321733036805,
                -0.7488408179558761,
                1.2421810235648036,
                -1.5615996326025259,
                0.14459746322931552,
                0.436243163681092,
                -0.03755520693488968,
                1.4708313794854644,
                -0.7665212115377759,
                -1.5988311572740517,
                -1.612405156192131,
                0.03962631818631988,
                0.13397872423862664,
                -0.010267844630530324,
                0.21072105306881997,
                0.6668054431158328,
                0.15834697039601572,
            )
        )


class TestObservationAndConfiguration:
    def test_latest_joints_preserves_local_receive_time(self, monkeypatch):
        buffer = ObservationBuffer()

        with pytest.raises(ValueError, match="feedback is unavailable"):
            buffer.latest_joints()

        monotonic_values = iter((10.0, 20.0))
        monkeypatch.setattr(
            "projects.holobrain.scripts.hc_tj_async_client.time.monotonic",
            lambda: next(monotonic_values),
        )
        expected = np.arange(17, dtype=np.float64)
        buffer.add_joints(1_000_000_000, expected)
        latest = buffer.latest_joints()
        assert latest.timestamp_ns == 1_000_000_000
        assert latest.received_monotonic_s == 10.0
        assert np.array_equal(latest.value, expected)

    def test_nearest_timestamp_prefers_earlier_and_enforces_tolerance(self):
        values = [TimedValue(90, "early"), TimedValue(110, "late")]
        assert nearest_timed_value(values, 100, 10).value == "early"
        with pytest.raises(ValueError, match="outside"):
            nearest_timed_value(values, 200, 20)

    def test_parses_calibration_and_zhi_to_eye_extrinsics(self):
        intrinsic = parse_camera_calibration(
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
        zhi_to_eye = parse_robot_camera_extrinsics(
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
        assert np.array_equal(intrinsic[3], [0, 0, 0, 1])
        assert np.allclose(zhi_to_eye[:3, 3], [-1, -2, -3])

    def test_buffer_synchronizes_on_rgb_and_client_defaults_to_dry_run(
        self, urdf_path
    ):
        buffer = ObservationBuffer()
        calibration = {
            "schema": "hc_camera_calibration/v1",
            "camera": "eye",
            "resolution": {
                "color": {"width": 6, "height": 4},
                "depth": {"width": 6, "height": 4},
            },
            "rgb": {"fx": 4, "fy": 5, "cx": 3, "cy": 2},
        }
        extrinsics = {
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
        buffer.set_calibration(json.dumps(calibration))
        buffer.set_extrinsics(json.dumps(extrinsics))
        buffer.add_color(1_000_000_000, b"jpeg")
        buffer.add_depth(1_049_000_000, b"png")
        buffer.add_joints(1_019_000_000, np.zeros(17))
        snapshot = buffer.synchronized()
        assert len(snapshot) == 5
        assert snapshot[0].timestamp_ns == 1_000_000_000
        assert snapshot[1].timestamp_ns == 1_049_000_000
        assert snapshot[2].timestamp_ns == 1_019_000_000

        config = HcTjAsyncClientConfig(
            server_url="http://localhost/holobrain/hc_tj_async/v1",
            instruction="test",
            urdf_path=urdf_path,
            depth_units_per_meter=1000,
        )
        assert config.enable_control is False
        assert config.camera_timeout_s == 1.0
        assert config.color_topic == (
            "/hc_teleop/camera_head/color/compressed"
        )
        assert config.depth_topic == (
            "/hc_teleop/camera_head/depth/compressed"
        )
        assert config.joint_state_topic == "/hc_teleop/joint_states"
        assert config.joint_command_topic == "/hc_teleop/joint_cmd"
        assert config.right_hand_topic == ("/hc_teleop/joint_cmd_finger_right")
        assert config.left_hand_topic == ("/hc_teleop/joint_cmd_finger_left")

    def test_state_timeout_boundary(self):
        assert not _state_is_stale(10.25, 10.0, 0.25)
        assert _state_is_stale(10.250001, 10.0, 0.25)

    def test_competing_publisher_filter_excludes_only_self(self):
        infos = [
            SimpleNamespace(node_name="client", node_namespace="/robot"),
            SimpleNamespace(node_name="other", node_namespace="/robot"),
        ]

        publishers = _other_publisher_names(infos, "client", "/robot")

        assert publishers == ["/robot/other"]
