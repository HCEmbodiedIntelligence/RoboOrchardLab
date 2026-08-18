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

"""Replay HC-TJ MCAP arm joints through the local URDF.

Use ``--source command`` or ``--source state`` to inspect either teleoperation
commands or measured states. Both topics are always decoded and summarized.

# example usage
```bash
python own_test/replay_hc_tj_mcap_urdf.py  \
/home/hc-em/workspace/workspaces/RoboOrchardLab/own_test/dataset_mcap/hc_tj_data_0812/hc_tj_description_2608
12_134826_0.mcap   \
--urdf /data/workspaces/RoboOrchardLab/hc_assets/hc_tj_description/urdf/hc_tj_robot.urdf  \
--source state  \
--loop
```
"""

from __future__ import annotations

import argparse
import bisect
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = ["main"]

JOINT_STATE_TOPIC = "io_teleop/joint_states"
JOINT_COMMAND_TOPIC = "io_teleop/joint_cmd"
ARM_JOINT_NAMES = tuple(
    [f"Joint{index}_R" for index in range(1, 8)]
    + [f"Joint{index}_L" for index in range(1, 8)]
)
EXPECTED_SOURCE_JOINT_NAMES = ARM_JOINT_NAMES + ("leg_1", "leg_2", "zhi")
TORSO_JOINT_NAMES = ("leg_1", "leg_2", "zhi")
DEFAULT_URDF_PATH = (
    Path(__file__).resolve().parents[3]
    / "hc_assets/hc_tj_description/urdf/hc_tj_robot.urdf"
)


@dataclass(frozen=True, slots=True)
class JointSample:
    """One 17-DoF arm-and-torso sample from a ROS JointState message."""

    timestamp_ns: int
    position: np.ndarray


def main() -> None:
    """Parse arguments, validate samples, and optionally render a replay."""
    args = _parse_args()
    mcap_path = _resolve_mcap_path(args.mcap)
    command_samples, state_samples = _load_joint_samples(mcap_path)
    _print_summary(mcap_path, command_samples, state_samples)
    if args.no_gui:
        return
    _replay(
        command_samples,
        state_samples,
        urdf_path=args.urdf,
        playback_rate=args.playback_rate,
        loop=args.loop,
        source=args.source,
    )


def _parse_args() -> argparse.Namespace:
    """Return CLI arguments for one MCAP replay session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mcap",
        type=Path,
        help="An HC-TJ .mcap file, or a directory containing exactly one to replay.",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_URDF_PATH,
        help=f"HC-TJ URDF path (default: {DEFAULT_URDF_PATH}).",
    )
    parser.add_argument(
        "--playback-rate",
        type=float,
        default=1.0,
        help="Replay speed multiplier; must be positive (default: 1.0).",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Replay repeatedly until the viewer is closed or Ctrl+C is pressed.",
    )
    parser.add_argument(
        "--source",
        choices=("command", "state"),
        default="command",
        help="Joint topic to drive the rendered robot (default: command).",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Only decode and print a summary; useful on headless machines.",
    )
    args = parser.parse_args()
    if args.playback_rate <= 0:
        parser.error("--playback-rate must be positive")
    if not args.urdf.is_file():
        parser.error(f"URDF does not exist: {args.urdf}")
    return args


def _resolve_mcap_path(path: Path) -> Path:
    """Resolve one input MCAP file and reject ambiguous directories."""
    if path.is_file() and path.suffix == ".mcap":
        return path
    if path.is_dir():
        files = sorted(path.glob("*.mcap"))
        if len(files) == 1:
            return files[0]
        if not files:
            raise FileNotFoundError(f"No .mcap file found in {path}")
        raise ValueError(
            f"{path} contains {len(files)} MCAP files; pass the one to replay"
        )
    raise FileNotFoundError(f"MCAP file does not exist: {path}")


def _load_joint_samples(mcap_path: Path) -> tuple[list[JointSample], list[JointSample]]:
    """Load command and measured arm samples from the two required topics."""
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
    except ImportError as error:
        raise RuntimeError(
            "rosbags is required; run this script in the holobrain conda environment"
        ) from error

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    samples_by_topic = {JOINT_COMMAND_TOPIC: [], JOINT_STATE_TOPIC: []}
    with AnyReader([mcap_path]) as reader:
        connections = [
            connection
            for connection in reader.connections
            if connection.topic in samples_by_topic
        ]
        found_topics = {connection.topic for connection in connections}
        missing_topics = set(samples_by_topic) - found_topics
        if missing_topics:
            raise ValueError(f"Required MCAP topics are missing: {sorted(missing_topics)}")
        for connection, log_time_ns, raw in reader.messages(connections=connections):
            message = _deserialize_joint_state(
                typestore, raw, connection.msgtype, connection.topic
            )
            samples_by_topic[connection.topic].append(
                _to_joint_sample(message, log_time_ns, connection.topic)
            )

    command_samples = samples_by_topic[JOINT_COMMAND_TOPIC]
    state_samples = samples_by_topic[JOINT_STATE_TOPIC]
    if not command_samples or not state_samples:
        raise ValueError("Both joint topics must contain at least one message")
    command_samples.sort(key=lambda sample: sample.timestamp_ns)
    state_samples.sort(key=lambda sample: sample.timestamp_ns)
    return command_samples, state_samples


def _deserialize_joint_state(
    typestore: Any, raw: bytes | memoryview, msgtype: str, topic: str
) -> Any:
    """Deserialize a JointState, allowing HC-TJ command's known zero tail."""
    normalized_msgtype = _normalize_msgtype(msgtype)
    raw_bytes = bytes(raw)
    try:
        return typestore.deserialize_cdr(raw_bytes, normalized_msgtype)
    except Exception as original_error:
        if topic != JOINT_COMMAND_TOPIC or raw_bytes[-8:] != bytes(8):
            raise ValueError(f"Failed to deserialize {topic} as {msgtype}") from original_error
        try:
            return typestore.deserialize_cdr(raw_bytes[:-8], normalized_msgtype)
        except Exception as trimmed_error:
            raise ValueError(f"Failed to deserialize {topic} after zero-tail removal") from trimmed_error


def _normalize_msgtype(msgtype: str) -> str:
    """Normalize ROS 2 aliases stored in this MCAP's connection metadata."""
    package, separator, type_name = msgtype.partition("/")
    if "/msg/" in msgtype or not separator:
        return msgtype
    return f"{package}/msg/{type_name}"


def _to_joint_sample(message: Any, log_time_ns: int, topic: str) -> JointSample:
    """Validate JointState ordering and retain arm plus torso positions."""
    names = tuple(message.name)
    if names != EXPECTED_SOURCE_JOINT_NAMES:
        raise ValueError(
            f"{topic} joint order mismatch: expected {list(EXPECTED_SOURCE_JOINT_NAMES)}, "
            f"got {list(names)}"
        )
    position = np.asarray(message.position, dtype=np.float64)
    if position.shape != (len(EXPECTED_SOURCE_JOINT_NAMES),) or not np.all(
        np.isfinite(position)
    ):
        raise ValueError(f"{topic} position must contain 17 finite joint values")
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    timestamp_ns = (
        int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec) if stamp else 0
    )
    return JointSample(timestamp_ns if timestamp_ns > 0 else int(log_time_ns), position)


def _print_summary(
    mcap_path: Path, command_samples: Sequence[JointSample], state_samples: Sequence[JointSample]
) -> None:
    """Print the decoded recording range and an initial command/state error."""
    state_at_start = _nearest_sample(state_samples, command_samples[0].timestamp_ns)
    first_error = command_samples[0].position - state_at_start.position
    duration_s = (command_samples[-1].timestamp_ns - command_samples[0].timestamp_ns) / 1e9
    print(f"MCAP: {mcap_path}")
    print(f"joint_cmd samples: {len(command_samples)}")
    print(f"joint_states samples: {len(state_samples)}")
    print(f"command duration: {duration_s:.3f} s")
    print(f"initial max |command - state|: {np.max(np.abs(first_error)):.4f} rad")
    print("arm order:", ", ".join(ARM_JOINT_NAMES))
    print("torso order: leg_1, leg_2, zhi")


def _nearest_sample(samples: Sequence[JointSample], timestamp_ns: int) -> JointSample:
    """Return the sample nearest to a target timestamp."""
    timestamps = [sample.timestamp_ns for sample in samples]
    index = bisect.bisect_left(timestamps, timestamp_ns)
    if index == 0:
        return samples[0]
    if index == len(samples):
        return samples[-1]
    before, after = samples[index - 1], samples[index]
    return before if timestamp_ns - before.timestamp_ns <= after.timestamp_ns - timestamp_ns else after


def _replay(
    command_samples: Sequence[JointSample],
    state_samples: Sequence[JointSample],
    *,
    urdf_path: Path,
    playback_rate: float,
    loop: bool,
    source: str,
) -> None:
    """Render the selected source in the HC-TJ URDF viewer."""
    try:
        from yourdfpy import URDF
    except ImportError as error:
        raise RuntimeError(
            "yourdfpy is required; run this script in the holobrain conda environment"
        ) from error

    samples = command_samples if source == "command" else state_samples
    robot = URDF.load(str(urdf_path), build_scene_graph=True)
    _validate_urdf_joints(robot.actuated_joint_names)
    start_ns = samples[0].timestamp_ns
    duration_ns = max(samples[-1].timestamp_ns - start_ns, 1)
    wall_start = time.monotonic()

    def update_scene(_: Any) -> None:
        """Advance the selected stream from elapsed viewer time."""
        elapsed_ns = int((time.monotonic() - wall_start) * playback_rate * 1e9)
        elapsed_ns = elapsed_ns % duration_ns if loop else min(elapsed_ns, duration_ns)
        sample = _nearest_sample(samples, start_ns + elapsed_ns)
        robot.update_cfg(_source_joint_config(sample))

    try:
        robot.show(callback=update_scene)
    except ModuleNotFoundError as error:
        if error.name == "pyglet":
            raise RuntimeError(
                "yourdfpy rendering needs pyglet. Install it with "
                "`conda run -n holobrain pip install 'pyglet<2'`."
            ) from error
        raise


def _source_joint_config(sample: JointSample) -> dict[str, float]:
    """Map one MCAP sample to URDF joint values.

    The MCAP torso encoder direction is opposite to the HC-TJ URDF convention:
    its three torso joint axes are all defined as negative Y in the URDF.
    Arm values already use the URDF convention and are kept unchanged.
    """
    config = dict(zip(EXPECTED_SOURCE_JOINT_NAMES, sample.position, strict=True))
    for joint_name in TORSO_JOINT_NAMES:
        config[joint_name] = config[joint_name]
    return config


def _validate_urdf_joints(actuated_joint_names: Iterable[str]) -> None:
    """Ensure every arm and torso joint from the recording exists in the URDF."""
    missing = set(EXPECTED_SOURCE_JOINT_NAMES) - set(actuated_joint_names)
    if missing:
        raise ValueError(f"URDF is missing source joints: {sorted(missing)}")


if __name__ == "__main__":
    main()
