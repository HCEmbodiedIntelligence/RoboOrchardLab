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

"""Replay a converted HC-TJ RODataset through the local URDF.

The converted ``joints`` and ``actions`` columns both use HoloBrain's 16-axis
ordering: ``Joint1_R..Joint7_R, R_ban, Joint1_L..Joint7_L, L_ban``. Finger
values are binary commands: 0 drives the URDF finger joint to its maximum
position (1.5707 rad), while 1 drives it to its minimum position (0 rad).

Example:

.. code-block:: bash

   conda run -n holobrain python \
     projects/holobrain/scripts/replay_hc_tj_mcap_urdf.py \
     projects/holobrain/data/arrow_dataset/hc_tj_260818_fix \
     --episode-index 0 --source actions --loop
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

RIGHT_ARM_JOINT_NAMES = tuple(f"Joint{index}_R" for index in range(1, 8))
LEFT_ARM_JOINT_NAMES = tuple(f"Joint{index}_L" for index in range(1, 8))
HOLOBRAIN_JOINT_NAMES = (
    RIGHT_ARM_JOINT_NAMES + ("R_ban",) + LEFT_ARM_JOINT_NAMES + ("L_ban",)
)
FINGER_INDICES = (7, 15)
FINGER_JOINT_NAMES = ("R_ban", "L_ban")
FINGER_MIN_POSITION_RAD = 0.0
FINGER_MAX_POSITION_RAD = 1.5707
TORSO_JOINT_CONFIG = {
    "leg_1": 0.0884089146881689,
    "leg_2": 0.18213798336727732,
    "zhi": 0.7343025305957696,
}
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_PATH = (
    REPOSITORY_ROOT / "projects/holobrain/data/arrow_dataset/hc_tj_260818_fix"
)
DEFAULT_URDF_PATH = (
    REPOSITORY_ROOT / "hc_assets/hc_tj_description/urdf/hc_tj_robot.urdf"
)


@dataclass(frozen=True, slots=True)
class JointSample:
    """One 16-axis arm-and-gripper sample from a converted Arrow row."""

    timestamp_ns: int
    position: np.ndarray


def main() -> None:
    """Parse arguments, validate one dataset episode, and render its replay."""
    args = _parse_args()
    dataset_path = _resolve_dataset_path(args.dataset)
    actions, joints, episode_count = _load_joint_samples(
        dataset_path, args.episode_index
    )
    _print_summary(
        dataset_path,
        actions,
        joints,
        episode_index=args.episode_index,
        episode_count=episode_count,
    )
    if args.no_gui:
        return
    _replay(
        actions if args.source == "actions" else joints,
        urdf_path=args.urdf,
        playback_rate=args.playback_rate,
        loop=args.loop,
        source=args.source,
    )


def _parse_args() -> argparse.Namespace:
    """Return CLI arguments for one converted-dataset replay session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=DEFAULT_DATASET_PATH,
        help=f"Converted HC-TJ RODataset (default: {DEFAULT_DATASET_PATH}).",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Episode metadata index to replay (default: 0).",
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
        help=(
            "Replay repeatedly until the viewer is closed or Ctrl+C is "
            "pressed."
        ),
    )
    parser.add_argument(
        "--source",
        choices=("actions", "joints"),
        default="actions",
        help="Arrow column used to drive the robot (default: actions).",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Only load, validate, and summarize the selected episode.",
    )
    args = parser.parse_args()
    if args.episode_index < 0:
        parser.error("--episode-index must be non-negative")
    if args.playback_rate <= 0:
        parser.error("--playback-rate must be positive")
    if not args.urdf.is_file():
        parser.error(f"URDF does not exist: {args.urdf}")
    return args


def _resolve_dataset_path(path: Path) -> Path:
    """Resolve a converted RODataset directory and validate its core files."""
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"RODataset directory does not exist: {path}")
    missing = [
        name
        for name in ("state.json", "dataset_info.json")
        if not (path / name).is_file()
    ]
    if missing:
        raise ValueError(
            f"{path} is not a converted RODataset; missing files: {missing}"
        )
    return path


def _load_joint_samples(
    dataset_path: Path, episode_index: int
) -> tuple[list[JointSample], list[JointSample], int]:
    """Load ``actions`` and ``joints`` from one episode of a RODataset."""
    try:
        from robo_orchard_lab.dataset.robot import RODataset
        from robo_orchard_lab.dataset.robot.db_orm import Episode
    except ImportError as error:
        raise RuntimeError(
            "RoboOrchard dataset dependencies are required; run this script "
            "in the holobrain conda environment"
        ) from error

    dataset = RODataset(str(dataset_path))
    episodes = list(dataset.iterate_meta(Episode))
    episode = next(
        (item for item in episodes if item.index == episode_index), None
    )
    if episode is None:
        available = [item.index for item in episodes]
        raise ValueError(
            f"Episode index {episode_index} is unavailable; available "
            "indices: "
            f"{available}"
        )
    begin = episode.dataset_begin_index
    end = begin + episode.frame_num
    if begin < 0 or episode.frame_num <= 0 or end > len(dataset):
        raise ValueError(
            f"Episode {episode_index} has invalid frame range "
            f"[{begin}, {end}) "
            f"for dataset length {len(dataset)}"
        )

    actions: list[JointSample] = []
    joints: list[JointSample] = []
    for row_index in range(begin, end):
        row = dataset[row_index]
        if row["episode_index"] != episode_index:
            raise ValueError(
                f"Dataset row {row_index} belongs to episode "
                f"{row['episode_index']}, expected {episode_index}"
            )
        actions.append(_to_joint_sample(row["actions"], "actions", row_index))
        joints.append(_to_joint_sample(row["joints"], "joints", row_index))
    actions.sort(key=lambda sample: sample.timestamp_ns)
    joints.sort(key=lambda sample: sample.timestamp_ns)
    return actions, joints, len(episodes)


def _to_joint_sample(state: Any, column: str, row_index: int) -> JointSample:
    """Validate and decode one 16-axis ``BatchJointsState`` value."""
    names = tuple(state.names)
    if names != HOLOBRAIN_JOINT_NAMES:
        raise ValueError(
            f"Row {row_index} {column} joint order mismatch: expected "
            f"{list(HOLOBRAIN_JOINT_NAMES)}, got {list(names)}"
        )
    position = np.asarray(state.position, dtype=np.float64)
    if position.shape != (1, len(HOLOBRAIN_JOINT_NAMES)) or not np.all(
        np.isfinite(position)
    ):
        raise ValueError(
            f"Row {row_index} {column} position must have shape "
            f"(1, {len(HOLOBRAIN_JOINT_NAMES)}) with finite values"
        )
    finger_values = position[0, list(FINGER_INDICES)]
    if not np.all(np.isin(finger_values, (0.0, 1.0))):
        raise ValueError(
            f"Row {row_index} {column} finger values must be binary, got "
            f"{finger_values.tolist()}"
        )
    if len(state.timestamps) != 1 or int(state.timestamps[0]) <= 0:
        raise ValueError(
            f"Row {row_index} {column} must contain one positive timestamp"
        )
    return JointSample(int(state.timestamps[0]), position[0].copy())


def _print_summary(
    dataset_path: Path,
    actions: Sequence[JointSample],
    joints: Sequence[JointSample],
    *,
    episode_index: int,
    episode_count: int,
) -> None:
    """Print the selected episode range and initial action/state difference."""
    joint_at_start = _nearest_sample(joints, actions[0].timestamp_ns)
    first_error = actions[0].position - joint_at_start.position
    duration_s = (actions[-1].timestamp_ns - actions[0].timestamp_ns) / 1e9
    print(f"RODataset: {dataset_path}")
    print(f"episode: {episode_index} of {episode_count}")
    print(f"actions samples: {len(actions)}")
    print(f"joints samples: {len(joints)}")
    print(f"actions duration: {duration_s:.3f} s")
    print(f"initial max |actions - joints|: {np.max(np.abs(first_error)):.4f}")
    print(
        "initial fingers (R_ban, L_ban):",
        tuple(actions[0].position[index] for index in FINGER_INDICES),
    )
    print("joint order:", ", ".join(HOLOBRAIN_JOINT_NAMES))


def _nearest_sample(
    samples: Sequence[JointSample], timestamp_ns: int
) -> JointSample:
    """Return the sample nearest to a target timestamp."""
    timestamps = [sample.timestamp_ns for sample in samples]
    index = bisect.bisect_left(timestamps, timestamp_ns)
    if index == 0:
        return samples[0]
    if index == len(samples):
        return samples[-1]
    before, after = samples[index - 1], samples[index]
    return (
        before
        if timestamp_ns - before.timestamp_ns
        <= after.timestamp_ns - timestamp_ns
        else after
    )


def _replay(
    samples: Sequence[JointSample],
    *,
    urdf_path: Path,
    playback_rate: float,
    loop: bool,
    source: str,
) -> None:
    """Render one selected Arrow column in the HC-TJ URDF viewer."""
    try:
        from yourdfpy import URDF
    except ImportError as error:
        raise RuntimeError(
            "yourdfpy is required; run this script in the holobrain conda "
            "environment"
        ) from error

    robot = URDF.load(str(urdf_path), build_scene_graph=True)
    _validate_urdf_joints(robot.actuated_joint_names)
    start_ns = samples[0].timestamp_ns
    duration_ns = max(samples[-1].timestamp_ns - start_ns, 1)
    wall_start = time.monotonic()
    print(f"Opening HC-TJ viewer driven by '{source}'; close it to exit.")

    def update_scene(_: Any) -> None:
        """Advance the selected stream from elapsed viewer time."""
        elapsed_ns = int((time.monotonic() - wall_start) * playback_rate * 1e9)
        elapsed_ns = (
            elapsed_ns % duration_ns if loop else min(elapsed_ns, duration_ns)
        )
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
    """Map arm positions and binary finger commands to HC-TJ URDF values.

    The stored command convention is inverted relative to the URDF joint
    coordinate: ``0`` maps to the upper limit and ``1`` maps to the lower
    limit for both ``R_ban`` and ``L_ban``.
    """
    config = dict(zip(HOLOBRAIN_JOINT_NAMES, sample.position, strict=True))
    for joint_name in FINGER_JOINT_NAMES:
        config[joint_name] = (
            FINGER_MAX_POSITION_RAD
            if config[joint_name] == 0.0
            else FINGER_MIN_POSITION_RAD
        )
    config.update(TORSO_JOINT_CONFIG)
    return config


def _validate_urdf_joints(actuated_joint_names: Iterable[str]) -> None:
    """Ensure all replayed arm, gripper, and fixed torso joints exist."""
    expected = set(HOLOBRAIN_JOINT_NAMES) | set(TORSO_JOINT_CONFIG)
    missing = expected - set(actuated_joint_names)
    if missing:
        raise ValueError(f"URDF is missing replay joints: {sorted(missing)}")


if __name__ == "__main__":
    main()
