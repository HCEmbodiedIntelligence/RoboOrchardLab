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

"""Extract takeover intervals without decoding MCAP messages.

Use a ROS 2 environment with ``rosbag2_py``, the MCAP storage plugin,
installed ROS message definitions, and ``pydantic>=2``::

    python projects/holobrain/scripts/split_mcap_takeovers.py A B

Only immediate ``*.mcap`` children are processed. Each ``name.mcap`` needs
``name.takeovers.json`` using hc-dagger-takeovers/v1. Intervals use absolute
nanosecond MCAP log times, with inclusive starts and exclusive ends. Offsets
are informational; interval timestamps and serialized payloads are preserved.
ROS regenerates schemas from installed message definitions. Arbitrary MCAP
metadata, attachments, sequence numbers, and separate publish timestamps are
not preserved by this ROS reader/writer interface. Head-camera calibration
and extrinsics carry forward the latest message before each interval, stamped
at its start, unless that topic already has a message exactly at the start.
Other messages outside the interval (including TF) are not injected.
Segments missing required converter topics are discarded with a warning.
For other topic layouts, every registered source topic must have a message.
"""

from __future__ import annotations
import argparse
import logging
import os
import tempfile
from pathlib import Path
from typing import Literal

import rosbag2_py
from pydantic import BaseModel, Field

__all__ = ["split_mcap_takeovers", "main"]

_CAMERA_STATE_TOPICS = frozenset(
    f"{prefix}/{kind}"
    for prefix in (
        "/observation/cameras/head",
        "/io_teleop/camera_head",
    )
    for kind in ("calibration", "extrinsics")
)

# Keep the ROS-only splitter independent of the converter's ML dependencies.
_REQUIRED_TOPIC_LAYOUTS = (
    frozenset(
        (
            "/observation/cameras/head/color/compressed",
            "/observation/cameras/head/depth/compressed",
            "/observation/cameras/head/calibration",
            "/observation/cameras/head/extrinsics",
            "/observation/joint_states",
            "/action/joint_states",
            "/action/right_hand/joint_states",
            "/action/left_hand/joint_states",
        )
    ),
    frozenset(
        (
            "/io_teleop/camera_head/color",
            "/io_teleop/camera_head/depth",
            "/io_teleop/camera_head/calibration",
            "/io_teleop/camera_head/extrinsics",
            "io_teleop/joint_states",
            "io_teleop/joint_cmd",
            "/io_teleop/joint_cmd_finger_right",
            "/io_teleop/joint_cmd_finger_left",
        )
    ),
)
_logger = logging.getLogger(__name__)


class _Interval(BaseModel):
    start_ns: int = Field(strict=True, ge=0, lt=2**63)
    end_ns: int = Field(strict=True, ge=0, lt=2**63)


class _Takeovers(BaseModel):
    schema_name: Literal["hc-dagger-takeovers/v1"] = Field(alias="schema")
    recording: str
    complete: Literal[True]
    intervals: list[_Interval]


def split_mcap_takeovers(input_dir: Path, output_dir: Path) -> list[Path]:
    """Write one MCAP per annotated interval, in JSON interval order.

    Args:
        input_dir: Directory containing MCAPs and matching takeover JSON files.
        output_dir: Destination for ``<stem>.takeover_0001.mcap`` etc.

    Returns:
        Created paths, excluding discarded segments. Original interval numbers
        are retained, so output filenames may have gaps.

    Raises:
        ValueError: Invalid/incomplete annotations, mismatched recording name,
            reversed intervals, or no input MCAPs.
        FileExistsError: A destination already exists; never overwrite it.
        OSError: Input or output access fails, including missing annotations.

    All annotations and output collisions are checked before writing. Each
    failed segment is removed; earlier completed segments remain on failure.
    MCAP parse errors propagate, without attempting truncated-file recovery.
    Overlapping intervals independently retain their matching messages.
    Head-camera calibration/extrinsics are seeded from the latest preceding
    message in this source, with its payload unchanged and log time set to the
    interval start. Messages at the start supersede these seeds; updates inside
    the interval remain intact. Missing source calibration is not fabricated.
    Empty segments or segments missing converter-required topic messages after
    seeding are discarded with a warning; processing continues. For unrelated
    topic layouts, all registered source topics are required. This checks
    message presence only, not payload validity or synchronization tolerances.
    """
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    sources = sorted(input_dir.glob("*.mcap"))
    if not sources:
        raise ValueError(f"No MCAP files in {input_dir}")
    jobs: list[tuple[Path, _Interval, Path]] = []
    for source in sources:
        annotation = source.with_suffix(".takeovers.json")
        takeovers = _Takeovers.model_validate_json(annotation.read_bytes())
        if takeovers.recording != source.name:
            raise ValueError(f"{annotation}: recording must be {source.name}")
        for index, interval in enumerate(takeovers.intervals, 1):
            if interval.end_ns <= interval.start_ns:
                raise ValueError(
                    f"{annotation}: interval {index} needs end > start"
                )
            target = output_dir / f"{source.stem}.takeover_{index:04d}.mcap"
            if target.exists() or target.is_symlink():
                raise FileExistsError(target)
            jobs.append((source, interval, target))
    if jobs:
        output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for source, interval, target in jobs:
        if _write_segment(source, interval, target):
            results.append(target)
            print(f"{target}: [{interval.start_ns}, {interval.end_ns})")
    print(
        f"Kept {len(results)} segments; discarded {len(jobs) - len(results)}"
    )
    return results


def main() -> None:
    """Parse input and output directories and extract takeover segments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    paths = split_mcap_takeovers(args.input_dir, args.output_dir)
    print(f"Wrote {len(paths)} takeover segments")


def _write_segment(source: Path, interval: _Interval, target: Path) -> bool:
    """Publish a complete segment; log and return False for missing messages.

    Staged files are removed on discard or failure. I/O and parse failures
    propagate rather than being treated as missing data.
    """
    reader = rosbag2_py.SequentialReader()
    try:
        reader.open(
            rosbag2_py.StorageOptions(uri=str(source), storage_id="mcap"),
            rosbag2_py.ConverterOptions("", ""),
        )
        # ROS writes a directory containing MCAP and metadata.yaml. Stage it
        # on the destination filesystem so publication can be exclusive.
        with tempfile.TemporaryDirectory(
            prefix=".takeover-", dir=target.parent
        ) as staging:
            bag_dir = Path(staging) / "bag"
            writer = rosbag2_py.SequentialWriter()
            try:
                writer.open(
                    rosbag2_py.StorageOptions(
                        uri=str(bag_dir), storage_id="mcap"
                    ),
                    rosbag2_py.ConverterOptions("", ""),
                )
                source_topics = reader.get_all_topics_and_types()
                required_topics = set()
                topic_names = {topic.name for topic in source_topics}
                for layout in _REQUIRED_TOPIC_LAYOUTS:
                    if topic_names.intersection(layout):
                        required_topics.update(layout)
                if not required_topics:
                    required_topics = topic_names
                for topic in source_topics:
                    writer.create_topic(topic)
                count = 0
                written_topics: set[str] = set()
                camera_state: dict[str, tuple[bytes, int]] = {}
                topics_at_start: set[str] = set()
                while reader.has_next():
                    topic, payload, timestamp = reader.read_next()
                    if topic in _CAMERA_STATE_TOPICS:
                        if timestamp < interval.start_ns:
                            previous = camera_state.get(topic)
                            if previous is None or timestamp >= previous[1]:
                                camera_state[topic] = (payload, timestamp)
                        elif timestamp == interval.start_ns:
                            topics_at_start.add(topic)
                    if interval.start_ns <= timestamp < interval.end_ns:
                        writer.write(topic, payload, timestamp)
                        written_topics.add(topic)
                        count += 1
                # Choose by log time, not physical record order. MCAP indexes
                # let readers place these seeds before the interval samples.
                for topic, (payload, _) in camera_state.items():
                    if topic not in topics_at_start:
                        writer.write(topic, payload, interval.start_ns)
                        written_topics.add(topic)
                missing_topics = required_topics - written_topics
                if not count or missing_topics:
                    _logger.warning(
                        "Discarded %s (source=%s, interval=[%d, %d)): "
                        "missing topics=%s; interval messages=%d",
                        target.name,
                        source,
                        interval.start_ns,
                        interval.end_ns,
                        ", ".join(sorted(missing_topics)) or "none",
                        count,
                    )
                    return False
            finally:
                writer.close()
            segments = list(bag_dir.glob("*.mcap"))
            if len(segments) != 1:
                raise RuntimeError(
                    f"Expected one MCAP in {bag_dir}, found {len(segments)}"
                )
            # Unlike rename/replace, link refuses concurrent overwrites.
            os.link(segments[0], target)
            return True
    finally:
        reader.close()


if __name__ == "__main__":
    main()
