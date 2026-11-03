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

from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import rosbag2_py
    from rclpy.serialization import serialize_message
    from std_msgs.msg import String
except ImportError as exc:
    raise unittest.SkipTest("ROS 2 Python bindings are unavailable") from exc

from projects.holobrain.scripts.split_mcap_takeovers import (
    split_mcap_takeovers,
)

__all__ = []


def _messages(path: Path) -> list[tuple]:
    reader = rosbag2_py.SequentialReader()
    try:
        reader.open(
            rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
            rosbag2_py.ConverterOptions("", ""),
        )
        messages = []
        while reader.has_next():
            messages.append(reader.read_next())
        return sorted(messages)
    finally:
        reader.close()


class TestSplitMcapTakeovers(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input"
        self.source.mkdir()
        self.output = self.root / "output"
        bag = self.root / "bag"
        writer = rosbag2_py.SequentialWriter()
        try:
            writer.open(
                rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""),
            )
            writer.create_topic(
                rosbag2_py.TopicMetadata(
                    id=0,
                    name="/sample",
                    type="std_msgs/msg/String",
                    serialization_format="cdr",
                )
            )
            for i, timestamp in enumerate([9, 20, 10, 19, 30, 20]):
                writer.write(
                    "/sample",
                    serialize_message(String(data=str(i))),
                    timestamp,
                )
        finally:
            writer.close()
        next(bag.glob("*.mcap")).rename(self.source / "demo.mcap")
        self.annotation = self.source / "demo.takeovers.json"
        self.annotate([(10, 20), (20, 30), (19, 21)])

    def annotate(self, intervals):
        self.annotation.write_text(
            json.dumps(
                {
                    "schema": "hc-dagger-takeovers/v1",
                    "recording": "demo.mcap",
                    "complete": True,
                    "intervals": [
                        {"start_ns": start, "end_ns": end}
                        for start, end in intervals
                    ],
                }
            )
        )

    def test_boundaries_overlap_and_no_overwrite(self):
        outputs = split_mcap_takeovers(self.source, self.output)
        original = _messages(self.source / "demo.mcap")
        self.assertEqual(len(outputs), 3)
        for target, (start, end) in zip(
            outputs, [(10, 20), (20, 30), (19, 21)], strict=True
        ):
            self.assertEqual(
                _messages(target),
                [message for message in original if start <= message[2] < end],
            )
        before = [target.read_bytes() for target in outputs]
        with self.assertRaises(FileExistsError):
            split_mcap_takeovers(self.source, self.output)
        self.assertEqual([target.read_bytes() for target in outputs], before)

    def test_invalid_intervals(self):
        for intervals in [
            [(20, 10)],
            [(10, 10)],
            [(-1, 20)],
            [(10.5, 20)],
            [(True, 20)],
            [(10, 2**63)],
        ]:
            with self.subTest(intervals=intervals):
                self.annotate(intervals)
                with self.assertRaises(ValueError):
                    split_mcap_takeovers(self.source, self.output)
                self.assertFalse(self.output.exists())

    def test_empty_segment_cleanup(self):
        self.annotate([(1000, 2000)])
        with self.assertRaisesRegex(ValueError, "no messages"):
            split_mcap_takeovers(self.source, self.output)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_write_failure_cleanup(self):
        with patch(
            "projects.holobrain.scripts.split_mcap_takeovers.os.link",
            side_effect=OSError("publication failed"),
        ):
            with self.assertRaisesRegex(OSError, "publication failed"):
                split_mcap_takeovers(self.source, self.output)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_no_interventions(self):
        self.annotate([])
        self.assertEqual(split_mcap_takeovers(self.source, self.output), [])

    def test_invalid_annotations(self):
        for change in [
            {"complete": False},
            {"recording": "wrong.mcap"},
            {"schema": "unknown"},
            {"intervals": [{"start_ns": 10, "end_ns": None}]},
        ]:
            with self.subTest(change=change):
                self.annotate([(10, 20)])
                data = json.loads(self.annotation.read_text())
                data.update(change)
                self.annotation.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    split_mcap_takeovers(self.source, self.output)
                self.assertFalse(self.output.exists())

    def test_missing_annotation(self):
        self.annotation.unlink()
        with self.assertRaises(FileNotFoundError):
            split_mcap_takeovers(self.source, self.output)

    def test_multiple_recordings(self):
        (self.source / "second.mcap").write_bytes(
            (self.source / "demo.mcap").read_bytes()
        )
        data = json.loads(self.annotation.read_text())
        data["recording"] = "second.mcap"
        (self.source / "second.takeovers.json").write_text(json.dumps(data))
        self.assertEqual(
            len(split_mcap_takeovers(self.source, self.output)), 6
        )


if __name__ == "__main__":
    unittest.main()
