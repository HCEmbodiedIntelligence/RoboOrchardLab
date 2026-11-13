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
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
    from rosbags.rosbag2 import StoragePlugin, Writer
    from rosbags.typesys import Stores, get_typestore
except ImportError as exc:
    raise unittest.SkipTest("rosbags and numpy are required") from exc

from projects.holobrain.scripts import check_mcap_joint_states as checker

__all__ = []


class TestCheckMcapJointStates(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input"
        self.source.mkdir()
        self.report = self.root / "report.json"
        self.typestore = get_typestore(Stores.ROS2_HUMBLE)

    def make_mcap(
        self,
        stem: str,
        names: list[str],
        *,
        malformed: bool = False,
        omit_topic: bool = False,
        position_count: int | None = None,
    ) -> Path:
        """Write real CDR messages in a real MCAP, including one bad sample."""
        bag = self.root / f"{stem}_bag"
        types = self.typestore.types
        with Writer(
            bag, version=9, storage_plugin=StoragePlugin.MCAP
        ) as writer:
            for index, topic in enumerate(checker.DEFAULT_TOPICS):
                if omit_topic and index == 3:
                    continue
                conn = writer.add_connection(
                    "/" + topic,
                    "sensor_msgs/msg/JointState",
                    typestore=self.typestore,
                )
                for ordinal in range(2):
                    selected = (
                        names
                        if index == 0 and ordinal == 1
                        else list(checker.DEFAULT_JOINTS)
                    )
                    msg = types["sensor_msgs/msg/JointState"](
                        types["std_msgs/msg/Header"](
                            types["builtin_interfaces/msg/Time"](0, 0), ""
                        ),
                        selected,
                        np.zeros(
                            position_count
                            if position_count is not None
                            and index == 0
                            and ordinal == 1
                            else len(selected)
                        ),
                        np.zeros(0),
                        np.zeros(0),
                    )
                    raw = self.typestore.serialize_cdr(
                        msg, "sensor_msgs/msg/JointState"
                    )
                    if malformed and index == 1 and ordinal == 1:
                        raw = b"invalid"
                    writer.write(conn, 100 + index * 2 + ordinal, raw)
        result = self.source / f"{stem}.mcap"
        next(bag.glob("*.mcap")).rename(result)
        return result

    def run_check(self, *args: str) -> tuple[int, dict]:
        self.output = io.StringIO()
        with contextlib.redirect_stdout(self.output):
            status = checker.main(
                [
                    "--input-dir",
                    str(self.source),
                    "--report",
                    str(self.report),
                    *args,
                ]
            )
        return status, json.loads(self.report.read_text())

    def test_scan_reports_partial_and_all_missing_without_deleting(
        self,
    ) -> None:
        good = self.make_mcap("good", list(reversed(checker.DEFAULT_JOINTS)))
        partial = self.make_mcap("partial", list(checker.DEFAULT_JOINTS[:-1]))
        right_only = self.make_mcap(
            "right", [f"Joint{i}_R" for i in range(1, 8)]
        )
        status, report = self.run_check()
        self.assertEqual(status, 1)
        files = {Path(f["path"]).stem: f for f in report["files"]}
        self.assertEqual(files["good"]["issues"], [])
        self.assertEqual(
            files["partial"]["issues"][0]["missing_joints"], ["Joint7_L"]
        )
        issue = files["right"]["issues"][0]
        self.assertEqual(issue["missing_joints"], list(checker.DEFAULT_JOINTS))
        self.assertEqual((issue["count"], issue["first_ordinal"]), (1, 2))
        self.assertEqual(issue["first_log_time_ns"], 101)
        self.assertTrue(all(p.exists() for p in (good, partial, right_only)))
        self.assertIn("Mode: CHECK ONLY", self.output.getvalue())
        self.assertIn("No MCAPs were deleted", self.output.getvalue())
        self.assertIn("--delete", self.output.getvalue())
        self.assertIn(
            f"RETAINED INVALID MCAP: {right_only}", self.output.getvalue()
        )
        self.assertEqual(report["summary"]["retained_invalid_files"], 2)

    def test_delete_removes_only_confirmed_files(self) -> None:
        good = self.make_mcap("good", list(checker.DEFAULT_JOINTS))
        partial = self.make_mcap("partial", list(checker.DEFAULT_JOINTS[:-1]))
        right = self.make_mcap("right", ["Joint1_R"])
        status, report = self.run_check("--delete")
        self.assertEqual(status, 0)
        self.assertTrue(good.exists())
        self.assertFalse(partial.exists())
        self.assertFalse(right.exists())
        self.assertEqual(sum(f["deleted"] for f in report["files"]), 2)
        self.assertIn("Mode: DELETE", self.output.getvalue())
        self.assertEqual(report["summary"]["retained_invalid_files"], 0)

    def test_duplicate_names_and_position_mismatch_are_reported(self) -> None:
        duplicate = self.make_mcap(
            "duplicate", [*checker.DEFAULT_JOINTS, "Joint1_R", "Joint1_R"]
        )
        mismatch = self.make_mcap(
            "mismatch", list(checker.DEFAULT_JOINTS), position_count=6
        )
        empty = self.make_mcap(
            "empty_positions", list(checker.DEFAULT_JOINTS), position_count=0
        )
        status, report = self.run_check()
        self.assertEqual(status, 1)
        files = {Path(f["path"]).stem: f for f in report["files"]}
        self.assertEqual(
            files["duplicate"]["issues"][0]["duplicate_joints"], ["Joint1_R"]
        )
        self.assertEqual(files["duplicate"]["issues"][0]["missing_joints"], [])
        self.assertEqual(files["mismatch"]["issues"][0]["position_count"], 6)
        self.assertEqual(
            files["empty_positions"]["issues"][0]["position_count"], 0
        )
        self.assertTrue(all(p.exists() for p in (duplicate, mismatch, empty)))

    def test_delete_removes_duplicate_and_mismatched_messages(self) -> None:
        duplicate = self.make_mcap(
            "duplicate", [*checker.DEFAULT_JOINTS, checker.DEFAULT_JOINTS[0]]
        )
        mismatch = self.make_mcap(
            "mismatch", list(checker.DEFAULT_JOINTS), position_count=6
        )
        status, report = self.run_check("--delete")
        self.assertEqual(status, 0)
        self.assertFalse(duplicate.exists())
        self.assertFalse(mismatch.exists())
        self.assertTrue(all(f["deleted"] for f in report["files"]))

    def test_decode_errors_missing_topics_and_symlinks_are_retained(
        self,
    ) -> None:
        corrupt = self.make_mcap("corrupt", ["Joint1_R"], malformed=True)
        missing_topic = self.make_mcap(
            "missing_topic", list(checker.DEFAULT_JOINTS), omit_topic=True
        )
        outside = self.make_mcap("outside", ["Joint1_R"])
        outside = outside.rename(self.root / outside.name)
        link = self.source / "link.mcap"
        link.symlink_to(outside)
        status, report = self.run_check("--delete")
        self.assertEqual(status, 1)
        self.assertTrue(
            all(p.exists() for p in (corrupt, missing_topic, link, outside))
        )
        files = {Path(f["path"]).stem: f for f in report["files"]}
        self.assertTrue(files["corrupt"]["error"])
        self.assertTrue(files["corrupt"]["issues"])
        self.assertTrue(files["link"]["error"])
        self.assertEqual(
            files["missing_topic"]["missing_topics"],
            [checker.DEFAULT_TOPICS[3]],
        )
        self.assertFalse(any(f["deleted"] for f in report["files"]))

    def test_no_matching_topics_are_reported_without_decoding(self) -> None:
        path = self.make_mcap("other_topics", ["Joint1_R"])
        status, report = self.run_check("--delete", "--topics", "/not_present")
        self.assertEqual(status, 1)
        self.assertTrue(path.exists())
        result = report["files"][0]
        self.assertEqual(result["missing_topics"], ["not_present"])
        self.assertIsNone(result["error"])
        self.assertEqual(result["issues"], [])

    def test_changed_file_is_retained_before_deletion(self) -> None:
        bad = self.make_mcap("changed", ["Joint1_R"])
        original_scan = checker._scan

        def changed_after_scan(*args, **kwargs):
            result = original_scan(*args, **kwargs)
            with bad.open("ab") as handle:
                handle.write(b"changed")
            return result

        with patch.object(checker, "_scan", side_effect=changed_after_scan):
            status, report = self.run_check("--delete")
        self.assertEqual(status, 1)
        self.assertTrue(bad.exists())
        self.assertIn("changed", report["files"][0]["delete_error"])

    def test_existing_report_blocks_deletion(self) -> None:
        bad = self.make_mcap("bad", ["Joint1_R"])
        self.report.write_text("previous report")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exc:
                self.run_check("--delete")
        self.assertEqual(exc.exception.code, 2)
        self.assertTrue(bad.exists())
        self.assertEqual(self.report.read_text(), "previous report")


if __name__ == "__main__":
    unittest.main()
