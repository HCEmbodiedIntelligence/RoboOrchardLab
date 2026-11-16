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
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
    from rosbags.rosbag2 import Reader, StoragePlugin, Writer
    from rosbags.serde import SerdeError
    from rosbags.typesys import Stores, get_typestore
except ImportError as exc:
    raise unittest.SkipTest("rosbags and numpy are required") from exc

from projects.holobrain.own_test import fix_mcap_left_arm as fixer
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
        malformed: bytes | None = None,
        omit_topic: bool = False,
        empty_topic: bool = False,
        position_count: int | None = None,
        tail: bytes = b"",
        big_endian: bool = False,
        frame_id: str = "",
        first_position: float = 0,
        passthrough: bool = False,
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
                if empty_topic and index == 3:
                    continue
                for ordinal in range(2):
                    selected = (
                        names
                        if index == 0 and ordinal == 1
                        else list(checker.DEFAULT_JOINTS)
                    )
                    msg = types["sensor_msgs/msg/JointState"](
                        types["std_msgs/msg/Header"](
                            types["builtin_interfaces/msg/Time"](0, 0),
                            frame_id,
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
                    if ordinal == 0:
                        msg.position[0] = first_position
                    raw = (
                        bytes(
                            self.typestore.serialize_cdr(
                                msg,
                                "sensor_msgs/msg/JointState",
                                little_endian=not big_endian,
                            )
                        )
                        + tail
                    )
                    if malformed is not None and index == 1 and ordinal == 1:
                        raw = malformed
                    writer.write(conn, 100 + index * 2 + ordinal, raw)
            if passthrough:
                conn = writer.add_connection(
                    "/unrelated",
                    "std_msgs/msg/String",
                    typestore=self.typestore,
                )
                writer.write(conn, 200, b"opaque pass-through payload")
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

    def test_malformed_messages_are_deleted_after_complete_scan(self) -> None:
        corrupt = self.make_mcap(
            "corrupt", ["Joint1_R"], malformed=b"invalid", omit_topic=True
        )
        status, report = self.run_check("--delete")
        self.assertEqual(status, 0)
        self.assertFalse(corrupt.exists())
        result = report["files"][0]
        self.assertIsNone(result["error"])
        self.assertEqual(sum(result["counts"].values()), 6)
        self.assertEqual(len(result["issues"]), 2)
        self.assertIn("Expected plain CDR1", result["issues"][1]["reason"])
        self.assertTrue(result["deleted"])

    def test_container_errors_and_symlinks_are_retained(self) -> None:
        corrupt = self.source / "broken_container.mcap"
        corrupt.write_bytes(b"not an MCAP container")
        outside = self.make_mcap("outside", ["Joint1_R"])
        outside = outside.rename(self.root / outside.name)
        link = self.source / "link.mcap"
        link.symlink_to(outside)
        status, report = self.run_check("--delete")
        self.assertEqual(status, 1)
        self.assertTrue(all(p.exists() for p in (corrupt, link, outside)))
        self.assertTrue(all(f["error"] for f in report["files"]))
        self.assertFalse(any(f["deleted"] for f in report["files"]))
        self.assertEqual(report["summary"]["passed_files"], 0)
        self.assertEqual(report["summary"]["scan_errors"], 2)
        self.assertIn(
            "SCAN FAILED (retained, NOT passed)", self.output.getvalue()
        )

    def test_producer_tail_does_not_hide_later_invalid_messages(self) -> None:
        good = self.make_mcap(
            "good", list(checker.DEFAULT_JOINTS), tail=b"tail!"
        )
        bad = self.make_mcap(
            "bad", list(checker.DEFAULT_JOINTS[:-1]), tail=b"tail!"
        )
        # The old checker fails before reaching later invalid messages.

        with Reader(good) as reader:
            _, _, raw = next(reader.messages())
        with self.assertRaises((SerdeError, AssertionError)):
            self.typestore.deserialize_cdr(raw, "sensor_msgs/msg/JointState")
        status, report = self.run_check("--delete")
        self.assertEqual(status, 0)
        self.assertTrue(good.exists())
        self.assertFalse(bad.exists())
        files = {Path(f["path"]).stem: f for f in report["files"]}
        self.assertIsNone(files["bad"]["error"])
        issue = files["bad"]["issues"][0]
        self.assertEqual(issue["first_ordinal"], 2)
        self.assertEqual(issue["missing_joints"], ["Joint7_L"])
        self.assertEqual(report["summary"]["passed_files"], 1)

    def test_accepted_files_rewrite_and_preserve_unpatched_bytes(self) -> None:
        for big_endian in (False, True):
            with self.subTest(big_endian=big_endian):
                source = self.make_mcap(
                    f"good_{big_endian}",
                    list(reversed(checker.DEFAULT_JOINTS)),
                    tail=b"producer tail",
                    frame_id="base_link",
                    big_endian=big_endian,
                    first_position=1.25,
                    passthrough=True,
                )
                scan = checker._scan(
                    source, checker.DEFAULT_TOPICS, checker.DEFAULT_JOINTS
                )
                self.assertIsNone(scan.error)
                self.assertEqual(scan.issues, [])
                self.assertEqual(scan.missing_topics, [])
                targets = fixer._reference_positions(source)
                output = self.root / f"rewritten_{big_endian}"
                self.assertEqual(
                    fixer._rewrite(source, output, targets), scan.counts
                )
                with Reader(source) as before, Reader(output) as after:
                    for (c1, t1, raw1), (c2, t2, raw2) in zip(
                        before.messages(),
                        after.messages(),
                        strict=True,
                    ):
                        self.assertEqual((c1.topic, t1), (c2.topic, t2))
                        expected = bytearray(raw1)
                        topic = c1.topic.lstrip("/")
                        if topic in fixer.TOPICS:
                            endian, offsets = fixer._position_offsets(raw1)
                            for name, offset in offsets.items():
                                struct.pack_into(
                                    endian + "d",
                                    expected,
                                    offset,
                                    targets[topic][name],
                                )
                        self.assertEqual(raw2, bytes(expected))

    def test_nonfinite_reference_is_rejected(self) -> None:
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                path = self.make_mcap(
                    str(value),
                    list(checker.DEFAULT_JOINTS),
                    first_position=value,
                )
                result = checker._scan(
                    path, checker.DEFAULT_TOPICS, checker.DEFAULT_JOINTS
                )
                self.assertIsNone(result.error)
                self.assertEqual(len(result.issues), 4)
                self.assertTrue(
                    all("Non-finite" in i.reason for i in result.issues)
                )
                with self.assertRaisesRegex(ValueError, "Non-finite"):
                    fixer._reference_positions(path)

    def test_truncated_and_unsupported_cdr_are_invalid(self) -> None:
        for raw in (
            b"\x00\x01",
            b"\x00\x01\x00\x00" + bytes(20),
            b"\x00\x03" + bytes(50),
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(checker.InvalidJointStateError):
                    fixer._position_offsets(raw)

    def test_truncated_position_array_is_reported_and_deleted(self) -> None:
        sample = self.make_mcap("sample", list(checker.DEFAULT_JOINTS))
        with Reader(sample) as reader:
            _, _, raw = next(reader.messages())
        _, offsets = fixer._position_offsets(raw)
        truncated = bytes(raw[: max(offsets.values()) + 4])
        bad = self.make_mcap(
            "truncated",
            list(checker.DEFAULT_JOINTS),
            malformed=truncated,
        )
        status, report = self.run_check("--delete")
        self.assertEqual(status, 0)
        self.assertFalse(bad.exists())
        self.assertTrue(sample.exists())
        result = next(
            f for f in report["files"] if Path(f["path"]).stem == "truncated"
        )
        self.assertIsNone(result["error"])
        self.assertEqual(result["issues"][0]["position_count"], 7)
        self.assertEqual(result["issues"][0]["count"], 1)
        self.assertEqual(result["counts"][checker.DEFAULT_TOPICS[1]], 2)
        self.assertTrue(result["deleted"])

    def test_invalid_cdr_fields_do_not_abort_container_scan(self) -> None:
        payloads = (
            b"\x00\x01",
            b"\x00\x01\x00\x00" + bytes(20),
            b"\x00\x01\x00\x00"
            + bytes(8)
            + struct.pack("<I", 2)
            + b"\xff\x00",
        )
        for index, raw in enumerate(payloads):
            self.make_mcap(
                f"invalid_{index}", list(checker.DEFAULT_JOINTS), malformed=raw
            )
        status, report = self.run_check("--delete")
        self.assertEqual(status, 0)
        self.assertEqual(report["summary"]["deleted"], 3)
        self.assertEqual(report["summary"]["scan_errors"], 0)
        for result in report["files"]:
            self.assertEqual(sum(result["counts"].values()), 8)
            self.assertEqual(result["issues"][0]["first_ordinal"], 2)
            self.assertIsNone(result["issues"][0]["name_count"])

    def test_scripts_run_directly_from_project_directory(self) -> None:
        project = Path(checker.__file__).resolve().parents[1]
        for script in (
            "scripts/check_mcap_joint_states.py",
            "own_test/fix_mcap_left_arm.py",
        ):
            result = subprocess.run(
                [sys.executable, script, "--help"],
                cwd=project,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--input-dir", result.stdout)

    def test_missing_and_empty_topics_are_reported_without_deleting(
        self,
    ) -> None:
        missing = self.make_mcap(
            "missing", list(checker.DEFAULT_JOINTS), omit_topic=True
        )
        empty = self.make_mcap(
            "empty", list(checker.DEFAULT_JOINTS), empty_topic=True
        )
        status, report = self.run_check()
        self.assertEqual(status, 1)
        self.assertTrue(all(p.exists() for p in (missing, empty)))
        for result in report["files"]:
            self.assertEqual(
                result["missing_topics"], [checker.DEFAULT_TOPICS[3]]
            )
            self.assertEqual(result["issues"], [])
            self.assertFalse(result["deleted"])
            self.assertIn(
                f"RETAINED INVALID MCAP: {result['path']}",
                self.output.getvalue(),
            )
        self.assertEqual(report["summary"]["retained_invalid_files"], 2)

    def test_delete_removes_missing_and_empty_topics(self) -> None:
        good = self.make_mcap("good", list(checker.DEFAULT_JOINTS))
        missing = self.make_mcap(
            "missing", list(checker.DEFAULT_JOINTS), omit_topic=True
        )
        empty = self.make_mcap(
            "empty", list(checker.DEFAULT_JOINTS), empty_topic=True
        )
        status, report = self.run_check("--delete")
        self.assertEqual(status, 0)
        self.assertTrue(good.exists())
        self.assertFalse(missing.exists())
        self.assertFalse(empty.exists())
        self.assertEqual(report["summary"]["files_missing_topics"], 2)
        self.assertEqual(report["summary"]["deleted"], 2)
        self.assertEqual(report["summary"]["retained_invalid_files"], 0)

    def test_no_matching_topics_are_reported_without_decoding(self) -> None:
        path = self.make_mcap("other_topics", ["Joint1_R"])
        status, report = self.run_check("--delete", "--topics", "/not_present")
        self.assertEqual(status, 0)
        self.assertFalse(path.exists())
        result = report["files"][0]
        self.assertEqual(result["missing_topics"], ["not_present"])
        self.assertIsNone(result["error"])
        self.assertEqual(result["issues"], [])
        self.assertTrue(result["deleted"])

    def test_only_selected_topics_are_required_for_deletion(self) -> None:
        path = self.make_mcap(
            "selected", list(checker.DEFAULT_JOINTS), omit_topic=True
        )
        status, report = self.run_check(
            "--delete", "--topics", "/action/joint_states"
        )
        self.assertEqual(status, 0)
        self.assertTrue(path.exists())
        self.assertEqual(report["files"][0]["missing_topics"], [])
        self.assertEqual(report["summary"]["deleted"], 0)

    def test_changed_file_is_retained_before_deletion(self) -> None:
        bad = self.make_mcap(
            "changed", list(checker.DEFAULT_JOINTS), omit_topic=True
        )
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
        bad = self.make_mcap(
            "bad", list(checker.DEFAULT_JOINTS), omit_topic=True
        )
        self.report.write_text("previous report")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exc:
                self.run_check("--delete")
        self.assertEqual(exc.exception.code, 2)
        self.assertTrue(bad.exists())
        self.assertEqual(self.report.read_text(), "previous report")


if __name__ == "__main__":
    unittest.main()
