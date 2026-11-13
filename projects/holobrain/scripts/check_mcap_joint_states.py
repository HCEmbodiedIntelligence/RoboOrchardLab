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

"""Find invalid JointState messages and optionally delete their MCAP files.

Requires ``pip install rosbags``; ROS is not needed. Recursively inspect four
action/observation JointState topics for Joint1_L..Joint7_L by default.
Every selected message must have unique names, contain the required joints,
and have one position per name, as required by fix_mcap_left_arm.py.

Without --delete, only scan and report. With --delete, permanently unlink
only fully scanned, unchanged regular files containing invalid joint states.
Missing topics and read/decode errors are reported separately; they
alone do not qualify a file for deletion. Symlink files are retained.
A JSON report is saved before deletion and updated after each attempt.

Example::

    python check_mcap_joint_states.py --input-dir /path/to/mcap
    python check_mcap_joint_states.py --input-dir /path/to/mcap --delete

Exit codes: 0 for no unresolved findings (including successful deletion),
1 for remaining findings or scan/deletion errors, 2 for CLI/report errors.
"""

from __future__ import annotations
import argparse
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from stat import S_ISREG
from typing import TextIO

from rosbags.interfaces import ConnectionExtRosbag2
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.store import Typestore

__all__ = ["main"]

DEFAULT_TOPICS = (
    "action/joint_states",
    "observation/joint_states",
    "action/left_arm/joint_states",
    "observation/left_arm/joint_states",
)
DEFAULT_JOINTS = tuple(f"Joint{i}_L" for i in range(1, 8))


@dataclass(slots=True)
class _JointStateIssue:
    topic: str
    missing_joints: list[str]
    first_ordinal: int
    first_log_time_ns: int
    duplicate_joints: list[str]
    name_count: int
    position_count: int
    count: int = 0


@dataclass(slots=True)
class _ScanResult:
    path: str
    counts: dict[str, int]
    missing_topics: list[str] = field(default_factory=list)
    issues: list[_JointStateIssue] = field(default_factory=list)
    fingerprint: tuple[int, ...] | None = None
    error: str | None = None
    deleted: bool = False
    delete_error: str | None = None


def main(argv: list[str] | None = None) -> int:
    """Scan MCAPs and optionally delete files with invalid joint states.

    Args:
        argv (list[str], optional): CLI arguments. Defaults to sys.argv.

    Returns:
        int: Exit status as described in the module docstring. Argument and
        report failures raise SystemExit with status 2 through argparse.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Permanently delete matching MCAPs.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="New JSON report path; default: timestamped file in current dir.",
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        default=DEFAULT_TOPICS,
        help="JointState topics to inspect; leading / is optional.",
    )
    parser.add_argument(
        "--required-joints",
        nargs="+",
        default=DEFAULT_JOINTS,
        help="Joint names required in every selected message.",
    )
    args = parser.parse_args(argv)
    root = args.input_dir.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Not a directory: {root}")
    sources = sorted(root.rglob("*.mcap"))
    if not sources:
        parser.error(f"No MCAP files found in {root}")
    topics = tuple(dict.fromkeys(t.lstrip("/") for t in args.topics))
    joints = tuple(dict.fromkeys(args.required_joints))
    if any(not value for value in (*topics, *joints)):
        parser.error("Topic and joint names must not be empty")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = (
        (args.report or Path.cwd() / f"mcap_joint_check_{stamp}.json")
        .expanduser()
        .absolute()
    )
    if report_path.suffix.lower() != ".json":
        parser.error("Report path must end in .json")
    report: dict[str, object] = {
        "input_dir": str(root),
        "topics": topics,
        "required_joints": joints,
        "delete_requested": args.delete,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    results: list[_ScanResult] = []
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    if args.delete:
        print(
            "Mode: DELETE -- confirmed invalid MCAPs will be permanently "
            "deleted after scanning.",
            flush=True,
        )
    else:
        print(
            "Mode: CHECK ONLY -- all input MCAPs are retained. "
            "Add --delete to delete invalid files.",
            flush=True,
        )
    try:
        # Exclusive creation also prevents overwriting reports or source data.
        with report_path.open("x", encoding="utf-8") as handle:
            for index, source in enumerate(sources, 1):
                result = _scan(source, topics, joints, typestore)
                results.append(result)
                print(
                    f"[{index}/{len(sources)}] {source.name}: "
                    "invalid JointState messages="
                    f"{sum(i.count for i in result.issues)}",
                    flush=True,
                )
                for issue in result.issues:
                    print(
                        f"  /{issue.topic}: "
                        f"{issue.count}/{result.counts[issue.topic]}"
                        f" messages; missing={issue.missing_joints}; "
                        f"duplicates={issue.duplicate_joints}; "
                        f"names/positions={issue.name_count}/{issue.position_count}",
                        flush=True,
                    )
                if result.missing_topics:
                    print(
                        f"  Missing topics: {result.missing_topics}",
                        flush=True,
                    )
                if result.error:
                    print(
                        f"  Scan error (retained): {result.error}", flush=True
                    )
            _write_report(handle, report, results)
            if args.delete:
                for result in results:
                    if not result.issues or result.error:
                        continue
                    path = Path(result.path)
                    try:
                        if _fingerprint(path) != result.fingerprint:
                            raise OSError("File changed since scan; retained")
                        path.unlink()
                        result.deleted = True
                        print(f"DELETED {path}", flush=True)
                    except OSError as exc:
                        result.delete_error = f"{type(exc).__name__}: {exc}"
                        print(f"DELETE FAILED {path}: {exc}", flush=True)
                    _write_report(handle, report, results)
            for result in results:
                if result.issues and not result.deleted:
                    print(f"RETAINED INVALID MCAP: {result.path}", flush=True)
            retained = sum(bool(r.issues) and not r.deleted for r in results)
            if retained and not args.delete:
                print(
                    "No MCAPs were deleted. Rerun this command with --delete "
                    "to remove fully scanned invalid files.",
                    flush=True,
                )
            report["summary"] = {
                "files": len(results),
                "files_with_invalid_messages": sum(
                    bool(r.issues) for r in results
                ),
                "deleted": sum(r.deleted for r in results),
                "retained_invalid_files": retained,
                "scan_errors": sum(r.error is not None for r in results),
                "files_missing_topics": sum(
                    bool(r.missing_topics) for r in results
                ),
            }
            report["finished_at"] = datetime.now(timezone.utc).isoformat()
            _write_report(handle, report, results)
    except OSError as exc:
        parser.error(f"Report I/O failed at {report_path}: {exc}")
    print(
        f"Done: files={len(results)}, "
        "files with invalid joint states="
        f"{sum(bool(r.issues) for r in results)}, "
        f"deleted={sum(r.deleted for r in results)}, "
        f"retained invalid={retained}. "
        f"Report: {report_path}",
        flush=True,
    )
    return int(
        any(
            r.error
            or r.delete_error
            or r.missing_topics
            or (r.issues and not r.deleted)
            for r in results
        )
    )


def _scan(
    path: Path,
    topics: tuple[str, ...],
    joints: tuple[str, ...],
    typestore: Typestore,
) -> _ScanResult:
    """Fully scan one file; any read/decode/change error blocks its deletion.

    Check plain CDR1, required/unique names, and matching positions, using
    the same message constraints as fix_mcap_left_arm.py. Aggregate invalid
    message shapes per topic with their first occurrence.
    Other topics are ignored. Never modify the source file.
    """
    result = _ScanResult(str(path), dict.fromkeys(topics, 0))
    issues: dict[
        tuple[str, tuple[str, ...], tuple[str, ...], int, int],
        _JointStateIssue,
    ] = {}
    try:
        result.fingerprint = _fingerprint(path)
        with Reader(path) as reader:
            connections = [
                c for c in reader.connections if c.topic.lstrip("/") in topics
            ]
            for conn in connections:
                if (
                    not isinstance(conn.ext, ConnectionExtRosbag2)
                    or conn.ext.serialization_format != "cdr"
                    or conn.msgtype
                    not in (
                        "sensor_msgs/JointState",
                        "sensor_msgs/msg/JointState",
                    )
                ):
                    raise ValueError(
                        f"Expected ROS 2 JointState CDR: {conn.topic}"
                    )
            # Rosbags interprets an empty connection filter as all topics.
            messages = (
                reader.messages(connections=connections) if connections else ()
            )
            for conn, timestamp, raw in messages:
                topic = conn.topic.lstrip("/")
                result.counts[topic] += 1
                try:
                    if raw[:2] not in (b"\x00\x00", b"\x00\x01"):
                        raise ValueError("Expected plain CDR1 JointState")
                    msg = typestore.deserialize_cdr(
                        raw, "sensor_msgs/msg/JointState"
                    )
                except Exception as exc:
                    raise ValueError(
                        f"/{topic} message #{result.counts[topic]}, "
                        f"log_time_ns={timestamp}: {type(exc).__name__}: {exc}"
                    ) from exc
                missing = tuple(
                    name for name in joints if name not in msg.name
                )
                duplicate_names = tuple(
                    sorted(
                        name
                        for name, count in Counter(msg.name).items()
                        if count > 1
                    )
                )
                name_count, position_count = len(msg.name), len(msg.position)
                if missing or duplicate_names or name_count != position_count:
                    key = (
                        topic,
                        missing,
                        duplicate_names,
                        name_count,
                        position_count,
                    )
                    if key not in issues:
                        issues[key] = _JointStateIssue(
                            topic,
                            list(missing),
                            result.counts[topic],
                            timestamp,
                            list(duplicate_names),
                            name_count,
                            position_count,
                        )
                        result.issues.append(issues[key])
                    issues[key].count += 1
        if _fingerprint(path) != result.fingerprint:
            raise OSError("File changed during scan; retained")
        result.missing_topics = [t for t, n in result.counts.items() if not n]
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def _fingerprint(path: Path) -> tuple[int, ...]:
    """Capture file identity and modification state; reject symlinks."""
    info = path.lstat()
    if not S_ISREG(info.st_mode):
        raise OSError(
            f"Expected a regular file, not a symlink/directory: {path}"
        )
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _write_report(
    handle: TextIO, report: dict[str, object], results: list[_ScanResult]
) -> None:
    """Persist scan/deletion evidence before permitting the next deletion."""
    report["files"] = [asdict(result) for result in results]
    handle.seek(0)
    json.dump(report, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
    handle.truncate()
    handle.flush()
    os.fsync(handle.fileno())


if __name__ == "__main__":
    raise SystemExit(main())
