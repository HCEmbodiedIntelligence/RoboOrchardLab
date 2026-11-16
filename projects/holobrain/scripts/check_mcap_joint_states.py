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

"""Find invalid JointState data and optionally delete affected MCAP files.

Requires ``pip install rosbags``; ROS is not needed. Recursively inspect four
action/observation JointState topics for Joint1_L..Joint7_L by default.
Every selected message must have unique names, contain the required joints,
and have one position per name, as required by fix_mcap_left_arm.py.
Every selected topic must contain at least one message, and its first message
must have finite required positions so the file can also serve as a reference.
Parse only through position, preserving producer-specific trailing bytes just
like fix_mcap_left_arm.py, which imports these checks. All container messages
and connection metadata are read, including pass-through topics. Custom
--topics/--required-joints validate only the supplied requirements. Use the
defaults for the left-arm rewriter.

Without --delete, only scan and report. With --delete, permanently unlink
only fully scanned, unchanged regular files containing invalid joint states
(including malformed target CDR) or missing/empty selected topics. Container
read errors and unusable connection metadata block deletion and return 1;
retained scan failures have NOT passed and must be resolved before rewriting.
Symlink files are retained.
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
import math
import os
import struct
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from stat import S_ISREG
from tempfile import TemporaryDirectory
from typing import TextIO

from rosbags.interfaces import Connection, ConnectionExtRosbag2
from rosbags.rosbag2 import Reader, StoragePlugin, Writer
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

__all__ = [
    "main",
    "DEFAULT_JOINTS",
    "DEFAULT_TOPICS",
    "InvalidJointStateError",
    "position_offsets",
    "check_connection",
    "copy_connections",
]

DEFAULT_JOINTS = tuple(f"Joint{i}_L" for i in range(1, 8))
DEFAULT_TOPICS = (
    "action/joint_states",
    "observation/joint_states",
    "action/left_arm/joint_states",
    "observation/left_arm/joint_states",
)


class InvalidJointStateError(ValueError):
    """Confirmed target-message rejection with optional shape details."""

    def __init__(
        self,
        reason: str,
        *,
        missing_joints: tuple[str, ...] = (),
        duplicate_joints: tuple[str, ...] = (),
        name_count: int | None = None,
        position_count: int | None = None,
    ) -> None:
        super().__init__(reason)
        self.missing_joints = missing_joints
        self.duplicate_joints = duplicate_joints
        self.name_count = name_count
        self.position_count = position_count


@dataclass(slots=True)
class _JointStateIssue:
    topic: str
    missing_joints: list[str]
    first_ordinal: int
    first_log_time_ns: int
    duplicate_joints: list[str]
    name_count: int | None
    position_count: int | None
    reason: str
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

    With --delete, missing/empty selected topics also qualify for permanent
    file deletion. Container/metadata errors and changed files block deletion.

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
        help="Permanently delete MCAPs with invalid data or missing topics.",
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
        help="Required nonempty JointState topics; leading / is optional.",
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
    if args.delete:
        print(
            "Mode: DELETE -- MCAPs with invalid joint states or missing/empty "
            "required topics will be permanently deleted after scanning.",
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
                result = _scan(source, topics, joints)
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
                        f"names/positions={issue.name_count}/"
                        f"{issue.position_count}; "
                        f"first message=#{issue.first_ordinal}, "
                        f"log_time_ns={issue.first_log_time_ns}; "
                        f"reason={issue.reason}",
                        flush=True,
                    )
                if result.missing_topics:
                    print(
                        f"  Missing topics: {result.missing_topics}",
                        flush=True,
                    )
                if result.error:
                    print(
                        "  SCAN FAILED (retained, NOT passed): "
                        f"{result.error}",
                        flush=True,
                    )
            _write_report(handle, report, results)
            if args.delete:
                for result in results:
                    if result.error or not (
                        result.issues or result.missing_topics
                    ):
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
                if (
                    result.issues or result.missing_topics
                ) and not result.deleted:
                    print(f"RETAINED INVALID MCAP: {result.path}", flush=True)
            retained = sum(
                bool(r.issues or r.missing_topics) and not r.deleted
                for r in results
            )
            if retained and not args.delete:
                print(
                    "No MCAPs were deleted. Rerun this command with --delete "
                    "to remove fully scanned invalid files.",
                    flush=True,
                )
            report["summary"] = {
                "files": len(results),
                "passed_files": sum(
                    not (r.error or r.issues or r.missing_topics)
                    for r in results
                ),
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
        "files missing topics="
        f"{sum(bool(r.missing_topics) for r in results)}, "
        f"deleted={sum(r.deleted for r in results)}, "
        f"retained invalid={retained}, "
        f"scan errors={sum(r.error is not None for r in results)}. "
        f"Report: {report_path}",
        flush=True,
    )
    return int(
        any(
            r.error
            or r.delete_error
            or ((r.issues or r.missing_topics) and not r.deleted)
            for r in results
        )
    )


def position_offsets(
    data: bytes,
    required_joints: tuple[str, ...] = DEFAULT_JOINTS,
) -> tuple[str, dict[str, int]]:
    """Locate required float64 positions, preserving producer-specific tails.

    Return endian and offsets relative to the original encapsulated CDR1 bytes.
    Validate header strings, unique/required names and complete position data.
    Velocity, effort and any trailing bytes are deliberately left unparsed,
    matching the byte-preserving rewriter. Never mutate the input.

    Raises:
        InvalidJointStateError: Unsupported/truncated CDR, invalid strings
            or an incompatible name/position shape.
    """
    if data[:2] not in (b"\x00\x00", b"\x00\x01"):
        raise InvalidJointStateError(
            "Expected plain CDR1 JointState (big/little endian)"
        )
    endian = "<" if data[1] else ">"
    payload = memoryview(data)[4:]
    cursor = 8  # Header.stamp: int32 sec + uint32 nanosec.

    def read_u32() -> int:
        nonlocal cursor
        cursor = (cursor + 3) & ~3
        value = struct.unpack_from(endian + "I", payload, cursor)[0]
        cursor += 4
        return value

    def read_string() -> str:
        nonlocal cursor
        length = read_u32()
        end = cursor + length
        if length < 1 or end > len(payload) or payload[end - 1] != 0:
            raise InvalidJointStateError("Invalid CDR string in JointState")
        value = bytes(payload[cursor : end - 1]).decode("utf-8")
        cursor = end
        return value

    try:
        read_string()  # Header.frame_id.
        names = [read_string() for _ in range(read_u32())]
        size = read_u32()
    except (struct.error, UnicodeDecodeError) as exc:
        raise InvalidJointStateError(
            f"Malformed JointState CDR: {exc}"
        ) from exc
    cursor = (cursor + 7) & ~7  # CDR alignment excludes the 4-byte header.
    missing = tuple(name for name in required_joints if name not in names)
    duplicates = tuple(
        sorted(name for name, n in Counter(names).items() if n > 1)
    )
    truncated = cursor + size * 8 > len(payload)
    if missing or duplicates or size != len(names) or truncated:
        raise InvalidJointStateError(
            "JointState needs unique required names and "
            "matching, complete positions",
            missing_joints=missing,
            duplicate_joints=duplicates,
            name_count=len(names),
            position_count=size,
        )
    return endian, {
        name: 4 + cursor + names.index(name) * 8 for name in required_joints
    }


def check_connection(connection: Connection) -> None:
    """Reject selected connections that cannot carry ROS 2 JointState CDR."""
    if (
        not isinstance(connection.ext, ConnectionExtRosbag2)
        or connection.ext.serialization_format != "cdr"
        or connection.msgtype
        not in ("sensor_msgs/JointState", "sensor_msgs/msg/JointState")
    ):
        raise InvalidJointStateError(
            f"Expected ROS2 JointState CDR on {connection.topic}"
        )


def copy_connections(
    connections: Iterable[Connection],
    writer: Writer,
) -> dict[int, Connection]:
    """Register source connections, raising on writer incompatibility.

    Validate all connections, including pass-through topics. Normalize older
    type names and derive missing ROS 2 hashes from the recorded schema.
    Return source IDs mapped to writer connections. Used by both the rewriter
    and the checker's empty temporary bag to exercise actual writer rules.
    The caller owns the writer and removes partial outputs after failures.
    """
    typestore = get_typestore(Stores.EMPTY)
    result = {}
    for conn in connections:
        if not isinstance(conn.ext, ConnectionExtRosbag2):
            raise ValueError(f"Expected ROS 2 connection: {conn}")
        msgtype = conn.msgtype
        if msgtype.count("/") == 1:
            package, name = msgtype.split("/")
            msgtype = f"{package}/msg/{name}"
        digest = conn.digest
        if not digest:
            typestore.register(get_types_from_msg(conn.msgdef.data, msgtype))
            digest = typestore.hash_rihs01(msgtype)
        result[conn.id] = writer.add_connection(
            conn.topic,
            msgtype,
            msgdef=conn.msgdef.data,
            rihs01=digest,
            serialization_format=conn.ext.serialization_format,
            offered_qos_profiles=conn.ext.offered_qos_profiles,
        )
    return result


def _scan(
    path: Path,
    topics: tuple[str, ...],
    joints: tuple[str, ...],
) -> _ScanResult:
    """Check every message using the rewriter's byte and metadata contract.

    Aggregate rejected target payloads and continue scanning, so malformed
    messages qualify for deletion only after complete container traversal.
    First selected messages must also be usable as finite reference targets.
    Validate pass-through connection metadata and read all messages as the
    rewriter does. Container/metadata/read/change errors block deletion.
    Custom topics/joints scope these checks to the supplied requirements.
    Never modify the source file.
    """
    result = _ScanResult(str(path), dict.fromkeys(topics, 0))
    issues: dict[tuple, _JointStateIssue] = {}
    try:
        result.fingerprint = _fingerprint(path)
        with Reader(path) as reader, TemporaryDirectory() as temp:
            # An empty temporary bag checks the exact writer metadata rules.
            with Writer(
                Path(temp) / "metadata_check",
                version=9,
                storage_plugin=StoragePlugin.MCAP,
            ) as writer:
                copy_connections(reader.connections, writer)
            for conn, timestamp, raw in reader.messages():
                topic = conn.topic.lstrip("/")
                if topic not in topics:
                    continue
                result.counts[topic] += 1
                try:
                    check_connection(conn)
                    endian, offsets = position_offsets(raw, joints)
                    if result.counts[topic] == 1 and not all(
                        math.isfinite(
                            struct.unpack_from(endian + "d", raw, offset)[0]
                        )
                        for offset in offsets.values()
                    ):
                        raise InvalidJointStateError(
                            "Non-finite first-message reference positions"
                        )
                except InvalidJointStateError as exc:
                    key = (
                        topic,
                        exc.missing_joints,
                        exc.duplicate_joints,
                        exc.name_count,
                        exc.position_count,
                        str(exc),
                    )
                    if key not in issues:
                        issues[key] = _JointStateIssue(
                            topic=topic,
                            missing_joints=list(exc.missing_joints),
                            first_ordinal=result.counts[topic],
                            first_log_time_ns=timestamp,
                            duplicate_joints=list(exc.duplicate_joints),
                            name_count=exc.name_count,
                            position_count=exc.position_count,
                            reason=str(exc),
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
