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

"""Read-only startup clock check against the robot's wired NTP server.

This does not set time or authorize motion. It fails closed unless three or
more consecutive samples over two seconds have absolute offset <= 10 ms and
round-trip time <= 20 ms. Runtime ROS freshness checks remain mandatory.
The configured address must be the same robot that publishes ROS timestamps.
"""

from __future__ import annotations
import argparse
import ipaddress
import json
import socket
import struct
import time

__all__ = ["main"]

_NTP_EPOCH = 2_208_988_800
_MAX_OFFSET_S = 0.010
_MAX_RTT_S = 0.020


def main(argv: list[str] | None = None) -> int:
    """Return 0 only after stable measured synchronization; never set time.

    Args:
        argv (list[str], optional): CLI arguments, or process arguments.

    Returns:
        int: Zero on success, one on timeout; argparse rejects bad arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="192.168.2.33")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    try:
        ipaddress.IPv4Address(args.server)
    except ipaddress.AddressValueError:
        parser.error("--server must be the robot's numeric wired IPv4 address")
    if not 0 < args.timeout <= 30:
        parser.error("--timeout must be positive and at most 30 seconds")
    try:
        _wait_for_clock(args.server, args.timeout)
    except TimeoutError as error:
        print(f"[clock-preflight] BLOCKED: {error}", flush=True)
        return 1
    print(
        "[clock-preflight] PASS: stable robot/client clock alignment",
        flush=True,
    )
    return 0


def _query(server: str, timeout: float) -> tuple[float, float]:
    """Measure server-minus-client offset and RTT, in seconds.

    Owns one connected UDP socket, closed on every exit. Rejects malformed,
    unsynchronized, mismatched-origin responses and local wall-clock steps.
    NTP is unauthenticated here: this is a check on a trusted wired robot LAN,
    not an identity or permission mechanism. Raises OSError or ValueError.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(timeout)
        sock.connect((server, 123))
        start_mono = time.monotonic()
        start_ns = time.time_ns()
        seconds, nanos = divmod(start_ns, 1_000_000_000)
        stamp = struct.pack(
            "!II",
            (seconds + _NTP_EPOCH) % 2**32,
            nanos * 2**32 // 1_000_000_000,
        )
        request = bytearray(48)
        request[0] = (4 << 3) | 3
        request[40:48] = stamp
        sock.send(request)
        response = sock.recv(512)
        end_ns = time.time_ns()
        end_mono = time.monotonic()
    elapsed = end_mono - start_mono
    if abs((end_ns - start_ns) * 1e-9 - elapsed) > 0.001:
        raise ValueError("local wall clock stepped during NTP query")
    if len(response) < 48:
        raise ValueError("short NTP response")
    flags = response[0]
    if flags & 7 != 4 or (flags >> 3) & 7 not in (3, 4):
        raise ValueError("invalid NTP response mode/version")
    if flags >> 6 == 3 or not 1 <= response[1] <= 15:
        raise ValueError("NTP server is unsynchronized or refused request")
    if response[24:32] != stamp:
        raise ValueError("NTP response does not match this request")
    if response[32:40] == bytes(8) or response[40:48] == bytes(8):
        raise ValueError("NTP response has zero receive/transmit timestamp")
    now_ntp = end_ns * 1e-9 + _NTP_EPOCH
    times = []
    for start in (32, 40):
        sec, fraction = struct.unpack("!II", response[start : start + 8])
        raw = sec + fraction / 2**32
        # Unfold NTP's 32-bit era nearest the current local clock.
        times.append(raw + round((now_ntp - raw) / 2**32) * 2**32)
    received, sent = times
    processing = sent - received
    rtt = elapsed - processing
    if processing < 0 or rtt < -0.001:
        raise ValueError("NTP timestamps have impossible ordering")
    offset = (
        (received - (start_ns * 1e-9 + _NTP_EPOCH)) + (sent - now_ntp)
    ) / 2
    # Include server processing in the conservative startup latency bound.
    return offset, max(elapsed, rtt)


def _wait_for_clock(server: str, timeout: float) -> None:
    """Require consecutive healthy queries; bad samples reset stability."""
    deadline = time.monotonic() + timeout
    stable_since = None
    count = 0
    reason = "no valid NTP sample"
    while time.monotonic() < deadline:
        try:
            offset, rtt = _query(server, min(1.0, deadline - time.monotonic()))
            good = abs(offset) <= _MAX_OFFSET_S and 0 <= rtt <= _MAX_RTT_S
            reason = f"offset={offset * 1000:.3f} ms RTT={rtt * 1000:.3f} ms"
            print(
                "[clock-preflight] "
                + json.dumps(
                    dict(
                        server=server,
                        offset_ms=round(offset * 1000, 3),
                        rtt_ms=round(rtt * 1000, 3),
                        valid=good,
                    )
                ),
                flush=True,
            )
        except (OSError, ValueError) as error:
            good = False
            reason = str(error)
            print(f"[clock-preflight] waiting: {reason}", flush=True)
        now = time.monotonic()
        if good:
            stable_since = now if stable_since is None else stable_since
            count += 1
            if count >= 3 and now - stable_since >= 2 and now <= deadline:
                return
        else:
            stable_since, count = None, 0
        time.sleep(max(0, min(1.0, deadline - time.monotonic())))
    raise TimeoutError(
        f"clock not stable within {timeout:g}s ({reason}); "
        "check wired NTP synchronization before enabling model control"
    )


if __name__ == "__main__":
    raise SystemExit(main())
