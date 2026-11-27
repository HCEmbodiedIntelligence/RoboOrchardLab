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

"""Clock startup fence tests; all UDP and clock operations are simulated."""

import struct

import pytest

from projects.holobrain.scripts import hc_tj_clock_check as clock


@pytest.fixture
def udp(monkeypatch):
    state = {"offset": 0.003, "mutate": lambda response: response}
    mono = iter([10.0, 10.002])
    wall = iter([1_794_360_000_000_000_000, 1_794_360_000_002_000_000])
    monkeypatch.setattr(clock.time, "monotonic", lambda: next(mono))
    monkeypatch.setattr(clock.time, "time_ns", lambda: next(wall))

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            state["closed"] = True

        def settimeout(self, timeout):
            state["timeout"] = timeout

        def connect(self, endpoint):
            state["endpoint"] = endpoint

        def send(self, request):
            self.request = request

        def recv(self, size):
            response = bytearray(48)
            response[0], response[1] = 0x24, 10
            response[24:32] = self.request[40:48]
            sec, frac = struct.unpack("!II", self.request[40:48])
            # Symmetric 1 ms path in each direction, negligible server time.
            sent = sec + frac / 2**32 + 0.001 + state["offset"]
            stamp = struct.pack("!II", int(sent), int(sent % 1 * 2**32))
            response[32:40] = response[40:48] = stamp
            return state["mutate"](response)

    monkeypatch.setattr(clock.socket, "socket", lambda *args: Socket())
    return state


class TestNtpQuery:
    def test_measures_positive_offset_and_closes_socket(self, udp):
        offset, rtt = clock._query("192.168.2.33", 0.8)
        assert offset == pytest.approx(0.003, abs=1e-6)
        assert rtt == pytest.approx(0.002, abs=1e-6)
        assert udp["endpoint"] == ("192.168.2.33", 123)
        assert udp["timeout"] == 0.8
        assert udp["closed"]

    @pytest.mark.parametrize(
        "field,value",
        [
            (0, 0x23),
            (0, 0xE4),
            (1, 0),
            (1, 16),
            (24, 0),
        ],
    )
    def test_rejects_invalid_metadata(self, udp, field, value):
        def mutate(response):
            response[field] = value
            return response

        udp["mutate"] = mutate
        with pytest.raises(ValueError):
            clock._query("192.168.2.33", 1)
        assert udp["closed"]

    def test_rejects_zero_timestamp(self, udp):
        def mutate(response):
            response[32:40] = bytes(8)
            return response

        udp["mutate"] = mutate
        with pytest.raises(ValueError, match="zero"):
            clock._query("192.168.2.33", 1)

    def test_rejects_short_packet(self, udp):
        udp["mutate"] = lambda response: response[:20]
        with pytest.raises(ValueError, match="short"):
            clock._query("192.168.2.33", 1)

    def test_rejects_wall_clock_step(self, udp, monkeypatch):
        wall = iter([1_794_360_000_000_000_000, 1_794_360_000_052_000_000])
        monkeypatch.setattr(clock.time, "time_ns", lambda: next(wall))
        with pytest.raises(ValueError, match="stepped"):
            clock._query("192.168.2.33", 1)


@pytest.fixture
def monotonic(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(clock.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        clock.time, "sleep", lambda dt: now.__setitem__(0, now[0] + dt)
    )
    return now


class TestClockPreflight:
    def test_requires_consecutive_samples_for_two_seconds(
        self, monkeypatch, monotonic
    ):
        samples = iter(
            [(0, 0.001), (0.06, 0.001), (0, 0.001), (0, 0.001), (0, 0.001)]
        )
        monkeypatch.setattr(clock, "_query", lambda *args: next(samples))
        clock._wait_for_clock("192.168.2.33", 10)
        assert monotonic[0] == 4

    @pytest.mark.parametrize(
        "sample", [(0.06, 0.001), (-0.06, 0.001), (0, 0.021)]
    )
    def test_offset_or_rtt_failure_blocks(
        self, monkeypatch, monotonic, sample
    ):
        monkeypatch.setattr(clock, "_query", lambda *args: sample)
        with pytest.raises(TimeoutError, match="not stable"):
            clock._wait_for_clock("192.168.2.33", 3)
        assert monotonic[0] == 3

    def test_cli_failure_is_nonzero(self, monkeypatch, monotonic, capsys):
        def query(*args):
            raise OSError("NTP unreachable")

        monkeypatch.setattr(clock, "_query", query)
        assert clock.main(["--timeout", "3"]) == 1
        assert "BLOCKED" in capsys.readouterr().out

    def test_cli_pass(self, monkeypatch, monotonic):
        monkeypatch.setattr(clock, "_query", lambda *args: (0.001, 0.001))
        assert clock.main([]) == 0
        assert monotonic[0] == 2

    @pytest.mark.parametrize(
        "args",
        [["--server", "hostname"], ["--timeout", "nan"], ["--timeout", "31"]],
    )
    def test_cli_rejects_unbounded_or_invalid_inputs(self, args):
        with pytest.raises(SystemExit) as error:
            clock.main(args)
        assert error.value.code == 2
