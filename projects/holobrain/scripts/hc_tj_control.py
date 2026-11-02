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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fence action execution without stopping observation or inference.

HC remains the authority. RobotControlGate only narrows its permission, using
fresh ownership snapshots and a non-reusable local execution window. Forward
snapshots through ControlRelay, independently of model requests. A service
restart or interrupted relay requires a new robot-side window.
"""

from __future__ import annotations
import json
import math
import threading
import time
import uuid
from collections.abc import Callable
from socketserver import ThreadingMixIn
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen
from wsgiref.simple_server import WSGIServer, make_server

__all__ = [
    "ControlRelay",
    "ControlledInference",
    "RobotControlGate",
    "ServiceControlGate",
    "make_control_server",
]


class _ControlWSGIServer(ThreadingMixIn, WSGIServer):
    daemon_threads = True


def make_control_server(host: str, port: int, app: Callable) -> WSGIServer:
    """Serve control separately from the single admitted native model call."""
    return make_server(host, port, app, server_class=_ControlWSGIServer)


class RobotControlGate:
    """Own the local execution window under the final-publication lock.

    Call update_ownership with HC's published ROS timestamp and the receiving
    node's ROS clock. Compare resume_started to local receive monotonic times
    of every required observation stream. Holding lock across allows and final
    publication serializes revocation with publication.
    """

    def __init__(
        self,
        timeout_s: float = 0.5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("control timeout must be finite and positive")
        self.lock = threading.RLock()
        self.timeout_s, self.clock = timeout_s, clock
        self.sender_instance = uuid.uuid4().hex
        self.resume_started = math.inf
        self._window = uuid.uuid4().hex
        self._revision = 0
        self._allowed = False
        self._reason = "control_unavailable"
        self._received = -math.inf
        self._stamp = 0
        self._sequence = -1
        self._identity: tuple[str, str, int] | None = None
        self._retired_instances: set[str] = set()
        self._ownership: dict[str, Any] = {}

    def _close(self, reason: str) -> None:
        if self._allowed:
            self._window = uuid.uuid4().hex
        self._allowed, self._reason = False, reason
        self.resume_started = math.inf

    def _expire(self) -> None:
        if self.clock() - self._received > self.timeout_s:
            self._close("control_unavailable")

    def invalidate(self, reason: str) -> None:
        """Close now; only a strictly newer HC heartbeat can reopen."""
        with self.lock:
            self._close(reason)
            self._window = uuid.uuid4().hex

    def update_ownership(self, data: dict[str, Any], now_ns: int) -> bool:
        """Accept fresh ordered HC snapshots; report a changed window.

        Invalid payloads close the gate and raise ValueError. Old snapshots are
        ignored and never renew the lease. New controller instances retire the
        previous instance, preventing delayed messages from switching back.
        """
        try:
            instance, session = (
                data["controller_instance_id"],
                data["session_id"],
            )
            epoch, sequence = data["epoch"], data["sequence"]
            stamp = data["published_stamp_ns"]
            if (
                not isinstance(instance, str)
                or not instance
                or not isinstance(session, str)
                or any(
                    type(v) is not int or v < 0
                    for v in (epoch, sequence, stamp)
                )
                or not isinstance(data["state"], str)
                or not isinstance(data["phase"], str)
                or not isinstance(data["source"], str)
            ):
                raise ValueError("invalid ownership identity or state")
        except (KeyError, TypeError, ValueError) as error:
            self.invalidate("invalid_control")
            raise ValueError("invalid HC ownership snapshot") from error
        with self.lock:
            before = self._window
            self._expire()
            age = (now_ns - stamp) * 1e-9
            if not -0.05 <= age <= self.timeout_s:
                return before != self._window
            if instance in self._retired_instances:
                return before != self._window
            same_instance = self._identity and instance == self._identity[0]
            if same_instance and (
                stamp <= self._stamp or sequence < self._sequence
            ):
                return before != self._window
            identity = (instance, session, epoch)
            if identity != self._identity:
                if self._identity and instance != self._identity[0]:
                    self._retired_instances.add(self._identity[0])
                self._close("control_changed")
                self._window = uuid.uuid4().hex
            allowed = (
                data["state"] == "AUTO"
                and data["phase"] == "grant"
                and data["source"] == "motion"
                and bool(session)
            )
            self._identity, self._stamp, self._sequence = (
                identity,
                stamp,
                sequence,
            )
            self._received = self.clock()
            self._ownership = {
                key: data[key]
                for key in (
                    "controller_instance_id",
                    "session_id",
                    "epoch",
                    "sequence",
                    "state",
                    "phase",
                    "source",
                )
            }
            self._ownership.update(
                reason=str(data.get("reason", "")),
                dagger_enabled=bool(
                    data.get("dagger_enabled", data["state"] != "OFF")
                ),
            )
            if allowed and not self._allowed:
                self._window = uuid.uuid4().hex
                self.resume_started = self.clock()
            if not allowed:
                self._close(str(data.get("reason") or data["state"].lower()))
            else:
                self._allowed, self._reason = True, ""
            return before != self._window

    def allows(self, window_id: str) -> bool:
        """Recheck the local window and control heartbeat at publication."""
        with self.lock:
            self._expire()
            return self._allowed and window_id == self._window

    def snapshot(self) -> dict[str, Any]:
        """Produce one ordered relay heartbeat, expiring stale authority."""
        with self.lock:
            self._expire()
            self._revision += 1
            return {
                "sender_instance": self.sender_instance,
                "revision": self._revision,
                "window_id": self._window,
                "allowed": self._allowed,
                "reason": self._reason,
                "ownership": dict(self._ownership),
            }


class ServiceControlGate:
    """Mirror one authenticated robot relay using a local heartbeat lease.

    Expired or closed windows cannot be reopened. update reports
    resync_required until the robot produces a new execution window.
    """

    def __init__(
        self,
        timeout_s: float = 0.5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.lock = threading.RLock()
        self.clock, self.timeout_s = clock, timeout_s
        self.server_instance = uuid.uuid4().hex
        self._sender = ""
        self._revision = -1
        self._window = ""
        self._allowed = False
        self._reason = "control_unavailable"
        self._received = -math.inf
        self._retired_windows: set[str] = set()
        self._retired_senders: set[str] = set()

    def _expire(self) -> None:
        if self.clock() - self._received > self.timeout_s and self._allowed:
            self._retired_windows.add(self._window)
            self._allowed, self._reason = False, "control_unavailable"

    def update(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply an authenticated ordered heartbeat and return its receipt."""
        sender, window = data.get("sender_instance"), data.get("window_id")
        revision, allowed = data.get("revision"), data.get("allowed")
        if (
            not isinstance(sender, str)
            or not sender
            or len(sender) > 128
            or not isinstance(window, str)
            or not window
            or len(window) > 128
            or type(revision) is not int
            or revision < 0
            or type(allowed) is not bool
        ):
            raise ValueError("invalid control heartbeat")
        with self.lock:
            self._expire()
            if sender in self._retired_senders:
                raise ValueError("retired control relay")
            if sender == self._sender and revision <= self._revision:
                raise ValueError("reordered control heartbeat")
            if sender != self._sender:
                if self._sender:
                    self._retired_senders.add(self._sender)
                    self._retired_windows.add(self._window)
                self._sender = sender
            if window != self._window and self._window:
                self._retired_windows.add(self._window)
            self._window, self._revision = window, revision
            self._received = self.clock()
            resync = allowed and window in self._retired_windows
            self._allowed = allowed and not resync
            self._reason = str(data.get("reason") or "suppressed")
            if not allowed:
                self._retired_windows.add(window)
            if resync:
                self._reason = "new_control_window_required"
            return {
                "server_instance": self.server_instance,
                "window_id": window,
                "allowed": self._allowed,
                "resync_required": resync,
            }

    def allows(self, window_id: str, server_instance: str) -> bool:
        """Check boot identity, execution window, and current lease."""
        with self.lock:
            self._expire()
            return (
                self._allowed
                and window_id == self._window
                and server_instance == self.server_instance
            )


class ControlledInference:
    """Run one model operation independently of action authorization.

    The model runs in its HTTP worker, while the threaded server can handle
    control requests independently. There is no model backlog. Admission is
    atomic with closing the gate. Results are executable only when the same
    window allows actions both before and after computation; preview results
    cannot become executable if authorization arrives during computation.
    No state lock is held during model execution.
    """

    def __init__(self, gate: ServiceControlGate) -> None:
        self.gate = gate
        self._busy = False
        self.submitted = 0
        self.dropped = 0

    def run(
        self,
        window_id: str,
        server_instance: str,
        operation: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        """Always compute when idle; fence permission to execute the result."""

        def dropped(reason: str) -> dict[str, Any]:
            self.dropped += 1
            return {
                "status": "dropped",
                "reason": reason,
                "actions_allowed": False,
                "control_window": window_id,
                "control_server": server_instance,
            }

        with self.gate.lock:
            if self._busy:
                return dropped("inference_busy")
            allowed_at_start = self.gate.allows(window_id, server_instance)
            self._busy = True
            self.submitted += 1
        try:
            result = operation()
            with self.gate.lock:
                return {
                    **result,
                    "status": "ok",
                    "actions_allowed": allowed_at_start
                    and self.gate.allows(window_id, server_instance),
                    "control_window": window_id,
                    "control_server": server_instance,
                }
        finally:
            with self.gate.lock:
                self._busy = False


class ControlRelay:
    """Forward state independently of inference; close on uncertain transport.

    get_snapshot returns a robot gate snapshot. invalidate(reason) must close
    that robot gate (directly or through the robot HTTP gateway). start/close
    own the thread. ready(window) is required before accepting model actions.
    """

    def __init__(
        self,
        get_snapshot: Callable[[], dict[str, Any]],
        invalidate: Callable[[str], None],
        control_url: str,
        token: str,
        interval_s: float = 0.1,
    ) -> None:
        if not token or interval_s <= 0:
            raise ValueError(
                "control token and positive interval are required"
            )
        self.get_snapshot, self.invalidate = get_snapshot, invalidate
        self.control_url, self.token, self.interval_s = (
            control_url,
            token,
            interval_s,
        )
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._server = ""
        self._window = ""
        self._allowed = False
        self._last_ok = -math.inf
        self.last_error = ""

    @property
    def server_instance(self) -> str:
        """Return the boot identity from the last authenticated handshake."""
        with self._lock:
            return self._server

    def ready(self, window_id: str) -> bool:
        """Check the acknowledged window and relay freshness."""
        with self._lock:
            return (
                self._allowed
                and window_id == self._window
                and time.monotonic() - self._last_ok <= 0.5
            )

    def sync_once(self) -> None:
        """Send one heartbeat; failures revoke robot-side execution."""
        try:
            snapshot = self.get_snapshot()
            req = Request(
                self.control_url,
                json.dumps(snapshot).encode(),
                {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.token}",
                },
                method="POST",
            )
            with urlopen(req, timeout=0.4) as response:
                receipt = json.load(response)
            server = receipt["server_instance"]
            if not isinstance(server, str) or not server:
                raise ValueError("missing inference server identity")
            if receipt["window_id"] != snapshot["window_id"]:
                raise ValueError("control receipt window mismatch")
            with self._lock:
                restart = bool(self._server and self._server != server)
                self._server = server
                resync = restart or receipt.get("resync_required") is True
                self._allowed = receipt.get("allowed") is True and not resync
                self._window = snapshot["window_id"]
                self._last_ok = time.monotonic()
                self.last_error = ""
            if resync:
                self.invalidate("inference_control_resync")
        except (
            OSError,
            URLError,
            ValueError,
            KeyError,
            TypeError,
            RuntimeError,
        ) as error:
            with self._lock:
                self._allowed = False
                self.last_error = str(error)
            try:
                self.invalidate("inference_control_unavailable")
            except (OSError, ValueError, RuntimeError) as revoke_error:
                self.last_error += f"; robot revoke failed: {revoke_error}"

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sync_once()
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        """Start the independent relay once."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def close(self) -> None:
        """Stop the relay and revoke this execution window."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.invalidate("inference_control_closed")
