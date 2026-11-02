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

"""CPU tests for revocation, transport concurrency, and stale completions."""

import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen

from projects.holobrain.scripts.hc_tj_control import (
    ControlledInference,
    ControlRelay,
    RobotControlGate,
    ServiceControlGate,
    make_control_server,
)


def ownership(stamp=1_000_000_000, sequence=1, **changes):
    return {
        "controller_instance_id": "hc",
        "session_id": "session",
        "epoch": 1,
        "sequence": sequence,
        "published_stamp_ns": stamp,
        "state": "AUTO",
        "phase": "grant",
        "source": "motion",
        **changes,
    }


class RobotControlGateTest(unittest.TestCase):
    def setUp(self):
        self.time = 0.0
        self.gate = RobotControlGate(clock=lambda: self.time)

    def test_stale_latched_auto_cannot_open(self):
        self.gate.update_ownership(ownership(), 2_000_000_000)
        self.assertFalse(self.gate.snapshot()["allowed"])
        self.gate.update_ownership(ownership(), 1_000_000_000)
        self.assertTrue(self.gate.snapshot()["allowed"])

    def test_pause_release_and_new_auto_never_reuse_window(self):
        gate = self.gate
        gate.update_ownership(ownership(), 1_000_000_000)
        old = gate.snapshot()["window_id"]
        gate.update_ownership(
            ownership(1_100_000_000, 2, state="HOLD", epoch=2), 1_100_000_000
        )
        self.assertFalse(gate.allows(old))
        gate.update_ownership(ownership(), 1_100_000_000)
        self.assertFalse(gate.snapshot()["allowed"])
        gate.update_ownership(
            ownership(1_200_000_000, 3, epoch=3), 1_200_000_000
        )
        self.assertTrue(gate.snapshot()["allowed"])
        self.assertFalse(gate.allows(old))

    def test_same_authority_after_lease_loss_has_new_window(self):
        gate = self.gate
        data = ownership()
        gate.update_ownership(data, data["published_stamp_ns"])
        old = gate.snapshot()["window_id"]
        self.time = 0.6
        self.assertFalse(gate.allows(old))
        data["published_stamp_ns"] += 600_000_000
        gate.update_ownership(data, data["published_stamp_ns"])
        self.assertFalse(gate.allows(old))
        self.assertTrue(gate.snapshot()["allowed"])
        self.assertEqual(gate.resume_started, self.time)

    def test_duplicate_and_invalid_messages_do_not_renew(self):
        data = ownership()
        self.gate.update_ownership(data, data["published_stamp_ns"])
        self.time = 0.4
        self.gate.update_ownership(data, data["published_stamp_ns"])
        self.time = 0.6
        self.assertFalse(self.gate.snapshot()["allowed"])
        with self.assertRaises(ValueError):
            self.gate.update_ownership({}, 0)

    def test_controller_restart_retires_old_controller(self):
        gate = self.gate
        gate.update_ownership(ownership(), 1_000_000_000)
        gate.update_ownership(
            ownership(1_100_000_000, controller_instance_id="new"),
            1_100_000_000,
        )
        new = gate.snapshot()["window_id"]
        gate.update_ownership(ownership(1_200_000_000), 1_200_000_000)
        self.assertEqual(gate.snapshot()["window_id"], new)

    def test_invalidation_requires_new_heartbeat(self):
        data = ownership()
        self.gate.update_ownership(data, data["published_stamp_ns"])
        self.gate.invalidate("transport_lost")
        self.gate.update_ownership(data, data["published_stamp_ns"])
        self.assertFalse(self.gate.snapshot()["allowed"])


class ServiceControlGateTest(unittest.TestCase):
    def setUp(self):
        self.time = 0.0
        self.gate = ServiceControlGate(clock=lambda: self.time)
        self.robot = RobotControlGate(clock=lambda: self.time)
        self.robot.update_ownership(ownership(), 1_000_000_000)
        self.initial = self.robot.snapshot()
        self.gate.update(self.initial)

    def test_expired_window_requires_resynchronization(self):
        self.time = 0.6
        window = self.initial["window_id"]
        self.assertFalse(self.gate.allows(window, self.gate.server_instance))
        replay = {**self.initial, "revision": 2}
        self.assertTrue(self.gate.update(replay)["resync_required"])
        self.robot.update_ownership(ownership(1_600_000_000), 1_600_000_000)
        fresh = self.robot.snapshot()
        fresh["revision"] = 3
        self.assertTrue(self.gate.update(fresh)["allowed"])

    def test_old_heartbeat_and_old_service_instance_are_rejected(self):
        with self.assertRaises(ValueError):
            self.gate.update(self.initial)
        self.assertFalse(self.gate.allows(self.initial["window_id"], "old"))

    def test_suppressed_requests_keep_computing_preview(self):
        executor = ControlledInference(self.gate)
        self.robot.invalidate("user_pause")
        self.gate.update(self.robot.snapshot())
        for _ in range(8):
            result = executor.run(
                self.initial["window_id"],
                self.gate.server_instance,
                lambda: {"actions": [[1.0]]},
            )
            self.assertEqual(result["status"], "ok")
            self.assertFalse(result["actions_allowed"])
            self.assertEqual(result["actions"], [[1.0]])
        self.assertEqual(executor.submitted, 8)
        self.assertEqual(executor.dropped, 0)

    def test_grant_during_preview_never_promotes_result(self):
        gate = ServiceControlGate(clock=lambda: self.time)
        executor = ControlledInference(gate)

        def operation():
            gate.update(self.initial)
            return {"actions": [[1.0]]}

        result = executor.run(
            self.initial["window_id"], gate.server_instance, operation
        )
        self.assertTrue(
            gate.allows(self.initial["window_id"], gate.server_instance)
        )
        self.assertFalse(result["actions_allowed"])
        next_result = executor.run(
            self.initial["window_id"],
            gate.server_instance,
            lambda: {"actions": [[2.0]]},
        )
        self.assertTrue(next_result["actions_allowed"])

    def test_pause_and_resume_while_model_blocked_discards_late_result(self):
        executor = ControlledInference(self.gate)
        entered, release = threading.Event(), threading.Event()

        def operation():
            entered.set()
            self.assertTrue(release.wait(3))
            return {"actions": [[1.0]]}

        with ThreadPoolExecutor(max_workers=1) as pool:
            task = pool.submit(
                executor.run,
                self.initial["window_id"],
                self.gate.server_instance,
                operation,
            )
            self.assertTrue(entered.wait(2))
            try:
                self.robot.invalidate("pause")
                self.gate.update(self.robot.snapshot())
                self.robot.update_ownership(
                    ownership(1_100_000_000), 1_100_000_000
                )
                fresh = self.robot.snapshot()
                self.gate.update(fresh)
                self.assertEqual(
                    executor.run(
                        fresh["window_id"],
                        self.gate.server_instance,
                        lambda: self.fail("second model operation started"),
                    )["reason"],
                    "inference_busy",
                )
            finally:
                release.set()
            self.assertFalse(task.result()["actions_allowed"])
        self.assertEqual(executor.submitted, 1)

    def test_close_before_atomic_admission_computes_preview(self):
        executor = ControlledInference(self.gate)
        with self.gate.lock:
            self.robot.invalidate("pause")
            self.gate.update(self.robot.snapshot())
            result = executor.run(
                self.initial["window_id"],
                self.gate.server_instance,
                lambda: {"actions": [[1.0]]},
            )
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["actions_allowed"])
        self.assertEqual(executor.submitted, 1)


class ControlTransportTest(unittest.TestCase):
    def test_real_http_control_ack_does_not_wait_for_model(self):
        gate = ServiceControlGate(timeout_s=2)
        robot = RobotControlGate()
        robot.update_ownership(ownership(), 1_000_000_000)
        initial = robot.snapshot()
        gate.update(initial)
        executor = ControlledInference(gate)
        entered, release = threading.Event(), threading.Event()

        def operation():
            entered.set()
            release.wait(3)
            return {"actions": [[1]]}

        def app(environ, start_response):
            if environ["PATH_INFO"] == "/control":
                data = json.loads(
                    environ["wsgi.input"].read(int(environ["CONTENT_LENGTH"]))
                )
                result = gate.update(data)
            else:
                result = executor.run(
                    initial["window_id"], gate.server_instance, operation
                )
            payload = json.dumps(result).encode()
            start_response(
                "200 OK",
                [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(payload))),
                ],
            )
            return [payload]

        server = make_control_server("127.0.0.1", 0, app)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"

        def infer():
            with urlopen(base + "/infer", timeout=4) as response:
                return json.load(response)

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(infer)
                self.assertTrue(entered.wait(2))
                robot.invalidate("user_pause")
                req = Request(
                    base + "/control", json.dumps(robot.snapshot()).encode()
                )
                with urlopen(req, timeout=0.5) as response:
                    self.assertFalse(json.load(response)["allowed"])
                self.assertFalse(result.done())
                release.set()
                self.assertFalse(result.result()["actions_allowed"])
        finally:
            release.set()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_relay_error_closes_robot_gate(self):
        robot = RobotControlGate()
        robot.update_ownership(ownership(), 1_000_000_000)

        def broken():
            raise RuntimeError("robot gateway unavailable")

        relay = ControlRelay(
            broken,
            robot.invalidate,
            "http://127.0.0.1:1/control",
            "test-token",
        )
        relay.sync_once()
        self.assertFalse(robot.snapshot()["allowed"])
        self.assertIn("gateway unavailable", relay.last_error)


if __name__ == "__main__":
    unittest.main()
