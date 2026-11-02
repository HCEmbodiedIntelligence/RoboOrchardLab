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

"""Route-level tests using the existing model-server import fixture."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from projects.holobrain.scripts.hc_tj_async_core import ProtocolError
from projects.holobrain.scripts.hc_tj_control import RobotControlGate


def _server():
    spec = importlib.util.spec_from_file_location(
        "_control_test_server_helpers",
        Path(__file__).with_name("test_inference_server.py"),
    )
    module = importlib.util.module_from_spec(spec)
    # These route tests mock model computation entirely; do not require or
    # initialize PyTorch/CUDA just to test authentication and action fencing.
    with patch.dict(sys.modules, {"torch": types.ModuleType("torch")}):
        spec.loader.exec_module(module)
        server = module._load_script_module()
    server._require_control = True
    server._control_token = "test-control-token"
    return server


class ControlledInferenceServerTest(unittest.TestCase):
    def setUp(self):
        self.server = _server()
        self.robot = RobotControlGate()
        self.robot.update_ownership(
            {
                "controller_instance_id": "hc",
                "session_id": "session",
                "epoch": 1,
                "sequence": 1,
                "published_stamp_ns": 1_000_000_000,
                "state": "AUTO",
                "phase": "grant",
                "source": "motion",
            },
            1_000_000_000,
        )

    def test_control_authentication_precedes_state_update(self):
        self.server.request = types.SimpleNamespace(headers={})
        self.assertEqual(self.server.hc_tj_control_update()[1], 401)
        snapshot = self.robot.snapshot()
        self.server.request = types.SimpleNamespace(
            headers={"Authorization": "Bearer test-control-token"},
            get_json=lambda **_: snapshot,
        )
        receipt = self.server.hc_tj_control_update()
        self.assertTrue(receipt["allowed"])
        self.assertEqual(receipt["window_id"], snapshot["window_id"])

    def test_legacy_routes_cannot_bypass_protected_service(self):
        self.server.request = object()
        self.assertEqual(self.server.model_infer()[1], 409)
        self.assertEqual(self.server.hc_tj_async_model_infer()[1], 409)

    def test_inference_authentication_precedes_decoding(self):
        decoder = Mock()
        self.server._infer_hc_tj_async = decoder
        self.server.request = types.SimpleNamespace(headers={})
        self.assertEqual(self.server.hc_tj_controlled_model_infer()[1], 401)
        decoder.assert_not_called()

    def test_paused_upload_keeps_computing_preview(self):
        decoder = Mock(return_value={"actions": [[1.0]]})
        self.server._infer_hc_tj_async = decoder
        self.server.request = types.SimpleNamespace(
            headers={"Authorization": "Bearer test-control-token"},
            files={"head_color": object()},
            form={"control_window": "old", "control_server": "old"},
        )
        for _ in range(5):
            result = self.server.hc_tj_controlled_model_infer()
            self.assertEqual(result["status"], "ok")
            self.assertFalse(result["actions_allowed"])
            self.assertEqual(result["actions"], [[1.0]])
        self.assertEqual(decoder.call_count, 5)

    def test_preview_does_not_hide_invalid_model_input(self):
        self.server._infer_hc_tj_async = Mock(
            side_effect=ProtocolError("invalid observation")
        )
        self.server.request = types.SimpleNamespace(
            headers={"Authorization": "Bearer test-control-token"},
            files={},
            form={},
        )
        result, status = self.server.hc_tj_controlled_model_infer()
        self.assertEqual(status, 400)
        self.assertEqual(result["error"], "invalid observation")

    def test_success_echoes_execution_identity(self):
        snapshot = self.robot.snapshot()
        receipt = self.server._control_gate.update(snapshot)
        self.server._infer_hc_tj_async = Mock(
            return_value={"actions": [[1.0]]}
        )
        self.server.request = types.SimpleNamespace(
            headers={"Authorization": "Bearer test-control-token"},
            files={},
            form={
                "control_window": snapshot["window_id"],
                "control_server": receipt["server_instance"],
            },
        )
        result = self.server.hc_tj_controlled_model_infer()
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["actions_allowed"])
        self.assertEqual(result["control_window"], snapshot["window_id"])
        self.assertEqual(result["control_server"], receipt["server_instance"])


if __name__ == "__main__":
    unittest.main()
