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

import importlib.util
import json
import sys
import types
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from projects.holobrain.scripts.hc_tj_async_core import HC_TJ_JOINT_NAMES

SCRIPT_PATH = (
    Path(__file__).resolve().parents[4]
    / "projects"
    / "holobrain"
    / "scripts"
    / "inference_server.py"
)


class _FakeSettingConfig:
    pass


class _FakeLogger:
    def info(self, *_args, **_kwargs):
        return None

    warning = info
    error = info
    exception = info


class _FakeLoggerManager:
    def get_child(self, _name):
        return _FakeLogger()


class _FakeModel:
    def __init__(self):
        self.decoder = types.SimpleNamespace(async_inference_plugin=None)

    def eval(self):
        return None


class _FakePipeline:
    load_kwargs = None

    def __init__(self):
        self.model = _FakeModel()

    @classmethod
    def load_pipeline(cls, **kwargs):
        cls.load_kwargs = kwargs
        return cls()

    def __call__(self, _model_input):
        return types.SimpleNamespace(action=torch.zeros((64, 16)))


class _FakeMultiArmManipulationInput:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeFlaskApp:
    def route(self, *_args, **_kwargs):
        def _decorator(func):
            return func

        return _decorator


def _load_script_module(inference_prefix: str = "inference"):
    fake_cli_module = types.ModuleType("robo_orchard_core.utils.cli")
    fake_cli_module.SettingConfig = _FakeSettingConfig
    fake_cli_module.pydantic_from_argparse = lambda _config, _parser: (
        types.SimpleNamespace(
            model_dir="/tmp/model",
            inference_prefix=inference_prefix,
            port=2000,
            server_name="holobrain",
            num_joints=7,
            valid_action_step=64,
            input_profile="tri_camera_dual_arm",
        )
    )

    fake_logging_module = types.ModuleType("robo_orchard_core.utils.logging")
    fake_logging_module.LoggerManager = _FakeLoggerManager

    fake_pipeline_module = types.ModuleType(
        "robo_orchard_lab.models.holobrain.pipeline"
    )
    fake_pipeline_module.HoloBrainInferencePipeline = _FakePipeline

    fake_processor_module = types.ModuleType(
        "robo_orchard_lab.models.holobrain.processor"
    )
    fake_processor_module.MultiArmManipulationInput = (
        _FakeMultiArmManipulationInput
    )

    fake_flask_module = types.ModuleType("flask")
    fake_flask_module.Flask = lambda *_args, **_kwargs: _FakeFlaskApp()
    fake_flask_module.Response = object
    fake_flask_module.jsonify = lambda payload: payload
    fake_flask_module.request = object()

    fake_gevent_module = types.ModuleType("gevent.pywsgi")
    fake_gevent_module.WSGIServer = object

    spec = importlib.util.spec_from_file_location(
        "holobrain_inference_server_script", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None

    patched_modules = {
        "robo_orchard_core.utils.cli": fake_cli_module,
        "robo_orchard_core.utils.logging": fake_logging_module,
        "robo_orchard_lab.models.holobrain.pipeline": fake_pipeline_module,
        "robo_orchard_lab.models.holobrain.processor": fake_processor_module,
        "flask": fake_flask_module,
        "gevent.pywsgi": fake_gevent_module,
    }
    sys.modules[spec.name] = module
    with patch.dict(sys.modules, patched_modules):
        spec.loader.exec_module(module)
    return module


class InferenceServerScriptTest(unittest.TestCase):
    @staticmethod
    def _npy(value):
        payload = BytesIO()
        np.save(payload, value)
        payload.seek(0)
        return payload

    def _valid_async_request(self):
        return {
            "head_color": self._npy(np.zeros((4, 6, 3), dtype=np.uint8)),
            "head_depth": self._npy(np.ones((4, 6), dtype=np.float32)),
            "head_intrinsic": self._npy(np.eye(4, dtype=np.float64)),
            "zhi_to_eye": self._npy(np.eye(4, dtype=np.float64)),
            "current_state": self._npy(np.zeros(16, dtype=np.float32)),
            "joint_names": json.dumps(HC_TJ_JOINT_NAMES),
            "instruction": "pick the block",
        }

    def test_loads_pipeline_with_cli_inference_prefix(self):
        _FakePipeline.load_kwargs = None

        _load_script_module(inference_prefix="robotwin2_0")

        self.assertIsNotNone(_FakePipeline.load_kwargs)
        self.assertEqual(
            _FakePipeline.load_kwargs["inference_prefix"],
            "robotwin2_0",
        )
        self.assertEqual(_FakePipeline.load_kwargs["directory"], "/tmp/model")

    def test_decodes_and_returns_native_16_axis_actions(self):
        module = _load_script_module()
        model_input = module._decode_hc_tj_async_request(
            self._valid_async_request()
        )
        self.assertEqual(model_input.history_joint_state[0].shape, (16,))

        output = module._infer_hc_tj_async(self._valid_async_request())

        self.assertEqual(np.asarray(output["actions"]).shape, (64, 16))
        self.assertEqual(output["joint_names"], list(HC_TJ_JOINT_NAMES))
        self.assertEqual(output["action_hz"], 30)

    def test_rejects_wrong_shape_and_rtc_without_plugin(self):
        module = _load_script_module()
        bad = self._valid_async_request()
        bad["current_state"] = self._npy(np.zeros(14, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "current_state"):
            module._decode_hc_tj_async_request(bad)

        request_data = self._valid_async_request()
        request_data["remaining_actions"] = self._npy(
            np.zeros((3, 16), dtype=np.float32)
        )
        request_data["delay_horizon"] = "1"
        with self.assertRaisesRegex(ValueError, "RTCInferencePlugin"):
            module._infer_hc_tj_async(request_data)

    def test_passes_remaining_actions_when_rtc_is_configured(self):
        module = _load_script_module()
        rtc_type = type(
            "RTCInferencePlugin",
            (),
            {"__module__": ("robo_orchard_lab.models.rtc_plugin.rtc_plugin")},
        )
        module.pipeline.model.decoder.async_inference_plugin = rtc_type()
        request_data = self._valid_async_request()
        request_data["remaining_actions"] = self._npy(
            np.zeros((3, 16), dtype=np.float32)
        )
        request_data["delay_horizon"] = "1"

        model_input = module._decode_hc_tj_async_request(request_data)

        self.assertEqual(model_input.remaining_actions.shape, (1, 3, 16))
        request_data = self._valid_async_request()
        request_data["remaining_actions"] = self._npy(
            np.zeros((3, 16), dtype=np.float32)
        )
        request_data["delay_horizon"] = "1"
        output = module._infer_hc_tj_async(request_data)

        self.assertEqual(output["action_horizon"], 64)

    def test_async_endpoint_returns_protocol_error_response(self):
        module = _load_script_module()
        module.request = types.SimpleNamespace(files={}, form={})

        payload, status = module.hc_tj_async_model_infer()

        self.assertEqual(status, 400)
        self.assertEqual(payload["error_type"], "protocol")
        self.assertIn("missing required fields", payload["error"])

    def test_legacy_infer_response_format_is_unchanged(self):
        module = _load_script_module()
        with patch.object(module, "decode_request", return_value=object()):
            response = module.infer({})
        self.assertEqual(
            set(response),
            {"left_arm_actions", "right_arm_actions", "action_horizon"},
        )


if __name__ == "__main__":
    unittest.main()
