# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
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

import argparse
import json
import logging
import os
import sys
import time
from io import BytesIO
from threading import Lock
from typing import Any

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from flask import Flask, Response, jsonify, request  # noqa: E402
from gevent.pywsgi import WSGIServer  # noqa: E402
from robo_orchard_core.utils.cli import (  # noqa: E402
    SettingConfig,
    pydantic_from_argparse,
)
from robo_orchard_core.utils.logging import LoggerManager  # noqa: E402

from projects.holobrain.scripts.hc_tj_async_core import (  # noqa: E402
    ACTION_HORIZON,
    ACTION_HZ,
    HC_TJ_JOINT_NAMES,
    ProtocolError,
    validate_inference_request,
)
from robo_orchard_lab.models.holobrain.pipeline import (  # noqa: E402
    HoloBrainInferencePipeline,
)
from robo_orchard_lab.models.holobrain.processor import (  # noqa: E402
    MultiArmManipulationInput,
)

logger = LoggerManager().get_child(__name__)

CONTROL_HZ = 200  # Hz for realbot control
TRAINING_HZ = 30  # Hz for training
INTERPOLATION = CONTROL_HZ / TRAINING_HZ


class Config(SettingConfig):
    model_dir: str = "models"
    """The directory of the model, including the model_config,
    ckpt and pipeline, you could get it from the scripts/export.py script."""
    inference_prefix: str = "inference"
    """The prefix of the inference pipeline config file."""
    port: int = 2000
    """The port of the server."""
    server_name: str = "holobrain"
    """The name of the server."""
    num_joints: int = 7
    """The number of joints."""
    valid_action_step: int = 64
    """The number of valid action steps."""


parser = argparse.ArgumentParser(description=Config.__doc__)
try:
    args: Config = pydantic_from_argparse(Config, parser)
except SystemExit as e:
    # Handle the case where the script is run with --help
    if e.code == 2:
        parser.print_help()
    exit(0)

pipeline = HoloBrainInferencePipeline.load_pipeline(
    directory=args.model_dir,
    inference_prefix=args.inference_prefix,
    device="cuda",
    load_weights=True,
    load_impl="native",
)
pipeline.model.eval()
logger.info(f"Model server {args.server_name} started on port {args.port}")

app = Flask(__name__)
_pipeline_lock = Lock()


def _decode_hc_tj_async_request(
    request_data: dict[str, Any],
) -> MultiArmManipulationInput:
    """Decode the existing own-branch HC-TJ v1 multipart wire contract."""
    required = (
        "head_color",
        "head_depth",
        "head_intrinsic",
        "zhi_to_eye",
        "current_state",
        "joint_names",
        "instruction",
    )
    missing = [key for key in required if key not in request_data]
    if missing:
        raise ProtocolError(f"missing required fields: {missing}")
    try:
        arrays = {
            key: np.load(BytesIO(request_data[key].read()), allow_pickle=False)
            for key in (*required[:5], "remaining_actions")
            if key in request_data
        }
        names = json.loads(request_data["joint_names"])
        raw_delay = request_data.get("delay_horizon")
        delay = int(raw_delay) if raw_delay is not None else None
        if raw_delay is not None and str(delay) != raw_delay:
            raise ValueError("delay_horizon must be an integer")
        if not isinstance(names, list) or not all(
            isinstance(name, str) for name in names
        ):
            raise ValueError("joint_names must be a JSON string array")
    except (KeyError, TypeError, ValueError) as error:
        raise ProtocolError(str(error)) from error
    value = validate_inference_request(
        head_color=arrays["head_color"],
        head_depth=arrays["head_depth"],
        head_intrinsic_mat=arrays["head_intrinsic"],
        zhi_to_eye_mat=arrays["zhi_to_eye"],
        current_state=arrays["current_state"],
        joint_names=names,
        instruction=request_data["instruction"],
        remaining_actions=arrays.get("remaining_actions"),
        delay_horizon=delay,
    )
    return MultiArmManipulationInput(
        image={"head": [value.head_color]},
        depth={"head": [value.head_depth]},
        history_joint_state=[value.current_state],
        intrinsic={"head": value.head_intrinsic_mat},
        t_world2cam={"head": value.zhi_to_eye_mat},
        instruction=value.instruction,
        remaining_actions=value.remaining_actions[None]
        if value.remaining_actions is not None
        else None,
        delay_horizon=value.delay_horizon,
    )


def _infer_hc_tj_async(request_data: dict[str, Any]) -> dict[str, Any]:
    """Return native 30 Hz, 16-axis HC-TJ actions without resampling."""
    model_input = _decode_hc_tj_async_request(request_data)
    if model_input.remaining_actions is not None:
        model = pipeline.model
        if hasattr(model, "module"):
            model = model.module
        plugin = getattr(
            getattr(model, "decoder", None), "async_inference_plugin", None
        )
        if (
            type(plugin).__name__ != "RTCInferencePlugin"
            or type(plugin).__module__
            != "robo_orchard_lab.models.rtc_plugin.rtc_plugin"
        ):
            raise ProtocolError(
                "remaining_actions requires a model with RTC enabled"
            )
    started = time.perf_counter()
    with _pipeline_lock, torch.inference_mode():
        output = pipeline(model_input)
    actions = torch.as_tensor(output.action).detach().cpu().numpy()
    if (
        actions.ndim != 2
        or actions.shape[1] != 16
        or not 0 < len(actions) <= ACTION_HORIZON
        or not np.isfinite(actions).all()
    ):
        raise RuntimeError(
            "pipeline returned invalid HC-TJ actions; "
            "expected finite [1..64,16]"
        )
    return dict(
        actions=actions.astype(np.float32).tolist(),
        joint_names=list(HC_TJ_JOINT_NAMES),
        action_hz=ACTION_HZ,
        action_horizon=len(actions),
        inference_ms=(time.perf_counter() - started) * 1000,
    )


@app.route(f"/{args.server_name}/hc_tj_async/v1", methods=["POST"])
def hc_tj_async_model_infer() -> Response | tuple[Response, int]:
    """Serve HC-TJ v1 alongside the existing tri-camera endpoint."""
    try:
        return jsonify(_infer_hc_tj_async({**request.files, **request.form}))
    except ProtocolError as error:
        return jsonify(error=str(error), error_type="protocol"), 400
    except Exception as error:
        logger.exception("HC-TJ asynchronous inference failed")
        return jsonify(error=str(error), error_type="inference"), 500


def decode_request(request_data) -> MultiArmManipulationInput:
    images = {
        "left": [
            np.load(BytesIO(request_data["left_color"].read())).astype(
                np.uint8
            )
        ],
        "right": [
            np.load(BytesIO(request_data["right_color"].read())).astype(
                np.uint8
            )
        ],
        "middle": [
            np.load(BytesIO(request_data["middle_color"].read())).astype(
                np.uint8
            )
        ],
    }

    depths = {
        "left": [
            np.load(BytesIO(request_data["left_depth"].read())).astype(
                np.float64
            )
            / 1000.0
        ],
        "right": [
            np.load(BytesIO(request_data["right_depth"].read())).astype(
                np.float64
            )
            / 1000.0
        ],
        "middle": [
            np.load(BytesIO(request_data["middle_depth"].read())).astype(
                np.float64
            )
            / 1000.0
        ],
    }

    left_arm_state = np.load(
        BytesIO(request_data["left_arm_state"].read())
    ).astype(np.float32)
    right_arm_state = np.load(
        BytesIO(request_data["right_arm_state"].read())
    ).astype(np.float32)
    joint_state = np.concatenate([left_arm_state, right_arm_state], axis=-1)[
        None, :
    ]

    intrinsics = np.eye(4)[None].repeat(3, axis=0)
    intrinsics[0, :3] = np.load(
        BytesIO(request_data["left_intrinsic"].read())
    ).astype(np.float64)
    intrinsics[1, :3] = np.load(
        BytesIO(request_data["right_intrinsic"].read())
    ).astype(np.float64)
    intrinsics[2, :3] = np.load(
        BytesIO(request_data["middle_intrinsic"].read())
    ).astype(np.float64)
    intrinsics = {
        "left": intrinsics[0],
        "right": intrinsics[1],
        "middle": intrinsics[2],
    }

    remaining_actions = (
        np.load(BytesIO(request_data["remaining_actions"].read())).astype(
            np.float32
        )[None]
        if request_data.get("remaining_actions") is not None
        else None
    )

    if remaining_actions is not None and remaining_actions.size > 0:
        # downsample the remaining actions
        remaining_actions = (
            torch.nn.functional.interpolate(
                torch.from_numpy(remaining_actions).permute(0, 2, 1),
                scale_factor=1 / INTERPOLATION,
                mode="linear",
                align_corners=True,
            )
            .permute(0, 2, 1)
            .numpy()
        )
    else:
        remaining_actions = None

    return MultiArmManipulationInput(
        image=images,
        depth=depths,
        history_joint_state=joint_state,  # type: ignore
        intrinsic=intrinsics,
        instruction=request_data.get("instruction", ""),
        remaining_actions=remaining_actions,
        delay_horizon=int(request_data.get("delay_horizon", 0)),
    )


def infer(request_data):
    try:
        model_input = decode_request(request_data)
        with _pipeline_lock:
            output = pipeline(model_input)
        actions = output.action

        actions = torch.nn.functional.interpolate(
            actions.permute(1, 0)[None],
            scale_factor=INTERPOLATION,
            mode="linear",
            align_corners=True,
        )[0].permute(1, 0)
        actions = actions[: int(args.valid_action_step * INTERPOLATION)]

        res = dict(
            left_arm_actions=actions[:, : args.num_joints]
            .cpu()
            .numpy()
            .tolist(),
            right_arm_actions=actions[:, args.num_joints :]
            .cpu()
            .numpy()
            .tolist(),
            action_horizon=len(actions),
        )
        return res
    except Exception as e:
        logging.exception(f"Error during inference: {e}")
        return None


@app.route(f"/{args.server_name}", methods=["POST"])
def model_infer():
    try:
        data = {**request.files, **request.form}

        required_keys = [
            "left_color",
            "middle_color",
            "right_color",
            "left_depth",
            "middle_depth",
            "right_depth",
            "left_intrinsic",
            "middle_intrinsic",
            "right_intrinsic",
            "left_arm_state",
            "right_arm_state",
            "instruction",
        ]
        for key in required_keys:
            if key not in data:
                return jsonify({"error": f"Missing key: {key}"}), 400

        res = infer(data)
        if res is None:
            return jsonify({"error": "Inference failed"}), 500
        return Response(json.dumps(res), mimetype="application/json")
    except Exception as e:
        logging.error(f"Error in endpoint: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    http_server = WSGIServer(("", args.port), app)
    http_server.serve_forever()
