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
from io import BytesIO
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
TRI_CAMERA_PROFILE = "tri_camera_dual_arm"
HC_TJ_PROFILE = "hc_tj"


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
    input_profile: str = TRI_CAMERA_PROFILE
    """Request schema: ``tri_camera_dual_arm`` or hc_tj-specific ``hc_tj``."""


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


def _load_npy(
    request_data: dict[str, Any], name: str, dtype 
) -> np.ndarray:
    """Load one multipart ``.npy`` field with an explicit dtype."""
    return np.load(BytesIO(request_data[name].read())).astype(dtype)


def _decode_hc_tj_request(
    request_data: dict[str, Any],
) -> MultiArmManipulationInput:
    """Decode hc_tj's head RGB-D, zhi_Link-to-eye and 14-joint schema."""
    intrinsic = np.eye(4, dtype=np.float64)
    intrinsic[:3, :3] = _load_npy(request_data, "head_intrinsic", np.float64)
    right_arm_state = _load_npy(request_data, "right_arm_state", np.float32)
    left_arm_state = _load_npy(request_data, "left_arm_state", np.float32)
    joint_state = np.concatenate([right_arm_state, left_arm_state], axis=-1)[
        None, :
    ]
    return MultiArmManipulationInput(
        image={"head": [_load_npy(request_data, "head_color", np.uint8)]},
        depth={
            "head": [
                _load_npy(request_data, "head_depth", np.float64) / 1000.0
            ]
        },
        history_joint_state=joint_state,
        intrinsic={"head": intrinsic},
        t_world2cam={
            "head": _load_npy(request_data, "head_world2cam", np.float64)
        },
        instruction=request_data.get("instruction", ""),
    )


def decode_request(request_data: dict[str, Any]) -> MultiArmManipulationInput:
    """Decode a multipart request according to the configured robot profile."""
    if args.input_profile == HC_TJ_PROFILE:
        return _decode_hc_tj_request(request_data)
    if args.input_profile != TRI_CAMERA_PROFILE:
        raise ValueError(f"Unsupported input profile: {args.input_profile!r}")
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
        output = pipeline(model_input)
        actions = output.action

        actions = torch.nn.functional.interpolate(
            actions.permute(1, 0)[None],
            scale_factor=INTERPOLATION,
            mode="linear",
            align_corners=True,
        )[0].permute(1, 0)
        actions = actions[: int(args.valid_action_step * INTERPOLATION)]

        action_values = actions.cpu().numpy()
        if args.input_profile == HC_TJ_PROFILE:
            res = dict(
                right_arm_actions=action_values[:, : args.num_joints].tolist(),
                left_arm_actions=action_values[:, args.num_joints :].tolist(),
                action_horizon=len(actions),
            )
        else:
            res = dict(
                left_arm_actions=action_values[:, : args.num_joints].tolist(),
                right_arm_actions=action_values[:, args.num_joints :].tolist(),
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
        if args.input_profile == HC_TJ_PROFILE:
            required_keys = [
                "head_color",
                "head_depth",
                "head_intrinsic",
                "head_world2cam",
                "right_arm_state",
                "left_arm_state",
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
