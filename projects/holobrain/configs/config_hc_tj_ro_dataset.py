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

"""HoloBrain dataset configuration for the hc_tj dual-arm recording."""

from __future__ import annotations
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from dataset_factory import (
    processor_register,
    train_dataset_register,
    validation_dataset_register,
)

__all__ = ["build_datasets", "build_processors", "build_transforms"]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_URDF_PATH = (
    _REPO_ROOT
    / "hc_assets"
    / "hc_tj_description"
    / "urdf"
    / "hc_tj_robot.urdf"
)
_DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "arrow_dataset" / "hc_tj"
)
_RIGHT_JOINT_NAMES = [f"Joint{i}_R" for i in range(1, 8)]
_LEFT_JOINT_NAMES = [f"Joint{i}_L" for i in range(1, 8)]
_ARM_JOINT_NAMES = _RIGHT_JOINT_NAMES + _LEFT_JOINT_NAMES


def _joint_scale_shift_from_urdf(urdf: Path) -> list[list[float]]:
    root = ET.parse(urdf).getroot()
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if (
            limit is None
            or "lower" not in limit.attrib
            or "upper" not in limit.attrib
        ):
            continue
        limits[joint.attrib["name"]] = (
            float(limit.attrib["lower"]),
            float(limit.attrib["upper"]),
        )
    missing = [name for name in _ARM_JOINT_NAMES if name not in limits]
    if missing:
        raise ValueError(f"hc_tj URDF is missing joint limits for {missing}")
    return [
        [(upper - lower) / 2, (upper + lower) / 2]
        for lower, upper in (limits[name] for name in _ARM_JOINT_NAMES)
    ]


def _base_to_zhi_from_zero_pose(urdf: Path) -> list[list[float]]:
    """Return base_link -> zhi_Link at the excluded torso's zero pose."""
    import pytorch_kinematics as pk
    import torch

    chain = pk.build_chain_from_urdf(urdf.read_bytes())
    parameter_names = chain.get_joint_parameter_names()
    parameter_indices = {
        name: index for index, name in enumerate(parameter_names)
    }
    right_indices = [parameter_indices[name] for name in _RIGHT_JOINT_NAMES]
    left_indices = [parameter_indices[name] for name in _LEFT_JOINT_NAMES]
    if right_indices != list(range(5, 12)) or left_indices != list(
        range(13, 20)
    ):
        raise ValueError(
            "hc_tj URDF joint parameter order changed: expected right "
            "[5..11] and left [13..19]"
        )
    zero = torch.zeros((1, len(parameter_names)), dtype=torch.float32)
    # pytorch_kinematics returns zhi_Link -> base_link. The transform chain's
    # historical T_base2world field needs base_link -> zhi_Link.
    zhi_to_base_mat = chain.forward_kinematics(zero)["zhi_Link"].get_matrix()[
        0
    ]
    return torch.linalg.inv(zhi_to_base_mat).tolist()


dataset_config = {
    "hc_tj": {
        "data_paths": [str(_DATASET_PATH)],
        "urdf": str(_URDF_PATH),
        "cam_names": ["head"],
        "joint_scale_shift": _joint_scale_shift_from_urdf(_URDF_PATH),
        "T_base2world": _base_to_zhi_from_zero_pose(_URDF_PATH),
    }
}


def _expand_ro_data_paths(patterns: list[str]) -> list[str]:
    from glob import glob

    paths: list[str] = []
    for pattern in patterns:
        for matched in glob(pattern):
            path = Path(matched)
            if (path / "state.json").exists():
                paths.append(str(path))
            else:
                paths.extend(
                    str(state.parent)
                    for state in sorted(path.rglob("state.json"))
                    if state.is_file()
                )
    return sorted(set(paths))


def build_transforms(config, mode, data_config):
    """Build hc_tj transforms with all geometry expressed in zhi_Link."""
    import numpy as np

    from robo_orchard_lab.dataset.horizon_manipulation.transforms import (
        AddItems,
        CalibrationToExtrinsic,
        ConvertDataType,
        ExtrinsicNoise,
        GetProjectionMat,
        ItemSelection,
        JointStateNoise,
        MoveEgoToCam,
        MultiArmKinematics,
        RandomCropPaddingResize,
        Resize,
        SimpleStateSampling,
        ToTensor,
        UnsqueezeBatch,
    )

    state_weight = [1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1]
    loss_weights = np.asarray([[state_weight] * 14], dtype=np.float32)
    add_items = {
        "type": AddItems,
        "T_base2world": data_config["T_base2world"],
        "joint_mask": [True] * 14,
        "joint_scale_shift": data_config["joint_scale_shift"],
    }
    if mode == "training":
        add_items.update(
            state_loss_weights=loss_weights.tolist(),
            fk_loss_weight=loss_weights.tolist(),
        )

    state_sampling = {
        "type": SimpleStateSampling,
        "hist_steps": config["hist_steps"],
        "pred_steps": config["pred_steps"],
        "use_master_gripper": True,
        "use_master_joint": True,
    }
    destination_wh = config.get("dst_wh", (392, 252))
    patch_size = config.get("patch_size", 1)
    destination_wh = tuple(
        max(patch_size, size // patch_size * patch_size)
        for size in destination_wh
    )
    resize = {"type": Resize, "dst_wh": destination_wh}
    to_tensor = {"type": ToTensor}
    move_ego_to_cam = {"type": MoveEgoToCam, "cam_idx": "head"}
    projection = {"type": GetProjectionMat, "target_coordinate": "ego"}
    convert_dtype = {
        "type": ConvertDataType,
        "convert_map": {
            "imgs": "float32",
            "depths": "float32",
            "image_wh": "float32",
            "projection_mat": "float32",
            "embodiedment_mat": "float32",
        },
    }
    kinematics_config = {
        "urdf": data_config["urdf"],
        "arm_joint_id": [list(range(5, 12)), list(range(13, 20))],
        "arm_link_keys": [
            [f"Link{i}_R" for i in range(1, 8)],
            [f"Link{i}_L" for i in range(1, 8)],
        ],
        "finger_keys": [[], []],
    }
    kinematics = {"type": MultiArmKinematics, **kinematics_config}

    common_keys = [
        "imgs",
        "depths",
        "image_wh",
        "projection_mat",
        "embodiedment_mat",
        "hist_robot_state",
        "joint_scale_shift",
        "kinematics",
        "text",
        "uuid",
        "joint_mask",
    ]
    if mode == "training":
        return [
            add_items,
            state_sampling,
            {
                "type": RandomCropPaddingResize,
                "range_w": (-30, 30),
                "range_h": (-20, 30),
                "range_scale": None,
            },
            resize,
            to_tensor,
            {
                "type": ExtrinsicNoise,
                "noise_range": (0.04, 0.04, 0.04, 0.015, 0.015, 0.015),
            },
            move_ego_to_cam,
            projection,
            {
                "type": JointStateNoise,
                "noise_range": [[-0.02, 0.02]] * 14,
                "add_to_pred": True,
            },
            convert_dtype,
            kinematics,
            {
                "type": ItemSelection,
                "keys": common_keys
                + [
                    "pred_robot_state",
                    "pred_mask",
                    "fk_loss_weight",
                    "state_loss_weights",
                ],
            },
        ]
    if mode == "validation":
        return [
            add_items,
            state_sampling,
            resize,
            to_tensor,
            move_ego_to_cam,
            projection,
            convert_dtype,
            kinematics,
            {
                "type": ItemSelection,
                "keys": common_keys + ["pred_robot_state", "pred_mask"],
            },
        ]
    if mode == "deploy":
        return [
            add_items,
            state_sampling,
            resize,
            to_tensor,
            {
                "type": CalibrationToExtrinsic,
                "calibration": None,
                "cam_names": data_config["cam_names"],
                **kinematics_config,
            },
            move_ego_to_cam,
            projection,
            convert_dtype,
            kinematics,
            {
                "type": ItemSelection,
                "keys": common_keys + ["remaining_actions", "delay_horizon"],
            },
            {"type": UnsqueezeBatch},
        ]
    raise ValueError(f"Unsupported hc_tj dataset mode: {mode}")


@train_dataset_register()
@validation_dataset_register()
def build_datasets(config, dataset_names, mode, **kwargs):
    """Build the registered hc_tj training or validation dataset."""
    from robo_orchard_lab.dataset.hc_tj import HcTjRODataset
    from robo_orchard_lab.utils.build import build
    from robo_orchard_lab.utils.misc import as_sequence

    datasets = []
    for dataset_name, data_config in dataset_config.items():
        if dataset_name not in dataset_names:
            continue
        paths = _expand_ro_data_paths(data_config["data_paths"])
        if not paths:
            raise FileNotFoundError(
                "hc_tj RODataset was not found. Run "
                "scripts/convert_hc_tj_mcap.py first."
            )
        transforms = build_transforms(config, mode, data_config)
        dataset = HcTjRODataset(
            paths=paths,
            hist_steps=config["hist_steps"],
            pred_steps=config["pred_steps"],
            transforms=[build(item) for item in as_sequence(transforms)],
        )
        dataset.flag = int(
            uuid.uuid5(uuid.NAMESPACE_DNS, dataset_name).hex[:4], 16
        )
        datasets.append(dataset)
    return datasets


@processor_register()
def build_processors(config, dataset_names):
    """Build the hc_tj deployment processor when requested by name."""
    from robo_orchard_lab.models.holobrain import (
        HoloBrainProcessor,
        HoloBrainProcessorCfg,
    )

    processors = {}
    for dataset_name, data_config in dataset_config.items():
        if dataset_name not in dataset_names:
            continue
        processors[dataset_name] = HoloBrainProcessor(
            HoloBrainProcessorCfg(
                load_image=True,
                load_depth=config["with_depth"],
                valid_action_step=None,
                cam_names=data_config["cam_names"],
                transforms=build_transforms(config, "deploy", data_config),
            )
        )
    return processors
