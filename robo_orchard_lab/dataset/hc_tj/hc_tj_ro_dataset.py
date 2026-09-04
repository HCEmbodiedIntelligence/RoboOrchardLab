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

"""RODataset reader for 14-axis and 16-axis hc_tj dual-arm embodiments."""

from __future__ import annotations
from typing import Callable, Sequence

import cv2
import numpy as np
from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms import Compose

from robo_orchard_lab.dataset.horizon_manipulation.row_sampler import (
    EpisodeChunkSamplerConfig,
)
from robo_orchard_lab.dataset.robot.dataset import (
    ConcatRODataset,
    ROMultiRowDataset,
)
from robo_orchard_lab.dataset.robot.db_orm import Episode

__all__ = ["HcTjRODataset"]

_CAMERA_NAME = "head"
_REFERENCE_FRAME = "zhi_Link"
_CAMERA_FRAME = "eye"
_DEFAULT_JOINT_NAMES = (
    *[f"Joint{i}_R" for i in range(1, 8)],
    "R_ban",
    *[f"Joint{i}_L" for i in range(1, 8)],
    "L_ban",
)


class _HcTjArrowDataParse:
    """Decode hc_tj Arrow rows into the existing HoloBrain input contract.

    The stored camera pose has ``parent_frame_id="zhi_Link"`` and
    ``child_frame_id="eye"``. It represents ``eye -> zhi_Link`` and is
    inverted here to produce the transform code's historical
    ``T_world2cam`` field, whose "world" is hc_tj's ``zhi_Link`` reference.
    """

    def __init__(
        self,
        *,
        depth_scale: int,
        hist_steps: int,
        joint_names: tuple[str, ...],
    ) -> None:
        self.depth_scale = depth_scale
        self.hist_steps = hist_steps
        self.joint_names = joint_names

    @staticmethod
    def _instruction(data: dict) -> str:
        instruction = data.get("instruction")
        if instruction is None or instruction.json_content is None:
            raise ValueError("hc_tj frame is missing instruction metadata")
        try:
            text = instruction.json_content["description"]
        except KeyError as error:
            raise ValueError(
                "hc_tj instruction metadata is missing 'description'"
            ) from error
        if not isinstance(text, str) or not text:
            raise ValueError("hc_tj instruction description must be non-empty")
        return text

    def _joint_array(self, data: dict, column: str) -> np.ndarray:
        states = data[column]
        for state in states:
            if tuple(state.names) != self.joint_names:
                raise ValueError(
                    f"hc_tj {column} joint order mismatch: {state.names}"
                )
        values = np.stack([state.position for state in states])
        values = values.squeeze(1).astype(np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.joint_names):
            raise ValueError(
                f"hc_tj {column} must have shape "
                f"(steps, {len(self.joint_names)}), got "
                f"{values.shape}"
            )
        return values

    @staticmethod
    def _camera(
        data: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        camera = data[_CAMERA_NAME]
        image = cv2.imdecode(
            np.frombuffer(camera.sensor_data[0], np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            raise ValueError("Failed to decode hc_tj head RGB image")
        depth_camera = data[f"{_CAMERA_NAME}_depth"]
        depth = cv2.imdecode(
            np.frombuffer(depth_camera.sensor_data[0], np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        if depth is None or depth.ndim != 2 or depth.dtype != np.uint16:
            raise ValueError("Failed to decode hc_tj 16UC1 head depth image")
        if depth.shape != image.shape[:2]:
            raise ValueError("hc_tj RGB and depth image shapes do not match")

        intrinsic = np.eye(4, dtype=np.float64)
        intrinsic[:3, :3] = camera.intrinsic_matrices[0]
        pose = camera.pose
        if pose is None:
            raise ValueError("hc_tj head camera pose is missing")
        if (
            pose.parent_frame_id != _REFERENCE_FRAME
            or pose.child_frame_id != _CAMERA_FRAME
            or camera.frame_id != _CAMERA_FRAME
        ):
            raise ValueError(
                "hc_tj head camera must use eye -> zhi_Link frame semantics"
            )
        zhi_to_eye_mat = np.linalg.inv(
            pose.as_Transform3D_M().get_matrix()[0].numpy()
        ).astype(np.float64)
        return image, depth.astype(np.float64), intrinsic, zhi_to_eye_mat

    def __call__(self, data: dict) -> dict:
        """Decode one metadata-materialized, multi-row hc_tj sample."""
        image, depth, intrinsic, zhi_to_eye_mat = self._camera(data)
        episode = data.get("episode")
        task = data.get("task")
        if episode is None or task is None:
            raise ValueError("hc_tj frame is missing episode or task metadata")
        episode_info = episode.info or {}
        data.update(
            text=self._instruction(data),
            imgs=image[None],
            depths=(depth / self.depth_scale)[None],
            intrinsic=intrinsic[None],
            T_world2cam=zhi_to_eye_mat[None],
            joint_state=self._joint_array(data, "joints"),
            master_joint_state=self._joint_array(data, "actions"),
            step_index=data["frame_index"],
            step_index_in_chunk=self.hist_steps - 1,
            task_name=task.name,
            uuid=episode_info.get("uuid"),
            cam_names=[_CAMERA_NAME],
        )
        return data


class HcTjRODataset(TorchDataset):
    """Read hc_tj RODatasets for HoloBrain training.

    This adapter is specific to the hc_tj embodiment: it materializes
    RODataset metadata, samples configured arm or arm-and-gripper chunks,
    decodes the single aligned head RGB/depth pair, and preserves
    ``zhi_Link`` as the
    camera and kinematics reference.

    Args:
        paths: hc_tj RODataset directories.
        hist_steps: Number of historical state rows per sample.
        pred_steps: Number of future action rows per sample.
        depth_scale: Divisor converting uint16 depth values to meters.
        transforms: HoloBrain transforms applied after hc_tj row decoding.
        joint_names: Ordered joint names expected in the ``joints`` and
            ``actions`` columns. Defaults to the 16-axis arm-and-gripper
            layout.
    """

    def __init__(
        self,
        paths: list[str],
        hist_steps: int,
        pred_steps: int,
        depth_scale: int = 1000,
        transforms: list[Callable] | None = None,
        joint_names: Sequence[str] = _DEFAULT_JOINT_NAMES,
    ) -> None:
        if not paths:
            raise ValueError("hc_tj dataset paths must not be empty")
        joint_names = tuple(joint_names)
        if not joint_names or len(set(joint_names)) != len(joint_names):
            raise ValueError("hc_tj joint_names must be non-empty and unique")
        row_sampler = EpisodeChunkSamplerConfig(
            target_columns=["joints", "actions"],
            hist_steps=hist_steps,
            pred_steps=pred_steps,
        )
        datasets = [
            ROMultiRowDataset(
                dataset_path=path,
                row_sampler=row_sampler,
                meta_index2meta=True,
            )
            for path in paths
        ]
        composed = Compose(
            [
                _HcTjArrowDataParse(
                    depth_scale=depth_scale,
                    hist_steps=hist_steps,
                    joint_names=joint_names,
                ),
                *(transforms or []),
            ]
        )
        for dataset in datasets:
            dataset.set_transform(composed)
        self._concat = ConcatRODataset(datasets)

    @property
    def num_episode(self) -> int:
        """Return the number of episodes across all source datasets."""
        return sum(dataset.episode_num for dataset in self._concat.datasets)

    def get_episode_range(self, episode_index: int) -> tuple[int, int]:
        """Return the concatenated half-open frame range for one episode."""
        global_episode = 0
        frame_offset = 0
        for dataset in self._concat.datasets:
            for episode in dataset.iterate_meta(Episode):
                if global_episode == episode_index:
                    start = frame_offset + episode.dataset_begin_index
                    return start, start + episode.frame_num
                global_episode += 1
            frame_offset += len(dataset)
        raise KeyError(f"Episode index {episode_index} not found")

    def __len__(self) -> int:
        return len(self._concat)

    def __getitem__(self, index):
        """

        Args:
            index (int): 索引

        Returns:
            dict: 形式如下：
            {
                joints: <class 'list'>-shape: (65,)
                actions: <class 'list'>-shape: (65,)
                head: <class 'robo_orchard_core.datatypes.camera_data.BatchCameraDataEncoded'>-shape: None
                head_depth: <class 'robo_orchard_core.datatypes.camera_data.BatchCameraDataEncoded'>-shape: None
                index: <class 'int'>-shape: None
                frame_index: <class 'int'>-shape: None
                timestamp_min: <class 'int'>-shape: None
                timestamp_max: <class 'int'>-shape: None
                episode: <class 'robo_orchard_lab.dataset.robot.db_orm.episode.Episode'>-shape: None
                task: <class 'robo_orchard_lab.dataset.robot.db_orm.task.Task'>-shape: None
                robot: <class 'robo_orchard_lab.dataset.robot.db_orm.robot.Robot'>- 放着urdf的内容
                instruction: <class 'robo_orchard_lab.dataset.robot.db_orm.instruction.Instruction'> - 内容和task差不多
                --- add from transform ---
                text: <class 'str'>-shape: None
                imgs: <class 'numpy.ndarray'>-shape: (1, 400, 640, 3)
                depths: <class 'numpy.ndarray'>-shape: (1, 400, 640)
                intrinsic: <class 'numpy.ndarray'>-shape: (1, 4, 4)
                T_world2cam: <class 'numpy.ndarray'>-shape: (1, 4, 4) zhi_Link -> eye
                joint_state: <class 'numpy.ndarray'>-shape: (65, 16) 机器人关节实际返回位置
                master_joint_state: <class 'numpy.ndarray'>-shape: (65, 16) 下发给机器人的关节位置
                step_index: <class 'int'>-shape: None
                step_index_in_chunk: <class 'int'>-shape: None
                task_name: <class 'str'>-shape: None
                uuid: <class 'str'>-shape: None
                cam_names: <class 'list'>-shape: (1,)
                dataset_index: <class 'int'>-shape: None
            }
        """
        return self._concat[index]

    def __getitems__(self, indices):
        return self._concat.__getitems__(indices)
