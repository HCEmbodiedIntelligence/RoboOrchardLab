<!--
Project RoboOrchard

Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# 为 HoloBrain 添加新机器人数据集并从 MCAP 启动训练

本文说明如何把一个新机器人的 ROS 2 MCAP 轨迹接入 HoloBrain。内容分为通用接入方法和仓库中 `hc_tj` 的完整实例。

最重要的原则是：**每种机器人必须使用自己的 URDF、关节定义、关节顺序、关节限位、运动学链、相机内外参和参考坐标系。不能把 Agilex 或其他机器人的配置直接套到新机器人上。**

## 1. 接入链路概览

完整链路如下：

```text
每个 MCAP 文件（一条 episode）
        │
        ▼
检查话题、消息类型、时间戳、关节顺序、图像和标定
        │
        ▼
按相机时间戳同步 RGB / 深度 / 观测状态 / 监督动作
        │
        ▼
EpisodePackaging + DatasetPackaging
        │
        ▼
RODataset（Arrow + DuckDB 元数据 + state.json）
        │
        ▼
机器人专用 Dataset Adapter
        │
        ▼
训练 transforms：采样、归一化、相机投影、FK、数据增强
        │
        ▼
dataset_factory 注册
        │
        ▼
机器人专用训练配置
        │
        ▼
scripts/train.py
```

通常需要新增或修改以下文件：

```text
projects/holobrain/scripts/convert_<robot>_mcap.py
robo_orchard_lab/dataset/<robot>/<robot>_ro_dataset.py
robo_orchard_lab/dataset/<robot>/__init__.py
projects/holobrain/configs/config_<robot>_ro_dataset.py
projects/holobrain/configs/config_holobrain_qwen_<robot>.py
projects/holobrain/configs/dataset_factory.py
tests/.../test_convert_<robot>_mcap.py
```

## 2. 先定义数据契约

写转换代码前，应明确以下内容。

### 2.1 训练观测和监督动作

至少确定：

- 哪个话题提供机器人当前状态，例如 `joint_states`。
- 哪个话题提供监督动作，例如 `joint_cmd`。
- 训练保留哪些关节，以及它们的固定顺序。
- 动作是绝对关节角、关节增量、末端位姿，还是其他控制量。
- 单位是否一致，例如关节角为弧度、深度为毫米。

HoloBrain 不会自动猜测动作语义。对于当前 `hc_tj`：

- `joints` 来自 `joint_states`，表示观测到的绝对关节位置。
- `actions` 来自 `joint_cmd`，表示监督的绝对目标关节位置。
- 只保留双臂 14 轴，顺序为右臂 7 轴后左臂 7 轴。
- `leg_1`、`leg_2`、`zhi` 不进入动作向量。

### 2.2 相机与坐标系

必须明确：

- RGB、深度是否对齐以及各自编码。
- 相机内参对应哪一路图像。
- 外参的父子坐标系、矩阵方向和四元数顺序。
- FK、相机外参和投影使用哪个共同参考坐标系。

对于 `hc_tj`：

- 训练相机名为 `head`，图像坐标系为 `eye`。
- RGB 保存为 JPEG，深度保存为 16UC1 PNG。
- 共同参考坐标系为 `zhi_Link`。
- 数据适配器将深度除以 `1000`，把毫米转换为米。
- 相机、双臂 FK 和投影都在 `zhi_Link` 语义下对齐。

### 2.3 时间同步

推荐以主相机 RGB 时间戳作为帧时间，分别为深度、状态和动作寻找最近邻消息，并设置最大容许误差。不要把相距很远的消息强行拼成一帧。

当前 `hc_tj` 转换器默认阈值为：

- 深度：50 ms。
- 关节状态：20 ms。
- 关节命令：20 ms。

超限帧会被丢弃并写入 `conversion_report.json`。

## 3. 将 MCAP 转为 RODataset

### 3.1 检查 MCAP

先确认每个文件代表一条完整 episode，并检查话题和类型：

```bash
conda activate holobrain

python - <<'PY'
from pathlib import Path
from rosbags.highlevel import AnyReader

root = Path("<MCAP_DIR>")
for path in sorted(root.rglob("*.mcap")):
    with AnyReader([path]) as reader:
        print(f"\n{path}")
        for connection in reader.connections:
            print(
                connection.topic,
                connection.msgtype,
                connection.msgcount,
            )
PY
```

检查重点：

- 必需话题是否存在且非空。
- 消息类型是否和转换器声明一致。
- 状态和动作中的关节名称、顺序、长度是否固定。
- RGB、深度能否解码。
- 标定 JSON 是否包含合法内外参。
- MCAP 是否正常闭合。

当前 `hc_tj` 转换器严格接受以下输入：

| 话题 | ROS 2 类型 | 用途 |
|---|---|---|
| `/io_teleop/camera_head/color` | `sensor_msgs/msg/CompressedImage` | 头部 RGB 和帧基准时间 |
| `/io_teleop/camera_head/depth` | `sensor_msgs/msg/CompressedImage` | 对齐深度 |
| `io_teleop/joint_states` | `sensor_msgs/msg/JointState` | 观测关节状态 |
| `io_teleop/joint_cmd` | `sensor_msgs/msg/JointState` | 监督关节动作 |
| `/io_teleop/camera_head/calibration` | `std_msgs/msg/String` | 相机内参 JSON |
| `/io_teleop/camera_head/extrinsics` | `std_msgs/msg/String` | `zhi_Link` 到相机相关的外参 JSON |

其他机器人应修改转换器的话题和解析契约，不能仅靠把 MCAP 路径传给 `convert_hc_tj_mcap.py` 来完成适配。

### 3.2 实现转换器

当前实例是：

```text
projects/holobrain/scripts/convert_hc_tj_mcap.py
```

一个转换器通常需要实现四层逻辑：

1. **读取和校验**：用 `rosbags.highlevel.AnyReader` 读取 MCAP，严格验证话题、类型、关节顺序、图像编码和标定。
2. **同步**：以主传感器时间戳生成帧，最近邻匹配其他流，并统计丢帧原因。
3. **episode 包装**：继承 `EpisodePackaging`，每一帧产出 `DataFrame`。
4. **数据集包装**：使用 `DatasetPackaging` 将多个 episode 写成 Arrow 数据和元数据数据库。

当前 `hc_tj` 每帧写入四个 feature：

| 字段 | 含义 | 存储内容 |
|---|---|---|
| `joints` | 当前机器人状态 | 14 轴绝对关节位置 |
| `actions` | 监督动作 | 14 轴绝对目标关节位置 |
| `head` | 头部 RGB | JPEG 编码数据、内参和外参 |
| `head_depth` | 对齐深度 | 16UC1 PNG 编码数据 |

同时应写入：

- `RobotData`：机器人名称和本机器人的 URDF。
- `TaskData`：稳定的任务名和描述。
- `InstructionData`：训练使用的自然语言指令。
- `EpisodeData`：episode 边界、是否截断、来源摘要等。

当前 hc_tj 转换结果将 `EpisodeData.truncated` 标为 `True`，表示轨迹不应被解释为已经证明完成了完整任务；这和 MCAP 文件是否有合法 footer 是两个不同概念。如果新数据有可靠的任务成功/结束标记，应按真实语义写入 `truncated` 和 `success`。

转换多个文件时，应满足：

- 一个 MCAP 对应一个 episode。
- 输入顺序确定且可复现。
- episode 之间的 `frame_index` 重新从 0 开始。
- 一次只解析和持有一个 MCAP，避免内存随文件数量增长。
- 覆盖已有输出时先写 staging 目录；所有 episode 成功后再替换旧数据集。
- 不修改原始 MCAP。

### 3.3 转换 hc_tj 多 episode 数据

从仓库根目录设置路径：

```bash
export REPO_ROOT="$(pwd)"
export MCAP_DIR="<MCAP_DIR>"
```

进入 HoloBrain 项目并执行：

```bash
conda activate holobrain
cd "${REPO_ROOT}/projects/holobrain"

python scripts/convert_hc_tj_mcap.py \
  --input "${MCAP_DIR}" \
  --output ./data/arrow_dataset/hc_tj \
  --urdf ../../hc_assets/hc_tj_description/urdf/hc_tj_robot.urdf \
  --task-name hc_tj \
  --instruction "<与这些轨迹一致的真实自然语言任务指令>"
```

`--input` 支持一个或多个文件、一个或多个目录；目录会递归查找小写 `.mcap` 文件。每个文件成为一个 episode。

可选参数：

- `--recover-truncated`：MCAP 尾部截断时，只保留完整的顶层 record，并在临时副本中补结束记录。不要用它掩盖普通格式错误。
- `--force-overwrite`：替换已存在的输出数据集。转换器使用 staging，转换失败时保留旧数据集。

如果输出已经存在，确认确实需要替换后再执行：

```bash
python scripts/convert_hc_tj_mcap.py \
  --input "${MCAP_DIR}" \
  --output ./data/arrow_dataset/hc_tj \
  --urdf ../../hc_assets/hc_tj_description/urdf/hc_tj_robot.urdf \
  --task-name hc_tj \
  --instruction "<真实任务指令>" \
  --force-overwrite
```

不要给不同任务的所有 episode 随意填写同一个模糊指令。如果目录包含多种任务，应按任务分别转换，或者扩展转换器以从每个 episode 的可靠元数据中读取任务和指令。

### 3.4 输出结构与报告

输出目录是一个 RODataset，通常包含：

```text
data/arrow_dataset/hc_tj/
├── dataset_info.json
├── state.json
├── meta_db.duckdb
├── conversion_report.json
└── *.arrow
```

具体 Arrow 文件名由 Hugging Face Datasets 写入器决定，不应在代码中硬编码。

`conversion_report.json` 应重点检查：

- `episode_count`：转换出的 episode 数。
- `total_frames`：最终有效帧总数。
- `episodes[].source`：源文件路径、大小和 SHA-256。
- `episodes[].recovery`：是否恢复、恢复偏移和丢弃字节数。
- `episodes[].messages`：各话题消息数。
- `episodes[].frames`：候选帧、输出帧、同步误差和丢帧原因。

## 4. 添加机器人专用 Dataset Adapter

RODataset 保存的是通用结构；训练模型需要固定的张量字典。因此通常要新增机器人专用适配器，例如：

```text
robo_orchard_lab/dataset/hc_tj/hc_tj_ro_dataset.py
```

当前实现的核心类是 `HcTjRODataset`，内部使用：

- `EpisodeChunkSamplerConfig`：按 episode 采样历史状态和未来动作，防止跨 episode 取序列。
- `ROMultiRowDataset`：一次读取多行组成时序样本。
- `_HcTjArrowDataParse`：解码图像、深度、关节和元数据，并转换为 HoloBrain 约定字段。
- `ConcatRODataset`：合并一个或多个 RODataset 路径。

适配器至少要输出：

| 字段 | 典型形状/含义 |
|---|---|
| `imgs` | `(num_cams, H, W, 3)`，BGR 图像 |
| `depths` | `(num_cams, H, W)`，单位米 |
| `intrinsic` | 每个相机的 4×4 内参扩展矩阵 |
| `T_world2cam` | 共同参考系到相机的变换 |
| `joint_state` | `(hist_steps, num_joints)` |
| `master_joint_state` | `(pred_steps, num_joints)` |
| `text` | 自然语言指令 |
| `task_name` | 任务名 |
| `cam_names` | 相机名列表 |

必须验证采样器不会跨越 episode 边界，历史和预测不足的边缘帧要使用框架既定的 padding/mask 语义。

## 5. 添加数据 transforms 和数据集注册

新建：

```text
projects/holobrain/configs/config_<robot>_ro_dataset.py
```

`hc_tj` 示例为 `config_hc_tj_ro_dataset.py`。它承担三项职责。

### 5.1 定义机器人数据配置

```python
dataset_config = {
    "<robot>": {
        "data_paths": ["<RODATASET_PATH>"],
        "urdf": "<ROBOT_URDF>",
        "cam_names": ["head"],
        "joint_scale_shift": ..., 
        "T_base2world": ...,
    }
}
```

其中 `joint_scale_shift` 每个关节为 `[scale, shift]`：

```text
scale = (upper - lower) / 2
shift = (upper + lower) / 2
normalized = (joint - shift) / scale
joint = normalized * scale + shift
```

HoloBrain 输出的是按该机器人配置归一化的动作表示，部署侧通过同一组 `joint_scale_shift` 反归一化。关节顺序或限位一旦和训练数据不一致，即使维度相同，结果也没有物理意义。

推荐直接从本机器人 URDF 的 `<limit lower=... upper=...>` 生成参数，并在启动时检查：

- 所有关节均有上下限。
- URDF 关节名称和数据关节名称一致。
- 左右臂索引与 `pytorch_kinematics` 的参数顺序一致。

### 5.2 定义三套 transform

`build_transforms(config, mode, data_config)` 一般实现：

- `training`：时序采样、图像增强、外参噪声、关节噪声、投影矩阵、FK、训练标签和 loss weight。
- `validation`：保留确定性的 resize、投影、FK 和标签，不添加随机增强。
- `deploy`：处理在线观测、相机标定、投影、FK，并增加 batch 维度。

关键模块包括：

- `SimpleStateSampling`：把历史状态和未来动作整理成模型输入/标签。
- `AddItems`：加入关节 mask、归一化参数和参考系变换。
- `MoveEgoToCam`、`GetProjectionMat`：统一空间坐标并生成投影矩阵。
- `MultiArmKinematics`：根据本机器人 URDF、关节索引和 link 名计算 FK。
- `ItemSelection`：只保留模型实际使用的字段。

不要复制其他机器人的 `arm_joint_id`、`arm_link_keys`、`finger_keys` 或 `T_base2world`。这些值必须从新机器人的 URDF 和训练目标重新确定。

### 5.3 注册构建函数

在数据配置中注册：

```python
@train_dataset_register()
@validation_dataset_register()
def build_datasets(config, dataset_names, mode, **kwargs):
    ...


@processor_register()
def build_processors(config, dataset_names):
    ...
```

然后在 `projects/holobrain/configs/dataset_factory.py` 的 `apply_dataset_register()` 中导入新模块：

```python
import config_<robot>_ro_dataset  # noqa: F401
```

这个导入会执行装饰器注册。只创建配置文件但不在这里导入，训练器不会发现新数据集。

## 6. 添加专用训练配置

推荐复制通用配置，而不是修改所有机器人共享的 common 配置：

```text
projects/holobrain/configs/config_holobrain_qwen_<robot>.py
```

最小写法：

```python
import copy

from config_holobrain_qwen_common import (
    build_model,
    build_optimizer,
    build_processors,
    build_training_dataset,
    build_validation_dataset,
    config as _common_config,
)

config = copy.deepcopy(_common_config)
config.update(
    training_datasets=["<robot>"],
    deploy_datasets=["<robot>"],
)
```

按数据和硬件检查这些参数：

- `hist_steps`：历史观测长度。
- `pred_steps`：一次预测的动作长度。
- `chunk_size`：动作 decoder 的 chunk 设置，通常要求能合理整除 `pred_steps`。
- `with_depth`、`with_depth_loss`：数据是否真的有有效深度。
- `min_depth`、`max_depth`：有效深度范围。
- `batch_size`、`num_workers`：显存和主机资源。
- `vlm_pretrain`：本地 Qwen 权重路径。
- `checkpoint`：HoloBrain 预训练权重路径。
- `freeze_vlm`：是否冻结视觉语言模型。

`hc_tj` 专用配置为：

```text
projects/holobrain/configs/config_holobrain_qwen_hc_tj.py
```

它保留 Qwen common 模型配置，只把 `training_datasets` 和 `deploy_datasets` 改为 `hc_tj`。

## 7. 训练前验证

### 7.1 验证底层 RODataset

从仓库根目录执行：

```bash
conda activate holobrain

python - <<'PY'
from robo_orchard_lab.dataset.robot import RODataset
from robo_orchard_lab.dataset.robot.db_orm import Episode

path = "projects/holobrain/data/arrow_dataset/hc_tj"
dataset = RODataset(path)
episodes = list(dataset.iterate_meta(Episode))

print("frames:", len(dataset))
print("episodes:", len(episodes))
print("first keys:", dataset[0].keys())
print("episode frame counts:", [item.frame_num for item in episodes])
PY
```

episode 数应等于 MCAP 文件数，总帧数应等于转换报告的 `total_frames`。

### 7.2 验证训练样本

```bash
conda activate holobrain

python - <<'PY'
from robo_orchard_lab.dataset.hc_tj import HcTjRODataset

dataset = HcTjRODataset(
    ["projects/holobrain/data/arrow_dataset/hc_tj"],
    hist_steps=1,
    pred_steps=64,
)
sample = dataset[0]

for key in (
    "imgs",
    "depths",
    "joint_state",
    "master_joint_state",
    "intrinsic",
    "T_world2cam",
    "text",
):
    value = sample[key]
    print(key, getattr(value, "shape", None), type(value))
PY
```

还应检查：

- 所有数值均有限，没有 NaN/Inf。
- 相机数、图像分辨率和配置一致。
- 关节维度和顺序一致。
- 深度单位和有效范围正确。
- FK link 数量、投影矩阵和坐标系方向正确。
- 文本确实描述当前轨迹任务。

### 7.3 一步 smoke test

进入项目目录：

```bash
cd "${REPO_ROOT}/projects/holobrain"

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python scripts/train.py \
  --config configs/config_holobrain_qwen_hc_tj.py \
  --workspace ./workspace/hc_tj_smoke \
  --kwargs '{"max_step":1,"step_log_freq":1,"save_step_freq":1,"num_workers":0}'
```

成功标准：

- 能构建数据集和 dataloader。
- 能加载 VLM 与 HoloBrain checkpoint。
- 完成一次 forward、loss、backward 和 optimizer step。
- workspace 中产生 checkpoint。

## 8. 启动正式训练

```bash
conda activate holobrain
cd "${REPO_ROOT}/projects/holobrain"

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python scripts/train.py \
  --config configs/config_holobrain_qwen_hc_tj.py \
  --workspace ./workspace/hc_tj
```

离线模式要求 `vlm_pretrain` 和 `checkpoint` 指向已存在的本地权重。如果要修改训练步数等参数，可以编辑专用配置，或通过 `--kwargs` 覆盖：

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python scripts/train.py \
  --config configs/config_holobrain_qwen_hc_tj.py \
  --workspace ./workspace/hc_tj \
  --kwargs '{"max_step":10000,"batch_size":1,"num_workers":8}'
```

## 9. 常见问题

### 9.1 bf16 下 BCE 报错

如果出现：

```text
binary_cross_entropy and BCELoss are unsafe to autocast
```

当前配置先不要设置：

```bash
ACCELERATE_MIXED_PRECISION=bf16
```

不指定混合精度即可训练。长期修复应在深度 loss 实现中评估改用 logits 配合 `binary_cross_entropy_with_logits`，或只让这一小段 loss 在 FP32 下执行；不能只为了消除错误而改变损失语义。

### 9.2 数据集找不到

确认：

- 转换输出位于数据配置的 `data_paths`。
- 输出目录内存在 `state.json`。
- 新数据配置已在 `dataset_factory.apply_dataset_register()` 中导入。
- 训练配置中的 `training_datasets` 名称和 `dataset_config` key 完全一致。

### 9.3 关节维度相同但训练结果异常

维度相同不代表语义相同。逐项检查：

- 数据列的关节顺序。
- URDF 中的关节名称和 `pytorch_kinematics` 参数索引。
- `joint_scale_shift` 的顺序。
- `arm_joint_id` 和 `arm_link_keys`。
- 监督动作到底是绝对角还是增量。
- 左右臂顺序是否被交换。

### 9.4 相机投影或 FK 异常

检查每个变换的方向，而不只看数值：

- 外参是 `camera -> reference` 还是 `reference -> camera`。
- 四元数是 `xyzw` 还是 `wxyz`。
- `parent_frame_id` 和 `child_frame_id` 是否和矩阵语义一致。
- 机器人躯干自由度被排除后，参考坐标系是否仍固定。
- RGB 内参是否错误地用于未对齐的深度图。

### 9.5 某个 MCAP 损坏

- 正常情况下严格报错，不要静默跳过整个 episode。
- 只有确认是尾部截断时才使用 `--recover-truncated`。
- 恢复后检查报告中的 `discarded_bytes`、完整 chunk 数和输出帧数。
- 不从无法证明完整的损坏 chunk 中逐条抢救训练消息。

## 10. 推荐测试清单

新增机器人支持至少覆盖：

- 正常单 MCAP 转换。
- 目录内多个 MCAP 转成多个 episode。
- episode 边界和帧索引正确。
- 缺失话题、错误消息类型、错误关节顺序会失败。
- 非法相机内参、外参、图像或深度会失败。
- 同步超限帧会被统计或丢弃。
- 截断文件默认拒绝，显式恢复只保留完整记录。
- 恢复临时文件会清理，原 MCAP 摘要不变。
- 覆盖转换失败时旧数据集仍然存在。
- Dataset Adapter 输出形状、dtype、单位和数值有限。
- 归一化再反归一化能还原原始关节值。
- FK、相机外参和投影矩阵使用同一个参考坐标系。
- 完成一次 `max_step=1` 训练。

## 11. hc_tj 当前接入结果

仓库中的实现已经具备：

- 多个文件或目录递归输入。
- 一个 MCAP 对应一个 episode。
- 双臂 14 轴状态和动作。
- 单头部 RGB/对齐深度。
- `zhi_Link` 统一几何参考系。
- 使用 `hc_tj_robot.urdf` 的真实关节限位、FK 索引和 link。
- 截断 MCAP 显式恢复。
- 转换审计报告。
- staging 写入和失败回滚。
- 专用 Dataset Adapter、transforms、注册和 Qwen 训练配置。

接入其他机器人时，可以复用这条工程链路和 RODataset 包装方式，但必须重新定义机器人相关的数据语义和几何参数。
