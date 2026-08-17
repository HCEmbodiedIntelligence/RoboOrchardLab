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

# hc_tj：从 MCAP 到 HoloBrain 训练与离线推理

本文给出一条最短可复现链路：将一个或多个 hc_tj MCAP 转成 HoloBrain
RODataset，启动训练，再用训练得到的 checkpoint 对指定 episode 做离线推理评测。

## 1. 数据约定

- 一个 MCAP 文件对应一个 episode。
- 状态来自 `joint_states`，监督动作来自 `joint_cmd`。
- 只保留双臂 14 轴，顺序为右臂 `Joint1_R..Joint7_R`，再到左臂
  `Joint1_L..Joint7_L`。
- 训练相机只有头部相机 `head`，包含 RGB 和对齐深度。
- 双臂 FK、相机外参和投影统一使用 `zhi_Link` 参考坐标系。
- 必须使用 hc_tj 自身的 URDF，不能替换成 Agilex 或其他机器人的 URDF。

相关实现：

```text
scripts/convert_hc_tj_mcap.py
configs/config_hc_tj_ro_dataset.py
configs/config_holobrain_qwen_hc_tj.py
../../robo_orchard_lab/dataset/hc_tj/hc_tj_ro_dataset.py
```

## 2. 准备环境和路径

从仓库根目录开始：

```bash
conda activate holobrain

export REPO_ROOT="$(pwd)"
export MCAP_DIR="<包含一个或多个 .mcap 文件的目录>"
export QWEN_DIR="<Qwen2.5-VL-3B-Instruct 本地目录>"
export PRETRAIN_CKPT="<HoloBrain Qwen 预训练 model.safetensors>"

cd "${REPO_ROOT}/projects/holobrain"
```

离线运行要求 `QWEN_DIR` 和 `PRETRAIN_CKPT` 已经存在于本机。

## 3. 将 MCAP 转为 RODataset

### 3.1 转换一个或多个 episode

`--input` 支持文件、多个文件或目录；目录会递归查找小写 `.mcap` 文件。
每个 MCAP 会写成一个独立 episode：

```bash
python scripts/convert_hc_tj_mcap.py \
  --input "${MCAP_DIR}" \
  --output ./data/arrow_dataset/hc_tj \
  --urdf ../../hc_assets/hc_tj_description/urdf/hc_tj_robot.urdf \
  --task-name hc_tj \
  --instruction "<与轨迹一致的自然语言任务指令>" \
  --recover-truncated
```

参数说明：

- `--instruction` 必填，并会作为训练语言指令写入每一帧。
- `--recover-truncated` 只抢救完整 MCAP record，忽略损坏尾部，不修改源文件。
- 输出目录已存在时，确认需要替换后增加 `--force-overwrite`。
- 如果 MCAP 正常闭合，可以省略 `--recover-truncated`。

转换器以 RGB 时间戳为基准，最近邻同步深度、状态和动作。默认最大误差为：

- 深度 50 ms；
- `joint_states` 20 ms；
- `joint_cmd` 20 ms。

超限帧会被丢弃并记录在报告中。

### 3.2 检查转换结果

输出目录应至少包含：

```text
data/arrow_dataset/hc_tj/
├── state.json
├── dataset_info.json
├── meta_db.duckdb
├── conversion_report.json
└── *.arrow
```

快速检查：

```bash
python own_test/test_hc_tj_ro_dataset.py
```

重点查看 `conversion_report.json` 中的：

- `episode_count` 和 `total_frames`；
- 每个 episode 的源 MCAP、SHA-256 和输出帧数；
- `recovery` 是否发生尾部恢复；
- 各话题消息数、同步误差和丢帧原因。

## 4. 启动训练

### 4.1 先运行一步 smoke test

当前深度损失使用普通 BCE；不要设置
`ACCELERATE_MIXED_PRECISION=bf16`，否则 PyTorch autocast 会拒绝该算子。

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python scripts/train.py \
  --config configs/config_holobrain_qwen_hc_tj.py \
  --workspace ./workspace/hc_tj_smoke \
  --kwargs "{\
    \"vlm_pretrain\":\"${QWEN_DIR}\",\
    \"checkpoint\":\"${PRETRAIN_CKPT}\",\
    \"max_step\":1,\
    \"step_log_freq\":1,\
    \"save_step_freq\":1,\
    \"batch_size\":1\
  }"
```

日志应显示：

- `training_datasets: ["hc_tj"]`；
- checkpoint 的 `missing_keys` 和 `unexpected_keys` 均为 0；
- 完成至少一个训练 step 并保存 checkpoint。

### 4.2 正式训练

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python scripts/train.py \
  --config configs/config_holobrain_qwen_hc_tj.py \
  --workspace ./workspace/hc_tj_train \
  --kwargs "{\
    \"vlm_pretrain\":\"${QWEN_DIR}\",\
    \"checkpoint\":\"${PRETRAIN_CKPT}\",\
    \"batch_size\":1,\
    \"max_step\":100000,\
    \"step_log_freq\":50,\
    \"save_step_freq\":5000\
  }"
```

这里的网络监督目标是未来 64 步、14 维的绝对目标关节角，不是头部相机坐标系下
的末端位姿变化量。模型内部会按 hc_tj URDF 做 FK，用于几何特征和 FK loss。

训练 checkpoint 目录中会包含 `model.safetensors`。离线评测需要传入这个文件，
而不是 checkpoint 目录：

```bash
export TRAINED_CKPT="<训练 checkpoint 目录>/model.safetensors"
```

## 5. 使用 RODataset 做离线推理评测

`train.py --eval_only` 会执行扩散推理，并把预测关节轨迹与数据中的真实
`joint_cmd` 比较；不会更新模型参数。

### 5.1 评测全部 episode

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python scripts/train.py \
  --config configs/config_holobrain_qwen_hc_tj.py \
  --workspace ./workspace/hc_tj_eval_all \
  --eval_only \
  --kwargs "{\
    \"vlm_pretrain\":\"${QWEN_DIR}\",\
    \"checkpoint\":\"${TRAINED_CKPT}\",\
    \"validation_datasets\":[\"hc_tj\"],\
    \"batch_size\":1\
  }"
```

### 5.2 只评测指定 episode

episode 编号从 0 开始。例如只评测编号为 1 和 8 的两个 episode：

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python scripts/train.py \
  --config configs/config_holobrain_qwen_hc_tj.py \
  --workspace ./workspace/hc_tj_eval_ep_1_8 \
  --eval_only \
  --kwargs "{\
    \"vlm_pretrain\":\"${QWEN_DIR}\",\
    \"checkpoint\":\"${TRAINED_CKPT}\",\
    \"validation_datasets\":[\"hc_tj\"],\
    \"validation_episode_indices\":[1,8],\
    \"batch_size\":1\
  }"
```

空列表、重复编号、负数和越界编号会直接报错。选择多个 episode 时，指标会在
这些 episode 的全部帧上汇总；如果需要逐 episode 指标，应每次只传一个编号并
分别运行。

评测输出包括 16、32、64 步预测长度下的：

- `average_joint`、`final_joint`：关节角误差；
- `average_xyz`、`final_xyz`：FK 后的位置误差；
- `average_quat`、`final_quat`：FK 后的姿态误差；
- `jerk`、`jerk_xyz`：关节和空间轨迹平滑度。

## 6. 结果解释与常见问题

1. **训练集评测不等于泛化能力。** 如果评测使用的就是训练数据，结果只能用于
   检查数据链路、收敛和过拟合情况。正式实验应单独转换未参与训练的 MCAP，
   并注册为独立验证数据集。
2. **不要使用错误 URDF。** hc_tj 的关节顺序、限位、FK 索引和 `zhi_Link`
   外参都依赖 `hc_tj_robot.urdf`。
3. **不要把 checkpoint 目录传给 `checkpoint`。** 应传具体的
   `model.safetensors`。
4. **离线模式仍需要 Qwen 基础权重。** 训练 checkpoint 不包含冻结的
   `vlm.*` 参数，推理时会从 `QWEN_DIR` 重新加载。
5. **显存占用较低是正常的。** 当前配置从 Qwen2.5-VL-3B 初始化，但只保留
   1 层语言 Transformer，并冻结 VLM；batch size 为 1 的纯推理通常只需约
   3 GiB 显存。

