# HC-TJ 异步真机推理与控制部署

本文描述 GPU 推理服务与 ROS 2 真机客户端分离部署。策略输出保持模型原生
30 Hz、64 步、16 维顺序：

```text
Joint1_R..Joint7_R, R_ban, Joint1_L..Joint7_L, L_ban
```

客户端在 100 Hz 控制周期内插值、限位和限速。默认是 dry-run；只有显式加入
`--enable-control` 才会创建命令 publisher。

## 前置安全条件

- 物理急停必须可触达，并安排一名操作员专门负责急停。
- 启动前清空机械臂和双手工作区，确认头部 `head_1=head_2=0 rad`。当前静态
  `zhi_Link→eye` 外参不支持头部运动。
- 停止 VR、IK、示教器等其他命令源。客户端会检查主关节和左右手三个命令
  话题；发现其他 publisher 会锁存故障并停发。
- `--allow-competing-publishers` 仅用于隔离调试，禁止在正常真机运行中使用。
- 故障不会自动恢复。排查原因、确认机器人安全后重启客户端。

## GPU 服务端

服务端继续保留旧的 `/{server_name}` 接口。HC-TJ 异步客户端只连接版本化端点
`/{server_name}/hc_tj_async/v1`。新端点接收 multipart `.npy`：

| 字段 | 契约 |
| --- | --- |
| `head_color` | `uint8 [H,W,3]`，与 Arrow/OpenCV 训练读取一致的 BGR |
| `head_depth` | `float32/float64 [H,W]`，单位米 |
| `head_intrinsic` | 有限 `float32/float64 [4,4]` |
| `zhi_to_eye` | 有限齐次 `float32/float64 [4,4]`，`zhi_Link→eye` |
| `current_state` | `float32 [16]`，严格使用上述关节顺序 |
| `joint_names` | 上述 16 个名字的 JSON 数组 |
| `instruction` | 非空字符串 |
| `remaining_actions` | 可选 `float32 [N,16]`，原生 30 Hz |
| `delay_horizon` | 有剩余动作时必需，`0..N` 的整数 |

响应包含 `actions [N,16]`、`joint_names`、`action_hz=30`、
`action_horizon` 和 `inference_ms`。服务端不会插值。共享 pipeline 串行执行。

导出的模型必须在 action decoder 中配置 RTC：

```json
{
  "async_inference_plugin": {
    "type": "robo_orchard_lab.models.rtc_plugin.rtc_plugin:RTCInferencePlugin",
    "max_horizon": 32
  }
}
```

HC-TJ 配置会把该字段导出到 `model/model.config.json` 的 decoder 配置。部署前先重新
加载导出 pipeline 验证配置；当请求带 `remaining_actions` 而模型未启用 RTC 时，
新端点会明确返回协议错误。

在 GPU 主机执行：

```bash
python projects/holobrain/scripts/inference_server.py \
  --model_dir <exported-model-dir>/model \
  --inference_prefix hc_tj \
  --server_name holobrain \
  --port 2000
```

服务器只应暴露在受控机器人网络；生产部署还应在反向代理或防火墙处限制来源。

## 机器人端 ROS 2 与 DDS

服务器和机器人客户端可位于不同主机，但 ROS 观测与控制只在机器人 ROS graph
中流动。先加载与硬件桥一致的 ROS 2 环境和 DDS domain：

```bash
source /opt/ros/humble/setup.bash
source <hc-io-suit-install>/setup.bash
export ROS_DOMAIN_ID=<robot-domain-id>
```

检查观测、标定和控制源：

```bash
ros2 topic hz /io_teleop/camera_head/color
ros2 topic hz /io_teleop/camera_head/depth
ros2 topic hz /io_teleop/joint_states
ros2 topic echo /io_teleop/camera_head/calibration --once
ros2 topic echo /io_teleop/camera_head/extrinsics --once
ros2 topic info --verbose /io_teleop/joint_cmd
```

客户端订阅 JPEG `CompressedImage`、16UC1 PNG `CompressedImage`、17 维
`JointState` 和两条 JSON `String` 标定消息。标定 publisher 与订阅端使用
reliable + transient-local QoS。RGB 是同步主轴；depth 最大相差 50 ms，关节反馈
最大相差 20 ms。

depth scale 必须显式给出。当前部署示例的原始 depth 是毫米，即
`1000 units/m`：

```bash
python projects/holobrain/scripts/hc_tj_async_client.py \
  --server-url http://<gpu-host>:2000/holobrain/hc_tj_async/v1 \
  --instruction "<fixed-task-instruction>" \
  --depth-units-per-meter 1000
```

这条命令是 dry-run：不会创建 `/io_teleop/joint_cmd` 或左右手命令 publisher。
保持至少一个完整任务周期，检查客户端无同步、HTTP、协议、动作跳变和状态过期
故障；同时检查 GPU 日志的输入、推理耗时和 64×16 输出。

## 离线与隔离验证

针对指定 Arrow 数据集调用真实 GPU 服务：

```bash
python projects/holobrain/scripts/eval_hc_tj_async.py \
  --server-url http://<gpu-host>:2000/holobrain/hc_tj_async/v1 \
  --dataset-path projects/holobrain/data/arrow_dataset/hc_tj_260818_fix
```

报告包含协议、范围、双臂 MAE/RMSE、左右手连续/二值误差、HTTP/模型延迟和
裁掉延迟前缀后的有效动作长度。

在已加载 ROS 2 的测试主机运行隔离回放：

```bash
python projects/holobrain/scripts/replay_hc_tj_async.py \
  --dataset-path projects/holobrain/data/arrow_dataset/hc_tj_260818_fix
```

回放只使用 `/holobrain_hc_tj_replay/*`，绝不连接真机控制话题。它验证 100 Hz
双臂命令、腰部反馈保持、左右完整 10 关节 OmniHand 映射和故障停发。

## 逐级启用

1. 完成 Arrow 离线请求和隔离 ROS 回放。
2. 真机保持急停可触达、工作区清空，运行 dry-run；确认没有其他控制 publisher。
3. 检查当前双臂反馈与每段预测首点差值小于 `0.10 rad`。
4. 加入 `--enable-control`。默认每关节每周期限速对应 `0.25 rad/s`，并应用
   HC-TJ URDF 位置限位。
5. 首次只允许单关节小幅目标，确认方向、单位、左右顺序和腰部保持。
6. 再逐步允许双臂和二值手部动作；全程保留物理急停和控制源互斥。

启用命令仅比 dry-run 多一个参数：

```bash
python projects/holobrain/scripts/hc_tj_async_client.py \
  --server-url http://<gpu-host>:2000/holobrain/hc_tj_async/v1 \
  --instruction "<fixed-task-instruction>" \
  --depth-units-per-meter 1000 \
  --enable-control
```
