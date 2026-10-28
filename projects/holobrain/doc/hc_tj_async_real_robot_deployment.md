# HC-TJ 异步真机推理与控制部署

本文使用独立的机器人 HTTP 网关连接 HoloBrain，不修改 HC_X1，也不要求 GPU
主机加入机器人 ROS 2 DDS 网络：

```text
HC_X1 驱动 ──本机 ROS──> hc_tj_http_robot_gateway.py
                              │ 压缩 RGB-D、关节状态
                              ▼ HTTP :8765
                     hc_tj_http_async_client.py ──HTTP──> inference_server.py
                              │
                              └──动作块 HTTP──> 网关 ──本机 ROS──> HC_X1
```

## 推荐：两端一键启动

首次运行时，机器人脚本会自动生成 Bearer token 并以 `600` 权限保存到
`~/.config/holobrain/hc_tj_gateway_token`。GPU 主机脚本缺少本地 token 时，会通过
SSH 自动从 `niic@192.168.2.33` 获取，不需要手动生成、复制或粘贴。自定义机器人
SSH 地址时设置 `HC_TJ_ROBOT_SSH`。

机器人侧只需运行一个脚本；它会先完成 sudo 检查，同时启动 HC_X1 和 HTTP
网关，并固定腰部只读、双臂和双手启用：

```bash
~/holobrain_http_gateway/start_robot.sh
```

GPU 主机也只需运行一个脚本；它会加载模型、等待推理端口就绪，使用真实观测
完成一次不下发动作的推理预热，再启动控制客户端，避免首次 CUDA 初始化延迟
影响控制首帧的时序和连续性检查：

```bash
./projects/holobrain/scripts/start_hc_tj_http_local.sh \
  'Pick up the object in front of the robot.'
```

一键脚本默认加载仓库中的 `260823_w/export/model`；需要使用其他导出模型时，
可通过 `HC_TJ_MODEL_DIR` 覆盖。

本机脚本默认使用 `first_point_jump_rad=0.3`、`inference_hz=2.0` 和
`latency_ewma_alpha=0.25`。两端都用 `Ctrl+C` 停止；应先停止 GPU 主机脚本，再
停止机器人脚本。执行本机脚本即表示允许开始真机控制，不再要求额外交互确认；
运行前必须自行确认急停和工作区安全。脚本只终止自己创建的进程。下面的手动步骤
保留用于部署和排错。

机器人 ARM 主机只转发已有压缩图像、状态和动作，不安装 PyTorch、HoloBrain 或
OpenCV。JPEG 与 16UC1 PNG 保持压缩格式跨机传输，在 GPU 主机解码。网关和客户端
默认都是 dry-run；只有两端都显式加入 `--enable-control` 才可能发布控制命令。

策略输出保持模型原生 30 Hz、最多 64 步、16 维顺序：

```text
Joint1_R..Joint7_R, R_ban, Joint1_L..Joint7_L, L_ban
```

客户端在 100 Hz 控制周期内插值、限位和限速。默认是 dry-run；只有显式加入
`--enable-control` 才会创建命令 publisher。

启用控制后，客户端会先将 17 轴主关节复位到内置默认姿态，再开始发送正式推理
请求。复位顺序为 `Joint1_R..Joint7_R, Joint1_L..Joint7_L, leg_1, leg_2, zhi`；
双臂以已配置的速度限制逐步靠近目标，反馈在每轴距离目标不超过 `0.02 rad` 前，
不会调用推理服务。手部不会在复位阶段发布命令。

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

RTC 的融合公式、异步时序与逐层调用链见 [RTC 原理与调用链](rtc_principle_and_call_flow.md)。

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
conda activate ./<environment-dir>
python projects/holobrain/scripts/inference_server.py \
  --model_dir <export-dir>/model \
  --inference_prefix hc_tj \
  --server_name holobrain \
  --port 2000
```

客户端使用版本化端点：

```text
http://127.0.0.1:2000/holobrain/hc_tj_async/v1
```

服务接收 BGR `uint8 [H,W,3]`、米制 depth、4×4 内参、`zhi_Link→eye`
外参、16 维状态，以及可选 RTC 剩余动作；响应必须为原生 30 Hz 的 `[N,16]`
动作。模型若要复用上一个动作块，导出配置中的 action decoder 必须启用
`RTCInferencePlugin`。

## 2. 独立部署机器人网关

不要把文件写进 HC_X1。先从 GPU 主机复制三个纯 Python 文件到独立目录：

```bash
ssh <robot-user>@<robot-host> \
  'mkdir -p ~/holobrain_http_gateway/projects/holobrain/scripts'

rsync -av \
  projects/holobrain/scripts/hc_tj_async_core.py \
  projects/holobrain/scripts/hc_tj_http_protocol.py \
  projects/holobrain/scripts/hc_tj_http_robot_gateway.py \
  <robot-user>@<robot-host>:~/holobrain_http_gateway/projects/holobrain/scripts/

rsync -av \
  projects/holobrain/scripts/start_hc_tj_http_robot.sh \
  <robot-user>@<robot-host>:~/holobrain_http_gateway/start_robot.sh

ssh <robot-user>@<robot-host> \
  'chmod +x ~/holobrain_http_gateway/start_robot.sh'
```

两台主机设置相同且至少 16 字符的随机 token。不要把生产 token 写入仓库或
命令行历史：

```bash
export HC_TJ_GATEWAY_TOKEN='<random-secret>'
```

保持现有 `HC_X1/start.sh` 正常运行。另开机器人终端，以 dry-run 启动独立网关：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=<robot-domain-id>
export PYTHONPATH="$HOME/holobrain_http_gateway"

/usr/bin/python3 \
  "$HOME/holobrain_http_gateway/projects/holobrain/scripts/hc_tj_http_robot_gateway.py" \
  --host 0.0.0.0 \
  --port 8765
```

日志必须出现 `dry-run gateway: no ROS control publishers were created`。检查：

```bash
curl http://127.0.0.1:8765/healthz
ros2 topic info /hc_teleop/joint_cmd
ros2 topic info /hc_teleop/joint_cmd_finger_right
ros2 topic info /hc_teleop/joint_cmd_finger_left
```

健康响应应为 `status=ok`、`control_enabled=false`，三个命令话题不应出现网关
publisher。

## 3. 在 GPU 主机完整 dry-run 推理

GPU 主机无需 ROS 环境。使用与机器人相同的 token，并显式给出 depth scale。
当前毫米 depth 对应 `1000 units/m`：

```bash
export HC_TJ_GATEWAY_TOKEN='<same-random-secret>'

python projects/holobrain/scripts/hc_tj_http_async_client.py \
  --robot-url http://<robot-host>:8765 \
  --server-url http://127.0.0.1:2000/holobrain/hc_tj_async/v1 \
  --instruction '<fixed-task-instruction>' \
  --depth-units-per-meter 1000
```

客户端每次读取一个新 RGB 时间戳，在 GPU 上解码 RGB-D、构造 16 维状态并调用
HoloBrain。正常日志会持续显示 observation id、推理延迟和剩余动作数。此命令不
发送动作 POST；机器人网关也没有控制 publisher，因此可以验证完整真实图像和
状态推理链路而不驱动机器人。

## 4. 逐级启用物理控制

物理控制必须由现场操作员单独确认。至少满足：急停可触达、工作区清空、头部保持
标定姿态、机器人已安全到达训练初始姿态，并停止 VR、IK、示教器等其他命令源。
网关不会自动移动到默认姿态；如果当前双臂与预测首点任一轴相差超过默认
`0.3 rad`，动作会被拒绝。

先把导出的 HC-TJ URDF 复制到网关独立目录。机器人端重启网关，并同时提供两个
控制参数：

```bash
python projects/holobrain/scripts/replay_hc_tj_async.py \
  --dataset-path projects/holobrain/data/arrow_dataset/hc_tj_260818_fix
```

回放只使用 `/holobrain_hc_tj_replay/*`，绝不连接真机控制话题。它验证 100 Hz
双臂命令、腰部反馈保持、左右完整 10 关节 OmniHand 映射和故障停发。
回放取一个 Arrow 样本的图像和初始双臂状态，以模拟关节反馈跟随客户端命令
完成启动复位，随后使用本地模拟推理服务产生动作并执行上述检查；不会逐帧
播放数据集中记录的动作。复位命令不计入推理阶段的命令数量或频率检查。

## 逐级启用

1. 完成 Arrow 离线请求和隔离 ROS 回放。
2. 真机保持急停可触达、工作区清空，运行 dry-run；确认没有其他控制 publisher。
3. 检查当前双臂反馈与每段预测首点差值小于 `0.10 rad`。
4. 加入 `--enable-control`。客户端会先复位到内置默认姿态；默认双臂每关节每
   周期限速对应 `0.25 rad/s`，并应用 HC-TJ URDF 位置限位。确认复位完成日志后，
   才会开始正式推理。
5. 首次只允许单关节小幅目标，确认方向、单位、左右顺序和腰部保持。
6. 再逐步允许双臂和二值手部动作；全程保留物理急停和控制源互斥。

启用命令仅比 dry-run 多一个参数：

```bash
python projects/holobrain/scripts/hc_tj_http_async_client.py \
  --robot-url http://<robot-host>:8765 \
  --server-url http://127.0.0.1:2000/holobrain/hc_tj_async/v1 \
  --instruction '<fixed-task-instruction>' \
  --depth-units-per-meter 1000 \
  --enable-control
```

网关把模型原生 30 Hz 动作插值到 100 Hz，应用 URDF 限位和默认
`0.25 rad/s` 每轴限速。主关节命令严格只包含 14 个双臂关节；`leg_1`、
`leg_2`、`zhi` 只用于观测，永远不会进入 HoloBrain 命令消息。首次只应进行受限
的小幅动作测试，全程保留物理急停。左右手指 publisher 默认也不创建；只有明确
加入 `--enable-hands` 才会启用手指控制。

## 离线验证

针对 Arrow 数据集调用真实 GPU 服务：

```bash
python projects/holobrain/scripts/eval_hc_tj_async.py \
  --server-url http://127.0.0.1:2000/holobrain/hc_tj_async/v1 \
  --dataset-path <hc-tj-arrow-dataset>
```

隔离 ROS 回放仍可使用 `replay_hc_tj_async.py`；它只使用
`/holobrain_hc_tj_replay/*`，不会连接真机控制话题。
