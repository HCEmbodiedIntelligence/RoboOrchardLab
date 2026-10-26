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

## 接口与安全边界

机器人网关仅在机器人本机订阅这些现有接口：

```text
/hc_teleop/camera_head/color/compressed
/hc_teleop/camera_head/depth/compressed
/hc_teleop/joint_states
/io_teleop/camera_head/calibration
/io_teleop/camera_head/extrinsics
```

启用控制后才创建以下 publisher：

```text
/hc_teleop/joint_cmd
/hc_teleop/joint_cmd_finger_right
/hc_teleop/joint_cmd_finger_left
```

HTTP API 使用 Bearer token，包括二进制观测 `GET
/hc_tj/observation/v1`、最新关节反馈 `GET /hc_tj/joint_feedback/v1`、动作块
`POST /hc_tj/action_chunk/v1` 和只读健康检查 `GET /healthz`。生产部署应再用
防火墙限制 8765 端口只能由 GPU 主机访问；HTTP
明文只适合受控机器人局域网，跨不可信网络必须加 TLS/VPN。

客户端在每次推理完成后重新读取最新关节反馈，再检查预测首点与双臂反馈差值；
机器人网关在接收动作时会用自己的最新反馈重复执行同一项独立检查。控制侧还会
检查：关节反馈新鲜度、动作有限值和维度、URDF 位置限位、每轴速度限制、动作序号
递增、轨迹耗尽，以及其他命令 publisher。网关仅在首次控制和 deadman 暂停恢复时
用最新反馈初始化限速器；正常 RTC 动作块更新会保留上一条已发布命令，避免周期性
回拉到存在跟踪延迟的反馈位置。
动作流还带有默认 0.75 秒软件 deadman；客户端或网络失联后会立即清空旧轨迹并
进入 `control_state=paused`，但不会锁存全局 fault。客户端收到暂停响应后会丢弃
尚未执行的旧动作和 RTC 续接状态，自动用下一帧最新观测重新推理，并从
`sequence_id=1` 建立新会话；新动作重新通过最新反馈、首点连续性、URDF 和限速
检查后才能恢复。关节反馈异常、竞争 publisher 等其他安全故障仍会锁存，必须
排查并重启网关。

## 1. 启动 GPU 推理服务

在 GPU 主机、项目虚拟环境中执行：

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
/usr/bin/python3 \
  "$HOME/holobrain_http_gateway/projects/holobrain/scripts/hc_tj_http_robot_gateway.py" \
  --host 0.0.0.0 \
  --port 8765 \
  --enable-control \
  --urdf-path "$HOME/holobrain_http_gateway/hc_tj_robot.urdf"
```

确认 `/healthz` 返回 `control_enabled=true` 且没有其他命令 publisher 后，GPU
客户端才加入第二道开关：

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
