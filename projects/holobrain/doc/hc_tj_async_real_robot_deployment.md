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

## 推荐：独立推理与 DAgger 动作授权（v2）

v1/v2 是通信和控制协议版本，使用同一个模型。v2 动作带有当前控制窗口标识，
暂停、接管和重新授权后旧窗口的动作不能继续执行。模型推理与动作授权相互独立：
只要机器人观测可用，OFF/HOLD/MANUAL 下也会持续推理，但结果不会下发。

X1 的 HC-teleop-robotic 实机入口现在管理 DAgger 网关：先单独启动 HC_X1，
然后在 HC-teleop-robotic 目录运行：

```bash
ROS_DOMAIN_ID=13 ./run.sh teleop
```

不再另外运行 `holobrain_http_gateway/start_robot.sh`。X1 legacy DAgger 配置下，
实机入口自动使用 `~/holobrain_http_gateway` 的网关代码；安装在其他位置时设置
`HC_TJ_GATEWAY_ROOT`。网关日志为本次 HC 会话的 `middleware/holobrain_gateway.log`。
端口占用会在启动前报错，需人工停止旧网关。退出 HC 时自动停止其网关，外部
HC_X1 仍由用户管理；普通 Dashboard 和仿真入口不会启动此网关。
网关只向 `/hc_teleop` 输入发程序动作，由 HC 独占 `/action` 硬件输出；
DAgger legacy 模式会拒绝绕过这一仲裁入口的命令话题配置。

DAgger 网关从 `/observation/cameras/head/calibration` 接收标定，
从 `/observation/cameras/head/extrinsics` 接收外参。标定采用
reliable/transient-local，周期外参采用 reliable/volatile，以匹配 HC 的发布方式。
普通 v1 默认仍使用旧元数据话题和 transient-local 外参；手动配置网关时可用
`--extrinsics-durability volatile` 接收周期外参。图像和关节数据齐全但元数据缺失时，
完整观测接口会返回 204，此时模型没有可用观测，不能仅凭启动提示判断已经在推理。

两端 ROS 域应保持一致。网关需要完整的双臂和腰部状态；`waist:=false` 会关闭
腰部反馈，保留状态但禁止腰部命令和回零应使用 `waist_control:=false`。

GPU 主机启动脚本默认启用 DAgger v2：

```bash
./projects/holobrain/scripts/start_hc_tj_http_local.sh 'Pick up the object in front of the robot.'
```

在 HC Dashboard 的“DAgger 动作授权与人工接管”中点击“开启会话”，再点击
“开始程序”授权动作下发。
安全开关只控制动作下发，不控制模型计算；自动模式无需头显在线。VR 接管仍需要
有效跟踪及新鲜的 Grip 释放/握持事件，人工控制时断线会停止输出。关闭、暂停、
故障和急停都不允许模型动作；恢复只回到保持，需要再次明确开启程序控制。

授权打开后只使用新观测生成的结果；未授权时开始计算的结果不会在之后自动变成
可执行动作。硬件就绪、双臂反馈时效、动作限位限速及急停检查仍然有效。
DAgger 客户端与网关默认均使用 0.3 rad 的首点偏差上限。首点超限时丢弃本次
结果，清除旧轨迹并用新观测重新推理；只有当前 AUTO 授权仍然有效且新结果通过
检查才会继续下发，不会放宽上限或自动解除 HC 的 FAULT。
手部沿用普通 v1 的开合指令方式，不要求手位置反馈。模型的手部输入初始为 0，
随后按正在执行的动作轨迹估计；该估计不是实测反馈。手部遥测仅用于诊断，缺失
或过期不会阻止 AUTO。手部仍检查目标范围、命令时效和授权，已有同一控制窗口内
的下发目标时保留限速；首个目标按原来的开环方式发送。

使用 `start_hc_tj_ros_local.sh` 的 ROS 直连 legacy 客户端也遵守上述开环约定：
启用 DAgger 不会额外要求手部位置反馈，手部输入按最后已发布的手指令估计。
启动和人工接管后的恢复均不等待手反馈；RGB-D、双臂状态及控制窗口检查仍然有效。
显式选择 `--robot-interface hc_motion` 的独立后端仍使用实测手部输入。

脚本自动将网关主机及本机加入 `no_proxy`/`NO_PROXY`，同时覆盖 curl 和 Python
请求。只有 HTTP 401 才会自动刷新已有 TOKEN；HTTP 503 表示关节反馈尚未就绪，
应检查 ROS 域和关节话题，而不是重新输入 SSH 密码。

## 普通 v1：独立硬件与网关一键启动

以下为未使用 HC DAgger 仲裁器的 v1 部署方式，不适用于已经单独运行 HC_X1
和 HC-teleop-robotic 的 v2 组合。

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
HC_TJ_ENABLE_DAGGER=0 ./projects/holobrain/scripts/start_hc_tj_http_local.sh \
  'Pick up the object in front of the robot.'
```

通过 `HC_TJ_MODEL_DIR` 指定要使用的导出模型，`HC_TJ_PYTHON` 指定 Python 环境。

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

HTTP 路径需要双臂及腰部共 17 轴的真实观测，腰部状态用于计算相机外参。
网关的机械臂动作仅包含双臂 14 轴，手部使用独立命令话题；不会将腰部观测回写为
腰部命令。HTTP 客户端不会执行下文 ROS 直连示例中的自动复位步骤。

## 前置安全条件

- 物理急停必须可触达，并安排一名操作员专门负责急停。
- 启动前清空机械臂和双手工作区，确认头部 `head_1=head_2=0 rad`。当前静态
  `zhi_Link→eye` 外参不支持头部运动。
- 停止 VR、IK、示教器等其他命令源。客户端会检查主关节和左右手三个命令
  话题；发现其他 publisher 会锁存故障并停发。
- `--allow-competing-publishers` 仅用于隔离调试，禁止在正常真机运行中使用。
- 故障不会自动恢复。排查原因、确认机器人安全后重启客户端。

## GPU 服务端与 v1 观测格式

普通服务端保留旧的 `/{server_name}` 和 `/{server_name}/hc_tj_async/v1` 接口。
DAgger 启动器使用经过鉴权的 `/{server_name}/hc_tj_async/v2`，控制窗口只决定结果
是否可执行，不阻止模型计算。v1/v2 观测字段相同；v2 另外携带控制窗口标识。
观测端点接收 multipart `.npy`：

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
