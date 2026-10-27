# HC 公共接口接入

本次只适配推理通信和现有 HC 控制接口。HC 使用原有 Motion/DAgger 功能，
无需新增 policy 模式、绑定请求或修改 HC 源码。训练、数据转换和录制逻辑不变。

`hc_motion` 支持 X1 和 OpenArmX；机型由生成文件的 `profile_id` 确定。
OpenArmX 复用现有 HC 抢占流程和 `openarmx_hc_robot_adapter`，控制双臂及左右夹爪。

`hc_tj_async_client.py`、`hc_tj_async_core.py` 来自原 `own` 分支；
客户端新增 `hc_motion` 选项，服务端补入其配套的 `/hc_tj_async/v1` 路由。
原服务端入口及客户端 `legacy` 默认模式保留。

## 启动

客户端环境需要 ROS 2/rclpy、NumPy、requests、PyYAML 和 OpenCV。

推理机器使用已导出的 HC-TJ 16 维模型；沿用原异步客户端所需的 RTC 配置：

```bash
python projects/holobrain/scripts/inference_server.py \
  --model_dir /path/to/exported/model --port 2000 --server_name holobrain
```

机器人机器先按 HC 原有流程启动 Motion、驱动和相机。客户端与 HC 使用相同的
ROS 域，并 source ROS 2 环境。取得当前 HC 配置生成的 `robot_api.resolved.yaml`。
若需离线生成，在 HC 仓库内执行（profile 必须与运行中选择的一致）：

```bash
python3 middleware/profile_cli.py generate \
  --profile adapters/robots/openarmx --output /tmp/hc-client-api
```

X1 使用 `adapters/robots/x1`。OpenArmX 需要先按 HC/OpenArmX 原有入口运行
Motion、机器人 adapter 和相机；两侧 ROS 域一致。客户端复用这些服务。

回到 RoboOrchardLab，先检查通信；此命令不创建控制 publisher：

```bash
python projects/holobrain/scripts/hc_tj_async_client.py \
  --robot-interface hc_motion \
  --robot-api-file /tmp/hc-client-api/robot_api.resolved.yaml \
  --server-url http://INFERENCE_HOST:2000/holobrain/hc_tj_async/v1 \
  --instruction '拿起物体' \
  --depth-units-per-meter 1000
```

确认观测和服务端响应正常后，在同一命令末尾增加 `--enable-control`。
在 HC 原有 DAgger 页面开启会话并点击“开始程序”，进入 AUTO 后客户端开始发送。
客户端本身不启动会话、不使能硬件、不自动执行 17 轴回位。
非 AUTO 时停发；恢复 AUTO 后清除旧预测，重新获取观测并推理。
客户端出现锁定错误时会打印原因，修正接口/服务问题后重启客户端。

## 接口对应

| 内容 | HC 现有接口 |
| --- | --- |
| 头部 RGB-D | `/observation/cameras/head/{color,depth}/compressed` |
| 相机内外参 | `/observation/cameras/head/{calibration,extrinsics}` |
| 关节反馈 | `/observation/joint_states`，按名称取双臂并转换为模型顺序 |
| 配置校验 | `/status/robot/info` 与生成文件的 profile/revision 一致 |
| 右臂、左臂目标 | 生成配置中的两路 `servo_j`，通常为 `/control/motion/{right,left}_arm/servo_j` |
| X1 右手、左手目标 | `/action/{right,left}_hand/joint_states`，每手分别发送 10 个关节 |
| OpenArmX 左右夹爪 | 经 HC 现有权限检查后，由 HC 发布到上述独立夹爪话题 |
| 接管状态 | `/status/control/ownership`，使用原有 AUTO/MANUAL/HOLD 等状态 |

X1 手部维持原客户端语义：模型右手、左手各有一个独立的 `[0,1]` 开合值，
沿用现有 `hand_command()` 映射成各自 10 个关节位置。它不是 20 维独立手指预测。
传入模型的手部状态仍是原客户端的命令态，没有改成实测状态或修改标定。
HC 模式只控制双臂和这两路手部话题，不发送腰部或头部目标。

HC 原 `main` 的 X1 人工接管支持双臂；其人工灵巧手能力未在本次修改中扩展。
客户端收到人工接管后会停止自己的双臂及手部发送。

## OpenArmX 对应关系

HTTP 继续使用现有 `/hc_tj_async/v1` 的 16 值布局和协议字段名：右臂 7 轴、
右夹爪开合量、左臂 7 轴、左夹爪开合量。机器人侧按生成配置映射到
`openarmx_{right,left}_joint1..7`，不会把这些名称当作 HC-TJ 的硬件关节名。
模型必须与 OpenArmX 的关节、相机和动作含义匹配；接口联通测试使用确定性测试模型。

OpenArmX 的开合量为 `0` 张开、`1` 闭合，按 profile 的主指关节限位换算。
当前配置对应 `0.044 m` 张开、`0 m` 闭合；两个夹爪独立控制。
观测使用实际主指反馈反算开合量。相机外参使用 profile 中的公共基座
`openarmx_body_link0`，HTTP 字段 `zhi_to_eye` 保留历史名称，承载该基座到相机的变换。

双臂仍发送公共 ServoJ。两个夹爪合为一条公共名、米制 `JointState`，通过现有
`/internal/control/joint_targets` 和 `hc1` 身份字段进入 HC 的最终权限检查；
当前会话、控制代际和 `motion` 来源必须匹配。此处明确依赖 HC 现有内部协议，
未增加新的 HC API。客户端只在 AUTO 时发送；Grip 抢占后由现有 VR 控制夹爪，
带旧身份的迟到命令被 HC 拒绝，松手按原流程交还。

## 验证

基础测试覆盖 HTTP 请求/响应、左右顺序、单位映射、额外身体关节及原客户端协议。
集成测试使用实际 HTTP、DDS、未修改的 HC、Motion 和驱动进程，模型和机器人由测试替身提供，
检查两臂运动、左右手不同目标、人工接管停发，以及旧 HTTP 失败在恢复后不污染新请求。
OpenArmX 项直接加载指定源码仓库的 adapter 和真实 VR 适配节点，检查左右控制器
第八个位置值、人工 Trigger 移动夹爪、MANUAL/AUTO 中旧身份拒绝和松手后恢复输出。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  --confcutdir=tests/test_integration/test_projects/holobrain \
  tests/test_integration/test_projects/holobrain/test_hc_tj_async_core.py \
  tests/test_integration/test_projects/holobrain/test_hc_tj_interface.py
```

集成项需要先 source HC 的 ROS 构建环境，并设置
`HC_INTERFACE_TEST_HC_ROOT=/path/to/unchanged/HC-teleop-robotic`。
OpenArmX 项还需设置 `HC_INTERFACE_TEST_OPENARMX_ROOT=/path/to/hc_openarmx`，
并将其 `src/openarmx_hc_robot_adapter` 及 HC 根目录、`.deps` 加入 `PYTHONPATH`。
测试会核对 adapter 的实际加载路径。它使用 localhost 隔离 ROS 域中的模拟机器人，
不进行真机或真实模型效果验收。
