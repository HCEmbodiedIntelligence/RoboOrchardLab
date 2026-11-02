# HC-TJ ROS 2 直连部署

HoloBrain 推理机直接加入机器人 ROS 2 DDS 域，不再经过机器人侧 HTTP Gateway：

```text
HC_X1 /observation/* ──DDS 192.168.2.x──> HoloBrain 推理客户端
HC Dashboard /status/control/ownership ──DDS──> HoloBrain 推理客户端
HoloBrain /hc_teleop/joint_cmd* ──DDS──> HC 中间件安全仲裁 ──> /action/*
```

图像、深度、相机内外参和机器人反馈均保持原始标准话题。推理动作仍经过 HC
中间件的所有权、急停、反馈时效、限位、限速与碰撞检查，推理客户端不直接发布
`/action/*`。

## 网络与 DDS

- 机器人推理专网：`192.168.2.33`
- 推理机：`192.168.2.100`
- `ROS_DOMAIN_ID=13`
- 机器人配置：`/home/niic/shm_profile.xml`
- 推理机配置：`~/fastdds_inference_profile.xml`

机器人侧 DDS 配置在进程启动时读取；修改后必须重新启动 HC_X1 与 HC 中间件。

## 启动

机器人侧分别启动硬件和中间件：

```bash
cd ~/HC_X1 && ./start.sh
cd ~/HC-teleop-robotic && ./run.sh teleop
```

`run.sh teleop` 不再检查、启动或管理 HoloBrain HTTP Gateway，端口 `8765` 不参与
直连链路。

推理机启动：

```bash
cd ~/workspace/RoboOrchardLab
./projects/holobrain/scripts/start_hc_tj_ros_local.sh 'Pick up the object.'
```

旧命令 `start_hc_tj_http_local.sh` 保留为兼容入口，会直接转到上述 ROS 2 启动脚本。

客户端直接订阅：

- `/observation/cameras/head/color/compressed`
- `/observation/cameras/head/depth/compressed`
- `/observation/cameras/head/calibration`
- `/observation/cameras/head/extrinsics`
- `/observation/joint_states`
- `/observation/hands/joint_states`
- `/status/control/ownership`

模型可以在未授权时持续计算，但动作只有在 Dashboard 开启程序控制、所有权状态为
`AUTO/grant/motion` 且控制心跳有效时才会送入中间件。启动和重连不会自动授权。
