# HC-TJ 训练数据评估的 Grad-CAM

在 `projects/holobrain` 目录、原有 HoloBrain 环境中运行。先重启更新后的
`scripts/inference_server.py`，使用与你的数据匹配的 HC-TJ 16 轴导出模型：

```bash
python scripts/inference_server.py \
  --model_dir /path/to/export/model \
  --inference_prefix hc_tj --input_profile hc_tj --port 2000
```

评估时增加 `--grad-cam-dir` 即可开启：

```bash
python scripts/eval_hc_tj_async.py \
  --server-url http://127.0.0.1:2000/holobrain/hc_tj_async/v1 \
  --dataset-path data/arrow_dataset/hc_tj_260818_fix \
  --sample-index 0 --sample-index 300 \
  --grad-cam-dir workspace/grad_cam
```

每个样本输出 `sample_XXXXXX_head_rgb.png`（原图）、
`sample_XXXXXX_head_grad_cam.png`（叠加图）、
`sample_XXXXXX_head_grad_cam.npy`（原始空间网格，float32、范围 0–1），以及
`sample_XXXXXX_head_grad_cam.json`（归因目标、分数、文件路径和是否为空）。
评估 JSON 的每个样本也包含对应路径。

默认目标是返回动作序列全部时间步、全部关节数值的均方幅值。
可以添加 `--grad-cam-joint 0 --grad-cam-step 0`，只解释第一个时间步的第一个关节；
也可以只指定其中一个。索引从 0 开始，关节顺序与接口的 `joint_names` 一致，
范围为 0–15，时间步范围为 0–63（不得超过模型实际返回的动作长度）。
各关节的单位/尺度不同，整段动作的默认目标可能由幅值较大的关节主导。

实现对动作解码器接收的第一层空间视觉特征 `[B, camera, channel, H, W]`
计算目标梯度，空间平均得到通道权重，加权求和、ReLU，再除以本图最大值。
这是融合 RGB、深度和语言后的空间特征对动作目标的正向归因，不是分类概率或物体检测框。
默认 HC-TJ deploy 处理器只缩放图像，因此将网格双线性缩放回原图进行叠加；
若自行修改处理器加入裁剪、翻转，需要同步修改可视化坐标映射。

红黄色高亮的不透明度随归因值增加；不同图片分别归一化，不能用颜色比较绝对重要程度。
全零热力图保留原图，并标记 `empty: true`，表示该目标没有正向 Grad-CAM 响应。
同一图片在不同动作目标下可能有不同热点。

不开启选项时不会计算梯度或保存图片。开启后保留整个动作生成过程的梯度图，
会增加显存和耗时；评估输出的延迟包含归因计算，`effective_action_steps` 也随之减少，
不应将这一延迟当作常规部署性能。推理服务串行执行请求，在共享服务上会阻塞其他推理。
不需要额外安装 `pytorch-grad-cam`。

验证：独立数学和图像保存测试可通过以下命令运行（需要 pytest、torch、numpy、Pillow、requests）：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/test_hc_tj_grad_cam.py -q -c /dev/null -p no:cacheprovider
```

## 模拟机器人 RGB 窗口

在仓库根目录运行本地模拟脚本（先启动更新后的推理服务）：

```bash
python projects/holobrain/own_test/simulate_hc_tj_async_e2e.py \
  --grad-cam
```

可增加 `--grad-cam-joint 0 --grad-cam-step 0`，不指定则解释全部关节和时间步。
GUI 的 RGB 区域显示最近一次完成推理所用的图像和归因叠加，标注请求编号及目标；
右侧深度也来自同一请求。等待下一次响应时保留上一张结果，避免把旧热力图叠到
新发布的图像上。初次响应前显示发布图像及等待提示。不加 `--grad-cam` 时仍显示
最新发布的 RGB-D；`--no-gui` 时也会请求并验证归因，结束报告包含
`grad_cam_responses` 次数。

服务端不支持 Grad-CAM 或返回非法热力图时，客户端会报告推理故障，不会静默
显示成普通 RGB。归因计算和本地渲染会增加动作响应延迟，仍按现有策略裁剪过期动作。
