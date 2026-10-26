# HoloBrain 中的 RTC：原理、调用链与时序

本文按当前仓库代码解释 RTC，并以 HC-TJ 异步真机链路为完整例子。这里的 RTC 是
`RTCInferencePlugin`：类注释称 *real-time chunking*，部署指南称 *Real-Time
Correction*。两种叫法指向同一段实现：把上一轮尚未执行的动作，与本轮新预测的动作
在重叠时间段内融合。它是**推理时的动作连续性处理**，不是网络协议、控制器，也不
参与训练。

## 先用一个场景理解

模型一次预测未来一段动作（action chunk）。机器人执行第 1 段动作时，下一次推理
可以同时进行。新结果回来时，旧动作通常还有一部分未执行。如果立即丢掉旧轨迹、
直接切换到新轨迹，关节目标可能突然跳变。

RTC 让新轨迹的开头先贴近旧轨迹，再逐步转向新预测。它解决的是**两段动作交接处
的连续性**；它不会消除推理耗时，也不会根据机器人反馈重新规划或保证动力学安全。

```text
时间向右 →
旧 chunk：已执行 ─────────│ 仍未执行的旧动作 ────────────────
新请求：                    ↑ 携带 remaining_actions 和 delay_horizon
新预测：                    │ 新动作 ─────────────────────────
融合后：                    │ 先沿旧动作 → 渐变 → 采用新动作
```

## 两个输入各是什么意思

| 名称 | 含义 | HC-TJ 线上格式 |
| --- | --- | --- |
| `remaining_actions` | 在**发送新请求时**，旧 chunk 还没执行的动作，从当前时刻向未来排列 | 可选 `float32 [N,16]`，原生 30 Hz |
| `delay_horizon` | 预计新请求完成前会经过多少个动作步；这些步优先沿用旧预测 | 有剩余动作时必填，整数 `0..N` |
| `max_horizon` | 插件最多处理多少个重叠步，不改变模型总预测长度 | 插件默认 `32` |

这里的 `delay_horizon` 是**步数**，不是毫秒，也不是推理次数。HC-TJ 客户端用
`ceil(估计请求耗时 × 30)` 计算它，并裁剪到剩余动作长度。第一次请求没有旧动作；
后续请求如果旧 chunk 已执行完，也不带这两个字段。服务端的 v1 协议会校验形状、
关节顺序、数值有限性以及 `delay_horizon` 范围。见
[`validate_inference_request`](../scripts/hc_tj_async_core.py) 和
[`_decode_hc_tj_async_request`](../scripts/inference_server.py)。

## 融合算法：哪些步听旧的，哪些步听新的

插件接受新预测 `pred [B,T,J,C]` 和旧剩余动作
`remaining_actions [B,R,J,C_old]`。实际融合长度为
`H = min(T, R, max_horizon)`；保护长度为 `D = min(delay_horizon, H)`。
对前 `H` 步、旧动作具有的通道，按下面的权重逐元素计算：

```text
融合动作[i] = w[i] × 旧动作[i] + (1 - w[i]) × 新动作[i]

0 <= i < D：w[i] = 1                 完全沿用旧动作
D <= i < H：w[i] 从接近 1 衰减到接近 0
i >= H：保持原始新预测
```

后半段的权重**并非线性下降**。源码令 `h = H - D`，取 `h` 个从 1 向 0
排列且不包含端点的 `c`，再算
`w = c × (exp(c) - 1) / (e - 1)`。所以 `D..H-1` 的权重仍大于 0，
而 `H` 及以后直接使用新预测。举例：若旧动作剩 20 步、`delay_horizon=6`、
`max_horizon=32`，则第 0–5 步完全沿旧动作，第 6–19 步渐变，第 20 步起完全
使用新预测。若 `D=H`，这 `H` 步全部沿旧动作，源码会记录“剩余动作过短”的
警告。源码类注释中的“线性下降”和形状参数名描述与实际实现有出入，应以
[`RTCInferencePlugin.forward`](../../../robo_orchard_lab/models/rtc_plugin/rtc_plugin.py)
的计算为准。

插件**原地修改** `pred` 的前 `H` 步；它只改旧动作包含的通道。HC-TJ 回传的是
16 个关节位置，因此进入插件时 `C_old=1`，只融合关节位置通道，模型预测的姿态
等其他通道仍来自新预测。

## 一次请求从机器人到模型再回来

1. **客户端继续执行旧轨迹。** HC-TJ ROS 客户端以默认 100 Hz 运行控制定时器，
   推理定时器每 20 ms 检查一次；推理请求交给单线程 worker，未完成时不会再发新
   请求。到达设定的推理频率后，它按旧 chunk 的起始时间和 30 Hz 去掉已经过期的
   动作，得到 `remaining_actions`；根据往返耗时的指数滑动平均估计
   `delay_horizon`。见 [`_remaining_actions`、`_inference_tick`](../scripts/hc_tj_async_client.py)。
2. **HTTP 请求传递原生动作。** 客户端用 multipart `.npy` 上传观测和可选的
   `remaining_actions [N,16]`，以表单字段传 `delay_horizon`。此时旧动作保持
   **30 Hz**，没有先插值到 100 Hz。见 [`_build_request`](../scripts/hc_tj_async_client.py)。
3. **服务端校验并装配模型输入。** `/{server_name}/hc_tj_async/v1` 解码和校验
   请求，把旧动作加一个 batch 维，放入 `MultiArmManipulationInput`。若携带旧动作
   而已加载的 decoder 没有 `RTCInferencePlugin`，服务端直接返回协议错误；共享
   pipeline 在锁内串行调用。见 [`inference_server.py`](../scripts/inference_server.py)。
4. **processor 保留这两个字段。** `Struct2Dict` 把它们放入输入字典；HC-TJ
   deploy transforms 选中这两个字段并加入 batch 维。随后
   `InferencePipeline` 预处理、把数据移到模型设备、调用模型；模型经图像/文本
   特征提取和空间增强后调用 action decoder。见
   [`processor.py`](../../../robo_orchard_lab/models/holobrain/processor.py)、
   [`config_hc_tj_ro_dataset.py`](../configs/config_hc_tj_ro_dataset.py)、
   [`basic.py`](../../../robo_orchard_lab/pipeline/inference/basic.py)、
   [`structure.py`](../../../robo_orchard_lab/models/holobrain/structure.py)。
5. **decoder 在去噪循环中调用 RTC。** decoder 把旧动作转换成与模型预测相同的
   关节归一化空间；每一步扩散去噪先通过 `get_prediction` 得到绝对状态预测，
   再调用插件融合，最后把融合结果交给噪声调度器推进下一步。也就是说 RTC
   **每个去噪步都会参与**，并非只在最终结果上混合一次。只有插件存在且两个
   字段同时出现时才执行。见
   [`HoloBrainActionDecoder.forward`](../../../robo_orchard_lab/models/holobrain/action_decoder.py)。
6. **输出再回到物理关节单位。** decoder 后处理将关节位置反归一化；processor
   取第一条轨迹的关节通道，得到 `[T,J]`。HC-TJ v1 服务端返回 30 Hz、16 维
   `actions`，不做插值。客户端收到响应后，按**本次真实 HTTP 耗时**再次删除
   已过期的新动作，检查首点相对当前反馈的跳变，再插值至控制频率执行。见
   [`post_process`](../../../robo_orchard_lab/models/holobrain/processor.py)、
   [`_infer_hc_tj_async`](../scripts/inference_server.py)、
   [`_inference_tick`](../scripts/hc_tj_async_client.py)。

把调用关系压缩成一行：

```text
ROS 观测/旧动作 → hc_tj_async_client._build_request → HTTP v1
→ inference_server._decode_hc_tj_async_request → pipeline
→ HoloBrainProcessor.pre_process → HoloBrain_Qwen2_5_VL.forward/predict/_forward
→ HoloBrainActionDecoder.forward → RTCInferencePlugin.forward（每个去噪步）
→ decoder/processor 后处理 → HTTP actions → 客户端裁掉过期步、插值、限速执行
```

## 时序例子：为什么还要在客户端“裁掉过期步”

假设旧 chunk 还剩 24 步，客户端估计一次请求耗时 0.20 秒。30 Hz 下
`delay_horizon = ceil(0.20 × 30) = 6`。请求发出后，机器人仍在执行旧动作；
模型生成新 chunk 时，前 6 步由旧动作保护，后面渐变。如果响应实际耗时
0.23 秒，客户端会从返回的新 chunk 中删除 `floor(0.23 × 30) = 6` 个已经
过时的步，然后执行余下部分。由此可以区分两个机制：

- RTC 的 `delay_horizon` 是**提前估计**，用于模型内交接；
- `trim_elapsed_actions` 使用**实际耗时**，用于客户端确定现在还能执行哪些步。

两者都按模型原生 30 Hz 计算。估计不准时，融合不能独自保证首点安全；客户端
仍会检查相对反馈的首点跳变，并在 100 Hz 控制侧限位、限速。相关实现分别在
[`hc_tj_async_core.py`](../scripts/hc_tj_async_core.py) 和
[`hc_tj_async_client.py`](../scripts/hc_tj_async_client.py)。

## 启用条件与容易混淆的边界

- **训练不走 RTC。** decoder 的训练分支计算训练目标与损失；插件调用只在
  `else:  # inference` 分支。首次推理没有旧动作，也不会触发插件。
- **配置插件不等于每次必用。** `async_inference_plugin` 被构建在 decoder 上；
  只有请求同时提供 `remaining_actions`、`delay_horizon` 才调用。`max_horizon`
  限制融合窗口，不是模型预测步数，也不是控制频率。
- **当前配置需要核验。** `config_holobrain_qwen_common.py` 使用
  `config.get("async_inference_plugin")` 构建 decoder；当前
  `config_holobrain_qwen_hc_tj.py` 没有显式设置该字段。因此仅凭该配置文件名
  不能断言导出的模型已启用 RTC。部署前检查导出目录中的
  `model/model.config.json`，确认 decoder 内有
  `robo_orchard_lab.models.rtc_plugin.rtc_plugin:RTCInferencePlugin`；
  可按[HC-TJ 部署文档](hc_tj_async_real_robot_deployment.md)提供的配置启用。
- **不要把两条 HTTP 路由混作一条。** 通用旧接口 `/{server_name}` 会把输出
  插值，并对传入的剩余动作做下采样；本文逐步讲解的是 HC-TJ 的
  `/{server_name}/hc_tj_async/v1`，它在网络两端都保持模型原生 30 Hz。
- **时间对齐是调用方的责任。** 插件按数组下标融合，没有时间戳对齐或对旧动作
  真实性的验证。`delay_horizon` 是往返耗时估计，实际延迟、排队和控制时钟可能
  偏离；窗口结束处也直接切换到新预测。阅读融合效果时应结合客户端的响应裁剪、
  跳变检查与控制限速一起看。

## 推荐的源码阅读顺序

1. [`rtc_plugin.py`](../../../robo_orchard_lab/models/rtc_plugin/rtc_plugin.py)：
   权重公式和修改范围。
2. [`action_decoder.py`](../../../robo_orchard_lab/models/holobrain/action_decoder.py)：
   插件的启用条件、归一化与去噪循环。
3. [`processor.py`](../../../robo_orchard_lab/models/holobrain/processor.py)：
   输入字段和最终动作形状。
4. [`hc_tj_async_client.py`](../scripts/hc_tj_async_client.py)、
   [`inference_server.py`](../scripts/inference_server.py)：异步时序与 HTTP 边界。
5. [`hc_tj_async_core.py`](../scripts/hc_tj_async_core.py)：协议校验、过期动作裁剪
   和控制侧插值。
