# LoRa 观测坐标的同包源 V 对照

固定同一 Tri-set `2026092702` 和源 V 面板。原始 IQ 坐标的观测列采用去扫频相位传输，朴素对照在去扫频域直接估计。两者共享异折、三复系数、Ridge、幅度约束、同一实收门及固定三次逆更新。输出保留原载波，中心回写半量。并行比较直通、原冻结 DURA 半量、扩容半量、两种独立观测校正以及两种校正接续扩容 DURA。组合在同一激活包上执行，旁路包逐值复用原扩容结果。 [实收模型](../../../experiments/pamf_lora_main/receive_frame_action.py) · [源波形证据](../20261125-phase1-lora-dura-dual-aperture-s2026112991-r01/report.md)

门限由合法源 U 中同一 2,500 个地面包，从 256 和 384 kS/s 的六窗异折误差降幅取合并最大值乘 1.05 后冻结。预测只输入当前实收 IQ。前导相位来自当包去扫频，任何实验编号或干扰真量均不进入估计。零优化步，新增学习参数为零。源 V 使用同一固定的 900 个包，五台接收机各 180。每套实际 256 kS/s 视图均含 clean 与六个 practical 场景。两条路线盲预测都闭合后，独立 CPU 才读取源真值。目标 IQ 和目标真值访问为零。 [运行入口](../../../experiments/pamf_lora_main/receive_frame_action_source.py)

源 V 发布前修正了限幅。六个 chirp 共同按最大系数范数缩放，保持逐项相对系数零均值和上限 0.55。定向回归复现并关闭了逐窗限幅产生共同分量的问题。同一真实源 U 重跑的门限仍为 0.543297，两种采样率均为 0/500 clean 触发。强相对作用的 9–32 kHz 独有复误差中位差为 −3.30、−3.37 dB，各激活 487、488 个包，激活包整包误差均下降。弱档分别为 −6.08、−6.06 dB，各激活 500 包。弱共享档仍有 21、8 个激活包的整包误差上升。完整复算保留在 `source_receive_frame_action_r03.json`。 [诊断代码](../../../tools/lora_receive_frame_action_probe.py)

记录同包正确数、逐接收机净变化、纠错改错和配对区间。源 U 实检查保留真权重读出、激活和旁路两组时延。N607 普通账号，优先选相对空闲 GPU，两条路线独立并行。工作目录为 `${N607_PROJECT}/releases/20261125-phase1-lora-receive-frame-action-source-m2-r01-release`。命令 `python -m experiments.pamf_lora_main.receive_frame_action_source publish --repo <checkout> --output <artifact-root>/release --context <runtime_source.json> --ssh-config <config>`。实际提交、执行环境和运行根由已推送 HEAD 与发布记录固定。数据契约和权重绑定、非有限量、无目标检查或闭合失败时停止本运行。预期为 `gate.json`、`smoke.json`、两套 `*-predictions.json`、两套 `*-source-scores.json` 和 `SOURCE_SCORED` 状态。结果用于研判是否把坐标变换纳入下一版端到端训练。当前 W24 和双采样率训练及正式评测沿用各自登记。

## 运行与同包结果

N607 已独立核对执行提交 `3e4747310555a4f5c72382a03b5d1f4a9d1b9c8f`、归档和远端编译。15 项聚焦测试通过，独立审查发现的相对限幅问题已修复并复核。GPU 6 在 3.87 秒内冻结源门限 0.5432968765。GPU 3 的真实权重检查中，2 个源地面包全部旁路，10 个通用受扰包全部激活。RTX 3090 上单包成对工程读出的旁路与激活中位时延分别为 7.30、15.42 ms，仅计这对读出本身。随后 GPU 6、2 分别完成 Full、Residual，CPU 在两套预测闭合后评分，调度器以 `SOURCE_SCORED` 退出。

| 读出 | Full 六景 | Full 低城区 | Full clean | Residual 六景 | Residual 低城区 | Residual clean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 直通，两种独立坐标校正 | 3,028 | 360 | 667 | 2,848 | 387 | 667 |
| 原冻结 DURA 半量 | 3,068 | 363 | 667 | 2,960 | 403 | 665 |
| 扩容半量，两种坐标校正接续扩容 | 3,125 | 389 | 667 | 2,914 | 385 | 667 |

六景分母为每路线 5,400，低城区和 clean 各为每路线 900。两路线全部 12,600 个包视图的门控触发数为零。两种独立坐标校正逐预测与直通一致，两种接续扩容逐预测与扩容半量一致。坐标变换相对匹配朴素对照，两路线六景均为 0 次纠错、0 次改错，95% 配对区间为 [0，0]。两路线和五台接收机的正确数、逐景纠错与改错已用本地同序源 V 真值独立复计，证据见 `readout/independent-recount.json`、`full-source-scores.json` 与 `residual-source-scores.json`。

固定门可识别专门施加的相对镜像和回波，但在本次登记 practical 源 V 上未触发。受控波形增益因此没有转化为设备识别增益。这一工程对照结果保留在源域。正式识别验收仍等待 W24 和双采样率候选各自的冻结与既定测试。
