# 恢复域监督剂量与轻噪声的源域诊断

PLANNED。此 run 仅完成既定恢复 IQ 计划的剂量及后扰动可行性分析，不创建 optimizer，不更新 F/G 参数，不访问目标 IQ/标签。矩阵唯一维护于 experiment.json。

固定原 F0 ordinary-PL E30：student seed2026092701，三冻结 raw 教师 seeds2026092701/02/03，原同契约 G-LU。复用已验证 E20→own10 来源、原 L/U/V 角色和 G 来源；加载仍检查完整 checkpoint header/实际 native U/EMA 关闭。source V 不求梯度。

## 12 个源 L/U 批次

原 L128 完整 T1 六前向损失保持；额外 raw U256 主项为原普通自 PL（自身 eval argmax/confidence≥.95，原始输入 CE，权重1，按完整 U batch 平均）。本诊断把这两项梯度相加作为 primary，不把新增教师规则混入原主项。

新增分支的 q 来自冻结三 E30 对 raw U 的平均概率，只接受三者同类且各≥.95；不用 G 自身置信度再筛选。对相同 U/标签/接受集合分别计算 raw 和 G(H(U)) 两次 student forward 的平均软 CE，各只求一次梯度。旧 native U/EMA 关闭，U 标签必须为隐藏 -1。H 使用原三个源 practical residual 家族循环；U namespace 为 PAMF/restored_dose/U/fixed，不改变上次 V 物理视图。

12 个独立源批次、L1536/U3072 曝光、48次 autograd.grad；不构造 optimizer。每批原生 train-mode 临时 buffers 在批后复原，参数/保留 buffers 最终逐项不变，student 随后丢弃。所有 V 预测使用原冻结 eval 教师。

预定剂量公式：`min(0.25,0.25*median(primary_norm)/median(restored_extra_norm))`。主项/新增梯度有限、非零且12批完整；median cosine≥−0.25 为本源梯度可行性条件。只根据这一固定源规则产生建议剂量，不访问目标分数。

梯度对象明确为全部实际可训练参数的 pre-clip 梯度：Sinc 的 CENTER_25K 两坐标梯度除以25000，与原 CENTER_UNITS 全局物理梯度 clip5 保持相同单位；其他参数保留原单位。这不是 AdamW/momentum 更新方向估计，不以坐标尺度造成的范数优势调整剂量。

## 完整 V 的恢复后轻噪声

复用上一 paired-source run 的完整 source logits，先核对同一 context、源物理顺序和各场景首批部署恢复/原默认头 logits。对全部 V27000、三个源场景、三冻结 F 测按包功率标定的40 dB复高斯噪声。噪声 seed20261025301，按物理记录/场景稳定生成；不再过 G 或额外归一化。

预定 D 组准入：9个 seed×scene 单元全部满足准确率变化≥−0.1pp、预测改变率≤0.5%。不通过则 D 不进入后续训练矩阵；C 不依赖 D 成立。该判定仅是源身份稳定性，不是目标收益证明。

## 发布和预期产物

7条相关单元/协议负向测试 PASS；一次独立 P0/P1 审查 PASS。真实权重 L2 无 query smoke 是 launcher 首步，通过即继续。Git commit 为代码/config身份，runtime记录实际commit，不追加逐文件/数据哈希。

命令：`PYTHON -u -m experiments.pamf_channel_pl.restored_dose_source dispatch --config PROJECT/releases/20261025-diagnostic-PAMF-DURA-dose-noise-release-r01/runtime.json`。

N607 普通账户；CWD为上述不可覆盖 release，输出 PROJECT/runs/20261025-diagnostic-PAMF-DURA-dose-noise-m1-r01；GPU0一条本任务worker，全卡CUDA/pending<2且空闲≥10000MiB，CPU2threads，保留所有健康任务。一次 archive传输SHA比较、远端compile及启动后独立PID/CWD/argv/GPU/loggrowth核对。

第一处技术/协议/来源/非有限/基线复用或持久状态错误停止本run，保留部分结果，不自动重试。预期 smoke.json、gradient_complete.json、progress.json、source_noise_predictions.npz、complete.json、pipeline_state.json；公开源梯度/噪声汇总和解释。下一训练矩阵、实际剂量、A基线是否可合法复用须在任何新目标访问前另行冻结，本run不启动训练。
