# 九个冻结识别器的恢复域退步诊断

状态：PLANNED。上轮三方法、三种子的正式测试已经完成，直接恢复域软 CE 没有达到目标。本次只分析合法源数据，不启动训练或新的目标测试。

固定 `PLControl / RawSoft / DURASoft × 2026092701/02/03` 九份 E50 默认头识别器，以及同契约原 F0 G-LU。每个 F 通过既有 reader 核实 own E20→E30→E50 的来源、数据角色和默认头；原 E30 仅作为自身分类头方向的参考。复用 VALIDATED_ONCE 数据，不重验数据或增加哈希层。

完整 source V 的 27000 个物理记录逐个使用相同八视图：raw、G(raw)，以及 low suburban、mid urban、low urban 的 H(x) 与 G(H(x))。信道 seed 20261016421、硬件 seed 20261016423、namespace `PAMF/restored_source/V/fixed`，与原 paired-source 诊断一致。G 固定 `reuse=True, graph=True, native=False`，FP32/TF32off。三种子共享同一组物理记录，不能把模型前向次数视为独立样本数。

观测指标为默认头准确率、真类 margin、相对同种子 PLControl 的纠错/破坏计数、raw 到各视图的特征余弦距离，以及同 TX 跨 RX 与异 TX 同 RX 的类中心距离。类中心仅作冻结特征统计，不拟合探针或新分类头。对 G 输出另外按对齐 NMSE 的固定区间 `[0,.01,.05,.1,.2,.4,1.00001]` 分层；对齐只消去一个整体复标量以测量误差，不改变预测 IQ。保留每种方法、种子、视图、TX、RX、day 和原 E30 raw 一致接受集合的完整结果。

全部 F/G 为 eval 且冻结，无 optimizer，无 U IQ/真值或目标读取，无 V 梯度或持久统计更新。结束核对所有参数与全部 registered buffers（包括非持久 bank）不变。预算为 1944000 次 F 包前向与 108000 次 G 包前向；真实 checkpoint 无 query smoke 在前：source L 的 2 包通过九个默认头，并核对部署 G 路径。

技术判停：首个来源/角色/物理顺序不匹配、非有限值、状态变更或运行错误即停止本诊断，保留部分产物，不自动重复提交，不操作其他健康 owner。科学诊断无新增通过门槛。NMSE、特征几何和分类头变化只提供关联，不能单独证明身份损失、接收机捷径或因果机制。

发布命令：`PYTHON -u -m experiments.pamf_channel_pl.restored_source_shift dispatch --config PROJECT/releases/20261025-diagnostic-PAMF-RestoredPL-sourceShift-release-r01/runtime.json`。CWD 为该不可覆盖 release；输出为 `PROJECT/runs/20261025-diagnostic-PAMF-RestoredPL-sourceShift-m9-r01`。环境为 N607 既有普通账户 Python/Torch2.1；GPU0 一条本 run worker，准入时全卡 CUDA/pending 总数小于 2、空闲至少 10000 MiB，CPU 2 threads。保留其他 owner。

两条聚焦单元/协议负向测试 PASS；一次独立 P0/P1 审查 PASS。Git commit 是代码/config 身份，记录实际 release commit；一次 archive SHA 传输比较、远端 compile 和启动后独立 PID/CWD/argv/GPU/log-growth 核实。`REJECTED_EXTRA_GATE`：上一目标 pipeline 全部结束不属于本源诊断输入条件，已移除该额外前置门槛，仍核实每份冻结 checkpoint 的合法来源。

预期产物：smoke.json、progress.json、私有 source_V_predictions.npz、complete.json、pipeline_state.json，以及本 run 的公开源域汇总、分层关联和限制。后续新方法须根据源证据另行固定机制与矩阵；本次诊断不将既有目标分数用于改权重、挑父模型或选择性重跑。
