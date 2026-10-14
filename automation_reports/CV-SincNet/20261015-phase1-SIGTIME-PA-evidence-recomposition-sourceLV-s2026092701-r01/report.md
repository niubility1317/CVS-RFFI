# 同身份跨 RX 的 PA 证据重组：固定源域原型

本轮检验一个明确假设：有标签样本的 PA 分支如果能与同身份、不同 RX 的基干证据兼容，辅助学习就可能减少各分支对同一接收机背景的共同依赖。这里的 PA 仍是模型的物理分支表征，不能解释为已恢复真实发射机参数。

最近完整 E20 矩阵已封闭：CROSS_GAP 默认头三种子 clean 均值 81.0853%，比匹配 BASE_SSL 低 0.0480 个百分点。因此不继续扫描 sharpness 的扰动半径、权重或时序。保留 SIG_TIME、固定 CENTER25K 和历史已验证正收益架构，各历史结果不覆盖、不混作同预算结论。

实际部署特征按原浮点顺序为 `((identity+tanh(PA_gain)*physical)+tanh(reference_gain)*reference_raw)+.2*tanh(SIG_gain)*sequence_raw`。只交换 `physical`，其余来自 anchor。辅助 CE 沿真实特征路径回传，既有默认 W 和三个融合 gain 在辅助图中 detach；主 CE 的原联合训练方式保持。本原型没有真实训练或另训分类头。

MixStyle 的原工作混合实例特征统计，FSDCL 将低频风格操作与语义保持结合。两者说明分解、重组和语义保护具有既有研究基础；本轮只主张一个待验证的 CVS 具体训练假设，不宣称已证明原创性或泛化效果。[MixStyle](https://arxiv.org/abs/2104.02008)，[FSDCL](https://arxiv.org/abs/2108.08596)。

固定矩阵为 BASE_CE 第一种子 2026092701 的自有初始权重与 final20，每状态四个 L90（ordinal 0/17/35/52），每单元 ZERO、SELF、SWAP_PA。配对按同 TX/eq/day 的五个源 RX 组成固定双射循环；缺任一完整组则整个辅助批跳过。SELF 区分额外普通 CE 与重组产生的独立学习信号。三模式均计算两个默认头读出，实际主网络前向相同。

梯度只读取 L。固定 V180（90 cell 各 ordinal 0/1）只作无梯度机制诊断。临时更新后的 V 读出恢复更新前默认 W 和全部融合增益，直接评估实际 joint160 的身份余弦间隔与 pre-W CE。初始优化器为空矩的假想原配方 AdamW，末态使用实际保存矩与末态 LR；均为 CPU 临时副本，分别 clip5，不修改原模型和优化器，不冒充同时发生的 CUDA 训练更新。

预登记停止条件：8 个单元至少 2 个同时满足 SWAP 相对 SELF 的梯度相对差 >0.05、V 身份间隔增加 >1e-5、pre-W CE 减少 >1e-6、V 特征 RMS 差 >1e-7、至少 1 个 Sinc 实际滤波器产生辅助变化，辅助梯度范数 >1e-8，W/三个 gain 的辅助梯度均 None。该门槛只筛查信号与单步方向，不能证明训练收益。门槛失败时关闭这个固定变体，不按结果重排阈值、权重或配对。

预算为 360 个独立 L 和 180 个独立 V、5404 packet 主网络前向、48 次临时 AdamW 更新、0 次真实更新/另训头/EMA/U/target。真实 checkpoint no-query smoke 是入口第一阶段，两状态 PASS 后立即进行源诊断。源角色与 ancestry 在加载前核对，复用既有 VALIDATED_ONCE 数据。

run_id：`20261015-phase1-SIGTIME-PA-evidence-recomposition-sourceLV-s2026092701-r01`。唯一 owner：`codex/root/PA-evidence-source-20261015`。执行代码使用发布时 Git HEAD，写入 `release_commit.txt`；父 E20 执行代码 `6693e4480ef990bc23c5021ff1440b547fe5fcaf`。环境为 N607 普通账户、CVS-RFFI Python、CPU 2 线程、GPU 0。CWD 为 `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_PA_evidence_recomposition_sourceLV_20261015_r01`。命令为 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_PA_evidence_recomposition_sourceLV_20261015_r01/experiments/cvs_evidence_recomposition20/runner.py /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_PA_evidence_recomposition_sourceLV_20261015_r01/resolved_request.json`。输出为 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-SIGTIME-PA-evidence-recomposition-sourceLV-s2026092701-r01/source_diagnostic`，日志为 `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_PA_evidence_recomposition_sourceLV_20261015_r01/source_probe.stdout.log`。矩阵、阈值、预期产物与技术停止规则详见 [experiment.json](experiment.json)。

状态：PLANNED，尚无实际数据结论。后续训练只有通过这个固定诊断才另行预登记，遵守 scratch20 或合规 E20+20 的当前预算，保留匹配纯 CE 对照、隐藏 U TX、原默认头与冻结后 clean truth-last 流程。


## 实际结果：固定原型关闭

状态为 **ANALYZED / VERIFIED**，不是机制晋级。固定源 gate **0/8**：INITIAL 0/4，FINAL20 0/4。执行代码 `047dfc2f4e88d18c5ebe0f584f88f8fc48ccfd64` 已推送并独立核对 remote OID。单一 archive SHA、远程编译、绑定进程 CWD／命令／CPU 与日志增长通过；随后独立读回确认该进程自然终止，完整 summary 与 8 个记录存在，无 failure。没有停止其他任务。

| 状态 | L ordinal | 梯度相对差 | V 身份间隔增量 | V pre-W CE 减少 | Sinc 辅助变核数 |
|---|---:|---:|---:|---:|---:|
| INITIAL | 0 | 0.005784 | -3.378e-08 | +1.390e-06 | 4/24 |
| INITIAL | 17 | 0.004381 | -2.742e-07 | -1.955e-05 | 6/24 |
| INITIAL | 35 | 0.004767 | -3.241e-08 | +1.074e-04 | 4/24 |
| INITIAL | 52 | 0.004457 | -2.717e-08 | -1.900e-05 | 6/24 |
| FINAL20 | 0 | 1.263735 | -2.669e-08 | +3.003e-07 | 3/24 |
| FINAL20 | 17 | 10.727818 | +2.533e-07 | +2.185e-09 | 5/24 |
| FINAL20 | 35 | 2.604050 | +3.590e-07 | +8.704e-07 | 6/24 |
| FINAL20 | 52 | 1.053592 | +1.991e-07 | -4.154e-08 | 2/24 |

初始态的梯度相对差仅 0.004381 至 0.005784，低于固定 0.05，四个单元的 V 身份间隔均为负。末态相对差为 1.053592 至 10.727818，说明重组确实产生不同方向；但间隔增量最高只有 3.5901e-7，低于 1e-5，pre-W CE 减少最高 8.7035e-7，低于 1e-6。出现不同梯度和非零参数变化，仍不足以说明新的身份证据得到改善。

上述间隔与 CE 增量以 SELF 为对照，不是相对更新前。更新前末态 V pre-W CE 为 0.1142069981，全部四 cohort、三模式的临时更新后均高于此值。初始普通 CE 位移能显著降低 pre-W CE，但四 cohort 的身份余弦间隔均比更新前降低。两个度量描述不同性质，不能把其中一个改善当作另一项已改善；这也是后续需要直接检查真实特征位移的原因。该观察来自 V180、两个时刻的稀疏诊断，不是全程训练有害或接收机捷径的因果证明，V 也不能成为后续投影的梯度数据。

每单元 90/90 PA 特征发生变化；原 W 与三个 gain 的辅助梯度均 None，四项 SELF 精确还原，主 CE 的 W 梯度非零。SWAP 临时更新相对 base 改变 2 至 6 个 Sinc 核，V 相对 SELF 的原特征 RMS 差为 1.04e-6 至 1.29e-4。因此，这不是辅助未执行或更新完全被 FP32 抹掉。初始参考 gain 为零，真实参考投影辅助梯度也为零；末态参考路径有非零梯度，不能把初始参考零梯度误报为接线故障。

实际读取 360 个 L、180 个 V，5404 packet 网络前向，48 次临时 CPU AdamW，0 次真实训练／分类头拟合／EMA 更新／U IQ／U 真值／target 访问。原模型、优化器、梯度与 RNG 未变。没有新测试预测或 truth-side 评分，因为此 run 只授权源机制诊断，不能据此更新 clean 性能。

两状态之间没有中期 checkpoint，本结果不证明训练期间所有证据重组都无用。末态 LR 为 1e-6，初始时刻为原配方 2e-4 且为空矩假想优化器；不能将初始与末态量级当作相同优化条件。但 8 个固定单元的既定条件均未通过，因此按预登记关闭这个完整 PA 交换变体，不扫描混合系数、loss 权重或延伸隐藏 U donor。

最新附件提出的集合监督和重复观测预测仍是不同机制：本仓库对应固定集合版本曾错误排除源真类，重复周期预测和正交补版本也未通过既定源条件。它们的负结果均保留，本次不重新启动这些原配方。旧 PA 小幅插值三种子训练也保留，不与这里的 SIG_TIME 完整交换、单步诊断混合统计。

下一项独立训练机制为直接约束 AdamW 实际部署特征位移的源 RX 身份风险。只完成设计和纯数学原型，真实源验证和主训练尚未发布，见 [下一步方案](../../../analysis/phase1_rx_update_projection_20261015.md)。SIG_TIME、固定 CENTER25K、COMPACT_LINEAR 和 LONG_SILU 原正收益路线独立保留。84% 默认头 clean 三种子目标仍 active、未完成。

完整标量记录见 [result_evidence.json](result_evidence.json)，八单元表见 [source_cells.csv](source_cells.csv)，发布／终态证据见 [launch_evidence.json](launch_evidence.json)。原始运行与日志在私有 run 产物中保留。
