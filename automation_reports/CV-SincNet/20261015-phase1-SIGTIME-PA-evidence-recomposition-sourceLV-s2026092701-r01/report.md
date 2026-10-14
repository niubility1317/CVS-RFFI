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
