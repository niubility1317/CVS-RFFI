# WiSig 原冻结 R69 作用与方差隔离

固定原 R69 `stage2/native_map_prior` E20/4,940 和 Main split Tri-SIG-TIME SSL 种子 `2026092701`。复用此前阶段归因的同一 1,800 个源 V 物理包，两路线各 clean 和六景。每条路都重跑直通、原 Graph-RR 和五个干预读出。原恢复结果和同包直通决策必须逐包重现归档。 [R69 原训练与推理](../../../docs/DURA_V1.md) · [同包阶段对照](../20261124-diagnostic-wisig-r69-stages-source-m2-r01/report.md)

前三个干预将 R69 用于两个阶段的作用分别替换为单位算子、只保留原估计 FIR、只保留原逐点主复响应和共轭项。后两个保留完整原作用，分别使用 4,096 个独立源 U 上原输出方差的中位数，以及当前包短训练重复的相位补偿差分功率折算的每实分量方差。这两个替换同时影响先验的方差输入和实际求解权重。短训练差分还包含前导错位、干扰及补偿残差，作为有效方差对照。保留原粗恢复、先验权重和两阶段四次 CG，干预结果不用于训练新权重。 [实现](../../../experiments/pamf_unrolled_restore/wisig_r69_observation_isolation.py)

本次是隔离机制诊断，R69 历史继承了模拟作用和噪声监督。运行阶段读数只用源 U 及当包实收 IQ，零训练、零目标读取。两路线 14 个源 V 矩阵全部冻结后，独立 CPU 才读取源标签，报告按场景和接收机的正确数、纠错改错及物理包配对区间。

N607 普通账号和空闲优先策略，两路独立 GPU，CPU truth-last。工作目录为 `${N607_PROJECT}/releases/20261125-diagnostic-wisig-r69-observation-source-m2-r01-release`，运行目录为 `${N607_PROJECT}/runs/20261125-diagnostic-wisig-r69-observation-source-m2-r01`。发布使用 `python -m experiments.pamf_unrolled_restore.wisig_r69_observation_isolation publish --repo <checkout> --output <artifact-root>/release --context <runtime_source.json> --ssh-config <config>`。归档匹配、真实权重检查、原输出复现或预测闭合失败时仅停止本运行并保留产物。固定方差先写 `constant-variance.json`，然后加载识别器执行无目标检查。预期输出 `smoke.json`、14 个盲矩阵、`predictions/complete.json` 及 `source-scores.json`。
