# 原噪声分布下正负配对：源域梯度方差反证

状态：PLANNED，技术恢复 r02，尚未启动。训练预算上限 E30；本行是零更新源诊断，读取固定全源 scratch E20 checkpoint，不是 E30 训练结果。

唯一假设：保持原 practical 完整处理与每个观测噪声分布，e/-e 负相关配对相对 e/e2 独立配对，可降低既有 RDrop 监督项的梯度估计方差。Gaussian 对称性只支持边际分布保持，不保证非线性梯度方差下降。[Wu 等，2019](https://proceedings.mlr.press/v89/wu19c.html) 的变分推断研究提供动机，不能作为 CVS 身份收益证据。旧独立 MC2、N0、FIR 等负结果保持关闭，本行不改噪声幅度或传播先验。

## 数据与最小矩阵

父为完整源 CONCAT-s2026092701 的 fixed-final20，L6300/U56700 隐藏 TX/V27000、RX1/3/4/6/8、day1/2/3、split392005，与原 source_contract 匹配；无未知祖先或目标选模。诊断取每个 6TX×5RX×3day 单元按 opaque ID 排序的前 2 个 L 样本，共 180 个；不是源 RX 留出或新的训练契约。U/V 不使用，U TX 不解封。

6 个固定 B30，3 个既有训练场景（low suburban、mid urban、low urban），每条件 8 次噪声重复。相同 H/几何/状态/接收硬件/质量/补偿，生成 e1/e2/-e1 三个完整观测；各自重算 AGC，不平均 IQ。每视图两次 RDrop 复用同 IQ；slot1 与 slot2 mask 独立，e2/-e1 共用 slot2 随机流，比较平均损失梯度。默认 cosine W6×160 与骨干均不更新，不拟合诊断头。

主量为所有原 trainable 参数原坐标下的梯度协方差迹，默认 W 单列；其数值依赖参数坐标，不等于 AdamW 更新方差、全部 L/U 目标方差或识别率。保存每组 24×24 梯度 Gram，支持零 IQ/CP/NN 独立重算，不保存完整梯度向量。

## 固定条件与停止规则

整体 full-gradient 和默认 W 方差比均 ≤0.8，mid/low urban 的 full-gradient 方差比均 <1，才支持源估计器假设。失败关闭固定 coupling，不换 draw/scene/seed/阈值救跑；通过也只是源估计器证据，后续匹配全源 E30 识别实验须独立登记，不自动扩种子或训练。不能替代 A_g clean 净收益、A_s 身份收益、完整 B00/B10/B01/B11 或 low>60% 联合目标。

## 执行与费用

- run：`20261017-diagnostic-F-antitheticNoise-sourceL-s2026092701-m1-r02`，owner：`codex/root/F-antithetic-noise-source`。
- CWD：`PROJECT/releases/cvs_F_antithetic_noise_source_20261017_r02`；命令：`CVS-RFFI/bin/python -u experiments/cvs_antithetic17/runner.py`。既有固定 full-source F runtime，FP32，CPU 2 线程；dispatcher 按全部 owner 每 GPU 最多 3 个进程且空闲至少 12 GB 选择单 GPU。
- 输出：`PROJECT/runs/20261017-diagnostic-F-antitheticNoise-sourceL-s2026092701-m1-r02`，一次不可覆盖根目录。实际 code/config commit 为发布时 Git HEAD，写入 release_commit.txt；单 archive 本地/远端 SHA 比对和远端 compile。
- 预算：25922 包 NN（含真实 checkpoint 无 query 合成 smoke 2），432 次 autograd，12960 次完整物理处理；1 CP/1 既有源 cache，0 optimizer/teacher/headfit/U/V/target。
- 技术停止：源契约不符、非有限数、输出冲突或启动失败，保留自有产物，不自动重试，不影响其他任务。
- 预期：resolved_config.json、real_checkpoint_smoke.json、progress.json、result.json（Gram/损失/方差）、completion.json、dispatcher_complete.json、日志及同 row 源报告。

## 已验证与待完成

本地 compile PASS；8 个协议负例、2 个纯假进程启动回归、48 次合成原生物理处理 PASS，12 个正视图回放最大差 0，Gram 方差与直接重算误差 <1e-12；无真实 IQ/CP/NN/更新。独立 P0/P1 审查指出一次启动接口 P1，已修复，一次定向复查 PASS，P1 关闭。尚未发布、读取真实 CP 或得到结果。


## 唯一技术恢复差异

r01 在真实 CP smoke 后的 NumPy→Torch 处失败，0梯度/0更新，无科学结果，原根保留，见 [r01 证据](../20261017-diagnostic-F-antitheticNoise-sourceL-s2026092701-m1-r01/report.md)。相同原生 runtime 的纯合成复现确认 ndarray 桥错误；采用已有源实现的 byte-buffer 桥，float32 B30 逐元素一致。本行所有抽样、科学阈值、CP、预算与判定均不变；额外失败成本单独记 2NN/90物理处理/1CP/1cache。r02 仍预期 25922NN/432autograd/12960物理处理/0更新，既定 E30 上限不改变。

恢复修复一次定向 P0/P1 复查 PASS；本地恢复检查全部 PASS（旧 Torch 对 NumPy2 的警告保留，桥不使用该 C API），编译与登记字段检查 PASS。
