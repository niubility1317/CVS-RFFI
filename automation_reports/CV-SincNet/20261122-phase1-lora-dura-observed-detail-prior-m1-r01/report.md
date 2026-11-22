# LoRa DURA 逐阶段非载波监督对照

运行名 `20261122-phase1-lora-dura-observed-detail-prior-m1-r01`，对照为 [同种子观察相位版本](E:/codex/home/worktrees/pamf-repl-weight-diagnosis/standard_ce_baselines_20261010/automation_reports/CV-SincNet/20261122-phase1-lora-dura-wisig-observed-phase-m1-r01/report.md)。在固定的 120 个源 V 包上，保留观察相位 R17 的载波、替换被真实配对 clean 支持的非载波细节，六景源正确数从 341/720 到 545/720。保留接收载波只补真实非载波得到 507/720。只替换载波得到 403/720，直通 395/720。城区低的 R17 载波加真实非载波为 90/120，原 R17 为 42/120。此源反事实用于决定新增损失，实际输入无对应 clean。六景平均保持固定，目标不参与研发。

## 训练差异与源权限

- 种子 2026092702。粗恢复和作用估计沿用同一观察相位父 run、同一合法 L/U 合同的两个冻结 E20/4,940 步。原权重、源日志与冻结元数据完整复制，配对父身份与原诊断一致。R17、R38、R44、R46、R65、R69 共六段各自完整重训，新增 29,640 次更新，链总 39,520 步。
- 令观测 IQ 为 y。用已实现的 LoRa 载波跟踪从 y 取一个单位幅度模板，投影掉每条 chirp 沿载波的均值。以同一个模板计算恢复 IQ 和已观察相位配准的源 clean IQ 的正交残差。每步误差按原 clean 正交残差能量加 0.002 倍的 clean 总功率归一化，再对当前展开输出各步平均，以固定权重 0.10 加到原配对和蒸馏损失。原赛程的 KD 权重和频谱项完整保留。新增项只监督正交残差，原配对项继续监督完整 IQ。投影坐标只来自合法当前接收 IQ。恒等 clean 配对同样保留各步监督。
- 保持原 R17 和 R69 网络、各阶段动力学和求解预算、物理算子形式及识别器。部署版 R69 仍为两阶段、共四次 CG。训练读取合法 L/U IQ。不读取 F、TX 真值、源 V 或目标来训练。源 V 只用于冻结后独立的识别与波形诊断。

## 正式登记与技术停止

普通 N607 账户，原 CUDA 环境和独立发布/输出根。单 worker 优先查 GPU 2、1、4、5、7、6、0、3 的可用显存与容量。预发源 smoke 使用实际同合同冻结 ActionRestorer、六景真实源 IQ 和 R17 的先验反传。技术停止条件为同契约父版身份或预算不符、源 smoke 失败、非有限梯度、正式矩阵缺项；保留其他健康任务。源诊断不改变运行内待测试矩阵。

冻结后同一 Tri-set 2026092702 读出 15 台未见接收机、30,000 个物理包 × clean 加六景 practical。完整输出和旧源门控输出，以及八条已冻结同配对控制。七景预测全固定后本地独立评分；包、接收机、场景和基线语义不变。首轮为单种子验证，最终按三种子既定规则汇总。

发布 `python -m experiments.pamf_lora_main.full_dura_lineage_publish ... --reuse-actions ORIGINAL_SUBMIT --detail-prior-submit OBSERVED_PHASE_SUBMIT`。预期产物包括运行配置、继承父证明、各阶段训练日志及冻结 checkpoint、源波形诊断、1,050 个盲预测文件及独立 `scores.json`。

[源域逐片真实细节反事实](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/dura_wisig_observed_phase_source_panel/r17_detail_oracle.json) · [逐阶段损失实现](E:/codex/home/worktrees/pamf-repl-weight-diagnosis/standard_ce_baselines_20261010/experiments/pamf_lora_main/full_dura_stage_detail.py)

发布 **VERIFIED**。代码提交 `d10646e88c31bf089fea6e6a17df76422ce64bd3` 的归档传输 SHA 比对和远端编译通过。六景实测 source-only smoke 使用既有冻结作用估计器，R17 先验梯度范数 0.001813，冻结父段未产生梯度。独立复核六个新训练阶段的每次展开均收到有限非零的新增损失梯度。GPU 2 的训练控制器、worker、实际 CWD、命令、GPU 和日志增长已核对。R17 完成首轮 247 步，联合损失 0.054873。阶段源读出和正式独立评分均在等待已登记产物。

[发布产物](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/dura_observed_detail_prior_release/submit.json)

## 冻结 R17 源读出与幅度反事实

R17 完成 20 轮、4,940 次实际更新。同一 120 个源 V 包、同一冻结 Tri-set。逐点验证粗恢复和作用估计在 R17 冻结内仍与继承父权重一致。

| 读出 | 观察相位代理作用 | Full 作用监督 | 逐阶段非载波监督 |
|---|---:|---:|---:|
| 固定粗恢复先验，六景 | 353/720 | 351/720 | 353/720 |
| R17 第一步，六景 | 354/720 | 339/720 | 327/720 |
| R17 第二步，六景 | 354/720 | 345/720 | 342/720 |
| R17 第三步，六景 | 341/720 | 355/720 | 336/720 |
| 完整 R17，clean | 68/120 | 65/120 | 68/120 |
| 完整 R17，城区低 | 42/120 | 44/120 | 42/120 |

直通六景、clean 和城区低分别为 395/720、94/120 和 44/120。新增监督降低六景非载波 NMSE 的均值，观察相位对照约 0.2756，逐阶段监督约 0.2408。与源参考对应的归一化非载波输出能量均值从约 0.9176 降为约 0.7308。总体 IQ 能量的城区低比率仍约 0.958，下降主要在非载波输出中。损失改善没有带来判别结果同步改善。

在同一个逐阶段监督 R17 已固定的载波上，逐 chirp 用已知源参考校正学习残差的 RMS 后为 241/720，只校正全窗最优复增益为 240/720。换成真实配对参考的完整非载波残差可达到 542/720。三个反事实都保留 R17 的载波，使用输入时不可见的真实 source clean，仅用于定位误差。它们表明调一个幅度或整体相位不能修复该恢复方向，细节随时间的结构仍然是关键。各实验按原矩阵继续，正式测试不使用这些参考信息。

[逐步识别](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/dura_observed_detail_prior_source_panel/r17_prior_readout.json) · [逐步波形](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/dura_observed_detail_prior_source_panel/r17_detail_wave_compare.json) · [幅度与复增益反事实](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/dura_observed_detail_prior_source_panel/r17_detail_energy_oracle.json)
