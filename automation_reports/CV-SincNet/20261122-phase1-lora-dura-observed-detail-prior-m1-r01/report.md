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
