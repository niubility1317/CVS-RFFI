# LoRa DURA 观察相位对照

运行名 `20261122-phase1-lora-dura-wisig-observed-phase-m1-r01`。父对照为同一物理数据、同一源动作缓存和同模型种子的 [原坐标八阶段版本](E:/codex/home/worktrees/pamf-repl-weight-diagnosis/standard_ce_baselines_20261010/automation_reports/CV-SincNet/20261122-phase1-lora-dura-wisig-lineage-m1-r02/report.md)。原坐标版本粗恢复的源校验显示准确性严重下降，同时四个 practical 场景的输出能量降至配对 clean 的约一半或以下。单纯整体相位校正对识别的源反事实只有 +1.53 个百分点。这次检查的假设是训练不应主要消耗容量去拟合与实际观测不一致的绝对相位，且保留已验证的原网络结构和阶段损失。

## 唯一训练变量

在合法源 L/U IQ 的每个已知配对上，以内积求单位复数 `p = exp(i·angle(sum(y·conj(x))))`。训练目标使用 `p*x`，输入始终是未加工的同一 `y`。幅值与 chirp 形状保持原量。完全相同的 clean 配对及近零相关情形逐点保留原参考。作用监督同步将主系数乘共轭 `p`、镜像系数乘 `p`，因此算子作用后的无噪观测相同。噪声方差、FIR、采样顺序、优化器、损失系数、教师训练和各阶段 4,940 步不变。全新训练，不加载其他训练态。只读复用 `20261122-phase1-lora-dura-wisig-lineage-m1-r02` 已完成的 LoRa L/U 作用代理缓存。

## 正式对照和运行

- 种子 2026092702，N607 普通账户，程序只用现有 CUDA 环境和独立 release。训练阶段只读取源 IQ，不读 F、TX 真值或目标。
- 八阶段从粗恢复到 R69 共 39,520 步。与原坐标实验使用同一冻结 Tri-set，15 台未见接收机、30,000 个物理包、clean 和六景 practical。原版完整输出与同输出加既有盲门控各一条，加八条冻结对照，共 70 个评分单元。与原坐标版的同读出结果在各自全部预测冻结后匹配比较，报告净纠错、clean 和每景/每 RX。
- 路由可用 GPU 0 至 7，按当时进程和显存状态分散，保留所有已有健康进程。同一时刻仅一个本运行 GPU worker。技术停止规则为源数据、动作来源、冻结权重或固定对照不符，真实源 smoke 失败，训练非有限或预测矩阵缺项。
- 发布入口为 `python -m experiments.pamf_lora_main.full_dura_lineage_publish ... --reuse-actions ORIGINAL_SUBMIT`。固定根下取 `smoke.json`、训练日志、每阶段权重、源 V 波形诊断和完整盲预测。由本地 `full_dura_score` 等待并独立评分。首轮用于单种子验证，最终限差仍需三种子确认。

[已冻结的粗恢复归因](E:/codex/home/worktrees/pamf-repl-weight-diagnosis/standard_ce_baselines_20261010/automation_reports/CV-SincNet/20261122-phase1-lora-dura-wisig-lineage-m1-r02/source-stage-analysis.md)
