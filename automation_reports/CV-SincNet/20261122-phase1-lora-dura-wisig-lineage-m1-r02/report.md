# LoRa WiSig 八阶段 DURA 正式续发

`20261122-phase1-lora-dura-wisig-lineage-m1-r02` 完全继承 [r01 训练预算、同配对预测矩阵和验收口径](E:/codex/home/worktrees/pamf-repl-weight-diagnosis/standard_ce_baselines_20261010/automation_reports/CV-SincNet/20261122-phase1-lora-dura-wisig-lineage-m1-r01/report.md)。改动仅为在作用代理出口将接收 IQ、FIR、主/镜像系数和方差统一为 `float32`，计算轨迹仍使用原精度。其余数据、种子、训练损失、固定 R69 输出和评分保持原预登记。独立目录保留首发失败证据。

同一 N607 环境使用真实 LoRa 源 IQ 和已验证的 LoRa 冻结 R69 完成冒烟检验。六景作用监督反传得到有限梯度范数 7.1090，只更新作用估计器。源 smoke 不读 F、TX 真值或目标。

[训练与物理差异分析](E:/codex/home/worktrees/pamf-repl-weight-diagnosis/standard_ce_baselines_20261010/automation_reports/CV-SincNet/20261122-phase1-lora-dura-wisig-lineage-m1-r01/diagnosis.md)
