# LoRa WiSig 八阶段 DURA 正式续发

`20261122-phase1-lora-dura-wisig-lineage-m1-r02` 完全继承 [r01 训练预算、同配对预测矩阵和验收口径](E:/codex/home/worktrees/pamf-repl-weight-diagnosis/standard_ce_baselines_20261010/automation_reports/CV-SincNet/20261122-phase1-lora-dura-wisig-lineage-m1-r01/report.md)。改动仅为在作用代理出口将接收 IQ、FIR、主/镜像系数和方差统一为 `float32`，计算轨迹仍使用原精度。其余数据、种子、训练损失、固定 R69 输出和评分保持原预登记。独立目录保留首发失败证据。

同一 N607 环境使用真实 LoRa 源 IQ 和已验证的 LoRa 冻结 R69 完成冒烟检验。六景作用监督反传得到有限梯度范数 7.1090，只更新作用估计器。源 smoke 不读 F、TX 真值或目标。

启动 **VERIFIED**。提交 `6f4f1770d0c0da0e7e0b6c89bbb68cdd726c9c36`，发布归档 SHA 比对和远端编译通过。正式续发 smoke 梯度范数为 7.8108。已经完整写入 84,000 组源作用代理。核对了控制器和训练进程的 PID、工作目录、命令、GPU 0 登记和日志增长。粗恢复前两轮完成 494 步，平均损失从 0.698818 降到 0.550320，累计处理了 12,593 个 L 窗次和 113,871 个 U 窗次。独立评分已开始等待盲预测矩阵冻结。

[提交与传输记录](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/dura_wisig_lineage_r02_release/submit.json)

[训练与物理差异分析](E:/codex/home/worktrees/pamf-repl-weight-diagnosis/standard_ce_baselines_20261010/automation_reports/CV-SincNet/20261122-phase1-lora-dura-wisig-lineage-m1-r01/diagnosis.md)
