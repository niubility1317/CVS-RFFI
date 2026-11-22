# LoRa DURA 同源 IQ 教师与分阶段保护训练

此前 LoRa 同一 G 的保护投影在真实目标上净纠错提高。源波形审计显示扩大单一残差幅度会损伤识别。该实验按三个种子依次学习 LoRa 专属三阶段 IQ 教师与双阶段学生。教师从 R65 骨架冷启，学生继承同源教师骨架的粗恢复、作用估计及最后两阶段先验。学生蒸馏对齐教师后两阶段按部署投影对应的 IQ。两者都保留共享非载波结构，均用配准复 IQ、0.05 多尺度谱误差、独立 clean 恒等项。学生的阶段误差权重按 1.0 至 0.1 衰减。所有权重来自本 LoRa 源合同。

各 seed 教师 2400 步与学生 2400 步，每步 24 包。每步 8 条原有合格 practical 配对、12 条第二次独立信道 realization、4 条 clean 恒等配对，L 与盲 U 等量。原保留 U 在两个 draws 与 clean 路径均排除。训练旁路样本仅作为训练使用 0.60 强度，部署沿用原门控和限幅及 shape 保护。训练不读 F、TX、目标以及 V。记录更新、配对误差、谱误差、恒等误差和教师阶段误差，实验评价以目标准确率与净纠错为准。该行合并了配对多样性、阶段蒸馏、谱损失与加长训练，后续用分解实验判断各项贡献。

全部三种子接新版冻结 Tri，在相同 30000 个目标包、15 未见 RX、clean 和六景 practical 上评估。原样复用联合对照的四路盲预测，另加 staged_protect，105 个单元固定后独立读真值。N607 普通账户、GPU0–7，最多四个本任务 worker。发布 `python -m experiments.pamf_lora_main.full_dura_publish --draw-submit ... --stack-submit ...`，传入原数据、额外源 IQ、原 G、保护重训 G、新版 Tri 和联合对照 handle。工作目录 `CV-SincNet/releases/20261122-phase1-lora-dura-source-distillation-m6-r01-release`，唯一输出 `CV-SincNet/runs/20261122-phase1-lora-dura-source-distillation-m6-r01`。真实源 smoke 后自动等齐第二次源配对，依次训练教师与学生，冻结后执行正式测试。来源、步数、非有限、原门控或盲预测异常为技术停止条件。
