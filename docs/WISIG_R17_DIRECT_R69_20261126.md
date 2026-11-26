# WiSig R17 直蒸馏至 R69

实验 `20261126-phase1-wisig-r17-direct-r69-adaptive-m1-r01` 检验能否跳过 R38、R44、R46 和 R65 四次中间训练，直接得到 R69 结构的两阶段学生。固定单恢复器种子 `2026092701`。教师来自 `20261126-phase1-wisig-original-dura-operator-m1-r01/G/r17/final.pt`，E20／4,940，沿用冻结 Adaptive 估计监督的 R07 祖先。教师不更新。训练只读取同合同合法源 L/U 及原 20 轮的 clean/received IQ。

- 直接按原有分组规则切片 R17 的粗恢复、作用头和第 2、3 个先验，写入原版 R69 宽度，再共享两个先验的长卷积权重。初始化只读 R17，未加载 R38 至 R69 的权重。学生 266,951 个参数、两阶段、4 步 CG，联合优化全部学生模块。
- 沿用 R69 的配对 clean IQ NMSE、0.05 倍多尺度谱误差及两阶段蒸馏。学生第 1、2 阶段分别拟合 R17 第 2、3 阶段，蒸馏权重从 1 降到 0.1。AdamW、原学习率与批次固定，训练 20 轮、4,940 步并固定最后权重。相对同一 R17 之后的逐级谱系少 19,760 个优化步。模拟器参数、目标 IQ、TX 标签与 PAMF 不进入训练。
- 完成后对比 Raw、原版 R69、相同 Adaptive R17 谱系的逐级 R69 和本次直蒸馏学生。所有方法匹配默认/加重、Full/Residual、clean 和六种 Practical，Main split 126,000 cross-RX 与 42,000 cross-both，三份冻结 PAMF。已经完成的逐级 R69 冻结预测只在独立评分进程按相同包和基线对齐。学生的测试 IQ 每条走完整 R69 CUDA Graph 推理，全部八片闭合后读真值。报告六景均值、Low Urban、clean、同包净纠错与训练墙钟时间。
- N607 普通账户，训练单卡可用显存至少 12 GiB。实际权重的源 IQ 无 query 检查通过后训练，冻结后最多双 GPU 预测。不可变工作目录为 `releases/<run>-release`，输出为 `runs/<run>`。停止条件为数据/谱系不匹配、非有限值、系统性故障或预测无法合法闭合。

发布入口 `experiments.pamf_unrolled_restore.wisig_r17_direct_distill publish`。预期产物为 `source_smoke.json`、`student/first_update.json`、`student/training.jsonl`、`student/final.pt`、八个冻结预测分片与 `scores.json`。判定基于逐级 R69 同包差异，目标结果不回流训练。
