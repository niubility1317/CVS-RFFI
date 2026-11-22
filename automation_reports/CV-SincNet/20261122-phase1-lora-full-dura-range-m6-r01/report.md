# 完整 LoRa DURA 恢复强度消融

本试验在全量目标分数可用前登记。此前三个全量 G 的 406 条剂量留出 U，其配对波形最优系数全部到达现有上限 1。独立源 V 的 800 个物理包子集上，冻结 G 在城区低可恢复 215 包中，去掉公共载频后的对齐残差从约 0.388 降至 0.219，最小二乘最优系数的中位数约为 2.84–2.87。对源场景的观察表明，验证更强的修正有现实依据。解析剂量只在真实已知 clean 的源配对上诊断，运行时不读该信息。

实验冻结复用 `20261122-phase1-lora-full-dura-dose-m3-r01` 的同源 G 和门控，复用 `20261122-phase1-lora-full-adaptive-m12-r01` 的同 seed adaptive_tri F 与其原始盲预测。对允许恢复的包，固定、1.5 倍、2 倍三路以相同冻结残差 `d=G(x)-x` 计算 `x+a*d`，绝对波形 RMS 最大修改依次不超过原输入的 25%、37.5%、50%。对旁路包不改动 IQ，标签精确复用原 F。direct 是同一 F 的原始盲预测。此处无再训练、无标签和仪器通道类别驱动的分档。三个 `a` 在目标预测之前固定，不按目标选择或剔除读出。除系数外不改门控、接收机、L/U、F、G 或信道物理试验，配对为相同 30,000 包 × 七景 × 三种子 × 四读出，共 84 格。

先用源 V 的相同 1000 包固定子集作无 TX 波形独立审计，再等待原 Tri 的整套预测冻结。每个种子同初始信号同时输出直接、固定、1.5 倍、2 倍。正式独立评分核对每种子、模式和 receiver/view 的 2000 预测，之后才读官方真值。重点看单调性、相对固定剂量和 direct 的净纠错、六景各表现和 clean 旁路，不能由 MSE 单项宣布改善。共用的全量对齐输入来自 `20261122-phase1-lora-full-aligned-data-r01`。本地使用三个真实冻结 G 验证，clean 完全等同，2 倍 RMS 确实≤50%，全部 84 计分行的合成配对通过。

命令 `python -m experiments.pamf_lora_main.full_dura_publish --dose-submit FROZEN_G_RECEIPT` 和已有 repo、context、data、tri、ssh-config 参数。输出 `CV-SincNet/runs/20261122-phase1-lora-full-dura-range-m6-r01`，工作目录 `CV-SincNet/releases/20261122-phase1-lora-full-dura-range-m6-r01-release`。真值侧按 `full_dura_score --sweep` 独立评分。N607 普通 CVS-RFFI 环境，GPU0–6，按真实显存容量跨卡，GPU7 排除。冻结血缘不匹配、实际源波形/盲预测异常、非有限值或缺失矩阵停止新派发并保留已有产物。

[源配对剂量分解](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/dura_wave_diagnosis/summary.json)
