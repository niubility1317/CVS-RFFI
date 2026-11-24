# LoRa 与 WiSig 可部署星地信道恢复阶段结果

截至 2026-11-24，原冻结基线的两套数据和两条信道路线已补齐。当前候选尚未达到正式验收，目标保持进行中。

## LoRa 三种子基线与验收差距

每个种子沿用相同的 15 台未见接收机、30,000 个物理包和对应冻结 Tri-set 默认头。表中每路线的六景分母为 540,000 次判定，低城区和 clean 各为 90,000 次。Residual 的两个缺失种子由同一个原冻结 residual R69 补齐，原 seed02 决策复用。两个路线的历史恢复权重沿用各自登记的身份。 [Full 三种子原分析](../../automation_reports/CV-SincNet/20261123-phase1-lora-dura-complex-response-confirm-m4-r01/analysis.md) · [Residual 原 R69 补测](../../automation_reports/CV-SincNet/20261123-phase1-lora-residual-frozen-r69-replay-m2-r01/report.md)

| 路线与方法 | 六景 | 低强度城区 | Clean 正确数 |
| --- | ---: | ---: | ---: |
| Full 直通 | 55.1180% | 40.6667% | 64,490 |
| Full 既有 DURA 半量 | 55.8517% | 41.7733% | 64,490 |
| Full 复响应半量，现有最好 | 56.0024% | 42.0500% | 64,490 |
| Full 正式验收 | 67.0000% | 53.0000% | 至少 64,490 |
| Residual 直通 | 52.9263% | 41.3500% | 64,490 |
| Residual 原 R69 半量 | 53.5485% | 42.8511% | 64,307 |

Full 当前最好版本距六景和低城区验收分别差 10.9976 和 10.9500 个百分点。Residual 原 R69 半量较直通在六景净多 3,360 次正确，按物理包和接收机计算的 95% 配对区间为 +2,693 至 +4,008 次，clean 净少 183 次。两条路线均待新方法用同一冻结权重、门限与推理规则完成正式配对验证。

LoRa 实收复谱方向修正的两臂已各完成 40,000 步。双方向终步在 Full、Residual 六景比匹配 DURA 净减 66、198 次，clean 均为 667/900。同权等量相较标量净减 11、16 次，两个配对区间均跨零。低城区五台源接收机的相位回写中位数均为零。候选停在源域。 [LoRa 实收复谱修正设计](LORA_RECEIVED_SPECTRAL_UPDATE_DESIGN_20261124.md) · [冻结结果、配对统计与幅相审计](../../automation_reports/CV-SincNet/20261124-phase1-lora-received-update-source-m4-r01/report.md)

## WiSig Main split 原冻结 R69

三种子沿用原 Tri-SIG-TIME SSL 的 E50 学生及默认头。每个种子 `cross_rx` 126,000 物理包，`cross_both` 42,000 物理包，两者互斥且并集为 168,000。各路线的 6 个信道场景均独立生成预测。两路线的三种子共 6 行均完成固定预测后，独立评分。聚合结果报告三模型种子的均值。Warmup、计算和逐景读出见原 run，逐种子、RX、场景和纠错分项见本机证据 `E:/type10-7/local_artifacts/wisig_r69_paired_main_20261123/verified_paired_readback.json`。

| 路线 | 划分 | 直通六景 | R69 六景 | 直通 clean | R69 clean |
| --- | --- | ---: | ---: | ---: | ---: |
| Full | cross_rx | 71.1670% | 79.9027% | 333,748 | 333,884 |
| Full | cross_both | 68.8069% | 76.8267% | 104,323 | 104,402 |
| Full | 合并 | 70.5769% | 79.1337% | 438,071 | 438,286 |
| Residual | cross_rx | 71.1649% | 79.9047% | 333,748 | 333,884 |
| Residual | cross_both | 68.8083% | 76.8291% | 104,323 | 104,402 |
| Residual | 合并 | 70.5758% | 79.1358% | 438,071 | 438,286 |

R69 v1.0 有 266,951 个参数。历史两阶段学生由 R65 固定父版初始化，在合法源 IQ 上以配对 clean 波形、中间迭代蒸馏及谱损失联合优化 E20/4,940 步。上游作用和新增噪声方差训练使用模拟器生成监督，报告中与本轮训练权限明确对应。推理只输入当前 256 点实收 IQ，内部估计信道作用与方差。已登记的部署读出使用 FP32、TF32 关闭及 Graph-RR。 [原 DURA v1.0 定义](../DURA_V1.md)

## 源域验证

- LoRa 将同一条 chirp 的对称邻点构造成独立相位方向。匹配的两组各训练 40,000 步，邻点组降低了固定源 U 相位和复谱损失，但 Full、Residual 六景较半量 DURA 净减 99［−147，−52］、216［−288，−145］次。两路 clean 均为 667/900。同权等 IQ RMS 的邻点开关净差为 −29、−30 次，区间均跨零。低城区新增几何方向在隔离对照中对应 9.55% 和 1.83% 的相位能量，两路弱独有复谱 NMSE 都升高。候选未晋级目标矩阵。 [LoRa 邻点相位与逐接收机冻结识别](../../automation_reports/CV-SincNet/20261124-phase1-lora-neighbor-phase-source-m4-r01/report.md)

- LoRa 幅相联合对照固定 40,000 步局部相位。实收分组前导幅度与相位合用，Full 和 Residual 相对仅修相位净减 15、32 次，区间均跨零。低城区两路的外带估计功率与精确功率之比中位为 0.993、1.011，但外带去偏幅度的编辑方向与隔离参考反向；分组幅度方向余弦为 0.887、0.876。共同剂量下精确幅度仅净增 4、13 次，区间均跨零。全量精确幅度相对仅修相位净增 59、55 次，Residual 仍比 DURA 少 28 次。两路 clean 均为 667/900。 [幅相联合测试及实收误差](../../automation_reports/CV-SincNet/20261124-diagnostic-lora-phase-radius-source-m2-r01/report.md)

- LoRa 相位位置分配的四个模型各完成 40,000 步。基础通用扰动下，局部与全包读出逐包正确性一致。加入局部和跨 chirp 相关突发后，Residual 的局部全量相对匹配全包净增 136 次，95% 区间 +91 至 +178。同权、同 IQ RMS 仍净增 100 次，区间 +61 至 +138，五台源 RX 均改善。固定噪声代理使 Residual 净减 134 次。主读出的 Full 和 Residual 六景为 3,087、2,877/5,400，分别比 DURA 少 7、83；两路 clean 都为 667/900。该机制获得源域独立收益，候选仍未达到正式晋级条件。 [基础扰动测试](../../automation_reports/CV-SincNet/20261124-phase1-lora-phase-dose-source-m4-r01/report.md) · [突发扰动和匹配结果](../../automation_reports/CV-SincNet/20261124-phase1-lora-phase-dose-bursts-source-m4-r01/report.md)

- LoRa 时域相位对照复用已有 40,000 步权重。实收重复相位在 Full 相对直通净增 65 次，95% 区间 +26 至 +103，较 DURA 少 1 次。Residual 较 DURA 净减 157 次。关闭相位裕量使低城区两路相位非零系数中位份额从 0% 到 66.7%，六景结果无稳定改善。隔离 10% RMS 精确相位相对 DURA 两路净增 152、201 次；在实收两臂的共同预算下，隔离方向相对等量学习仍净增 30、37 次。所有读出 clean 为 667/900。 [冻结相位对照及配对结果](../../automation_reports/CV-SincNet/20261124-diagnostic-lora-phase-observation-source-m2-r01/report.md)

- WiSig 实收观测求解两臂各完成 40,000 步。两路线耦合带门分别为 4,736、4,735/10,800，较原 R69 净减 5,458、5,460，两路 clean 均为 1,775/1,800。同权换成单位作用挽回 2,544、2,546 次，换成逐点块只挽回 119、118 次。估计作用可降低正向残差，但该改善未转化成冻结识别收益。两臂均未晋级。 [WiSig 两阶段求解测试、配对归因及波形](../../automation_reports/CV-SincNet/20261124-phase1-wisig-observed-solve-source-m4-r01/report.md)

- LoRa 局部 Wiener 对照覆盖同一 900 个源 V 包与原 Tri-set 单种子。完整复协方差全量在 full 相对成熟 DURA 净增 42 次，95% 配对区间为 −3 至 +92，在 residual 净减 249 次，区间约为 −324 至 −176。两路线 clean 均保持 667/900。匹配半量消融中，复协方差相对对角读出分别净减 24 次和净增 31 次。同包隔离归因显示，换成真实总噪声功率后，原 Wiener 六景在 full 和 residual 分别再少判对 3 和 138 次。保载波精确扣噪则分别多判对 423 和 871 次，五台源接收机均受益。Residual 上更准确的功率未能解决复谱恢复造成的识别损失。该读出未进入正式目标矩阵。 [LoRa 局部 Wiener 对照](../../automation_reports/CV-SincNet/20261123-phase1-lora-local-wiener-source-m2-r01/report.md) · [LoRa 隔离噪声归因](../../automation_reports/CV-SincNet/20261123-diagnostic-lora-noise-floor-source-m2-r01/report.md)
- LoRa 分块复谱实验中，匹配遮挡组的粗恢复在 full 和 residual 比全谱组净多判对 93 和 257 次，五台源接收机均正向。遮挡组全量细化相对同组粗恢复则净少 13 和 128 次。两条路线最好的联合半量比成熟 DURA 分别少 49 和 126 次，clean 均为 667/900。两组各 34,693 参数、1,600 步。遮挡先验的独立收益在源域消融成立，两组均未进入正式矩阵。 [LoRa 分块复谱消融](../../automation_reports/CV-SincNet/20261123-phase1-lora-spectral-block-source-m4-r01/report.md)
- LoRa 同范数方向消融中，保护实收共享复谱在 full 全量读出净增 34 次，在 residual 半量读出净减 105 次。保护半量在 full 比直通多 28 次，95% 配对区间为 +5 至 +52，仍比成熟 DURA 少 38 次。Clean 均为 667/900。相同约束的源域收益未跨路线复现。 [LoRa 同量复数修正](../../automation_reports/CV-SincNet/20261123-phase1-lora-repeat-protection-source-m4-r01/report.md)
- LoRa 的源 U 经验复谱库中，跨半组一致性在 residual 相对交叉查询半量净增 38 次，区间 +19 至 +57。该读出仍比成熟 DURA 少 86 次。对四个已完成试验的 19 个固定恢复读出加直通做标签后验覆盖，full 和 residual 在源 V 六景分别为 69.70% 与 74.22%，超过配对成熟对照 670 与 1,048 次。其中经验库独占 85 与 109 次。覆盖用于判断互补性，尚无可部署规则取得所列覆盖。 [LoRa 经验模板与覆盖归因](../../automation_reports/CV-SincNet/20261124-phase1-lora-empirical-chirp-source-m2-r01/report.md)
- LoRa 将前导分成三组，轮换留出一组检验复数修正方向。保留方向和只保留幅值的两臂在所有激活包上都选择本组细化。六景匹配硬选净差为 −2 和 +3，95% 配对区间均跨零。各路线最强固定提议分别少于成熟半量 DURA 79 和 315 次，clean 均为 667/900。方向臂有 36,233 参数，训练 57.99 秒，单包旁路和激活中位数为 1.75 ms、16.55 ms。该实验未晋级。 [三组交叉留出与逐接收机评分](../../automation_reports/CV-SincNet/20261124-phase1-lora-crossfit-choice-source-m4-r01/report.md)
- LoRa 的局部相位中段、两端两臂各完成 40,000 步与 20,000 个源 U 包的拟合预算。两端半量的 full 六景 3,057/5,400，中段粗恢复的 residual 六景 2,872/5,400，分别低于同包 DURA 37 次和 88 次。所有候选保持 clean 667/900。两端全量推理保留外带置信信息，相对遮蔽该量净增 27 和 110 次，95% 区间分别约 +2 至 +53、+63 至 +159。强受扰时相位读出的复波形误差下降，识别正确数仍有退化。无候选晋级。 [40,000 步局部相位恢复](../../automation_reports/CV-SincNet/20261124-phase1-lora-local-phase-source-m4-r01/report.md)
- LoRa 联合幅相长训的同架构、同数据、同预算消融中，加入源 U 跨 chirp 弱复谱关系损失后，全量读出两路线分别多判对 72 和 201 次，95% 配对区间约 +39 至 +104、+135 至 +263，两路线五台源接收机方向一致。效果最好的关系组半量仍比成熟 DURA 少判对 112 和 404 次，clean 都为 667/900。Residual 低城区的 9–32 kHz 复误差明显下降，同包设备识别仍下降。没有读出进入正式评测。 [40,000 步联合幅相消融与投影归因](../../automation_reports/CV-SincNet/20261124-phase1-lora-complex-geometry-source-m4-r01/report.md)
- LoRa 双孔径相对同步相对旧独立估计在 full 六景净增 15 次，区间 +2 至 +28，residual 净增 2 次且区间跨零。换用自适应坐标的同方差 Wiener 仍比成熟 DURA 少判对 24 和 117 次，两路 clean 均为 667/900。扣噪参考表明，residual 低城区包内实际有效频偏离差中位数为 147.19 Hz，旧估计对参考的误差中位数为 8.21 Hz，自适应估计为 18.53 Hz。此固定工程对照保留原独立估计。 [双孔径评测与隔离误差](../../automation_reports/CV-SincNet/20261124-phase1-lora-aperture-tracking-source-m2-r01/report.md)
- LoRa 固定 5,000 个无标签模板，两臂各从零训练 40,000 步并排除所有建库包。保护局部真实弱复谱方向后，粗恢复相对同剂量标量组在 residual 多判对 109 次，区间 +40 至 +177，full 多判对 36 次且区间跨零。Residual 的增益来自郊区，三个城区均小幅下降，保护组后续细化也少判对 98 次。两路线最佳固定读出各少于成熟 DURA 47、131 次，clean 仍为 667/900。两路均未晋级。 [经验源方向保护评测](../../automation_reports/CV-SincNet/20261124-phase1-lora-empirical-tangent-source-m4-r01/report.md)
- LoRa 两组各训练 40,000 步。将共同复谱与逐 chirp 剩余复谱分开衰减后，全量相对同门同编辑量的标量消融在 full 和 residual 分别净增 27 和 70 次，95% 配对区间为 +9 至 +45 和 +44 至 +97，五台接收机均正向，clean 均为 667/900。最佳分开半量仍比匹配 DURA 少 38 和 192 次。两臂粗细输出在正确性上没有变化，低城区外带衰减达到观测限制。候选停在源域。 [LoRa 分开复谱恢复设计与测试](../../automation_reports/CV-SincNet/20261124-phase1-lora-split-shrink-source-m4-r01/report.md)
- LoRa 短时留一 chirp 复谱恢复两臂各训练 40,000 步。时间平均粗恢复的 full 六景 3,087/5,400，较 DURA 净减 7 次，低城区净增 13 次。局部粗恢复的 residual 六景 2,857/5,400，较 DURA 净减 103 次。两路线所有读出 clean 均为 667/900。同权局部噪声估计使 residual 粗恢复在低城区净增 13 次，95% 区间 +5 至 +21。局部全量细化相对粗恢复在两路线六景净减 41 和 134 次。低城区局部噪声功率代理误差降低，先验预测失配仍高于所估方差，候选停在源域。 [LoRa 短时不确定性设计与同包结果](../../automation_reports/CV-SincNet/20261124-phase1-lora-lapped-posterior-source-m4-r01/report.md)
- LoRa 实收前导交叉预测风险两臂各训练 40,000 步。加入观测风险后，低城区先验误差与估计方差的比值在 full 从 4.27 降至 2.19、在 residual 从 10.18 降至 3.95。同权同迹方向粗恢复较标量在六景分别净减 7 和净增 7 次，区间均跨零。标量粗恢复在 full 低城区比 DURA 净增 22 次，95% 区间 +4 至 +41，但六景净减 11 次。Residual 最佳读出为标量半量，较 DURA 净减 108 次。Clean 均为 667/900，试验停在源域。 [LoRa 交叉预测风险与方向消融](../../automation_reports/CV-SincNet/20261124-phase1-lora-held-chirp-precision-source-m4-r01/report.md)
- LoRa 无标签全包对关系约束的两臂各训练 40,000 步。两次独立通用扰动下，固定源 U 留出的细化距离误差从 2.526 降至 2.315。源 V 低城区的全局距离误差也有小幅改善，但五邻域重合与同物理包检索在两路均下降，9–32 kHz 相对扣噪参考的幅相更差。全量相对同预算波形组的六景净差为 −19 和 −106 次。最佳关系半量相对 DURA 净差为 −41 和 −128 次，clean 均为 667/900。未晋级。 [LoRa 包间关系消融与近邻复核](../../automation_reports/CV-SincNet/20261124-phase1-lora-packet-relation-source-m4-r01/report.md)
- LoRa 受控复数相消两臂各完成 40,000 步。按三对实收 chirp 计算留出剂量后，全量在 residual 较固定剂量净增 68 次，95% 同包区间 +27 至 +108，五台源 RX 均正向。Full 对应净增 1 次。同权重关闭相消只改变六景净正确数 1 次和 0 次。验证组主读出仍比匹配 DURA 少 64 和 124 次，两路 clean 均为 667/900，候选停在源域。 [LoRa 受控相消设计、匹配长训及同包复核](../../automation_reports/CV-SincNet/20261124-phase1-lora-reference-cancel-source-m4-r01/report.md)
- LoRa 零优化步方差匹配使用源 U 分别估计公共和独有复谱协方差。相对物理 RMS 等修改量 Wiener，在 full 和 residual 分别净增 55 和 106 次，95% 同包区间 +34 至 +77 和 +74 至 +138，两路线五台源 RX 均改善。公共分量的作用明确，额外处理独有分量未增加六景正确数。方差匹配仍比 DURA 少 81 和 438 次，两路 clean 均为 667/900，工程读出停在源域。 [LoRa 方差匹配与等修改量复核](../../automation_reports/CV-SincNet/20261124-phase1-lora-variance-transport-source-m2-r01/report.md)
- LoRa 公共复谱区间两臂均完成 40,000 步。固定权重和原 IQ 物理修改量后，方向保护在 full 和 residual 只净增 2 和 0 次。主读出较匹配 DURA 净减 60 和 102 次，clean 均保持 667/900。低城区 9–32 kHz 受实收两组支持的频点比例在 full 为 25.66%，residual 为 0.74%，低覆盖限制了保护收益。 [LoRa 相干区间长训及逐频带归因](../../automation_reports/CV-SincNet/20261124-phase1-lora-coherent-interval-source-m4-r01/report.md)
- LoRa 隔离扣噪中，保载波联合扣除公共和独有复谱，两路线六景较 DURA 分别净增 465 和 622 次，95% 同包区间为 +399 至 +532、+539 至 +714。Residual 单独扣公共、独有项分别比直通少 243、464 次。联合处理在两个单项之外带来 +239、+1,441 次识别交互，五台源 RX 方向一致。保实收幅度、只换扣噪相位也分别较 DURA 净增 211、321 次。所有读出 clean 均为 667/900。这些是隔离归因结果。下一版 LoRa 设计比较同预算的复谱独立更新和联合更新。 [LoRa 公共、独有和幅相隔离对照](../../automation_reports/CV-SincNet/20261124-diagnostic-lora-noise-split-source-m2-r01/report.md)
- LoRa 复谱独立与联合更新两臂各完成 40,000 步。同权同物理 RMS 切换到联合上下文后，Full 和 Residual 六景分别净增 103、175 次，95% 同包区间为 +66 至 +137、+121 至 +228。独立重训匹配比较在 Full 净增 27 次，Residual 净减 25 次且区间跨零。联合半量比 DURA 少 84、442 次，clean 均保持 667/900。Residual 低城区联合上下文提高公共复相关，独有复相关仍低于直通，候选未晋级。 [LoRa 联合复谱长训、同量消融与幅相复核](../../automation_reports/CV-SincNet/20261124-phase1-lora-joint-spectrum-source-m4-r01/report.md)
- LoRa 拆分实际冻结模型的修正后，只用独有分量相对原联合全量净增 279、593 次，五台源 RX 均正向。只用独有分量相对 DURA 仍少 5、515 次。固定公共修正及独有物理 RMS，用相位导出更新替换幅度导出更新后，Residual 六景净增 171 次，低城区净减 17 次。该实收隔离诊断促成后续从零训练的公共谱锚定求解。 [LoRa 公共、独有和等量幅相归因](../../automation_reports/CV-SincNet/20261124-diagnostic-lora-learned-split-source-m2-r01/report.md)
- LoRa 实收公共锚定与无约束两臂各 40,000 步。全量锚定相对同预算无约束读出在 Full、Residual 净增 241 和 627 次，两路五台源 RX 均改善。同权同 RMS 净增 76、39 次，Residual 区间跨零。最强固定锚定读出在两路仍比 DURA 少 26、99 次。Clean 均为 667/900，未晋级。 [LoRa 公共谱锚定及观测方差复核](../../automation_reports/CV-SincNet/20261124-phase1-lora-common-anchor-source-m4-r01/report.md)
- WiSig 的直接前导均衡和绝对 CFO 校正使 clean 与受损识别均下降。两路短训练重复残差可反映观测失配，估计方差含相位变化和硬件失配。 [接收量及工程对照](../../automation_reports/CV-SincNet/20261123-phase1-wisig-stf-engineering-source-s2026092701-r02/report.md)
- 与历史 F50 配对时，接收端时标检测和协议后缀外推在源 V 把两路线六景提高到 7,121/10,800，同时保持 1,773/1,800 clean。换成正式对比所用的 Tri-SIG-TIME 单种子后，同一方法的两路线六景分别净减 66 和 67 次正确。拟合一个短长前导共享复响应也下降，其中标量法净减 176 和 172，三抽头法净减 204 和 206。Tri-SIG-TIME 的 R69 同包基线为 10,194 和 10,195/10,800。两条路线的 clean 正确数均为 1,775/1,800，时标方法未晋级。 [F50 时标对照](../../automation_reports/CV-SincNet/20261123-phase1-wisig-timing-engineering-source-s2026092701-r01/report.md) · [Tri-SIG-TIME 联合前导对照](../../automation_reports/CV-SincNet/20261123-phase1-wisig-joint-preamble-source-s2026092701-r01/report.md)
- 合法源 U 训练使用通用波形扰动，粗恢复、观测约束和细化联合训练。与匹配波形损失组相比，弱复谱约束在冻结 Tri-SIG-TIME 单种子的无门联合源读出中，两路线六景分别多判对 1,524 和 1,526 次，全部场景和源接收机方向一致。复评沿用原 227,268 参数、1,200 步权重。该读出仍低于直通与 R69，固定门在两条路线各净少 2 次六景正确与 1 次 clean 正确。此前把接收端时标放进训练和推理可改善历史 F50 的匹配训练对照，联合细化仍低于只做时标校正。 [Tri-SIG-TIME 弱复谱权重复评](../../automation_reports/CV-SincNet/20261123-phase1-wisig-tri-learned-replay-s2026092701-r01/report.md) · [F50 弱复谱消融](../../automation_reports/CV-SincNet/20261123-phase1-wisig-uncertain-prior-source-s2026092701-m2-r01/report.md) · [时标联合训练消融](../../automation_reports/CV-SincNet/20261123-phase1-wisig-aligned-recovery-source-s2026092701-m2-r01/report.md)
- WiSig 使用实收短前导重复信息的频块遮挡粗恢复相对匹配全谱臂在 full 与 residual 均多判对 4,479 次。两路五台源接收机方向一致，绝对识别仍低于直通。遮挡臂继续细化各少判对约 5,600 次，只改单一幅度或相位可部分挽回。固定门在两路六景各只触发 1/10,800 次，逐景正确数与直通一致。此结果与 LoRa 一起定位了粗预测与联合细化的收益差异，尚无候选通过跨路线要求。 [WiSig 重复前导频块消融](../../automation_reports/CV-SincNet/20261123-phase1-wisig-repeat-block-source-m3-r01/report.md)
- WiSig 的另一项同包隔离诊断使用相同冻结 Tri-SIG-TIME。实收幅度与配对 clean 相位结合后，两路线六景是 87.45% 和 87.47%，仍低于同次 R69 的 94.39% 和 94.40%。仅替换 clean 幅度分别是 72.81% 和 72.89%。两者都取配对 clean 时为 98.61%。这些读出用于归因，配对 clean 未用于训练、冻结或候选输入。 [WiSig 幅相因子报告](https://github.com/niubility1317/CVS-RFFI/blob/4538bbb21ed6e22bad9ca2c6599f6d0ecee879be/automation_reports/CV-SincNet/20261123-phase1-wisig-phase-magnitude-source-m4-r01/report.md)
- WiSig 的匹配训练中，在优化循环内保护实收幅度使两路线的门控读出各比原拟合多判对 2,005 次，仍分别比直通少 184 和 185 次。直通和多个归档恢复读出的事后同包覆盖达到 98.20% 和 98.24%。这个覆盖使用了源 V 标签，显示各恢复读出的互补性，尚无接收端选择规则达到该表现。 [幅度保护及候选覆盖](https://github.com/niubility1317/CVS-RFFI/blob/5f56af5f65eb158ee7aa5c28ec61918a487c9532/automation_reports/CV-SincNet/20261123-phase1-wisig-observed-amplitude-source-m3-r01/report.md)
- WiSig 波形风险动态选择在两路各激活约 1,280 次，超过 99% 的激活读出为同一细化提议。与静态选择相比，动态规则的六景净差为 +2 和 +1，配对区间均跨零。隔离诊断在已知配对 clean 时按同一波形风险选提议，六景只有 65.64% 和 65.57%。该网络的六个提议即使用标签统计覆盖也低于 R69，表明这组读出的互补性和波形风险排序均有限。 [波形排序与隔离诊断](https://github.com/niubility1317/CVS-RFFI/blob/90d9b29dca07a0f449470b0f4c9d7e7ad19d53a8/automation_reports/CV-SincNet/20261124-phase1-wisig-waveform-choice-source-m3-r01/report.md)
- WiSig 加入实测源包的频谱关系约束后，旁路读出相对同架构对照在两路线各净多判对 78 次，95% 同物理包区间分别为 +34 至 +122、+35 至 +122。相对 R69 仍净少 1,243 和 1,240 次，clean 也各低 2 次。候选未进入正式 Main split。 [频谱关系消融](https://github.com/niubility1317/CVS-RFFI/blob/1b807962d06453b95166888de9dbe00f31e8719f/automation_reports/CV-SincNet/20261124-phase1-wisig-instance-geometry-source-m3-r01/report.md)
- WiSig 可见 STF/LTF 波形证据两臂各训练 40,000 步。前导组仅时标相对只用重复证据，在 full 和 residual 六景分别净增 322 和 326 次，95% 配对区间 +199 至 +437 和 +205 至 +444，两路五台 RX 都改善。低城区与隔离配对时标误差 ≤2 点的包从 803 增至 960。前导组联合相对仅时标分别净减 2,031 和 2,032 次。带门读出保持 clean 1,775/1,800，六景较 R69 净减 1,490 和 1,485 次。未晋级正式 Main split。 [WiSig 联合截窗恢复、时标与幅相归因](../../automation_reports/CV-SincNet/20261124-phase1-wisig-anchored-window-source-m4-r01/report.md)
- 配对 clean 只进隔离源诊断。用当前 Main split 冻结识别器和两路 1,800 个源 V 包，换 R69 重叠区相对只做配对时标使六景净增 1,001 和 997 次，只换 R69 边界净增 374 和 371 次。固定 R69 边界后仅换重叠区相位净增 543 和 540 次，仅换幅度净增 150 和 139 次，幅相合用另有 +175 和 +186 次交互增益。两路原始决策与同识别器归档逐样本一致。隔离读出不进入部署候选；旧版 F50 的配对诊断独立保留。 [当前 Main split 边界与重叠区](../../automation_reports/CV-SincNet/20261124-diagnostic-wisig-main-overlap-source-m2-r01/report.md) · [历史 F50 配对隔离归因](../../automation_reports/CV-SincNet/20261123-diagnostic-wisig-source-paired-time-edge-s2026092701-r01/report.md)
- WiSig 短训练非线性漂移两臂各训练 40,000 步。固定权重和时标后，显式工程补偿在两路线六景净增 7、9 次，95% 同包区间均跨零。预登记带门读出两路 clean 同为 1,775/1,800，六景只各触发 9/10,800 次，较 R69 净减 1,140 和 1,136 次，未晋级。低城区的重叠区复 NMSE 由仅时标的 1.4173、1.4085 变为显式补偿后的 1.4168、1.4070，R69 为 0.0194、0.0193。 [WiSig 幅相联合长训及同包复核](../../automation_reports/CV-SincNet/20261124-phase1-wisig-repeat-drift-source-m4-r01/report.md)
- WiSig 原 R69 的两次观测一致性求解分别使两路线的六景相对各自先验净增 3,665/3,673 和 2,056/2,054 次。两次求解后的 clean 均为 1,775/1,800。完整 R69 比第一次求解多判对 688、693 次，五台接收机均改善。低城区的重叠复误差从第一次求解的 0.0356、0.0356 降至 0.0194、0.0193，实际 R69 终点与原 Graph-RR 逐样点相同。读出只用于归因，未训练新权重。 [WiSig 两阶段独立归因](../../automation_reports/CV-SincNet/20261124-diagnostic-wisig-r69-stages-source-m2-r01/report.md)

## 相关工作与本轮验证口径

| 原始工作 | 算法、输入与监督 | 与本轮的实质差异 |
| --- | --- | --- |
| Tapparel 与 Burg，LoRa 两遍同步 | 使用当前捕获前导联合估计时偏、频偏和采样频偏，以通信接收为目标。GNU Radio 实现包含上、下扫频与 SFO 前导补偿。 | 本项目的 LoRa 发布方片段保存 8 个上扫频窗，现有主比较冻结读出取其中 6 个。上扫频的单一去扫峰会混合频偏和时偏。恢复输出还要保留冻结识别依赖的细节，因而把估计可观测性和指纹损伤分别验证。 [原论文](https://arxiv.org/abs/2502.08485) · [作者实现](https://github.com/tapparelj/gr-lora_sdr/blob/master/lib/frame_sync_impl.cc) |
| Hanna 等，WiSig 发布方同步均衡 | 从更长原始记录检测前导，估计粗细 CFO，利用完整 L-LTF 估计响应和噪声，执行 MMSE 均衡后加回 CFO。 | 主对比 `equalized=1` 的 256 点窗口覆盖完整短训练字段和不完整长训练字段。追加传播后的可用估计来自短训练重复和可见长训练前缀。已实测短窗直接均衡损失，接收端时标更正和保留原幅相带来源域正收益。 [WiSig 原论文](https://arxiv.org/html/2112.15363) · [作者均衡实现](https://github.com/WiSig-dataset/wisig-process-raw/blob/master/matlab_equalization/equalize_channel.m) |
| Edfors 等，OFDM 低秩 LMMSE | 用频域相关结构和信噪信息估计 OFDM 信道，以信道估计误差和符号判决评价。 | 目前前导估计采用正则化的短窗复响应。采集时的发射机、接收机、此前均衡及追加传播共同影响该响应。工程负结果说明符号恢复有效性与冻结设备识别收益需要分别验证。 [原论文](https://doi.org/10.1109/VETEC.1996.501446) |
| Shen 等，LoRa 信道独立特征及跨接收机识别 | 信道版本使用相邻谱图列的比值、信道增强和深度度量识别。接收机版本使用对抗训练，从多个接收机的捕获波形学习接收机无关特征。 | 本轮恢复器返回复 IQ，训练只见合法源波形和独立通用扰动。身份读出使用已冻结识别器。公开 LoRa 审计的指标用于理解特征和协议，不跨信道设置移用其准确率。 [信道鲁棒原论文](https://arxiv.org/pdf/2107.02867) · [多接收机原论文](https://arxiv.org/abs/2207.02999) · [作者代码入口](https://github.com/gxhen/LoRa_RFFI) |
| Yu 等，PSCDAE 射频指纹降噪 | 同步优化 ZigBee 重建和有标签分类，以高 SNR 波形为重建目标，验证 AWGN 下识别。 | 本轮没有训练用身份监督，恢复先独立冻结，随后接两套既有分类器对照。同一复波形损失改善仍可能改错，单独记录同包净纠错和 clean。 [原论文](https://arxiv.org/pdf/1907.08809) |
| Yin 等，扩散射频指纹降噪 | 以逐点合成高斯噪声监督训练噪声预测器，把接收 SNR 映射到反向扩散步，并以生成的降噪信号训练分类器。验证使用 6 台 Wi-Fi 设备和一台 N210。 | 本轮用实际接收估计控制单次恢复，训练目标是实测源波形的保真与配对差异，不向候选提供逐点噪声或模拟信道监督。多接收机、两 practical 路线与同一冻结识别器构成独立验收条件。 [作者论文](https://livrepository.liverpool.ac.uk/3190640/1/WCNC2025_RFFI_WiFi.pdf) |
| Lehtinen 等，Noise2Noise | 用独立受扰的成对观测训练同潜在信号恢复。 | 本轮借鉴同物理包跨扰动的一致性，并对共享扰动下的真实源 IQ 弱复谱差异增加传递约束。同包短训练残差同时包含失配，方差读出按有效残差解释。实验只支持当前匹配消融的源域结论。 [原论文](https://proceedings.mlr.press/v80/lehtinen18a.html) |

接收端时标校正的源域收益依赖所用冻结识别器。弱复谱约束的正向匹配消融已在历史 F50 和正式所用的 Tri-SIG-TIME 单种子上复现。两套数据的新方法正式准确率及跨设置通用性仍待验证。
