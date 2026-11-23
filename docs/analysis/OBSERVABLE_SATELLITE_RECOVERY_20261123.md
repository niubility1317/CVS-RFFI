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

- LoRa 局部 Wiener 对照覆盖同一 900 个源 V 包与原 Tri-set 单种子。完整复协方差全量在 full 相对成熟 DURA 净增 42 次，95% 配对区间为 −3 至 +92，在 residual 净减 249 次，区间约为 −324 至 −176。两路线 clean 均保持 667/900。匹配半量消融中，复协方差相对对角读出分别净减 24 次和净增 31 次。同包隔离归因显示，换成真实总噪声功率后，原 Wiener 六景在 full 和 residual 分别再少判对 3 和 138 次。保载波精确扣噪则分别多判对 423 和 871 次，五台源接收机均受益。Residual 上更准确的功率未能解决复谱恢复造成的识别损失。该读出未进入正式目标矩阵。 [LoRa 局部 Wiener 对照](../../automation_reports/CV-SincNet/20261123-phase1-lora-local-wiener-source-m2-r01/report.md) · [LoRa 隔离噪声归因](../../automation_reports/CV-SincNet/20261123-diagnostic-lora-noise-floor-source-m2-r01/report.md)
- LoRa 分块复谱实验中，匹配遮挡组的粗恢复在 full 和 residual 比全谱组净多判对 93 和 257 次，五台源接收机均正向。遮挡组全量细化相对同组粗恢复则净少 13 和 128 次。两条路线最好的联合半量比成熟 DURA 分别少 49 和 126 次，clean 均为 667/900。两组各 34,693 参数、1,600 步。遮挡先验的独立收益在源域消融成立，两组均未进入正式矩阵。 [LoRa 分块复谱消融](../../automation_reports/CV-SincNet/20261123-phase1-lora-spectral-block-source-m4-r01/report.md)
- LoRa 同范数方向消融中，保护实收共享复谱在 full 全量读出净增 34 次，在 residual 半量读出净减 105 次。保护半量在 full 比直通多 28 次，95% 配对区间为 +5 至 +52，仍比成熟 DURA 少 38 次。Clean 均为 667/900。相同约束的源域收益未跨路线复现。 [LoRa 同量复数修正](../../automation_reports/CV-SincNet/20261123-phase1-lora-repeat-protection-source-m4-r01/report.md)
- LoRa 的源 U 经验复谱库中，跨半组一致性在 residual 相对交叉查询半量净增 38 次，区间 +19 至 +57。该读出仍比成熟 DURA 少 86 次。对四个已完成试验的 19 个固定恢复读出加直通做标签后验覆盖，full 和 residual 在源 V 六景分别为 69.70% 与 74.22%，超过配对成熟对照 670 与 1,048 次。其中经验库独占 85 与 109 次。覆盖用于判断互补性，尚无可部署规则取得所列覆盖。 [LoRa 经验模板与覆盖归因](../../automation_reports/CV-SincNet/20261124-phase1-lora-empirical-chirp-source-m2-r01/report.md)
- LoRa 将前导分成三组，轮换留出一组检验复数修正方向。保留方向和只保留幅值的两臂在所有激活包上都选择本组细化。六景匹配硬选净差为 −2 和 +3，95% 配对区间均跨零。各路线最强固定提议分别少于成熟半量 DURA 79 和 315 次，clean 均为 667/900。方向臂有 36,233 参数，训练 57.99 秒，单包旁路和激活中位数为 1.75 ms、16.55 ms。该实验未晋级。 [三组交叉留出与逐接收机评分](../../automation_reports/CV-SincNet/20261124-phase1-lora-crossfit-choice-source-m4-r01/report.md)
- LoRa 的局部相位中段、两端两臂各完成 40,000 步与 20,000 个源 U 包的拟合预算。两端半量的 full 六景 3,057/5,400，中段粗恢复的 residual 六景 2,872/5,400，分别低于同包 DURA 37 次和 88 次。所有候选保持 clean 667/900。两端全量推理保留外带置信信息，相对遮蔽该量净增 27 和 110 次，95% 区间分别约 +2 至 +53、+63 至 +159。强受扰时相位读出的复波形误差下降，识别正确数仍有退化。无候选晋级。 [40,000 步局部相位恢复](../../automation_reports/CV-SincNet/20261124-phase1-lora-local-phase-source-m4-r01/report.md)
- LoRa 联合幅相长训的同架构、同数据、同预算消融中，加入源 U 跨 chirp 弱复谱关系损失后，全量读出两路线分别多判对 72 和 201 次，95% 配对区间约 +39 至 +104、+135 至 +263，两路线五台源接收机方向一致。效果最好的关系组半量仍比成熟 DURA 少判对 112 和 404 次，clean 都为 667/900。Residual 低城区的 9–32 kHz 复误差明显下降，同包设备识别仍下降。没有读出进入正式评测。 [40,000 步联合幅相消融与投影归因](../../automation_reports/CV-SincNet/20261124-phase1-lora-complex-geometry-source-m4-r01/report.md)
- WiSig 的直接前导均衡和绝对 CFO 校正使 clean 与受损识别均下降。两路短训练重复残差可反映观测失配，估计方差含相位变化和硬件失配。 [接收量及工程对照](../../automation_reports/CV-SincNet/20261123-phase1-wisig-stf-engineering-source-s2026092701-r02/report.md)
- 与历史 F50 配对时，接收端时标检测和协议后缀外推在源 V 把两路线六景提高到 7,121/10,800，同时保持 1,773/1,800 clean。换成正式对比所用的 Tri-SIG-TIME 单种子后，同一方法的两路线六景分别净减 66 和 67 次正确。拟合一个短长前导共享复响应也下降，其中标量法净减 176 和 172，三抽头法净减 204 和 206。Tri-SIG-TIME 的 R69 同包基线为 10,194 和 10,195/10,800。两条路线的 clean 正确数均为 1,775/1,800，时标方法未晋级。 [F50 时标对照](../../automation_reports/CV-SincNet/20261123-phase1-wisig-timing-engineering-source-s2026092701-r01/report.md) · [Tri-SIG-TIME 联合前导对照](../../automation_reports/CV-SincNet/20261123-phase1-wisig-joint-preamble-source-s2026092701-r01/report.md)
- 合法源 U 训练使用通用波形扰动，粗恢复、观测约束和细化联合训练。与匹配波形损失组相比，弱复谱约束在冻结 Tri-SIG-TIME 单种子的无门联合源读出中，两路线六景分别多判对 1,524 和 1,526 次，全部场景和源接收机方向一致。复评沿用原 227,268 参数、1,200 步权重。该读出仍低于直通与 R69，固定门在两条路线各净少 2 次六景正确与 1 次 clean 正确。此前把接收端时标放进训练和推理可改善历史 F50 的匹配训练对照，联合细化仍低于只做时标校正。 [Tri-SIG-TIME 弱复谱权重复评](../../automation_reports/CV-SincNet/20261123-phase1-wisig-tri-learned-replay-s2026092701-r01/report.md) · [F50 弱复谱消融](../../automation_reports/CV-SincNet/20261123-phase1-wisig-uncertain-prior-source-s2026092701-m2-r01/report.md) · [时标联合训练消融](../../automation_reports/CV-SincNet/20261123-phase1-wisig-aligned-recovery-source-s2026092701-m2-r01/report.md)
- WiSig 使用实收短前导重复信息的频块遮挡粗恢复相对匹配全谱臂在 full 与 residual 均多判对 4,479 次。两路五台源接收机方向一致，绝对识别仍低于直通。遮挡臂继续细化各少判对约 5,600 次，只改单一幅度或相位可部分挽回。固定门在两路六景各只触发 1/10,800 次，逐景正确数与直通一致。此结果与 LoRa 一起定位了粗预测与联合细化的收益差异，尚无候选通过跨路线要求。 [WiSig 重复前导频块消融](../../automation_reports/CV-SincNet/20261123-phase1-wisig-repeat-block-source-m3-r01/report.md)
- WiSig 的另一项同包隔离诊断使用相同冻结 Tri-SIG-TIME。实收幅度与配对 clean 相位结合后，两路线六景是 87.45% 和 87.47%，仍低于同次 R69 的 94.39% 和 94.40%。仅替换 clean 幅度分别是 72.81% 和 72.89%。两者都取配对 clean 时为 98.61%。这些读出用于归因，配对 clean 未用于训练、冻结或候选输入。 [WiSig 幅相因子报告](https://github.com/niubility1317/CVS-RFFI/blob/4538bbb21ed6e22bad9ca2c6599f6d0ecee879be/automation_reports/CV-SincNet/20261123-phase1-wisig-phase-magnitude-source-m4-r01/report.md)
- WiSig 的匹配训练中，在优化循环内保护实收幅度使两路线的门控读出各比原拟合多判对 2,005 次，仍分别比直通少 184 和 185 次。直通和多个归档恢复读出的事后同包覆盖达到 98.20% 和 98.24%。这个覆盖使用了源 V 标签，显示各恢复读出的互补性，尚无接收端选择规则达到该表现。 [幅度保护及候选覆盖](https://github.com/niubility1317/CVS-RFFI/blob/5f56af5f65eb158ee7aa5c28ec61918a487c9532/automation_reports/CV-SincNet/20261123-phase1-wisig-observed-amplitude-source-m3-r01/report.md)
- WiSig 波形风险动态选择在两路各激活约 1,280 次，超过 99% 的激活读出为同一细化提议。与静态选择相比，动态规则的六景净差为 +2 和 +1，配对区间均跨零。隔离诊断在已知配对 clean 时按同一波形风险选提议，六景只有 65.64% 和 65.57%。该网络的六个提议即使用标签统计覆盖也低于 R69，表明这组读出的互补性和波形风险排序均有限。 [波形排序与隔离诊断](https://github.com/niubility1317/CVS-RFFI/blob/90d9b29dca07a0f449470b0f4c9d7e7ad19d53a8/automation_reports/CV-SincNet/20261124-phase1-wisig-waveform-choice-source-m3-r01/report.md)
- WiSig 加入实测源包的频谱关系约束后，旁路读出相对同架构对照在两路线各净多判对 78 次，95% 同物理包区间分别为 +34 至 +122、+35 至 +122。相对 R69 仍净少 1,243 和 1,240 次，clean 也各低 2 次。候选未进入正式 Main split。 [频谱关系消融](https://github.com/niubility1317/CVS-RFFI/blob/1b807962d06453b95166888de9dbe00f31e8719f/automation_reports/CV-SincNet/20261124-phase1-wisig-instance-geometry-source-m3-r01/report.md)
- 配对 clean 只进隔离源诊断。依次归因时标、缺失样点和整体复增益后，最强的诊断仍低于原 R69。该读出不纳入可部署候选。 [配对隔离归因](../../automation_reports/CV-SincNet/20261123-diagnostic-wisig-source-paired-time-edge-s2026092701-r01/report.md)

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
