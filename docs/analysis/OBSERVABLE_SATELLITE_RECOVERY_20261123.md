# LoRa 与 WiSig 可部署星地信道恢复阶段结果

截至 2026-11-23，原冻结基线的两套数据和两条信道路线已补齐。当前候选尚未达到正式验收，目标保持进行中。

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

- WiSig 的直接前导均衡和绝对 CFO 校正使 clean 与受损识别均下降。两路短训练重复残差可反映观测失配，估计方差含相位变化和硬件失配。 [接收量及工程对照](../../automation_reports/CV-SincNet/20261123-phase1-wisig-stf-engineering-source-s2026092701-r02/report.md)
- 当前接收端时标检测和协议后缀外推在源 V 把两路六景提高到 7,121/10,800，同时保持 1,773/1,800 clean。合法估计对齐而不改动观测重叠的幅相，收益仍低于匹配 R69。 [时标工程对照](../../automation_reports/CV-SincNet/20261123-phase1-wisig-timing-engineering-source-s2026092701-r01/report.md)
- 新的源 U 训练使用通用波形扰动，粗恢复、观测约束和细化联合训练。匹配弱复谱损失消融带来源域增益，仍低于 R69。把接收端时标加入训练和推理可改善匹配训练对照，但联合细化仍低于只做时标校正。每模型 227,268 参数、1,200 步，训练秒数与逐景、逐 RX 指标留存。 [弱复谱消融](../../automation_reports/CV-SincNet/20261123-phase1-wisig-uncertain-prior-source-s2026092701-m2-r01/report.md) · [时标联合训练消融](../../automation_reports/CV-SincNet/20261123-phase1-wisig-aligned-recovery-source-s2026092701-m2-r01/report.md)
- 配对 clean 只进隔离源诊断。依次归因时标、缺失样点和整体复增益后，最强的诊断仍低于原 R69。该读出不纳入可部署候选。 [配对隔离归因](../../automation_reports/CV-SincNet/20261123-diagnostic-wisig-source-paired-time-edge-s2026092701-r01/report.md)
