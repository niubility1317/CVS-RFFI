# LoRa 与 WiSig 信道恢复研发复盘

LoRa 主测试固定星地增强 PAMF W24 E104、原默认头与 `8×2×432` 输入。Full 和 Residual 沿用已登记的同一批 30,000 个物理包与各自的接收处理。阶段验收以 Full 的六景总正确数至少 162,000、低城区至少 25,500、clean 至少 28,566 为准。

两条历史对话的主线是从接收端前导工程、逐谱恢复和门控试验，转回完整 DURA 联合恢复。LoRa 在固定增强 W24 后取得了正式同包收益，距离低城区目标仍有较大差距。WiSig 已证明原 R69 的收益，新训练的观察算子和估计式模型尚未复现这种收益。两条路线都显示，前导更平滑、对齐误差更小或 clean 更稳定时，受扰识别仍可能下降。

## R02–R17 的实现、训练与结果

早期版本运行于 ManySig。训练臂默认使用同一物理源 IQ 生成受扰与 clean 配对，训练 TX 标签和 F 监督均关闭。固定源 V 或历史冻结 F 的身份读出另行披露。R02、R03、R07、R10、R12、R15、R16、R17 的新增训练臂各 20 轮、4,940 次更新、批量 256，覆盖 6,300 条 L 和 56,700 条 U，除非表内另列继承或评分。R02 使用 `pamf_reception_restore`，R06–R07 的作用与 CG 入口在 `pamf_operator_restore` 和 `pamf_action_restore`，R10/R11 在 `pamf_uncertainty_restore`，R17 的实际阶段执行在 `pamf_unrolled_restore/models.py`。

| 版本 | 机制、训练和输出 | 已验证结果 |
|---|---|---|
| R02 | 从零训练时域残差 U-Net、复时频 U-Net、条件扩散与流匹配。确定性模型使用配对 IQ 和频谱损失，生成模型使用速度及配对目标，正式输出走完整 16 步采样。各路径输出同一份复 IQ。 | 复时频在 1,200 包历史面板的低城区增加 12.75 个百分点，clean 降低 0.50 个百分点。扩散及流匹配多次采样的身份切换率分别为 56.25%、51.56%。 [R02 结果](../../automation_reports/CV-SincNet/20261019-phase1-reception-restoration-selfsup-manysig-m6-r02/report.md) |
| R03 | 冻结 R02 复时频粗恢复，从零训练确定性、扩散及流匹配修正器。配对目标是 clean 与 coarse 之差，部署输出为粗估计加定标后的残差。加入父臂后组件合计 40 轮。 | 两种源规则均选择确定性修正。扩散与流匹配的源采样身份切换率分别 31.25%、7.03%。 [R03 结果](../../automation_reports/CV-SincNet/20261019-phase1-restoration-residual-refinement-selfsup-m3-r03/report.md) |
| R04–R05 | 原 R02 粗恢复及 R03 确定性修正的 168,000 条 × 七景正式接线，分别由冻结历史识别器读出，本轮无新增训练。 | 输入恢复和固定分类器的对照已完成，历史读出分别为 clean F 与 As F。 [R04](../../automation_reports/CV-SincNet/20261019-phase1-restoration-formal-frozen-coarse-s2026092701-r04/report.md) · [R05](../../automation_reports/CV-SincNet/20261019-phase1-restoration-formal-frozen-residual-s2026092701-r05/report.md) |
| R06 | 固定 coarse，源域控制对比已知模拟作用、各向同性代理和实收估计的 10/49 tap 算子。无新增训练，已知作用用于离线诊断。 | 已知作用软修正使平均 IQ NMSE 下降 13.73%；10/49 tap 实收估计分别增加 1.33% 和 6.47%。 [R06](../../automation_reports/CV-SincNet/20261019-phase1-restoration-operator-source-s2026092701-r06/report.md) |
| R07 | 冻结 R02 coarse，新训练由当前实收估计的复主路、镜像路和噪声量，以配对模拟作用／噪声代理监督；固定软 MAP 和六次 CG。使用 15% clean 与 85% residual。 | 对 coarse 的源六景平均 NMSE 增加约 1.55%，低城区身份只增加 0.11 个百分点。未晋级目标测试。 [R07](../../automation_reports/CV-SincNet/20261019-phase1-restoration-learned-action-source-s2026092701-r07/report.md) |
| R08–R09 | 重用 R04/R05 完整恢复 IQ，接原 F50 跨 RX 默认头。R08 在等待资源时撤下，R09 完成同一矩阵。两者无新增训练或目标信道重生成。 | R09 的直通／粗恢复／残差修正低城区为 30.2994%／63.1500%／63.9458%，残差 clean 84.1786%，直通 84.6762%。 [R08](../../automation_reports/CV-SincNet/20261019-phase1-restoration-formal-FcrossRX50-s2026092701-r08/report.md) · [R09](../../automation_reports/CV-SincNet/20261019-phase1-restoration-formal-FcrossRX50-s2026092701-r09/report.md) |
| R10 | 同一冻结 coarse，训练直接残差、原生方差 MAP 与可学习方差 MAP。物理组另继承 R07 action，输出从固定六步 CG 取得。配对 IQ／频谱及方差 NLL；有一项源 IQ 方差定标。 | 不确定性组波形改善，但最差低城区 TX 较旧对照下降 1.99 个百分点。直接组与物理组存在已记录的初始权重差异，两物理组初值一致。 [R10](../../automation_reports/CV-SincNet/20261019-phase1-restoration-uncertainty-selfsup-s2026092701-m3-r10/report.md) |
| R11 | 单独登记以源证据冻结 R10 原生方差 MAP，继承 coarse、action、prior 各 20 轮。正式评分新增训练为零。 | 低城区 65.2292%，clean 84.5863%，历史直通 clean 84.6762%。 [R11](../../automation_reports/CV-SincNet/20261019-phase1-restoration-formal-nativeMAP-F50-s2026092701-r11/report.md) |
| R12 | 同父 R10 native，冻结 coarse/action，只续训 prior。比较配对 IQ／FFT 与加频谱、PA、统计量、参考响应的解析保真。解析定标仅使用 L/U IQ。 | 源低城区两臂同为 82.9444%，目标确认未启动。 [R12](../../automation_reports/CV-SincNet/20261019-phase1-restoration-analytic-views-selfsup-s2026092701-m2-r12/report.md) |
| R13–R14 | 固定 R10 native，分别换用已知作用、方差代理、coarse 或 prior 读出。R13 因推理外层引发 Sinc 版本计数错误失败，R14 用新输出根完成原诊断。无新增训练。 | R14 的已知作用加模拟方差代理使六景 NMSE 下降 15.41%，低城区增加 0.56 个百分点，中城区降低 0.06 个百分点。 [R13](../../automation_reports/CV-SincNet/20261019-diagnostic-restoration-action-fidelity-F50-s2026092701-m5-r13/report.md) · [R14](../../automation_reports/CV-SincNet/20261019-diagnostic-restoration-action-fidelity-F50-s2026092701-m5-r14/report.md) |
| R15–R16 | 冻结 R07 coarse/action，匹配全分辨率扩张 TCN 和 U-Net。R15 使用 IQ NMSE 与 FFT，R16 从对应共同初值只用 IQ，固定原生方差与六次 CG。 | R15 TCN 的源低城区比新 U-Net 低 4.06 个百分点。R16 IQ-only 提升源低城区，U-Net 82.50%→84.39%，TCN 78.44%→82.33%，两轮均未过完整源准入。 [R15](../../automation_reports/CV-SincNet/20261020-phase1-restoration-fullresolution-selfsup-s2026092701-m2-r15/report.md) · [R16](../../automation_reports/CV-SincNet/20261020-phase1-restoration-iqonly-selfsup-s2026092701-m2-r16/report.md) |
| R17 | 冻结 R07 coarse/action/natural variance。三个学习先验与软一致性阶段，输入为实收、coarse、当前估计、反投影残差和对数方差。每步两次 warm-start CG，跨步反传，只用 IQ NMSE。共享或不共享先验具有相同初始输出，最终交付一份 IQ。 | 原 6 次单段 CG／共享三阶段／不共享三阶段的源低城区分别 84.3889%／86.3889%／86.8333%。不共享版过源门槛，正式低城区 67.2167%，其他五景 2/5 达到历史门槛。 [R17](../../automation_reports/CV-SincNet/20261020-phase1-restoration-unrolled-selfsup-s2026092701-m2-r17/report.md) |

## LoRa 权重与对照

- R02–R17 权重来自 ManySig 的物理划分与 256 点读出。LoRa 权重来源必须匹配 `lora_full_packets_v1`、`lora_rx5_main_v1:392005` 和原始源 IQ 角色。旧权重登记为 `CHECKPOINT_DATA_CONTRACT_MISMATCH`。复数时频粗恢复、实收作用估计、反投影残差和多步软一致性可用 LoRa 源 IQ 重新实现和训练。
- 已完成的原生八 chirp 两阶段恢复使用另一识别器 E108。它验证了接线和源端可训练性，E104 的正式比较复用已独立评分的同包归档。原生主带复谱正则的 E104 小规模对照六景净少 99 次正确，配对 95% 区间为 −161 至 −34 次，已保留。 [八 chirp](../../automation_reports/CV-SincNet/20261125-phase1-lora-w24-native-restore-s2026092702-r01/report.md) · [主带消融](../../automation_reports/CV-SincNet/20261125-phase1-lora-w24-native-mainband-m2-r01/report.md)
- 40,000 步原 W24 DURA 的半量读出已做双路线正式评分。固定 E104 时，门控版 Full 为 136,912/180,000、低城区 15,562/30,000、clean 28,566/30,000，Residual 为 117,175/180,000、低城区 12,294/30,000、clean 28,566/30,000。全包版 Full 为 137,631/180,000、低城区 15,660/30,000、clean 28,539/30,000，Residual 为 118,157/180,000、低城区 12,297/30,000、clean 28,539/30,000。全包版 Full 距当前三项目标分别差 24,369 次、9,840 包和 27 包。正式同包 Residual 的直通和历史 R69 对照分别为 114,903 与 114,005 次六景正确。 [门控归档](../../automation_reports/CV-SincNet/20261125-phase1-lora-w24-formal-both-routes-m3-r01/report.md) · [全包结果](LORA_W24_UNGATED_FORMAL_20261125.md)
- 从零训练的首尾损失候选在 40,000 步冻结。按原定源 V 规则，Full 六景 1,441/1,800 超过原全包半量 1,434/1,800，Residual 1,242/1,800 超过 1,217/1,800，两路 clean 均等于直通 297/300。取得资格后进入完整双路线。另一冻结权重的联合原生边缘读出在源 Full 为 1,423/1,800、clean 293/300，在 Residual 为 1,259/1,800、clean 293/300。其仅中心读出在两路 clean 均为 296/300。两组保持各自的权重、源测结果与使用范围。 [首尾损失](../../automation_reports/CV-SincNet/20261125-phase1-lora-w24-outer-objective-source-s2026112991-r01/report.md) · [联合边缘](../../automation_reports/CV-SincNet/20261125-phase1-lora-w24-joint-edges-source-s2026112991-r01/report.md)

## 首尾联合损失的正式复用

两路线同用一份冻结恢复权重和固定半量中心读出，保留 48 点原生边缘。候选与归档的全包半量对照采用相同 W24 E104、数据坐标及逐包评估口径。每条预测重新计算直通并逐包比对归档。正式运行复用已经登记的两路六分片，不重新训练候选。 [本轮正式运行](../../automation_reports/CV-SincNet/20261125-phase1-lora-w24-formal-outer-objective-s2026092702-r01/report.md)

## 两条旧对话的 LoRa 研发路线

| 路线 | 已做的匹配检查 | 得到的经验 |
|---|---|---|
| 公共／独有复谱、秩 4、跨频协方差与分带 | 历史 Tri-set 源面板上，同一权重开启公共与独有分量耦合，六景相对独立组 Full +53、Residual +147；秩 4 相关项的独立净增为 +52、+128。完整私有 Wiener 在同类匹配测试中却为 +36、−303。局部 Hankel 比整段滤波好，仍低于半量 DURA。 | 恢复器可以借公共结构估计扰动，同时需要保留 per-chirp 弱复谱。相关项的局部收益只与各自匹配对照比较。大多数候选未通过双路线源筛查。 |
| CFO、采样时钟、镜像、回波和时标 | 逐 chirp 细化使源低城区 CFO 误差 P90 在 Full 从 42.3 降至 22.4 Hz，在 Residual 从 72.3 降至 34.2 Hz。镜像/回波转换到正确去扫坐标，在受控条件下降低 9–32 kHz 误差约 4–6 dB，practical 上却可出现全部旁路、零判决变化。镜像校正在受控条件下的约 70% NMSE 改善，到 practical 激活包的误差改善中位数低于 0.01%。 | 同步与坐标纠正使反演更合理。保留受扰重放、接收机分组和冻结识别检查，才能确认它们是否改变实际决策。 |
| 相位、幅度、修正量、鲁棒门限与候选选择 | 同包同量时，固定 2 倍修正、将解析方向替换为精确扣噪方向，六景 Full +219、Residual +753；单纯加倍解析方向为 +41、−189。等量分带、奇偶先验、多专家路由、反向噪声训练、逐点相位、带间风险和只调写回强度大多未通过匹配双路对照。用旧源面板真值穷举 1,485 对候选加直通和半量 DURA，最好的逐包选择也只有 63.54%。 | 在显著扩大回写量前先证明复数方向有效。有些实验只改善某一路，源面板的候选选择潜力也受候选本身的共同错误限制。精确方向和真值选专家只用于离线诊断。 |
| 基于前导的实收噪声估计 | 方差匹配相对等量 Wiener 在源六景净增 +55、+106，仍低于匹配 DURA。旧 Tri-set 的正式 3,000 包局部方差消融相对原扩容版六景 −83、−156，两路 clean 持平。分侧协方差门限在同规模目标子集为 +23、−2，clean 各少 3 包。 | 按包估计应在两条接收路线使用同一推理规则，并报告 clean 误激活与轻扰动净损失。源端门限零误触发并未延续到部分目标 RTL 接收机。 |
| 六窗正式路线 | 旧 Tri-set 三种子同包 Full 的复响应半量、`guard_half`、扩容半量六景分别为 56.0024%、56.4131%、56.4759%，低城区为 42.0500%、42.6411%、43.4222%。扩容半量在 Residual 为 53.9015%，低于 `guard_half` 的 54.8617%。扩容从零联合训练粗恢复、两个独立学习先验和每阶段两步 CG。 | 完整 DURA 的提升可核验，扩容对 Full 和 Residual 的效果有分歧。旧 Tri-set 和 W24 两组冻结识别器、训练时间和输入不同，按各自对照记录。 |
| 双采样率与弱复谱训练 | 已核验旧 Tri-set 3,000 包独立评分。对原扩容半量，双采样率六景 Full +27、Residual −103，低城区 +9、−43，clean 两路各 +1。调整弱复谱损失分母的匹配组为六景 −85、+83，低城区 −32、+32，clean 持平。 | 双带宽接口与弱谱损失各自具有明确的路线差异。保留两条路线的独立源验证与同包评分，避免靠 Full 平均掩盖 Residual 退化。 |
| 原生 W24、固定半量与联合损失 | 星地增强 E104 的直通 Full 六景为 74.2850%，旧 clean-only E108 为 65.4100%。改进后的 1,271,038 参数 W24 DURA 沿八条前导、384 点中心与两端各 24 个原生点，从零用 20,000 个源 U 包训练 40,000 步。门控和逐包半量都带来正式收益，全包半量提升更多，同时 clean 少 27 包。首尾损失以相同结构和预算调整中心窗监督，过了双路源门槛。 | 识别器代际差只解释基线变化。评价恢复器时固定 E104；对每个版本披露实际调用覆盖、实际写回和在同包上的纠错与改错。 |

表中的早期探索主要来自同一批源域面板。多轮共用样本，汇总为机制证据，不按独立复现次数累加。代表性试验与原始登记可从两条 Codex 对话、[LoRa 共同失败归因](../../automation_reports/CV-SincNet/20261124-phase1-lora-common-failure-physics-source-m2-r01/report.md)、[W24 正式测试](LORA_W24_UNGATED_FORMAL_20261125.md)查回。

### LoRa 最有区分度的诊断

- **先降噪，再检验作用反演。** 固定 E104 的 300 个 Full 源 V 包中，仅去除仿真新增噪声，六景从 76.22% 到 94.56%，低城区从 144 到 263 个正确。扣除 90% 的新增噪声幅度时，六景已到 91.89%，低城区仍为 77.00%。历史 Tri-set 的另一个 900 包源面板中，123 个两路共同判错的低城区包只做已知作用反演时恢复 20、12 包，先精确去噪再反演时恢复 102、104 包。两项诊断共同指向低城区的噪声和逆解耦合。 [W24 噪声扫描](LORA_W24_SOURCE_NOISE_SWEEP_20261125.md) · [共同失败归因](../../automation_reports/CV-SincNet/20261124-phase1-lora-common-failure-physics-source-m2-r01/report.md)
- **保真要在实际读出上量。** 半量改全量时，低城区 0–100 kHz 复 NMSE 在 Full 从 0.01060 降到 0.00478，正确包从 164 降到 158；Residual 从 0.04636 降到 0.00579，正确包从 131 降到 102。主带单独承受 Wiener 处理的源复验中，频带功率中位数仍保留 98.7%，高郊区正确数却从 91/100 降为 33/100。正式 3,000 包 Wiener 半量 Full 六景只有 31.3944%，同包直通为 74.4833%。 [半量与全量](LORA_W24_READOUT_SWEEP_20261125.md) · [Wiener 正式子集](../../automation_reports/CV-SincNet/20261125-phase1-lora-w24-native-wiener-pilot-s2026092702-r01/report.md)
- **公共 chirp 和短边缘都需要单独检验。** 同能量增加公共修正时，六景比原半量少判对 35、31 次。固定半量中心后，只对原生边缘精确去除新增噪声，六景净增 32、36，实收平滑为 −1、+1。随后实际联合边缘解在源 Residual 有提升，在 Full 与 clean 存在损失。这说明原生边缘有可恢复空间，已有平滑与两级边缘网络的收益没有普适到双路。 [公共／独有消融](../../automation_reports/CV-SincNet/20261125-diagnostic-lora-w24-repeat-partition-s2026112559-r01/report.md) · [原生边缘消融](../../automation_reports/CV-SincNet/20261125-diagnostic-lora-w24-edge-denoise-source-s2026112559-r01/report.md)

## 两条旧对话的 WiSig 研发路线

| 路线 | 核验结果 | 得到的经验 |
|---|---|---|
| 原 DURA/R69 复跑 | 168,000 包、三个冻结 Tri-SIG-TIME 种子。Full 和 Residual 直通六景 70.5770%、70.5758%，R69 达 79.1337%、79.1358%，clean 为 438,286/504,000。 | 原 R69 是匹配的正式恢复对照，必须保留实际运行行为及训练来源。 |
| STF/LTF 定时、预测后缀、短延迟乘积与平滑前导 | 常规六景约 90.6% 的前导定时误差在 1 点内。短延迟读出使约 7,103 个 Full 包视图进入 ±2 点，对齐子群仍净减 68 次。真实 512 包后缀实验，短响应/共轭拟合有 8/9 条件增加误差。平滑前导正式测试两路六景约 40.04%，目标激活约 76%，激活包相对直通下降约 40.17 个百分点。 | 时标可用于回到同一坐标。对齐无法补回已有的幅相或波形破坏。实收匹配成功后还要分开检验重叠区与缺边。 |
| 盲恢复和估计式恢复、门控 | 受扰低覆盖的历史门控下，Estimated、Blind 两版正式六景比 R69 低约 8.56 个百分点。强制逐包运行后 Full 六景为 56.5018%、56.7603%，Residual 为 56.4997%、56.7611%，两版 clean 均超过直通。 | clean 高和低触发率可与受扰恢复能力很弱同时发生。活动数、触发后净变化和全包路径各有不同用途。 |
| 原 R69 作用分解与从零 FIR | 在源 V 固定原 R69，其 FIR 单独读出两路均为 9,955/10,800，比单位作用各多 1,306，但仍距完整 R69 约 240。直接把重复短训练差分作为方差使两路 clean 1775→1527；固定方差则让六景减少 783、787。新因果和居中 FIR 各 40,000 步，居中末步源六景 8,180、8,170，优于因果却低于直通 9,057、9,062 和 R69 10,194、10,195。 | 有用的跨采样作用需要与观测方差、粗恢复、先验及其数值尺度共同校准。原模型内组件的有效性与从零重建整套模型分别核验。 |
| 幅相、锚点、MAP 权重及匹配长训 | 居中 FIR 末步换入原 R69 相位在两路六景 +316、+317，换幅度为 +178、+184。把学习锚点直接换成实收，末步 −863、−843。只增加首次 MAP 观测权重，第一阶段 +305、+299，末步只剩 +137、+140，且 clean 各少 2。已完成源评分的固定权重、逐频及相位损失三组 Full 六景为 8,286、8,364、8,313，Residual 为 8,285、8,358、8,316（分母均为 10,800），均低于同面板直通。 | 相位和首阶段数据一致性是独立线索。实际收益依赖后续阶段是否保留改进，阶段之间要记录每包纠错和新损伤。 |

WiSig 的源域幅相互换、已知方差和精确去噪都属于归因。原 R69 使用的训练坐标、已有流水线和冻结参数各自保留，不移植为 LoRa 权重。 [WiSig 无门控正式评分](WISIG_UNGATED_CLEAN_MAIN_20261125.md) · [原生 FIR 与 MAP 消融](WISIG_NATIVE_FIR_SOURCE_RESULTS_20261125.md) · [前导和方差分析](WISIG_PREAMBLE_ACTION_AND_NOISE_SOURCE_20261125.md)

## 后续实验可沿用的结论

1. **LoRa 优先检验能否从实收估计正确的复数修正方向。** 一条有证据的研究假设是让可学习粗恢复先抑制新增噪声，再做受置信度约束的软数据一致性。采用 R17 式阶段分解时，应把观测估计、软一致性和先验都写进匹配消融。同一 LoRa 源数据、固定 W24、相同总求解迭代和训练预算可隔离一次反演与交替细化的贡献。
2. **在实际送入 W24 的输出上检查指纹保留。** 逐阶段与最终半量读出分别记录复幅相、近带弱谱、chirp 间差异、原生边缘与 clean 纠错/改错。信号功率、重建 NMSE 和公共谱一致性用于定位机制。模型训练、定标和冻结仍限合法源 IQ，冻结后的身份读出单独评分。
3. **把 Full、Residual、clean 和轻扰动纳入同一机制实验。** 两路共享权重及运行规则，逐接收机报告亮点和退化。源域先报低城区与六景总体净纠错，并记录实际运行覆盖。全量正式测试只按冻结结果作验收。门控、读出幅度、分频权重和增强日程在看到正式目标结果前固定。
4. **把工程共性保留在接线和诊断层。** 已有失败包括源冒烟漏掉 CPU 重放导入、历史包模块顺序错误、以及评分时重采样精度不一致。真实源冒烟应触发与正式预测相同的数据、子进程和坐标路径。预测冻结后，再核对物理 ID、读出覆盖和真值连接。旧 `w24-io` 与 `w24-sat-aug-full` 的 r01 源冒烟失败；两个 r02 有首批 clean 后，在受扰重放缺失模块 `experiments.pamf_unrolled_restore.denoise` 处失败。N607 未生成完整分片，保留为技术失败证据。当前完整双路线由独立运行负责。
