# 异 TX 接收变化的相对身份风险迁移

状态 **ANALYZED / VERIFIED：INITIAL 0/4、FINAL20 1/4，固定候选未通过**。下文保留原预登记。当前目标是学习能跨 RX 使用的实际 joint160 身份特征，保留原默认分类头正常联合训练；84% clean 三种子均值尚未达到。这是本轮第二个核心假设，失败后不再增加第三个假设。此前绝对概率迁移 INITIAL 和 FINAL20 各只有 1/4 通过全部源条件，固定候选关闭，不改变阈值。它仍留下明确证据：FINAL20 跨身份 RX 差分产生 316 至 462 个 active 判定、81 至 107 个不同 origin，辅助与主 CE 的 F 梯度余弦 .407 至 .688；初期则接近 .99975，主要放大普通分类方向。

本候选改变监督内容：**只修复借来的接收变化比真实目标 RX 同身份样本新增的分类风险。** 同 day、真实有标签 L，a 是 RX-r/side0 六 TX，b 是 RX-s/side1 六 TX。对 origin y 与每个 c≠y，正向 virtual=a_y−a_c+b_c，实际参照 factual=b_y；反向 virtual=b_y−b_c+a_c，factual=a_y。c 遍历全部 5 个异 TX，不先平均差分、不按结果选参照。目标来自 origin 真实 y，factual 也为真实同 y。物理包不假设相同报文。类相关接收作用、非线性及 packetnoise 都可能使迁移失败，所以这只是可检验的特征训练任务，不是硬件反事实或精确 RX/TX 分解证明。

用原 cosine W.detach、scale30、eps1e-4，固定 loss=.1×mean relu(CE(virtual,y)−stopgrad(CE(factual,y))−1e-6)。1e-6 是数值容差，不扫参数。ReLU 在相等处梯度为零；停止风险后不继续要求置信度达到 .9。virtual 的三个实际 joint 端点与三个实际增益完整反传，factual 风险和 W 停止梯度；mainCE 正常联合更新原 W/骨干/增益。factual 在其他 virtual 角色仍可能收到梯度，不能声称它的参数不变。没有新头、拟合分类器、生成器、教师关系复制、原型或部署特征变换。FP64 解析判别及同 view FP32 比较单列，不称额外模型前向。

采用既定三控制 ZERO/SAME_RX/CROSS_RX。CROSS 是 20 个有序跨 RX 组合×3 day=60 个唯一组合；SAME 是 15 个同 RX/day 两 physicalside 组合重复 4 次，披露只有 15 个唯一组合。每 pair 双向×6 y×5 c=60 virtual，共 3600/control；同数 factual 判别逐项计算，不复用 180 个 factual readout，因此 7200 解析判别/control、两非零控制每 call 14400，FP64 与 FP32 各计算一次。固定 loss 分母 3600，不随 active 数变化。每个 packet 作 origin/source-reference/destination-reference 各 20 次，groupendpoint 各 4 次；active_origin 在 180 个真实物理包中去重，重复视图不冒充覆盖增加。RX、targetTX、referenceTX 的 active、风险、错误与真实参照 CE 分层保存。

源矩阵 INITIAL/FINAL20×4 currentL180×3control，共 24 单元。复用此前 900 L：current ord2/3、19/20、37/38、54/55；held66/67，物理 ID 不交。原数据角色仍 L6300/U56700隐藏/V27000，RX1/3/4/6/8、day1/2/3、6TX。父 FINAL20 见过 heldL，这不是未见 RX 或新的独立盲测；V 不参与梯度，U 不读取 IQ/标签，target 不读取。复用已验证数据 handles，不新增数据重验或哈希链。

固定父为 own BASE_CE-s2026092701，执行6693e4480ef990bc23c5021ff1440b547fe5fcaf，原随机 initial_smoke 与 final_ssdg epoch20。source_proof 在两个加载前核对完整实际角色/ID、scratch ancestry、固定final冻结规则，初始 scratch_only 和 finalactualargs 严格验证。没有按 target成绩选择权重或回退其他父。INITIAL 保留原 optimizeroptions、moments 为空；FINAL 保留 savedm/v/step。两者同预定 LR2e-4/globalclip5/188parameters。各真实 CP 做 2 个全零 IQ noquery smoke，PASS 立即继续。eval 模式，原模型/optimizer/CP/grad/RNG不更新；临时 AdamW 副本执行对照一步，after 全188参数进入真实模型，ACTUAL afterW 和 PRE_W beforeW 分析同一 afterjoint/after3gains，不换头、不拟合。

**真实结果前固定资格**：每 state 至少 2/4 CROSS 同时相对 ZERO/SAME 满足以下全部条件。

- current active≥180/3600，distinct active origin≥18/180，辅助梯度非零，abs(auxF/mainCE cosine)≤.95，避免仅放大原方向。
- held PRE_W 的 mean positive excess risk 差≤−1e-4，且 raw virtual meanCE 差≤−1e-4，transfer 错数不增。后一条件防止只降低 factual 身份置信度就让 excess 变小。
- 原 held ACTUAL/PRE_W 准确率不降，CE 相对各控制的增加≤1%；门槛与旧候选不混用。
- actualF 差 norm>1e-6、heldactualjoint RMS 差>1e-5，至少一个浅/中层 tensor 改变。

FINAL20 合格优先固定同父 E20＋20；仅 INITIAL 合格才 scratch20。该优先级依据已经完成的前项初期近共线、晚期新方向证据，在本候选数据前固定。source PASS 只说明值得真实训练，必须继续发布固定预算的 CE 纯基线、原隐藏 U 旧门控 SSL、SAME/CROSS，原默认头 joint、final-only、每row自动clean预测和独立 truth-last。续训必须同父同20增量 CE 对照、完整继承核对及真实 optimizer状态说明，累计40不能冒充scratch20。此前绝对概率候选不按本规则重判；源不通过则关闭固定 loss，不扫容差/λ/LR/参照/强度。

预算 900uniqueL/6300metadata/6124 网络 packet 前向（4 smoke＋2×[180before＋4×(180current＋3×180after)]）、48 临时 CPUAdamW、60loss calls、864000 FP64＋864000 FP32 解析判别；0 实际训练更新/头拟合/EMA/U/V/target。风险降低不等于原身份识别提高，局部一步收益不等于完整20epoch，更不等于未见 RX 泛化。Sinc kernel 比较是 whole CE+aux proposal versuspre，不冒充纯 aux effect。

唯一 run `20261015-phase1-SIGTIME-identity-excess-transfer-sourceL-s2026092701-r01`，owner codex/root/identity-excess-transfer-20261015。普通 N607/CVS-RFFI Python/CPU2/GPU0，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_identity_excess_transfer_sourceL_20261015_r01`，命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_identity_excess_transfer_sourceL_20261015_r01/experiments/cvs_identity_excess_transfer20/runner.py /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_identity_excess_transfer_sourceL_20261015_r01/resolved_request.json`，不可覆盖输出 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-SIGTIME-identity-excess-transfer-sourceL-s2026092701-r01/source_diagnostic`，日志 `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_identity_excess_transfer_sourceL_20261015_r01/source_probe.stdout.log`。actualcode/config Gitcommit 写 release_commit.txt；一次 focusednegativecheck/唯一 independentP0P1review、一次 preflight/单归档 SHA比较/remotecompile、一次 PID/CWD/argv/GPU/loggrowth核对。权限/来源/非有限或技术错误保存产物并只停止ownedassay，先核对原状态再恢复，不触碰健康任务，不新增审批/签名/receipt。matrix/stop/artifacts 唯一登记见[experiment.json](experiment.json)。

跨类别借用 delta 已有[Delta-encoder](https://arxiv.org/abs/1806.04734)，relative risk 与本仓库 sharpness gap 思想相关；不能把这些概念本身称首创。这里待检验的组合是同day真实 RX差的跨身份功能迁移、真实同身份目标RX风险作为停止门槛、实际加性 joint 三端梯度和原默认头。跨 RX 射频指纹分离也已有[RIEI](https://arxiv.org/abs/2411.03636)；本任务不声称证明可分离条件。前述报告的集合监督、观测预测、PA重组后续完整负结果保留，不重新打开；结构/CENTER25K等历史正收益也按各自同row/预算保留，不能相加或按target最高挑父。


## 完整真实源结果

执行 `6d4f66c830cfb6ab471bf58e2cf45741ed08cc0c`；codecommit/push/remoteOID、归档SHA/remotecompile、两个realCPsmoke、PID/CWD/cmd/GPU/loggrowth与独立自然终态均 VERIFIED，无技术失败。wall 21.22 s，24cells、16paired、120RX记录。

| state | ordinal | CROSS active/3600 | SAME active/3600 | CROSS distinct/180 | SAME distinct/180 | CROSS auxF/main cos |
|---|---|---:|---:|---:|---:|---:|
| INITIAL | 2 | 1762 | 1800 | 178 | 149 | 0.995860 |
| INITIAL | 19 | 1778 | 1812 | 178 | 152 | 0.996684 |
| INITIAL | 37 | 1777 | 1792 | 175 | 145 | 0.996380 |
| INITIAL | 54 | 1778 | 1784 | 180 | 145 | 0.995631 |
| FINAL20 | 2 | 2328 | 2212 | 180 | 159 | 0.392784 |
| FINAL20 | 19 | 2359 | 2236 | 180 | 163 | 0.467907 |
| FINAL20 | 37 | 2398 | 2256 | 180 | 167 | 0.563323 |
| FINAL20 | 54 | 2322 | 2188 | 180 | 167 | 0.694862 |

| state | ordinal | CROSS−control | held positive-excess差 | rawvirtualCE差 | factualCE差 | transfer错误差 | originalACTUAL CE相对差% |
|---|---|---|---:|---:|---:|---:|---:|
| INITIAL | 2 | ZERO | -0.000103544 | +0.000152267 | +0.000181742 | +1 | +0.0071 |
| INITIAL | 2 | SAME_RX | -4.33486e-05 | -1.54981e-05 | -4.4335e-06 | +0 | -0.0001 |
| INITIAL | 19 | ZERO | -0.000214665 | +0.000286879 | +0.000380208 | +0 | +0.0147 |
| INITIAL | 19 | SAME_RX | -0.000132855 | +0.000310515 | +0.000377009 | +0 | +0.0143 |
| INITIAL | 37 | ZERO | -0.000477968 | +0.000133429 | +0.000310084 | +1 | +0.0128 |
| INITIAL | 37 | SAME_RX | -0.000264057 | -0.000154326 | -4.91667e-05 | -3 | -0.0017 |
| INITIAL | 54 | ZERO | -0.000165107 | +0.000600437 | +0.000702189 | +0 | +0.0262 |
| INITIAL | 54 | SAME_RX | -7.19316e-05 | +0.000245411 | +0.00028756 | +0 | +0.0109 |
| FINAL20 | 2 | ZERO | -0.000152432 | -0.000153395 | +0.000111118 | -1 | +1.3018 |
| FINAL20 | 2 | SAME_RX | -9.49493e-05 | -9.59227e-05 | +2.67889e-05 | +0 | +0.3107 |
| FINAL20 | 19 | ZERO | -0.000117521 | -0.000120685 | -7.65178e-06 | +0 | -0.0912 |
| FINAL20 | 19 | SAME_RX | -0.000122532 | -0.000125427 | -3.71743e-05 | +0 | -0.4397 |
| FINAL20 | 37 | ZERO | -3.19826e-05 | -3.30895e-05 | -1.56098e-05 | +0 | -0.1838 |
| FINAL20 | 37 | SAME_RX | -0.00014244 | -0.000146715 | -7.89685e-06 | +0 | -0.1026 |
| FINAL20 | 54 | ZERO | -8.21212e-05 | -8.30686e-05 | +8.84381e-05 | -1 | +1.0193 |
| FINAL20 | 54 | SAME_RX | -0.000109509 | -0.000110733 | -1.46505e-05 | +0 | -0.1699 |

固定资格：{"passed": false, "stage_signal": null, "qualified_ordinals_by_state": {"INITIAL": [], "FINAL20": [19]}, "cell_conditions": [{"state": "INITIAL", "ordinal": 2, "qualified": false, "failed_conditions": ["near_plain_CE_gradient_amplification", "ZERO:no_held_transfer_CE_difference_gain", "ZERO:transfer_error_harm", "SAME_RX:no_held_positive_excess_risk_difference_gain", "SAME_RX:no_held_transfer_CE_difference_gain"]}, {"state": "INITIAL", "ordinal": 19, "qualified": false, "failed_conditions": ["near_plain_CE_gradient_amplification", "ZERO:no_held_transfer_CE_difference_gain", "SAME_RX:no_held_transfer_CE_difference_gain"]}, {"state": "INITIAL", "ordinal": 37, "qualified": false, "failed_conditions": ["near_plain_CE_gradient_amplification", "ZERO:no_held_transfer_CE_difference_gain", "ZERO:transfer_error_harm"]}, {"state": "INITIAL", "ordinal": 54, "qualified": false, "failed_conditions": ["near_plain_CE_gradient_amplification", "ZERO:no_held_transfer_CE_difference_gain", "SAME_RX:no_held_positive_excess_risk_difference_gain", "SAME_RX:no_held_transfer_CE_difference_gain"]}, {"state": "FINAL20", "ordinal": 2, "qualified": false, "failed_conditions": ["ZERO:ACTUAL_relative_CE_harm", "ZERO:PRE_W_relative_CE_harm", "SAME_RX:no_held_positive_excess_risk_difference_gain", "SAME_RX:no_held_transfer_CE_difference_gain"]}, {"state": "FINAL20", "ordinal": 19, "qualified": true, "failed_conditions": []}, {"state": "FINAL20", "ordinal": 37, "qualified": false, "failed_conditions": ["ZERO:no_held_positive_excess_risk_difference_gain", "ZERO:no_held_transfer_CE_difference_gain"]}, {"state": "FINAL20", "ordinal": 54, "qualified": false, "failed_conditions": ["ZERO:no_held_positive_excess_risk_difference_gain", "ZERO:no_held_transfer_CE_difference_gain", "ZERO:ACTUAL_relative_CE_harm", "ZERO:PRE_W_relative_CE_harm"]}], "controls_not_fallback": true, "interpretation": "Local source excess-risk and identity gain with distinct gradient, not fulltraining or unseenRX/DG"}

INITIAL 四组 CROSS 有 1762 至 1778/3600 active、175 至 180/180 origin，但 auxF/main cos .995631 至 .996684 仍近共线，全部超过预登记 .95。相对 ZERO，positive excess 四组都下降（1.035e-4 至4.780e-4），rawvirtualCE 四组却全部增加（1.334e-4 至6.004e-4），factualCE 四组也增加（1.817e-4 至7.022e-4），两组 transfer错数多1；原ACTUAL CE也是四组略差，accuracy保持共用main一步23.8889%。由此实证区分**风险差变小**与**身份风险本身变小**。factual门槛detach只阻止本项直接通过该门槛梯度逃避；同一参数参与多种virtual角色，factual预测仍会随实际更新变化，不能把detach当作健康保证。不能宣称threshold上升完全解释所有positive-excess变化，因为该指标还包含逐view正部与active集合变化；rawvirtualCE/原CE结果已足以拒绝这组训练假设。

FINAL20 CROSS 有 2322 至2398/3600 active（64.5%至66.61%），180/180origin全部覆盖；SAME有2188至2256active，159至167origins。与此前 absoluteπ.9 的316至462active及81至107origins相比，约束规模明显扩大；但它是相同900L的新目标复用，不是五倍独立数据。CROSS auxF/main cos .392784至.694862，实际F71/浅中层18 tensors均有区别，auxnorm .081309至.116378，auxW仍None，三个gain/branch实际梯度均记录。source current约束数量与不同方向已成立，但独立held身份效用仍不稳定。

FINAL heldvirtualCE和positive-excess相对ZERO/SAME共8个比较全下降，virtualCE下降3.309e-5至1.534e-4，positive-excess下降3.198e-5至1.524e-4，transfer错误无增加（vsZERO两组少1，vsSAME全部相同）。原ACTUAL/PRE_W准确率始终100%。原CE相对ZERO只有ordinal19/37改善，2/54变差，ACTUAL相对变差1.3018%/1.0193%，超过固定1%健康范围；对SAME，ordinal2仍略差，其余三组改善。只有ordinal19同时满足全部条件；ordinal2的SAME gain不足1e-4，ordinal37/54的ZERO gain不足1e-4。不凑组、不把接近阈值判通过，INITIAL0/4、FINAL1/4仍未达到2。

两个本轮候选都关闭固定loss，不扫描π/λ/LR/容差/参照/partialstrength，不扩到完整训练；这是科学失败，不是资源或技术blocker。保留真实异TX参照产生的广覆盖、晚期不同梯度与8个transferCE正比较，但未确认对原默认身份判别的稳定收益，更没有正式20epoch或未见RX测试。此前同TX-PA三seed负结果、mean/ROW/BILATERAL/梯度过滤与完整sharpness负结果继续保留，不用旧单组正结果充当成功候选。

目前结果更支持先解释接收变化所携带的信号：raw三端点差分可能同时传递TX相关非线性和packetnoise，源结果尚未证明其跨TX身份保持假设。该解释是工作假设；没有直接分离噪声、类相关RX交互的因果证据。下一轮应先利用已保存 referenceTX/targetTX/RX 风险分布提出**新的监督内容假设**，不能继续仅扩active量或给当前loss换尺度、换停止门槛。暂不预先宣布新的有效创新路线，也不另训诊断分类头。

原报告强调的“足够覆盖≠有效更新≠身份判别改善≠泛化”在本次全部可分辨：相对风险覆盖扩大并改变晚期更新方向，但原CE两好两差；早期primary gap下降而真正身份风险变差。后续资格仍须分别保留primary任务与原default身份判别，不能仅以自设任务下降发布主训练。

历史结构/CENTER25K正收益按其同row/预算继续保留，尤其固定CENTER25K三seed约+.2425pp且三seed均正；SIG_TIME与COMPACT_LINEAR/LONG_SILU的已测结果不能拼到不同配方或作为这次loss收益，SIG_TIME历史CUDA初始化confound继续披露。既定SIG_TIME+CENTER25K父未改变，没有按target最高结果重选父。84%默认头clean三seed目标active，尚未达成。无query反馈调整候选、seed、父或训练日程。

900uniqueL/6300metadata/6124packetforwards/48临时CPUAdamW/60loss calls，各FP64/FP32 864000解析判别；0真实训练/头拟合/EMA/U/V/target。原live模型/optimizer/CP/grad/RNG不变。完整[24单元](source_cells.csv)、[120RX](source_RX_metrics.csv)、[16对照](paired_regimes.csv)、[源证据](result_evidence.json)、[运行核验](launch_evidence.json)。本source-only assay已经完整分析，但最终识别目标没有完成。
