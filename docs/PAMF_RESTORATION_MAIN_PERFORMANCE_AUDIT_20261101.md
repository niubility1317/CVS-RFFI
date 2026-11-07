# 星地恢复器 Main 性能核对

截至 2026-11-02，固定 DURA v1.0 为 1/6 景达标；本表 17 个已完成新候选中 0 个达到六景逐个距 raw clean ≤3 个百分点。完整验收还要求两个总体分别保持 clean。性能是主线，训练效率是支线；源域拟合、参数量和训练结束均不能替代正式验收。

所有数值来自已完成独立评分的公开同 row 结果。表格按研发顺序排列，不按测试分数选择候选，不合成不同模型的最佳单项，也不更换默认 DURA。原地面完整 Tri-SIG-TIME SSL 的同一冻结默认头为基座，Main 包含跨接收机 126000 条和跨接收机且跨日期 42000 条。每个新 G 固定最终轮次，七景全部预测完成后独立 truth-last。主参照 raw clean 为 86.538690%（145385/168000），故每景至少需要 140345 条正确预测，即 83.538690%。

|恢复方式|clean|高档郊区|中档郊区|低档郊区|高档城区|中档城区|低档城区|达标景数|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Raw received|86.5387|80.6155|78.1077|75.7446|77.0500|62.7958|51.6048|0/6|
|DURA v1.0（固定默认）|86.6155|84.3435|83.0161|81.3220|83.0494|75.7060|68.8012|1/6|
|MSAT-M20|86.4268|77.3893|75.8417|74.1494|76.3244|70.5042|65.0554|0/6|
|VarF|86.5851|77.4929|75.9690|74.1036|76.4738|70.7976|65.2518|0/6|
|PHY-MSAT|87.2161|79.9839|76.9726|74.0655|78.2512|69.6583|62.8315|0/6|
|FullFGrad|86.4845|78.3333|76.9643|75.4976|77.1988|71.4089|65.8494|0/6|
|FeatureCond|86.6190|77.9065|76.5119|74.8804|76.6440|70.3429|64.3726|0/6|
|FeatureStrong|86.8226|75.9792|75.0958|74.0399|74.9506|69.6185|63.9994|0/6|
|PowerAnchor|86.6446|78.9905|77.7149|76.1667|77.7994|71.5339|65.4339|0/6|
|DirectBridge|86.6036|77.6637|76.3911|74.9036|76.4595|70.2869|64.2065|0/6|
|StateBridge|86.1071|71.7530|70.9423|69.8857|70.6917|65.3214|59.6351|0/6|
|RolloutBridge|86.5750|75.9601|73.9298|71.8756|74.4619|67.4702|61.5494|0/6|
|SoftPower|86.5351|78.9137|77.5744|76.1042|77.7054|71.5756|65.5268|0/6|
|TF ResidualDiffusion|85.9137|73.6214|71.0083|68.4631|70.5935|58.0036|48.6881|0/6|
|Diffusion Early4（冻结推理）|86.8768|79.1982|77.9565|76.4214|77.0315|67.4256|58.4810|0/6|
|Diffusion SigmaMix|86.7185|73.2601|71.3012|69.2732|71.6690|63.9185|56.5030|0/6|
|CovarianceMAP|86.5726|83.5006|82.1393|80.5393|82.1833|75.0476|67.1982|0/6|
|CovarianceFeature|86.7429|83.9720|82.4649|80.5810|82.6083|74.8952|66.9625|1/6|
|CovarianceDetail|86.6357|80.1405|77.8077|75.4952|78.2137|69.4470|62.4012|0/6|

准确率单位为 %。完整可复算值和正确数见同名 JSON/CSV。

## clean 需分别保持两个总体

下表是相对原 raw clean 的正确数净变化。合并 clean 增加仍可能掩盖某一总体损伤；clean 保持与六个受损场景达标分别判断。

|新恢复器|跨接收机净正确数|跨接收机且跨日期净正确数|两个总体均保持|
|---|---:|---:|---|
|MSAT-M20|-111|-77|否|
|VarF|+79|-1|否|
|PHY-MSAT|+998|+140|是|
|FullFGrad|-75|-16|否|
|FeatureCond|+68|+67|是|
|FeatureStrong|+515|-38|否|
|PowerAnchor|+176|+2|是|
|DirectBridge|+163|-54|否|
|StateBridge|-551|-174|否|
|RolloutBridge|+41|+20|是|
|SoftPower|+22|-28|否|
|TF ResidualDiffusion|-821|-229|否|
|Diffusion Early4（冻结推理）|+635|-67|否|
|Diffusion SigmaMix|+293|+9|是|
|CovarianceMAP|+82|-25|否|
|CovarianceFeature|+377|-34|否|
|CovarianceDetail|+151|+12|是|

## “恢复成 clean 的几成”需区分三个问题

1. **识别准确率保留比例**可直接计算，为受损场景恢复后的正确数除以原 raw clean 正确数。这衡量同一冻结识别器的任务表现，不是波形信息恢复率。
2. **恢复 IQ 与同条 clean 的接近程度**需用配对 IQ 误差、频谱误差及特征误差描述。这些指标不保证 TX 识别收益，不能由准确率反推出 IQ 恢复比例。
3. **单次观测的理论极限**取决于信道、噪声、先验及可辨识性，现有有限模型测试无法给出可信的单一百分比。源端 true-H、clean-feature oracle 使用额外信息，有限求解器也不构成严格上界。

低档城区 Raw received 为 51.604762%，相当于原 clean 准确率的 59.632012%；距 3pp 验收还缺 53649 条正确预测。

低档城区 DURA v1.0（固定默认） 为 68.801190%，相当于原 clean 准确率的 79.503388%；距 3pp 验收还缺 24759 条正确预测。

恢复器也可能输出比某条原 clean 更利于 F0 判别的 IQ，因此原 F0 的 clean 准确率不是所有恢复方法的数学上限。是否突出发射机特质必须核对真实 TX 的净纠错、clean 损伤和逐 RX/day/TX 表现；只提高 F0 自信或模仿其 clean 判决不能证明身份特征真实增强。[理论边界说明](../automation_reports/CV-SincNet/20261101-phase1-Restore-FeatureCond-E30-s2026092701-r01/theoretical_limits.md)区分了条件均值误差、可逆性、任务准确率与信息界。

## 当前证据支持的判断

- 更强源端 F 特征拟合没有稳定转化为跨接收机收益。FeatureStrong 的单因素源 latent 权重对照、完整源 V 冻结验证与 Main 结果需一起解读，不能仅归因为训练包记忆或梯度冲突。
- 功率约束确实消除了既有投影前尺度自由度，但这项机制修复仍未达到正式性能目标。
- RolloutBridge 真实四步训练相对 StateBridge 六景均增加正确预测，但六景仍全部低于固定 DURA，0/6 达标。四步训练消除了既定训练/部署状态构造差异，却没有独立解决跨 RX 泛化；不能将源四步拟合改善视作恢复任务已经接近 clean。
- SoftPower 保留同 E0/网络/源数据/优化器/软 RMS，只删除硬功率投影，六景相对 PowerAnchor 为4降2升，clean 合并净少6条，跨接收机且跨日期净少28条；源 IQ NMSE 更低并未带来整体主性能收益。这不是“放开幅度一定有效”的证据，也不能据此认定原本硬投影在所有任务最优。
- 时频联合残差扩散已完成从零30及完整15调用部署，六景0/6、两个clean总体均损伤（合并净少1050条），低档城区48.688095%，低于同F0的received直接识别51.604762%。这条整体新架构尚未产生性能收益；较短训练不能抵消主性能退化。
- 同权重的冻结第4读出（0新训练/7调用）使六景较完整八步提高5.576786至9.792857个百分点，仍全部低于DURA、0/6达标。合并clean净+568，但跨接收机且跨日期净−67；这证明该固定模型的后段采样确实降低Main表现，并未达到最终目标。
- SigmaMix 仅改变训练噪声尺度分布，两个clean总体分别净+293、+9，低档城区较原完整扩散提高7.814881个百分点；六景仍0/6、全部较DURA低11.083333至12.298214个百分点。大噪声曝光增加解决了部分覆盖问题，但不足以支持继续扩大这条扩散路线。
- CovarianceMAP 用时频Transformer估计物理Action/噪声，并以全512维源IQ协方差求后验；G更新不执行F0或后验。六景较received直接识别均有收益，但仍全部低于DURA 0.658333至1.602976个百分点、0/6达标；clean合并净+57、跨日期净−25。高档郊区距raw clean3.038095个百分点也未达标，不能放宽3pp或选择性重跑。
- CovarianceFeature与物理估计对照保持同实际E0、原source mu/C、数据顺序/draw/LR、单阶段30/7410和部署路径，新增实际后验IQ的冻结F0连续特征指导。六景1/6达标、仍全部低于DURA；相对物理对照四个较温和场景提高0.041667至0.471429个百分点，中/低档城区下降0.152381/0.235714个百分点，后者同条净少396条正确。clean合并净+343，cross_rx净+377而cross_both净−34。训练源特征误差下降不能认定域差异是唯一起因，还需同源同病例冻结对照区分目标作用和泛化。
- CovarianceDetail将物理估计与4250882参数IQ细节分支分开，全部63000条源记录每轮参与后验与连续F0指导；scratch单阶段30/7410、5481223总参数。两个clean总体净+151/+12，但六景0/6，相对DURA降低4.202976至6.400000个百分点。冻结源V低档城区84.222222%→91.000000%，IQ/特征误差下降且TX类间/类内比提高，Main却低于纯物理版；这反驳“源特征更可分就保证跨RX更好”，不能单独归因为某一种域因素或认定细节分支本身必然无效。
- 冻结源四步轨迹显示，DirectBridge 低档城区首末步 Fclean 一致率均为 84.749304%；StateBridge 则从 86.350975% 降到 81.615599%，净损伤 68/1436，IQ NMSE 从 .07612306 增至 .11645158。后续迭代的退化已在训练源面板出现，不能全部归为 Main 域差异。训练用 clean/received 插值，而部署使用网络自行生成的状态，二者不一致是可检验假设，尚不独立证明原因。[完整轨迹诊断](../automation_reports/CV-SincNet/20261101-diagnostic-Restore-bridge-trajectory-source-s2026092701-r01/analysis.md)保留全部 96 组和配对。DirectBridge 的代数捷径仍只是一种可能限制，不能认定实际网络使用了精确解析式。
- 所有新 G 都在原 L_s/U_s 范围训练。包含冻结源 F0 连续特征指导的行明确不是 F-free；CovarianceMAP 的 G 更新只回归源物理量、无 F0 梯度，但方法设计参考过冻结源 F0 诊断，知识边界单列。TX 真标签、V/OE/novel/目标不进入 G 的更新、早停或选模。OE 仅用于冻结诊断。
- 当前仅一个模型 seed，且该基准已有历史曝光。这些是固定矩阵的研究结果，不宣称全新盲测或已知恢复极限。下一方案仍须依据合法源端机制证据预先固定，不能借已揭示的 Main 错误逐条优化或按场景切换模型。

## 可追溯来源

- [MSAT-M20 完整分析](../automation_reports/CV-SincNet/20261101-phase1-Restore-MSAT-M20-F0-E30-s2026092701-r01/analysis.md)，实际代码 `8720a8584803366d42d2879fb99502b8d220ac26`。
- [VarF 完整分析](../automation_reports/CV-SincNet/20261101-phase1-Restore-MSAT-VarF-E30-s2026092701-r03/analysis.md)，实际代码 `9fdb1f5b164484f2bc438cb1e45d1ee805d2473a`。
- [PHY-MSAT 完整分析](../automation_reports/CV-SincNet/20261101-phase1-Restore-PHYMSAT-E30-s2026092701-r01/analysis.md)，实际代码 `8198a2e70c655f24ebaf321f232a676cfb46d916`。
- [FullFGrad 完整分析](../automation_reports/CV-SincNet/20261101-phase1-Restore-MSAT-FullFGrad-E30-s2026092701-r01/analysis.md)，实际代码 `940fa78ba450b686b1b03c7530c47af4360faafc`。
- [FeatureCond 完整分析](../automation_reports/CV-SincNet/20261101-phase1-Restore-FeatureCond-E30-s2026092701-r01/analysis.md)，实际代码 `c8659b15bb3d316325867b154eede51f1a61e247`。
- [FeatureStrong 完整分析](../automation_reports/CV-SincNet/20261101-phase1-Restore-FeatureStrong-E30-s2026092701-r01/analysis.md)，实际代码 `3e5416df5b13dfcf9e16995988e988c5494f68b2`。
- [PowerAnchor 完整分析](../automation_reports/CV-SincNet/20261101-phase1-Restore-PowerAnchor-E30-s2026092701-r01/analysis.md)，实际代码 `42ea957db57d7d90df67dadbeffe11fc014909a0`。
- [DirectBridge 完整分析](../automation_reports/CV-SincNet/20261101-phase1-Restore-DirectBridge-E30-s2026092701-r01/analysis.md)，实际代码 `84a063c9ba1f9974cfe361e293a9e11fa93920b4`。
- [StateBridge 完整分析](../automation_reports/CV-SincNet/20261101-phase1-Restore-StateBridge-E30-s2026092701-r01/analysis.md)，实际代码 `c50a9159b1fb891c6f1a6f8ad4bfa2d7eb66222e`。
- [RolloutBridge 完整分析](../automation_reports/CV-SincNet/20261101-phase1-Restore-RolloutBridge-E30-s2026092701-r01/analysis.md)，实际代码 `7250d0a88160d6ab949dafbdc73edb7f302e290f`。
- [SoftPower 完整分析](../automation_reports/CV-SincNet/20261101-phase1-Restore-SoftPower-E30-s2026092701-r01/analysis.md)，实际代码 `1dd40c07970b664f18c57149a0a2ab02f000b114`。
- [TF ResidualDiffusion 完整分析](../automation_reports/CV-SincNet/20261102-phase1-Restore-TFResidualDiffusion-E30-s2026092701-r01/analysis.md)，实际代码 `5e5921c975678367a5d0a00ee72d4619e5f6927e`。
- [Diffusion Early4（冻结推理） 完整分析](../automation_reports/CV-SincNet/20261102-evaluation-Restore-DiffusionEarly4-main-s2026092701-r01/analysis.md)，实际代码 `3e5ecb4ffcc5658693bd9b6dc509711b4509a231`。
- [Diffusion SigmaMix 完整分析](../automation_reports/CV-SincNet/20261102-phase1-Restore-DiffusionSigmaMix-E30-s2026092701-r01/analysis.md)，实际代码 `3e5ecb4ffcc5658693bd9b6dc509711b4509a231`。
- [CovarianceMAP 完整分析](../automation_reports/CV-SincNet/20261102-phase1-Restore-CovarianceMAP-E30-s2026092701-r01/analysis.md)，实际代码 `bf29554a03e28a012eaaa134f5c7b3f7856f2268`。
- [CovarianceFeature 完整分析](../automation_reports/CV-SincNet/20261102-phase1-Restore-CovarianceFeature-E30-s2026092701-r01/analysis.md)，实际代码 `5359a1b82b1fa5d640a394c3e3a77ce3b945fe1e`。
- [CovarianceDetail 完整分析](../automation_reports/CV-SincNet/20261102-phase1-Restore-CovarianceDetail-E30-s2026092701-r01/analysis.md)，实际代码 `1fcbc3eb3cc16e7577ecc3cbedda99174b4c5cc6`。

本核对只复用公开汇总，新增神经前向、原始训练/目标数据读取、真值读取均为 0。完整目标保持 ACTIVE，默认 DURA 不变。


## 大幅改造与源端机制证据

独立的[时频联合 Transformer 条件残差扩散完整结果](../automation_reports/CV-SincNet/20261102-phase1-Restore-TFResidualDiffusion-E30-s2026092701-r01/analysis.md)已纳入上表。1614608参数，从零单阶段30/7410；训练1次denoiser/记录，部署固定8步Heun共15次，DURA结构和历史G均无继承。它未达到主性能门槛，完整84单元/2604组/所有配对复算PASS，raw/DURA判决差0；训练loss不与Main准确率混作同一指标。

[冻结源采样轨迹](../automation_reports/CV-SincNet/20261102-diagnostic-Restore-diffusion-source-path-s2026092701-r01/analysis.md)完成92160行/108组/所有配对复算，误差0。整体第4内部读出Fclean一致90.869141%→末步87.294922%，低档城区69.428969%→62.465181%，对应同条判决净减少366/10240和100/1436；它是源内Fclean参考，不是TX准确率或Main成绩。源端已经出现后半段采样损伤，训练单步加噪残差误差不能代替真实完整sampler质量。

基于该源诊断固定的[冻结第4读出完整结果](../automation_reports/CV-SincNet/20261102-evaluation-Restore-DiffusionEarly4-main-s2026092701-r01/analysis.md)已纳入上表；它复用原source-only scratch30权重，0新训练/7核心调用，祖先G30/F0 E50和源尺度知识仍计成本，不能冒充另一个新从零模型。[训练sigma均衡完整结果](../automation_reports/CV-SincNet/20261102-phase1-Restore-DiffusionSigmaMix-E30-s2026092701-r01/analysis.md)也已纳入上表，全部参数从零单阶段30，50%原lognormal加50%均匀8部署点，原15调用部署保持。两项均完整七景、0/6达标，不按目标分组选择或组合方案。SigmaMix全部84单元/2604组/60轮训练日志独立复核通过，raw/DURA缓存判决差0。

[已有源RX/day特征几何诊断](../automation_reports/CV-SincNet/20261101-diagnostic-Restore-Fgeometry-RXday-s2026092701-r01/analysis.md)提供了另一项反证：接收机均值相关变异占比从原始特征的0.890638%降至源方差加权后的0.612513%。这些clean特征统计不支持“方差加权放大接收机均值”这一具体解释，不据此推出删均值或投影方案；它也不能排除协方差和非线性域差异。

[源/主信道静态设置核对](PAMF_RESTORATION_SOURCE_MAIN_CONFIG_20261102.md)未发现三个核心模块的定义漂移。记录的载频差异及union中历史复制观测的来源边界单列，不能仅凭此项解释性能差距或重建数据。

[完整C真实信道源诊断](../automation_reports/CV-SincNet/20261102-diagnostic-Restore-source-covariance-inverse-s2026092701-r01/analysis.md)的原定支持条件满足后，CovarianceMAP已从零单阶段30并完整Main验证，纳入上表。[冻结四路参数归因](../automation_reports/CV-SincNet/20261102-diagnostic-Restore-covariance-parameter-source-s2026092701-r01/analysis.md)进一步表明：低档城区盲Fclean82.242340%，仅换真实Action87.325905%，仅换缓存真实噪声方差标量82.660167%；同条净变化分别+73/1436、+6/1436。估计Action值得优先研究，但这是源内Fclean参考，不是Main已证明的原因或严格恢复上限；本版真实盲源表现和Main均未超过DURA。

[RF细节敏感性证据](PAMF_RESTORATION_RF_DETAIL_ANALYSIS_20261101.md)来自历史R17固定源面板，说明部分可观察硬件方向和随机方向均被衰减。它不证明发射机真特征的逐样本恢复上限，也不能继承为新扩散模型证据。扩散多尺度训练是待检验假设；单条观测中已丢失的细节可能被先验合成，最终仍需同一冻结F0的真实TX净纠错及clean损伤，未来未知设备保真另作冻结诊断。


## 原 R17 IQ 教师路线的历史边界

原 [R17 最终 IQ 源缓存](../automation_reports/CV-SincNet/20261031-source-DURA-R17-nativeIQ-cache-s2026092701-r01/analysis.md)已有完成证据：原 R07 63000 个 L_s/U_s 物理记录×20 份源扰动，共1260000次教师输入，缓存生成972.193282秒。原教师4665073参数，历史 coarse/action/prior 各20轮、共60组件轮次，目标访问前曾按源V/F选择；缓存生产本身没有训练、F/TX/V/目标读取或状态更新。已有一次完整 ID/角色/shape/dtype/finiteness 验证沿用，本次只读公开报告，没有重新加载教师或重验数据。

后续 [nativeIQhint 从零单阶段30轮对照](../automation_reports/CV-SincNet/20261031-phase1-DURA-nativeIQhint-E30-s2026092701-r01/analysis.md)也已完成，使用冻结 R17 最终 IQ 缓存指导紧凑266951参数学生，不复制教师权重。旧 F50/defaulthead 的 low urban 为67.375595%，相对匹配E30 physical提高0.668452个百分点，相对当时DURA低1.514286个百分点；六星地0/6超过当时DURA。教师IQ还带有自身误差，不能等同clean或认为权重更大必然更好。

这些是旧F50证据，**不纳入本次历史检索时原完整Tri-SIG-TIME SSL/Main的15个已完成候选计数，也不改标成本基座成绩**。source-only来源合规与方法有效性是两个问题；历史教师、源V/F选择、缓存和学生成本均须披露。CovarianceFeature本次固定scratch30配方，只增加原Tri-SIG-TIME SSL对实际恢复IQ的连续源特征指导，没有R17教师输入，也没有因这次历史检索更改训练或新增候选。本次历史检索时该候选尚无完整Main成绩，没有提前纳入完成表。


## 原始论文的条件核对

[五篇原始论文核查](PAMF_RESTORATION_LITERATURE_CONDITIONS_20261102.md)区分编码特征分类、解码IQ接固定F0，以及需要目标接收机配对校准的不同问题。PSC-DAE使用TX监督，DRIFT训练新的TX/RX识别表示，TCNN采集目标校准响应，LTE-V2X使用完整协议参考结构；均不能直接继承为本任务的成绩或六景恢复上限。CDP理论的固定/最优分类器设定也不等价于当前六类未知信道数值上界。核查没有新增G/F/IQ/目标预测/真值读取或训练，没有更改CovarianceFeature固定配方。


## 实际后验源特征指导与训练计算优化

[CovarianceFeature完整结果](../automation_reports/CV-SincNet/20261102-phase1-Restore-CovarianceFeature-E30-s2026092701-r01/analysis.md)已纳入上表，42单元/1302组及raw、DURA、固定物理对照和clean的全部同条配对独立复算，最大差1.24e-14、baseline判决差0。原模型全部权重从零、原LsUs和单阶段30保持；该阶段16个已完成候选仍没有达到完整六景及两个clean保持要求，不将1/6或合并clean上涨当成目标完成。

[冻结源端计时](../automation_reports/CV-SincNet/20261102-diagnostic-Restore-factorized-source-cost-s2026092701-r01/analysis.md)验证另一条计算路径在同权重、同源批次的loss及全部参数梯度一致，0optimizer/新训练。前向-反向-clip区域中位wall382.360001→353.696042ms（减少7.496589%），10对7快3慢，allocated峰值减少4.008942%；不包含optimizer、全量IO、epoch日志，也不代表完整30轮同质量加速。它没有改变当前5359的训练/Main路径或产生另一个可计数的新恢复模型。默认DURA保持，目标ACTIVE。


## 独立物理与 IQ 细节网络的完整负结果

[CovarianceDetail 完整 Main](../automation_reports/CV-SincNet/20261102-phase1-Restore-CovarianceDetail-E30-s2026092701-r01/analysis.md)已纳入上表，当前17个完成候选仍无完整目标达成者。5481223参数、scratch单阶段30/7410；物理分支只受Action/noise监督，实际后验与lognoise条件detach，IQ/多分辨率频谱/原F0连续特征只指导独立细节分支。完整1890000次源输出/F0曝光，0旧G初始化、TX/logits/CE/原型/新身份头或目标训练。结构、梯度路由及完整曝光共同改变；物理E0同BF，但实际训练轨迹有数值差异，不能作单一组件因果归因。

Main六景为80.140476/77.807738/75.495238/78.213690/69.447024/62.401190%，0/6；两个clean分别净+151/+12。低档城区比DURA低6.4个百分点，距raw clean24.1375个百分点，距3pp线还缺35511条正确预测。该场景准确率相当于raw clean的72.107852%，只表示任务准确率保留，不能称为信息或波形恢复上限。

[冻结源 V](../automation_reports/CV-SincNet/20261102-diagnostic-Restore-CovarianceDetail-sourceV-s2026092701-r01/source_analysis.md)只新增最终G/F0的12600条输出，原1800条clean参考和三种控制逐值复用，所有28单元/420分组/105条件组在预测固定后独立真实TX评分。低档城区纯物理84.222222%→新版91.000000%，165修复/43损伤；IQ NMSE 0.069798→0.041880，F余弦距离0.035036→0.013077。同固定F空间的源TX类间/类内比2.092424→4.209074，RX×day格内比2.511658→5.657282。这些是源V表征与真实识别收益证据，不是跨RX迁移、硬件成分真实性或未知设备保真证明；源V本身低档城区也仍比源raw clean低7.611111个百分点。

[同原F0实测延迟](../automation_reports/CV-SincNet/20261102-diagnostic-Restore-CovarianceDetail-final-latency-s2026092701-r01/analysis.md)全部600次计时/30预热/5source2路径核对；B1新G+F 22.380381ms，对照DURA G+F 12.734163ms；B128分别189.174147/22.029280ms。训练耗时115.850539分钟。单阶段没有带来本版同质量提速；静态MAC不能替代真实延迟。共享GPU、FP64后验、预热与驻留显存边界详见该独立报告，不沿用旧F50组合时延。

本版42个Main单元/1302组及全部raw、DURA、物理、原特征指导、clean配对独立复算，baseline判决差0，最大差1.24e-14；完整30轮训练和实际最终权重另核对。三项实验均已结束并归档，默认DURA保持。源端拟合改善与Main退化同时存在，下一步机制假设须继续在合法源端建立证据，不能根据Main分组错例调参、选模或按场景组合。本段只复用已发布汇总，0新NN、原始IQ或真值读取。


## 新版接收机均值偏差的源端检验

[同源V冻结特征分析](../automation_reports/CV-SincNet/20261102-diagnostic-Restore-CovarianceDetail-sourceV-s2026092701-r01/receiver_geometry_analysis.md)完成52200条保存特征、87个汇总及667个RX/day/RX×day组统计。低档城区新细节网络相对各自clean的RX均值误差0.000825112，较纯物理0.004257533低80.619941%；均值模式对齐增益0.952200（clean自身1），RX均值误差只占总特征误差0.575860%。这不支持“新版放大RX均值偏差”的具体解释，不据此删均值、投影或添加RX对抗目标。源V只有已见5RX/3day，不能排除未见RX的其他域因素；没有读取TX标签作条件因果控制。分析0新NN/IQ/TX/目标读取，5项针对性检查通过，分解恒等式误差4.44e-16；17候选计数和全部Main分数不变，目标ACTIVE。


## 新版的缓存 Action 一致性检验

[冻结源V缓存Action报告](../automation_reports/CV-SincNet/20261102-diagnostic-Restore-CovarianceDetail-sourceV-s2026092701-r01/measurement_consistency_analysis.md)完成840分组/168条件组，复用保存IQ与Action代理，0新NN/TX/目标/训练。低档城区1731/1800条新版IQ更接近clean，其中1247条观测残差反而增加；相对同条clean的残差能量绝对差距均值从0.344762降至0.232462（−32.573363%）。更小观测残差不等于更好去噪，这项源证据不支持直接加强带噪观测拟合。5项针对性检查通过，252个已公开IQ/Fclean分组差0。缓存Action不是完整硬件Jacobian、标量方差不是完整噪声协方差；没有以此解释全部Main差距或估计信息上限。17候选、各Main结果、默认DURA和完整主目标均保持。

## IQ 坐标诊断已完成

480 条固定源 V 记录覆盖七景和四个 I/Q 坐标方向。恢复输出映回原坐标后交给原 Tri-SIG-TIME SSL。零度 3,360 条 G/F0 判决与 480 条 clean F0 参考复用原产物，新增 10,080 条 G/F0 输出，另有 8 条真实源 L_s 冒烟输出。独立复算覆盖 28 单元与 672 分组，指标差与原缓存变化均为 0。

旋转 180° 后，clean 的 IQ NMSE 从 0.000028 增至 0.188923，低档城区从 0.034477 增至 3.715660。低档城区的原 clean 判决一致率从 91.875% 降到 75.000%。这是同一源波形在可逆坐标变换下的实测差异。判决一致率以 F0 在原 clean 上的输出为参照，真实 TX 未读取。Main 差距的成因与理论信息上界仍待独立证据。

此次仅新增源端证据。Main 已完成候选仍为 17 个。 [完整诊断](../automation_reports/CV-SincNet/20261102-diagnostic-Restore-CovarianceDetail-source-coordinate-s2026092701-r01/analysis.md)

## 2026-11-06 至 11-07 同期独立 Main 报告核对

复核 22 个已有完整分析及 ANALYZED 记录的独立地面测试报告。每份低档城区均为同一冻结 Tri-SIG-TIME SSL 和 168,000 条 Main 记录，全部预测后独立评分。用 DURA 的 115,586 条正确预测加同样本修复、减损伤，逐项重算候选正确数并核对报告准确率。22 个候选均未达到 140,345 条门槛。该组最好的一版准确率 68.442262%，比 DURA 少对 603 条，距门槛仍少 25,362 条。这些是同期独立研究，源训练权限和训练预算各有差异，单列于本表 17 个候选之外。

| 独立测试 | 低档城区准确率 | 相对 DURA 修复／损伤 | 距验收少对条数 |
|---|---:|---:|---:|
| [direct-dualdomain](../automation_reports/CV-SincNet/20261106-phase1-direct-dualdomain-ground-test-s2026092701-r01/analysis.md) | 38.016071% | 14,970／66,689 | 76,478 |
| [direct-spectral-distill](../automation_reports/CV-SincNet/20261106-phase1-direct-spectral-distill-ground-test-s2026092701-r01/analysis.md) | 36.699405% | 13,624／67,555 | 78,690 |
| [dura-band-guard](../automation_reports/CV-SincNet/20261106-phase1-dura-band-guard-ground-test-s2026092701-r01/analysis.md) | 68.378571% | 1,126／1,836 | 25,469 |
| [dura-crosshalf-iq-ssl](../automation_reports/CV-SincNet/20261106-phase1-dura-crosshalf-iq-ssl-ground-test-s2026092701-r01/analysis.md) | 68.413095% | 1,761／2,413 | 25,411 |
| [dura-crossview-guard](../automation_reports/CV-SincNet/20261106-phase1-dura-crossview-guard-ground-test-s2026092701-r01/analysis.md) | 68.360714% | 1,287／2,027 | 25,499 |
| [dura-expert-mixture](../automation_reports/CV-SincNet/20261106-phase1-dura-expert-mixture-ground-test-s2026092701-r01/analysis.md) | 67.224405% | 3,702／6,351 | 27,408 |
| [dura-fixed-iq-ssl](../automation_reports/CV-SincNet/20261106-phase1-dura-fixed-iq-ssl-ground-test-s2026092701-r01/analysis.md) | 68.278571% | 1,761／2,639 | 25,637 |
| [dura-internal-guard](../automation_reports/CV-SincNet/20261106-phase1-dura-internal-guard-ground-test-s2026092701-r02/analysis.md) | 67.867857% | 5,824／7,392 | 26,327 |
| [dura-iq-geometry](../automation_reports/CV-SincNet/20261106-phase1-dura-iq-geometry-ground-test-s2026092701-r01/analysis.md) | 68.343452% | 1,977／2,746 | 25,528 |
| [dura-joint-iq-ssl](../automation_reports/CV-SincNet/20261106-phase1-dura-joint-iq-ssl-ground-test-s2026092701-r01/analysis.md) | 68.226190% | 2,077／3,043 | 25,725 |
| [dura-moment-relation](../automation_reports/CV-SincNet/20261106-phase1-dura-moment-relation-ground-test-s2026092701-r01/analysis.md) | 68.341071% | 1,984／2,757 | 25,532 |
| [dura-multiscale-spectral](../automation_reports/CV-SincNet/20261106-phase1-dura-multiscale-spectral-ground-test-s2026092701-r01/analysis.md) | 68.307143% | 1,827／2,657 | 25,589 |
| [dura-noise-gated](../automation_reports/CV-SincNet/20261106-phase1-dura-noise-gated-ground-test-s2026092701-r01/analysis.md) | 68.363690% | 1,957／2,692 | 25,494 |
| [dura-nonlinear-guard](../automation_reports/CV-SincNet/20261106-phase1-dura-nonlinear-guard-ground-test-s2026092701-r01/analysis.md) | 68.346429% | 1,146／1,910 | 25,523 |
| [dura-orthogonal](../automation_reports/CV-SincNet/20261106-phase1-dura-orthogonal-ground-test-s2026092701-r01/analysis.md) | 68.340476% | 1,335／2,109 | 25,533 |
| [dura-orthogonal-guard](../automation_reports/CV-SincNet/20261106-phase1-dura-orthogonal-guard-ground-test-s2026092701-r01/analysis.md) | 68.348810% | 1,140／1,900 | 25,519 |
| [dura-posterior-teacher](../automation_reports/CV-SincNet/20261106-phase1-dura-posterior-teacher-ground-test-s2026092701-r01/analysis.md) | 68.400595% | 1,344／2,017 | 25,432 |
| [dura-sim-action](../automation_reports/CV-SincNet/20261106-phase1-dura-sim-action-ground-test-s2026092701-r01/analysis.md) | 67.979167% | 5,895／7,276 | 26,140 |
| [dura-tail-excess](../automation_reports/CV-SincNet/20261106-phase1-dura-tail-excess-ground-test-s2026092701-r01/analysis.md) | 68.438690% | 1,390／1,999 | 25,368 |
| [dura-timevarying](../automation_reports/CV-SincNet/20261106-phase1-dura-timevarying-ground-test-s2026092701-r01/analysis.md) | 68.442262% | 1,577／2,180 | 25,362 |
| [dura-wide-source](../automation_reports/CV-SincNet/20261106-phase1-dura-wide-source-ground-test-s2026092701-r01/analysis.md) | 67.350000% | 3,462／5,900 | 27,197 |
| [dura-expert-distill](../automation_reports/CV-SincNet/20261107-phase1-dura-expert-distill-ground-test-s2026092701-r01/analysis.md) | 67.501190% | 3,310／5,494 | 26,943 |

## 固定配对的纠错覆盖上限

固定同一场景和一个候选，设 DURA 正确数为 D，候选修复 DURA 错误数为 R，损伤数为 H。候选正确数为 D+R−H。同一物理记录两条路径中至少一条判对的数量恰为 D+R，即候选正确数加 H。因此，如果决策只能从这两条已冻结的判决中选一条，D+R 给出这组固定配对的上限。下表每景对原 17 个候选，以及新增整体相位冻结推理、整包逆滤波、接收调制分量监督和非负条件方差恢复器逐一计算，低档城区再加入 22 份同期独立测试。最大值用于说明现有配对覆盖范围，没有生成逐条选择、路由或融合输出。

| 星地场景 | 固定配对数 | 两者至少一方正确的最高比例 | 距 3 pp 门槛仍少对 |
|---|---:|---:|---:|
| 高档郊区 | 21 | 88.644643% | 0 |
| 中档郊区 | 21 | 87.978571% | 0 |
| 低档郊区 | 21 | 87.081548% | 0 |
| 高档城区 | 21 | 87.969643% | 0 |
| 中档城区 | 21 | 83.466071% | 122 |
| 低档城区 | 43 | 78.725000% | 8,087 |

低档城区 DURA 判对 115,586 条。43 组配对中最大的额外修复量为 16,672 条，仍有 35,742 条两条路径均误判。即使保住全部 DURA 正确判决，最高也只有 78.725%，距 140,345 条门槛仍少 8,087 条。中档城区的现有固定配对也都达不到门槛。因此，强退化场景除了减少损伤，还需要产生能修复更多共同错误的新 IQ 输出。该数值仅约束已发布的固定两路判决，任意新恢复器和单次观测的信息上界仍待测定。

169 个候选和场景组合只复算公开的 truth-last 汇总，原 17 个候选另逐个核对了两个总体的计数恒等式。无新增目标真值、目标预测或训练。 [全部配对计数](PAMF_RESTORATION_PAIRED_BUDGET_20261107.csv)

## 2026-11-07 源端机制与下一版验证

以下结果全部使用原 480 条源 V、原冻结 Tri-SIG-TIME SSL 及完整预登记矩阵。先验统计只从原 L_s/U_s 拟合，源 V 的真实 TX 在全部预测固定后独立评分。每条源记录的重复干预共享物理身份，按配对计数报告。此处的源 raw clean 为 474/480，Main raw clean 为 145385/168000，两者属于不同总体。

| 源端干预 | 低档城区真实 TX 正确数 | 对下一步的作用 |
| --- | --- | --- |
| 仅改变模拟接收机参数或其 RX×day 分配 | 原验证 401，训练参数 399，验证参数轮换 400 | 六景最大净变化 2/480，量化该固定模拟参数范围的敏感性 |
| 同条 clean 分段替换 | 原输出 401，只替换包首 431，只替换后段 463，两段一起 473，raw clean 474 | 为独立整包网络分配包首与后段处理能力 |
| 等量先验是否包含当前源 RX | 盲估同 RX 412、其他 RX 341；已知信道和噪声时为 434、379 | 在固定信道以外检验先验失配，引入训练期源先验切换 |
| 单独替换均值或协方差 | 盲估同 RX 412，换均值 408，换协方差 364；已知条件为 434、425、394 | 检验从当前观测调整先验置信度的网络 |

模拟接收机干预固定传播、几何和底层噪声抽样。分段替换直接借用同条 clean 的真实误差。RX 先验干预按已知源 RX 选择统计，两个先验各用 12600 条训练记录且 L/U 配额相同。各项干预的额外信息和恢复函数不同，计数增益不能逐项累加。先验分量的低档城区交互为盲估 −19、已知条件 −6 条，完整因素表保留了这项交互。

[模拟接收机](../automation_reports/CV-SincNet/20261107-diagnostic-Restore-receiver-session-sourceV-s2026092701-r01/analysis.md) · [分段 IQ](../automation_reports/CV-SincNet/20261107-diagnostic-Restore-preamble-edges-sourceV-s2026092701-r01/analysis.md) · [接收机先验](../automation_reports/CV-SincNet/20261107-diagnostic-Restore-receiver-prior-sourceV-s2026092701-r02/analysis.md) · [均值与协方差](../automation_reports/CV-SincNet/20261107-diagnostic-Restore-prior-factor-sourceV-s2026092701-r01/analysis.md)

三条可执行方案已登记并发布，当前等待 GPU。整包时变复数逆滤波器直接从 received 预测三个区域的复数主路、共轭路和 IQ 残差。先验失配训练版在 30 轮内交替使用全源及排除当前源 RX 的先验，训练条件均值补偿。观测条件化协方差版从当前 received、先验和观测创新量预测非负逐点方差增量，梯度经过完整后验，并用实际带噪恢复 IQ、STFT 和批准的源连续 F0 特征学习。后者先检验 C_source 加对角非负增量这一具体表达。

[整包网络](../automation_reports/CV-SincNet/20261107-phase1-Restore-PacketInverseFilter-E30-s2026092701-r01/report.md) · [失配曝光](../automation_reports/CV-SincNet/20261107-phase1-Restore-MismatchedPrior-E30-s2026092701-r01/report.md) · [条件方差网络](../automation_reports/CV-SincNet/20261107-phase1-Restore-ConditionedCovariance-E30-s2026092701-r01/report.md)

三版均按从零单阶段 30 轮登记，沿原 L_s/U_s 范围，固定最终轮次后自动完成 Main 两总体七景与独立评分。性能验收仍由每景至少 140345 条正确及两个 clean 总体分别保持证明。源端诊断支持方法选择，实际性能以各版完整 Main 配对结果为准。

## 2026-11-07 E02 条件方差分支的源端诊断

VERIFIED / ANALYZED。固定E02源训练70条（实际全部L_s，五RX×七scene×2），两种源先验×四种固定方差，共56单元完成且独立重算。当前分支与零附加方差的损失差仅约1e-8量级，当前softplus平均斜率约2.2e-7；统一增大方差在global55/70、other-RX44/70条上局部不利，但约43%/46%的坐标仍有负梯度方向。结果支持表达限制与梯度衰减并存，尚不证明永久失效或Main识别收益。源2检查、3项针对性测试、一次独立P0/P1和实际CPU执行通过，零模型更新、零TX/V/OE/target读取。GPU6当前已完成3/30轮，原final30和Main七景矩阵保持。 [源训练配对机制分析](../automation_reports/CV-SincNet/20261107-diagnostic-Restore-conditional-variance-E02-sourceLU-s2026092701-r01/analysis.md)

## 2026-11-07 源协方差固定基的表达限制

VERIFIED / ANALYZED。只读既有源矩完成10单元，零新IQ/标签/V/OE/target/拟合/模型读取。global本底五RX等权：独立64约75.70%，独立512约75.91%，耦合64理想空间约98.35%的协方差差异Frobenius能量；不是识别率或IQ恢复比例。10个矩阵直接重建独立复算通过且耦合替换均正定。源端相对方差谱也超出当前0.1至1.9范围。支持后续研究方向间耦合与更充分的正定伸缩，不以该上限代替网络学习或正式Main收益。 [源统计分析](../automation_reports/CV-SincNet/20261107-analysis-Restore-source-covariance-capacity-s2026110707-r01/analysis.md)

## 2026-11-07 冻结整体相位对照的完整 Main 结果

VERIFIED / ANALYZED。原Main两总体168000条x七景、四流56单元全预测后独立truth-last，三阶段真实exit0且当前控制器/子进程均已退出。全部1736组、1302raw配对及BF/DURA/clean额外配对独立复算通过，最大指标差1.24e-14，三条控制流复用判决差0。六景依次83.427976/81.999405/80.531548/82.244048/75.323214/67.878571%，0/6；clean跨接收机净-1900、跨接收机且跨日期净-133。本对照六景与两个clean联合验收未通过。低档城区67.878571%，距140345条门槛少26309条正确。这是同一冻结原BF权重的整体相位推理对照，0新训练；实际预测/评分仍原代码，资源调度变化没有改变模型、数据或矩阵。后续在训候选保持原计划，默认DURA保持。 [完整结果和全部分组](../automation_reports/CV-SincNet/20261107-evaluation-Restore-CovariancePhaseOnly-main-s2026092701-r01/analysis.md)

## 2026-11-07 整包逆滤波从零30轮的完整Main结果

VERIFIED / ANALYZED。整包时变逆滤波恢复器490866参数从零单阶段30轮/7410完整结束，全部30轮日志/源顺序/预算、E0/E30有限状态、三组区域head实际更新、final逐值等于E30技术状态已核实。原Main两总体168000条x七景，三流42单元全部预测后独立truth-last；4阶段exit0、控制器及子进程均已退出。独立复算1302组/868raw配对及DURA/clean配对通过，最大差1.2e-14、raw/DURA复用判决差0。六景依次82.230357/80.775000/78.887500/80.596429/72.464286/63.987500%，0/6；clean两总体净+500/-92，联合验收未通过。实际训练阶段1161.801秒，完整两阶段预测246.631秒，均为共享负载整阶段计时，不代替B1时延或主性能。原source-only数据、连续F0指导和科学代码保持，资源调度未改变模型或测试范围；其他已登记候选继续，不按本项结果取消、组合或重排。默认DURA保持。 [完整结果与全部分组](../automation_reports/CV-SincNet/20261107-phase1-Restore-PacketInverseFilter-E30-s2026092701-r01/analysis.md)

## 2026-11-07 TF_ModulationCovariance 完整Main结果

VERIFIED / ANALYZED。固定scratch30/7410、全部源日志/顺序和final=E30核实；原Main42单元全部预测后独立truth-last，4阶段exit0且进程均已退出。1302组/868raw配对及DURA/clean配对复算PASS，最大差1.24e-14，baseline判决差0。六景83.657143/82.395238/80.766071/82.386310/75.182738/67.238095%，1/6；clean两个总体净-66/-29，联合验收False。原科学代码/模型/数据/预算保持，默认DURA不变，其他已登记候选继续。 [完整分析](../automation_reports/CV-SincNet/20261107-phase1-Restore-ModulationCovariance-E30-s2026092701-r01/analysis.md)

## 2026-11-07 调制参数化的已验证收益范围

同一固定Main测试中，调制参数化与分量监督版相对原CovarianceMAP六景均净增加正确预测，增幅0.039881至0.255952个百分点；clean净少152条。它仍低于DURA、六景1/6且两个raw clean总体净−66/−29，不能认定整个恢复目标已完成或该组合已具备跨seed稳定性。全部源顺序、draw、曝光与学习率相同；末轮Action相对MSE从0.00972509到0.00901296，噪声log MSE从0.04974598到0.04541007。物理源拟合改善确有本次正式识别小幅收益，但单次组合对照不能将收益单独分配给参数化或分量监督。其他预登记候选保持原科学计划。

## 2026-11-07 TF_ConditionedCovariance 完整Main结果

VERIFIED / ANALYZED。固定scratch30/7410、全部源日志/顺序和final=E30核实；原Main42单元全部预测后独立truth-last，4阶段exit0且进程均已退出。1302组/868raw配对及DURA/clean配对复算PASS，最大差1.2e-14，baseline判决差0。六景83.586310/82.238095/80.573214/82.334524/75.005952/67.087500%，1/6；clean两个总体净+154/-12，联合验收False。原科学代码/模型/数据/预算保持，默认DURA不变，其他已登记候选继续。 [完整分析](../automation_reports/CV-SincNet/20261107-phase1-Restore-ConditionedCovariance-E30-s2026092701-r02/analysis.md)

## 2026-11-07 非负条件方差版的实际收益与边界

完整Main六景依次83.586310/82.238095/80.573214/82.334524/75.005952/67.087500%，1/6；相对DURA六景均下降0.700000至1.713690个百分点（其中高档城区下降0.714881，最低实际降幅0.700000）。两个raw clean总体净+154/−12，合并上涨不构成clean均保持。低档城区距140345条门槛仍少27638条。末轮平均非负方差增量相对原先验约2.4114e-9，源端几乎关闭增量；该观察与正式未达标同时存在，不能单独证明全部Main差距的成因。预登记匹配的先验失配均值版仍在训练，具体匹配增量比较待其完整预测固定后补齐。当前完整Main/DURA结果不用于修改其他已登记候选。

## 2026-11-07 TF_CoupledCovariance 完整Main结果

VERIFIED / ANALYZED。固定scratch30/7410、全部源日志/顺序和final=E30核实；原Main42单元全部预测后独立truth-last，4阶段exit0且进程均已退出。1302组/868raw配对及DURA/clean配对复算PASS，最大差1.24e-14，baseline判决差0。六景84.041071/82.468452/80.612500/82.552381/74.165476/65.292857%，1/6；clean两个总体净+165/-30，联合验收False。原科学代码/模型/数据/预算保持，默认DURA不变，其他已登记候选继续。 [完整分析](../automation_reports/CV-SincNet/20261107-phase1-Restore-CoupledCovariance-E30-s2026092701-r01/analysis.md)

## 2026-11-07 TF_SignedCovariance 完整Main结果

VERIFIED / ANALYZED。固定scratch30/7410、全部源日志/顺序和final=E30核实；原Main42单元全部预测后独立truth-last，4阶段exit0且进程均已退出。1302组/868raw配对及DURA/clean配对复算PASS，最大差1.2e-14，baseline判决差0。六景83.489286/82.073810/80.443452/82.207738/74.348214/65.957143%，0/6；clean两个总体净+266/-35，联合验收False。原科学代码/模型/数据/预算保持，默认DURA不变，其他已登记候选继续。 [完整分析](../automation_reports/CV-SincNet/20261107-phase1-Restore-SignedCovariance-E30-s2026092701-r02/analysis.md)

## 2026-11-07 三类最终协方差源V与Main对照

VERIFIED：三版final30的固定源V480×7诊断完整完成，315指标组/315配对组通过，raw clean474/480。耦合版相对非负方差版中/低档城区源V净+14/+4，同一科学权重的Main却净−1412/−3015。源端真实TX局部收益不保证跨RX测试收益；源低城区85.0%本身也距源raw clean98.75%有13.75pp差距。零新训练、V拟合或目标预测，不把差异单独归因于某一种RX机制，其他已登记候选保持。 [完整源验证及Main对照](../automation_reports/CV-SincNet/20261107-diagnostic-Restore-CovarianceFamily-final30-sourceV-s2026092701-r01/analysis.md)

## 2026-11-07 TF_CoordinateMAPDetail 完整Main结果

VERIFIED / ANALYZED。固定scratch30/7410、全部源日志/顺序和final=E30核实；原Main42单元全部预测后独立truth-last，4阶段exit0且进程均已退出。1302组/868raw配对及DURA/clean配对复算PASS，最大差1.22e-14，baseline判决差0。六景83.677976/81.066667/78.604762/78.305952/59.244048/46.719643%，1/6；clean两个总体净-100/-30，联合验收False。原科学代码/模型/数据/预算保持，默认DURA不变，其他已登记候选继续。 [完整分析](../automation_reports/CV-SincNet/20261107-phase1-Restore-CoordinateMAPDetail-E30-s2026092701-r01/analysis.md)
