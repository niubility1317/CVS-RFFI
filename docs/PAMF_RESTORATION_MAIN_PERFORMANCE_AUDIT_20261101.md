# 星地恢复器 Main 性能核对

截至 2026-11-02，固定 DURA v1.0 为 1/6 景达标；本表 15 个已完成新候选中 0 个达到六景逐个距 raw clean ≤3 个百分点。完整验收还要求两个总体分别保持 clean。性能是主线，训练效率是支线；源域拟合、参数量和训练结束均不能替代正式验收。

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
- 冻结源四步轨迹显示，DirectBridge 低档城区首末步 Fclean 一致率均为 84.749304%；StateBridge 则从 86.350975% 降到 81.615599%，净损伤 68/1436，IQ NMSE 从 .07612306 增至 .11645158。后续迭代的退化已在训练源面板出现，不能全部归为 Main 域差异。训练用 clean/received 插值，而部署使用网络自行生成的状态，二者不一致是可检验假设，尚不独立证明原因。[完整轨迹诊断](../automation_reports/CV-SincNet/20261101-diagnostic-Restore-bridge-trajectory-source-s2026092701-r01/analysis.md)保留全部 96 组和配对。DirectBridge 的代数捷径仍只是一种可能限制，不能认定实际网络使用了精确解析式。
- 所有新 G 都在原 L_s/U_s 范围训练；冻结源 F0 提供连续特征指导，因此这些方法不是 F-free。TX 真标签、V/OE/novel/目标不进入 G 的更新、早停或选模。OE 仅用于冻结诊断。
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

本核对只复用公开汇总，新增神经前向、原始训练/目标数据读取、真值读取均为 0。完整目标保持 ACTIVE，默认 DURA 不变。


## 大幅改造已进入实际训练

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

这些是旧F50证据，**不纳入本次历史检索时原完整Tri-SIG-TIME SSL/Main的15个已完成候选计数，也不改标成本基座成绩**。source-only来源合规与方法有效性是两个问题；历史教师、源V/F选择、缓存和学生成本均须披露。当前CovarianceFeature继续自己的固定scratch30配方，只增加原Tri-SIG-TIME SSL对实际恢复IQ的连续源特征指导，没有R17教师输入，也没有因这次历史检索更改训练或新增候选。本次历史检索时该候选尚无完整Main成绩，没有提前纳入完成表。

## 原始论文的条件核对

[五篇原始论文核查](PAMF_RESTORATION_LITERATURE_CONDITIONS_20261102.md)区分编码特征分类、解码IQ接固定F0，以及需要目标接收机配对校准的不同问题。PSC-DAE使用TX监督，DRIFT训练新的TX/RX识别表示，TCNN采集目标校准响应，LTE-V2X使用完整协议参考结构；均不能直接继承为本任务的成绩或六景恢复上限。CDP理论的固定/最优分类器设定也不等价于当前六类未知信道数值上界。核查没有新增G/F/IQ/目标预测/真值读取或训练，没有更改CovarianceFeature固定配方。
