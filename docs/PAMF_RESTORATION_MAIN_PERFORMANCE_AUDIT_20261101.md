# 星地恢复器 Main 性能核对

固定 DURA v1.0 为 1/6 景达标；本表九个新候选中 0 个达到六景逐个距 raw clean ≤3 个百分点。完整验收还要求两个总体分别保持 clean。性能是主线，训练效率是支线；源域拟合、参数量和训练结束均不能替代正式验收。

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

本核对只复用公开汇总，新增神经前向、原始训练/目标数据读取、真值读取均为 0。完整目标保持 ACTIVE，默认 DURA 不变。

补充分析：[射频细节保留与可观察性边界](PAMF_RESTORATION_RF_DETAIL_ANALYSIS_20261101.md)整理历史源R31结果，区分局部响应、真实身份和理论恢复率；不增加当前模型性能证据或改变训练。
