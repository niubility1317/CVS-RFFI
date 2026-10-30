# 可选对照：源域 F 特征指导的单阶段恢复

状态：**AUTHORIZED_FOR_INDEPENDENT_SOURCE_F_GUIDED_CONTRAST**。用户已明确选择“允许新增独立 F 指导对照”。本文件固定该独立分支的权限和矩阵；实际发布、训练、评分状态按独立run登记及产物核实，不把授权当成已经运行或完成。现有 F-free 实验继续原矩阵。

## 为什么需要单独决定

会话和当前工作区 AGENTS.md 的 2026-10-20 约定明确要求：“不读 TX 真值、训练好的 F 权重/输出或 target 来优化恢复器；F 仅在冻结后的评测与无 query smoke 使用。”用户本次明确为独立对照放开其中的**源端训练读取冻结 F 权重/特征**，不继承为 F-free 方法，不修改 DURA 默认。科学权限与声明差异同步记录于`docs/PROJECT_PROTOCOL.md`。

现有结果尚未满足六个星地场景各距同一 raw clean ≤3pp。源 IQ 均值改善不能充分代理身份特征保持；这支持检验任务约束的价值，并不证明 F-free 路线已经达到不可突破的上限。本方案来自这一目标与损失缺项，不依据目标分数选择 checkpoint 或中间轮次。

## 具体候选与对照

|项目|固定方案|
|---|---|
|恢复器|完整266951参数、两prior/四CG、原生插值；普通随机初始化|
|源基础目标|既有noVarAux：最终IQ NMSE＋0.1内部prior NMSE＋0.05多尺度谱误差＋0.05源模拟器action probe NMSE；不训练logvariance辅助校准|
|源基础选择依据|已完成源梯度诊断中的方差辅助冲突与源IQ误差改善；不从目标分数排序选父；不加载任何已训练G|
|唯一新增项|0.1×逐例余弦特征误差：`1−cos(F.features(G(received)), stopgrad(F.features(clean)))`|
|F|同一原F50 seed2026092701/defaulthead，原source-final50/11100；全参数冻结、eval模式，不重训骨干或分类头|
|训练数据|原R07合法63000源IQ：6300 L_s＋56700 U_s，原7/63/30物理划分及epoch01..20模拟配对|
|直接TX标签读取|0；但通过F继承源有标签知识，必须明确披露，不能称为原F-free或完全无TX监督|
|预算|从零单阶段E20/4940、B256、相同seed/批序/两probe/LR峰1e-3/末2e-5/热身247、AdamW/裁剪/FP32/TF32off|
|对照|复用已完成且数据/初始化/源基础目标/预算匹配的noVarAux原七景；不同权限与不同运行时间明确标注|
|默认版本|保持DURA v1.0，不自动替换|

这里的F指`20261016-phase1-F-supcon-e50-clean-manysig-m9-r02/CROSS_RX-s2026092701`的原冻结识别器，不挪用当前Tri-SIG-TIME SSL身份或分数。来源按已有`load_reader`严格检查，必须与原源物理角色匹配；原数据VALIDATED_ONCE复用，不建立新的数据验证或签名链。

## 训练效率与实现边界

G只有一次联合训练，没有coarse/action/prior逐级预训练，也不从旧G蒸馏。可将每条源clean的冻结160维F特征计算一次：63000×160×FP32为40.32MB特征数值，另计索引及metadata；只在授权后生成，不将这些特征用于Phase2/3。

每次G更新还需一次`F(G(received))`，梯度只穿过F到G输入，F参数与状态保持不变。额外F前反向和缓存成本要单独计时；不能宣称与F-free同成本，也不能把F的既有50轮有标签训练成本隐藏为零。报告新增G成本、复用F来源/既有预算和整体配对语义。

特征约束实现位于`experiments/pamf_dura_efficiency/proposed_feature_loss.py`；它只接受冻结eval reader和160维参考，目标特征detach，保留IQ梯度。原型开发阶段只用随机线性ToyReader测试梯度隔离；真实F兼容性、来源与首步smoke在独立run的授权后最小发布流程中核实，实际状态以该run产物为准。

## 测试、停止与声明

训练仍不读V或target。固定最后E20，不按目标分数选模型、调权重、早停或选择性重跑。冻结后自动同F50/source1800×7、target168000×7，所有七景prediction固定后独立truth-last，保留全部正负结果。

逐景报告accuracy/MacroF1、correct、repair/damage、TX/RX/day、相对rawclean与DURA/R17差距、clean损伤和完整训练/推理成本。成功条件不变：六星地各距raw clean≤3pp且clean保持。任务特征保持不等于每个TX物理细节真实重建，不能仅凭F一致性或正式准确率宣称波形真实性。

普通账户N607；本goal最多2条GPU worker，每卡最多1条own，启动时free≥12GB且可见compute PID<2，保留其他owner健康进程。输出root/代码提交/矩阵在启动前独立登记；权限未获明确同意不得发布本候选。合规/来源/非有限/输出碰撞/预测覆盖失败时保留产物并按既有技术停止规则处理，不自动回退到其他权重或改配方。

用户已明确允许该独立F指导对照；原F-free研究和既有权重/结果保留。新增真实F调用和训练按本方案及最小发布流程执行，不扩大到目标端学习。

实际独立run已发布并通过真实F/G梯度及冻结检查，进入训练；后续训练/评分状态见[该run报告](../../automation_reports/CV-SincNet/20261030-phase1-DURA-source-Fguide-cos010-s2026092701-r01/report.md)。
