# Mitigating Receiver Impact DA核心机制重新审计

日期：2026-07-13

范围：只审计Yang等发表于IEEE IoTJ 2024的论文《Mitigating Receiver Impact on Radio Frequency Fingerprint Identification via Domain Adaptation》及当前`paper_reproduction/mitigating_receiver_impact_da`实现。本文不以复现精度是否接近论文作为方法正确性的判据。

## 结论

当前默认全组件路径可称为`bounded paper-equation reproduction`，不能称为严格或完整论文复现。式(5)-(11)和Algorithm 1的主要min-max机制已经对齐，但完整实验同构仍被未公开的数据划分、模型细节和训练配置阻断。本轮确认并修复了一个Table III消融错误：关闭class weighting时不应把源域CE系数从`mu`改成`1.0`。

## 论文事实与实现对应

|ID|论文事实|当前实现|判定|证据/边界|
|---|---|---|---|---|
|M01|`zeta=mean(T(E(xs)))-log mean(exp(T(E(xt))))`，优化为`min_E max_T zeta`|T子步最小化`-zeta`，E/C子步最小化`+lambda*zeta`|MATCH|`losses.py:7-15`; `algorithm.py:380-400,404-473`|
|M02|每个batch先更新T共`m`次，再更新E/C一次|默认`m=7`，更新顺序一致|MATCH|`algorithm.py:380-400,458-473`|
|M03|T更新不应改变E；E更新需通过固定T获得域对齐梯度|T子步detach特征；E/C子步冻结T参数但保留到E的梯度|MATCH|`algorithm.py:382-383,405-410`|
|M04|CPL为`beta_l(k)=sigma_{l-1}(k)/max sigma_{l-1}`、`tau_l(k)=beta_l(k)tau`，首batch统一`tau`|默认`paper`路径使用softmax概率和前一batch累计状态|PARTIAL|主公式一致；零计数类使用epsilon保护，论文未定义该边界|
|M05|class weight为`p_prior(k)/(sigma'_{l-1}(k)/n^t_{l-1})`|默认使用前一batch所有目标预测的累计频率；零计数回退1|PARTIAL|主公式和时序一致；零计数处理未由论文规定|
|M06|式(10)固定为`mu*L_s+(1-mu)*L_t`，class weighting消融只应令`omega=1`|全组件路径一致；本轮将no-class-weight路径的source scale由`1.0`修正为`mu`|FIXED|此前相关Table III消融结果不能继续作为有效论文消融证据|
|M07|外层为`while not convergent`，每轮重置计数|实现把外层解释为固定epoch并逐epoch重置|PARTIAL|论文没有给收敛判据，也没有明说outer loop等于epoch|
|M08|初始`h0`可为源域已训练模型|实现先做固定20个source-pretrain epoch|UNSPECIFIED|论文未给h0训练epoch、优化器、停止规则或是否复用优化器状态|
|M09|推理只保留E/C，T训练后丢弃|checkpoint推理状态只导出E/C|MATCH|`model.py:297-309`|

## 仍与论文无法证明一致的关键位置

|ID|当前做法|论文/官方证据|风险|
|---|---|---|---|
|U01|目标适配集与最终评估集是同一个dataset对象|论文未说明是否拆分；官方trainer明确具有`dataset_target`、`dataset_valid`和`dataset_test`三个独立入口，但仓库未公开配置|高：可能直接改变样本量、难度和Table II数值，当前不得称exact split|
|U02|旧运行在训练时读取目标真标签并记录逐batch伪标签准确率|这些标签不参与梯度或formal final checkpoint选择，但论文方法只要求目标无标签|高：旧结果已降级为target-exposed diagnostic；新默认路径用无标签target view，训练batch不再含`label`、`tx`或`tx_i`。底层ManySig.pkl仍按TX组织，不能声称原始文件无标签|
|U03|输入固定为256点IQ、left crop、再次去均值/RMS归一化|论文只说明energy detection、L-LTF MMSE equalization、减均值并按功率归一化；未给长度、裁剪和pkl处理来源|高：预处理等价性未证明|
|U04|采用一个具体1D-ResNet18 stem、通道、BN、ReLU和三层FC宽度|论文只给“ResNet18的2D卷积替换为1D卷积”和C/T为三层FC|高：架构仍欠定；template profile只能是诊断假设|
|U05|Adam、batch128、20+20 epoch、无scheduler、final checkpoint、特定seed|论文只给`lr=0.0006`、`lambda=0.005`、`mu=0.5`、`m=7`、`tau=0.7`|高：训练和停止配置不能由论文恢复|
|U06|默认使用完整DV公式|论文明确支持；官方trainer改用MINE移动平均surrogate、raw-logit CPL、当前批class weight和PyTorch weighted mean|中：论文公式路径与官方trainer路径相互矛盾，必须分开报告，不能混合挑选有利细节|
|U07|Table III用`use_pseudo`开关表示CPL组件|论文没有说明关闭CPL后是保留固定阈值伪标签还是完全删除目标伪标签CE；公开trainer把`use_pseudo=False`解释为删除目标CE|高：现有消融只能视为公开trainer式组件开关诊断，不能证明唯一的论文消融语义|

## 官方仓库证据边界

官方仓库`YannLeo/Cross_Receiver_RFFI_Network`在审计时HEAD为`6c4ed8de3e1575442a92136f4685d86f249d5800`，仅包含`README.md`和`mine_pseudo_classweight_trainer.py`。README称其为official implementation并要求结合另一个Pytorch-Template仓库使用，但没有发布WiSig配置、数据处理、主入口、MINE模型定义或Table II运行参数。因此它能证明trainer中的若干行为，不能补齐论文实验。

## 对既有结果的重新定性

1.默认全组件Proposed结果仍可用于“当前有界实现与论文数值的差距”分析，但不能证明完整论文方法已被精确复现。
2.`standard_da_only`不受本轮class-weight缩放修复影响。
3.任何关闭class weighting且由修复前代码生成的Table III行均失效，包括`standard_da_cw`名称以外需要按实际flags核实的no-class-weight行；必须用修复后代码重跑才能比较论文Table III。
4.所有Table III行还受CPL开关语义未公开的限制；即使重跑，也必须报告为公开trainer式消融或实现假设。
5.`official_compat`结果只代表公开trainer语义诊断，不代表论文式(5)-(11)复现。
6.目标标签选出的best epoch只能是oracle post-hoc诊断，不能作为正式UDA结果。

## 事实边界

- 论文明确：闭集UDA、源域有标签、目标域无标签、DV方向、CPL公式、class weight公式、GAD顺序，以及五个已报告超参数。
- 论文未明确：目标train/test拆分、样本数、batch size、epoch、优化器、scheduler、seed、停止规则、完整网络层配置和零计数处理。
- 禁止做法：根据论文精度倒推并选择目标标签checkpoint；把官方trainer与论文公式任意拼接；把架构假设或同集评估写成exact paper reproduction。
