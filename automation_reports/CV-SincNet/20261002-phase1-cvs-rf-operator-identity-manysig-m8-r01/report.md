# CVS 受约束射频行为算子

run_id：`20261002-phase1-cvs-rf-operator-identity-manysig-m8-r01`。状态 LOCAL_VERIFIED，尚未正式发布或训练。两项前瞻MP/GMP候选各4seed，普通CE唯一、身份骨干、无增强，性能主要、轻量次要。

|候选|总/有效CE参数|Conv/Linear MAC/包|常驻模型状态字节|
|---|---:|---:|---:|
|rf_mp|167153|10562852|669264|
|rf_gmp|167665|10824996|671312|

49项本地聚焦检查通过：手工系数AM/AM与AM/PM/记忆/共轭成分、相位变换、未来扰动、零与弱信号有限导数及gradcheck、逐包一致、实际CE梯度和配对初始化、源选择/角色/继承边界，以及39项既有与新增clean协议检查。RF复数收缩由绑定符号的实数Conv实现，包含在MAC中；基生成、径向保护与观测逐元素运算不计MAC，实测耗时/峰值包含它们。CPU资源只用于实现核实，N607与识别结果尚未测量。

已核实公开WiSig处理链：equalized=1保留加回的CFO，25 Msps输入和包RMS归一化不提供纯TX输入/输出对。当前旧PA结构与本轮差异、物理公式、前瞻预算和可辨识性边界见[物理设计](../../../docs/CVS_RF_OPERATOR_HYPOTHESIS_20261002.md)。单包received IQ无法凭CE保证TX/RX/信道解耦；学习系数为共享特征滤波器，不能称为实测TX硬件参数。因果性只保证RF算子，后续卷积/池化是整包分类；全局相位不变只保证物理观测，整个网络不保证CFO/任意多径/RX不变。

从零E200×50，完整L6300/epoch、U56700unused/V27000、原四seed和split392005。最终源V/最差源RX等权性能最高候选冻结；并列后才比较成本。默认选中4个clean预测加原20个固定对照，全部24份完成后独立truth-last评分；完整RX/TX/F1/混淆矩阵/成本和负结果保留，不按测试重排、调参或重跑。物理合成检查不作为训练增强或识别提升证明。[验证](evidence/local_validation.json) · [逐行预登记](experiment.json)。独立P0/P1审查及唯一P1修复定点复核通过；审查者独立13项测试PASS，compileall通过。

N607普通账户Torch2.1 CPU的8个合成模型检查通过；24次CE更新中梯度有限非零，最大FP32旋转观测差1.20e−5，单包一致/零常量有限，formal run/log/release均未存在。仅在内存运行本地算子，无真实数据或checkpoint/GPU任务；此证据证明运行时正确性，不证明识别收益。见[远端CPU验证](evidence/remote_cpu_rf_validation.json)及[独立审查](evidence/independent_review.json)。
