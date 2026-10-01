# CVS 受约束射频行为算子

run_id：`20261002-phase1-cvs-rf-operator-identity-manysig-m8-r01`。状态 ANALYZED，完整 E200 与独立 clean 确认均已完成。两项前瞻MP/GMP候选各4seed，普通CE唯一、身份骨干、无增强，性能主要、轻量次要。

|候选|总/有效CE参数|Conv/Linear MAC/包|常驻模型状态字节|
|---|---:|---:|---:|
|rf_mp|167153|10562852|669264|
|rf_gmp|167665|10824996|671312|

49项本地聚焦检查通过：手工系数AM/AM与AM/PM/记忆/共轭成分、相位变换、未来扰动、零与弱信号有限导数及gradcheck、逐包一致、实际CE梯度和配对初始化、源选择/角色/继承边界，以及39项既有与新增clean协议检查。RF复数收缩由绑定符号的实数Conv实现，包含在MAC中；基生成、径向保护与观测逐元素运算不计MAC，实测耗时/峰值包含它们。CPU资源只用于实现核实，N607与识别结果尚未测量。

已核实公开WiSig处理链：equalized=1保留加回的CFO，25 Msps输入和包RMS归一化不提供纯TX输入/输出对。当前旧PA结构与本轮差异、物理公式、前瞻预算和可辨识性边界见[物理设计](../../../docs/CVS_RF_OPERATOR_HYPOTHESIS_20261002.md)。单包received IQ无法凭CE保证TX/RX/信道解耦；学习系数为共享特征滤波器，不能称为实测TX硬件参数。因果性只保证RF算子，后续卷积/池化是整包分类；全局相位不变只保证物理观测，整个网络不保证CFO/任意多径/RX不变。

从零E200×50，完整L6300/epoch、U56700unused/V27000、原四seed和split392005。最终源V/最差源RX等权性能最高候选冻结；并列后才比较成本。默认选中4个clean预测加原20个固定对照，全部24份完成后独立truth-last评分；完整RX/TX/F1/混淆矩阵/成本和负结果保留，不按测试重排、调参或重跑。物理合成检查不作为训练增强或识别提升证明。[验证](evidence/local_validation.json) · [逐行预登记](experiment.json)。独立P0/P1审查及唯一P1修复定点复核通过；审查者独立13项测试PASS，compileall通过。

N607普通账户Torch2.1 CPU的8个合成模型检查通过；24次CE更新中梯度有限非零，最大FP32旋转观测差1.20e−5，单包一致/零常量有限，formal run/log/release均未存在。仅在内存运行本地算子，无真实数据或checkpoint/GPU任务；此证据证明运行时正确性，不证明识别收益。见[远端CPU验证](evidence/remote_cpu_rf_validation.json)及[独立审查](evidence/independent_review.json)。

## N607源训练发布已核实

代码`010ff27a20a3bb8af90580aa1eda4ab7d08427df`已push并独立核对远端OID。新release传输校验、远端compile与Torch2.1冷进程双候选CE反向检查通过。独立读回dispatcher PID1068124的CWD/argv；8行各占一块GPU，进程与resolved/log增长已核实，状态SOURCE_TRAINING。实际6300L/56700Uunused/27000V、每轮50步、纯CE/无增强/域骨干关闭与目标访问关闭一致。发布后源码保持不可变，后续clean按另一个不可覆盖run发布。详见[发布后读回](evidence/source_launch_readback.json)。

## 完整源训练与冻结

8 行均完成 E200/10000 步，完整 1600 epoch、80000 step、CSV 和全部 stdout 已解析，无运行期技术异常；实际源数据角色一致，scratch/noaug/nodomain/CE-only 核实。按登记的性能优先规则冻结 `rf_gmp`，源选择独立复算完全一致。源性能完全并列后才考虑参数和计算，不使用测试反馈。确认 run `20261002-phase1-cvs-rf-operator-clean-manysig-m24-r01` 已预登记，默认完成 4 个新预测和 20 个冻结控制预测的独立评分。见[evidence/source_selection.json](evidence/source_selection.json)和[evidence/source_completion_validation.json](evidence/source_completion_validation.json)。

## 独立 clean 确认完成

选中 `rf_gmp` 由独立子 run `20261002-phase1-cvs-rf-operator-clean-manysig-m24-r01` 完成四 seed clean 测试，原 20 预测只读复用，共 24 行、192 条评分。准确率 76.1238% ± 0.9616%，较上轮残差 CVS -2.3305 个百分点（0/4 seed 提升）。本轮没有进一步提高平均准确率，参数更少不能替代性能目标。源 run 未读取 target，未选候选不测试，测试评分不回流调参或重选。见[完整确认报告](../20261002-phase1-cvs-rf-operator-clean-manysig-m24-r01/report.md)。
