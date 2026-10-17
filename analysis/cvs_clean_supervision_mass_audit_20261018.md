# ordinary／As 监督份额：代码与完整源日志核对

2026-10-18，ANALYZED／CODE_AND_CLOSED_SOURCE_LOGS_ONLY。检查起点 commit d4f84de954c056b79da98de82f933d6cc43dd266。0 新 IQ／checkpoint／模型前向／教师前向／更新，未发布新实验，未修改旧运行。

**已证实 ordinary 监督发生重分配；尚未证明它导致 clean 下降。保留绝对原监督也不是充分保护条件，旧 additive As 的独立确认已提供反例。** 不把该审计改称新 Ag、As 改善或联合目标完成。

## 实际损失与视图份额

记 O(x) 为原 ordinary 增强，S(x) 为原完整 practical(raw)，G 为已经测试的新增 IQ 算子。R(z) 是同一已生成 IQ 的两次默认头平滑 CE 均值＋原 RDrop 双向 KL，CE／KL 都按 L128 物理样本均值。每对不重抽 IQ，SupCon 仍只看共有 ordinary 对浅／中层响应。

|实际行|L 前向/物理包|批平均 ordinary R 份额|新增 G(O) 份额|As 份额|R 份额之和|
|---|---:|---:|---:|---:|---:|
|NATIVE|2|1|0|0|1|
|REPEAT/B00|4|1|0|0|1|
|FULL_AS/CONCAT|4|0.5|0|0.5|1|
|B10|4|0.75|0.25|0|1|
|B01|4|0.75|0|0.25|1|
|B11|4|0.5|0.25|0.25|1|

NATIVE 为 R(O)；REPEAT/B00 为 .5R(O第一对)＋.5R(O第二对)。FULL_AS 为 .5R(O)＋.5R(S)，因此相比 REPEAT 的 ordinary 总系数1，FULL_AS 的 ordinary 变为.5；不是在完整 ordinary 目标之上追加 As。B01第二对中64个O／64个S，所以全批平均ordinary .75／As .25。B10/B11同理；这些是类别视图的批平均份额，不表示每个包都各接受.25的新算子，也不是额外物理样本。

源码：[fullsource hook](../experiments/cvs_fullsource_concat20/model.py)、[64:64 factorial hook](../experiments/cvs_clock_factorial20/model.py)、[E30原模型/算子偏移](../experiments/cvs_clock_factorial30/model.py)、[RDrop terms](../experiments/cvs_sigtime_rdrop_legacy/model.py)。本轮closed dropout两行沿用FULL_AS的.5/.5，并没有通过反射掩码改变这个份额。

累计E21至E30的SupCon绝对权重仍.1，U原lambda_u=.16／lambda_ent=.01等规则保持。因此在FULL_AS中，SupCon和U相对ordinary身份项的比例也随ordinary系数下降而变化；不能把所有差异单独归因为传播。系数变化不等于梯度范数或身份收益按比例变化，AdamW／裁剪／响应非线性也不允许推导“系数恢复就会保护clean”。

## 原生首 CE 没有被重复加入

原生loss_closed_l以loss_tx_l的系数1起始；hook返回registered_L−identity_loss。主循环只向总loss加一次这个correction，另一次赋给scaled_closed_loss是备用梯度表示，不是同一loss重复相加。实际入口见train_ssdg.py约9110、10179、11494行。

原生train/loss_labeled／train/loss_tx_labeled仍记录correction之前第一ordinary前向的CE，不能作为完整L目标。复用上一实验的完整逐步分析（2220步/行）：INDEPENDENT这两个原生日志均值为0.278988845，完整L均值0.653325310，实际optimizer总目标均值0.656828731；ANTITHETIC依次为0.278725174／0.653013149／0.656497172。两行全部closed scale为1，L目标逐步独立重算最大误差1.1920929e−7。单独的每步entropy没有完整独立日志，本笔记不声称重新数值分解了整个optimizer目标。

## ordinary 不等于未增强 x

上述保留项是O(x)，不能直接称“完全clean训练”。本E20＋10实际冻结原父E13增强日程，10个epoch遥测一致：scale0.2506003339、PA概率0.03508404675、循环移位概率0.08771011688／最大8点、CFO概率0.08771011688／最大0.0001002401336 cycle/sample、AWGN概率0.10024013357、multipath概率0.04510806011。DAC概率0、class_signature=false。基础配置的PA.14、移位32点不是实际阶段值。

这些是旧共有普通增强的实际执行值，不是PA标签保持证据；保留NoExplicitCFO也不等于训练CFO扰动关闭。此次没有把旧PA或CFO改称新Ag，没有按结果新增、删除或调幅。As仍直接作用raw，不把ordinary中的噪声和多径再施加一次；每对RDrop使用相同已生成IQ。完整实际阶段范围与数值见[机器可读核对](cvs_clean_supervision_mass_audit_20261018.json)。

## 旧 additive As 已排除简单“保留系数即可”解释

[首seed报告](../automation_reports/CV-SincNet/20261016-phase1-F-crossRX-additiveAs-sourceRX1-e20-s2026092701-m3-r01/report.md)曾通过源规则，但[固定seed02确认](../automation_reports/CV-SincNet/20261016-phase1-F-crossRX-additiveAs-confirm-sourceRX1-e20-s2026092702-m3-r01/two_seed_source_confirmation.md)失败：保留原raw.75＋ordinary.25，再加As.25，第二seed相对D00的clean−8.253968pp、相对等总loss/前向D_raw控制−6.825397pp；两seed相对D00均值−2.460317pp。完整源V健康通过不能覆盖这个失败。

该历史为4训练RX／L5040／U45360／V21600、held RX1原L1260的旧契约，不能替代现在5RX／L6300／U56700隐藏TX／V27000、无训练RX留出的证据。它反证该固定配方的充分保护解释，不证明所有可能的fullsource组织都失败；也不授权改系数、seed、RX或预算救跑。旧结果保留，本轮不重启它。

## 与另一对话分工

已读取引用对话，并独立核对其落盘[全源R25报告](E:/type10-7/automation_reports/CV-SincNet/20261017-phase1-FcrossRX-cleanretain-e30-s2026092701-m1-r01/report.md)：该对话负责ordinary.75／As.25、scratch30/20L10U及sourceV1800×2draw的源状态/新增噪声比分层。本对话不重复其训练或源读出，不把它与本E20＋10的不同配方成绩拼接成同预算因子证据，不据其目标结果设新系数。

## 本轮判断与下一步边界

“普通拼接减少ordinary份额”这一实现判断已证实；“这就是clean代价的唯一原因”未证明；“保留绝对份额即可同时保护clean和改善As”在旧固定双seed配方中已被否定。本轮最小检查已完成，无需为这个解释再开训练。

当前没有获得新源依据支持的IQ候选，不把泛化问题缩成调ordinary/As配比。完整目标仍要求有效Ag、As、同契约因子交互和跨seed联合证据。后续候选须有不同于已关闭路线的观测依据；本笔记不新增审批、源训练域留出或预算。
