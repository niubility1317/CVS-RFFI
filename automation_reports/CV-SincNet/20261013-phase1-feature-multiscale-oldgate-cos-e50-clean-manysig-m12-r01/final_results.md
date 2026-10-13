# 历史旧门控基线下的多尺度特征提取：完整结果

状态：ANALYZED / VERIFIED。12/12 模型、三个实际种子全部完成；预登记主方案 JOINT_MS50 未达到 clean 均值84%。

| 组别 | seed01 | seed02 | seed03 | 三seed均值 ± 样本SD | Macro-F1均值 |
|---|---:|---:|---:|---:|---:|
| OLD_GATE_COS50 | 80.6685% | 78.0815% | 82.8304% | 80.5268% ± 2.3776 pp | 80.0024% |
| TIME_MS50 | 82.4262% | 81.0988% | 81.5464% | 81.6905% ± 0.6753 pp | 81.1841% |
| PA_MS50 | 79.2571% | 77.5744% | 80.1780% | 79.0032% ± 1.3202 pp | 78.6648% |
| JOINT_MS50 | 80.4036% | 81.5369% | 81.9280% | 81.2895% ± 0.7917 pp | 80.7658% |

TIME−基线 +1.163690 pp（2/3 seed正）；PA−基线 −1.523611 pp（0/3正）；JOINT−基线 +0.762698 pp（1/3正）。JOINT−TIME −0.400992 pp。不能称为稳定三seed提升，也不能按最高单seed或控制组替代预登记目标。

全部从零E50，33label+17pseudo；共600 epoch、133200 successful optimizer/EMA updates。各组同历史EMA.999、batch_neighbor/window2/minconf.8、strongagreement、rx_day_quantile.86、lambda_u.16、lambda_ent.01、label_smoothing.01、AdamW.0002/wd.0001/clip5/cosine最低1e-6、L128/U256、222step/epoch。训练无卫星，NoExplicitCFO保留内部补偿；普通6×160 classifier结构不改。四组只改t1/2/3或PA1/2/3；源码代码commit 5fa536f9ca2d90e7d30d547d58503a41b612d57f。

full source log审计：每行50epoch/11100连续step、前33label后17pseudo、每步EMA成功counter严格连续、身份loss/总loss有限、无nonfinite跳步、U真值available=0；末222step均值与source completion一致。全source来自同物理契约，无外来checkpoint/target拟合，固定final50。Source训练CE下降不能代替跨RX的target或新类表征证据。

每行clean168000 predictions先固定后fresh independent scorer连接truth；总2016000 decisions。直接从raw predictions和truth重新计数216个overall/RX/day/TX confusion matrices、accuracy、Macro-F1，所有分层sum等于overall。记录覆盖source完整曲线、实际时间、逐seed和逐stratum，见result_summary.json与两个CSV。原始weights/predictions/logs仍在原run根。

Scope：ManySig原7/63/30、L6300/U56700隐藏TX/V27000，SourceRX1/3/4/6/8 day1/2/3；同固定六TX映射及clean heldtargetRX7组/day4组。run 20261013-phase1-feature-multiscale-oldgate-cos-e50-clean-manysig-m12-r01，split/scenario/物理ID均沿用原登记。仅clean研究，不声明卫星部署、新TX注册或新的独立盲测；这是已暴露测试基准的固定矩阵。目标分数未回流到源训练、选模、矩阵排序、停止或重跑。

新六路线原说明及NORM/COMP原型在读取这些目标数值前已固定；不是依据本轮target结果重新选择alpha、mask或epoch。后续先原基线/同参数普通残差/归一化恢复/竞争/组合，冻结原z重训同一普通cosine做表征诊断；其正式实验及Phase2旧新类对照尚未完成，不能提前宣称有效。

证据：local_artifacts/cvs_feature_multiscale50_20261013_r01/final_audit.json、readback.json；远端runs/20261013-phase1-feature-multiscale-oldgate-cos-e50-clean-manysig-m12-r01下逐行source/prediction/scores。全部负结果保留。
