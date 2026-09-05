# ADV3B02-DAOT-STN两批实验数据包

本目录保存ManySig、seed392005条件下，DAOT-STN-V1第一批A1～A7与RX-V2第二批P1～P5、E1/R1的GitHub可发布实验数据。完整方法、机制、公式、参数解释、结果、逐接收机分析、数值健康、资源消耗和结论边界见[`../../ADV3B02_TWO_BATCH_FULL_REPORT_20260905.md`](../../ADV3B02_TWO_BATCH_FULL_REPORT_20260905.md)。

## 1.覆盖范围

- 正式候选：A1～A7、P1～P5、E1、R1，共14行。
- 完成状态：13行完成200epoch及clean和3个LEO_WEAK场景评估；P5在E10/B122触发系统性非有限批次保护，无最终性能结果。
- 数据协议：ManySig equalized；`tx_rx_day_1_7_2`；source RX=`[1,3,4,6,8]`；target RX=`[0,2,5,7,9,10,11]`；`L_s/U_s/V=6300/56700/27000`；seed=`392005`。
- 评估规模：每个候选每场景168000条样本；完成行均分别报告`clean`、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`。

## 2.文件说明

|文件|内容|完整性|
|---|---|---|
|`summary.json`|14行最终指标、逐接收机结果、资源、数值健康、机制激活字段和异常|14/14|
|`epoch_curves.csv`|完整训练曲线的发布字段|2609个epoch记录|
|`per_receiver_scenario.csv`|逐候选、逐接收机、逐场景正确数、总数和准确率|364个单元|
|`checkpoint_readback.json`|A1、A7、P1、P3、P4、E1、R1代表性checkpoint的机制状态读回|7个代表性checkpoint|
|`experiment_matrix.csv`|14行实验机制、关键权重、运行状态和可归因边界|14/14|
|`run_configs/*.json`|N607每个正式候选目录中的不可变运行配置|14/14|

`epoch_curves.csv`保留全部2609轮，但仅保留报告所需发布字段；N607原始`metrics_epoch.jsonl`约90MB，完整训练/评估日志共118148行。依据仓库交付规则，原始大日志、checkpoint、数据集和运行目录不进入Git；它们的远端根路径、状态、计数、异常指纹和checkpoint身份已记录在主报告及`summary.json`中。

## 3.复现入口

- 第一批A1～A7：[`../../../code/scripts/launch_phase1_adv3b02_daot_stn_a1_a7_manysig_s392005_20260902.sh`](../../../code/scripts/launch_phase1_adv3b02_daot_stn_a1_a7_manysig_s392005_20260902.sh)
- 第二批P1～P5：[`../../../code/scripts/launch_phase1_adv3b02_daot_stn_rx_v2_p1_p5_manysig_s392005_20260903.sh`](../../../code/scripts/launch_phase1_adv3b02_daot_stn_rx_v2_p1_p5_manysig_s392005_20260903.sh)
- 可选扩展E1/R1：[`../../../code/scripts/launch_phase1_adv3b02_daot_stn_rx_v2_e1_r1_manysig_s392005_20260903.sh`](../../../code/scripts/launch_phase1_adv3b02_daot_stn_rx_v2_e1_r1_manysig_s392005_20260903.sh)
- V1设计追踪：[`../../CVS_PHASE1_ADV3B02_DAOT_STN_V1_TRACE_20260901.md`](../../CVS_PHASE1_ADV3B02_DAOT_STN_V1_TRACE_20260901.md)
- RX-V2设计追踪：[`../../CVS_PHASE1_ADV3B02_DAOT_STN_RX_V2_TRACE_20260903.md`](../../CVS_PHASE1_ADV3B02_DAOT_STN_RX_V2_TRACE_20260903.md)

## 4.解释边界

三视图教师仅属于A2/A3性能上界路径；RX-V2默认采用两次fresh教师前向与Temporal Orbit Memory。全部14行的orbit prototype loss均为0，不能声称prototype蒸馏实际贡献性能。R1虽然配置`lambda_subspace=0.05`，但最终basis/eigenvalues为null且subspace loss全程为0，因此不能把R1结果归因于选择性子空间正则。本批次没有同配置ADV3B02 A0基线、没有多seed确认，也没有Phase2域适应或新类注册结果。
