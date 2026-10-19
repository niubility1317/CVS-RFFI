# RIEI第一模型种子的固定F0配对结果

核验状态：VERIFIED；模型seed2026092701。仅完成三组配对中的第一组，不能报告为三seed结论。时间：2026-10-19T13:37:05+08:00。

run：20261019-phase1-pamf-dg-nativebudget-m24-r01。实际训练代码ac5163f6cb59b49a657c3ee02089f8e556ae2279；只读报告函数3ef79995d4a1ecdf0b90cb9af6feb34668da7cf9。历史run名称含PAMF，本组没有PAMF模型。

ManySig F0，6 TX，源RX索引1/3/4/6/8、源D1/D2/D3，目标RX索引0/2/5/7/9/10/11及全部4日。L_s/U_s/V=6300/56700/27000，U隐藏不用、V标签另列。equalized1、256点、unit-RMS；默认身份头随骨干原流程联合训练，冻结后直接预测。无星地训练增强、目标训练或目标选模。

## 七视图结果

| 视图 | full Accuracy（%） | CE Accuracy（%） | full−CE（百分点） | full F1（%） | CE F1（%） |
|---|---:|---:|---:|---:|---:|
| clean | 60.6286 | 66.5470 | -5.9185 | 58.9658 | 61.9518 |
| practical_high | 53.7708 | 61.9452 | -8.1744 | 47.8851 | 57.5931 |
| practical_mid | 51.4464 | 58.6798 | -7.2333 | 45.4785 | 54.4059 |
| practical_low_suburban | 49.5042 | 55.9637 | -6.4595 | 43.5073 | 51.8714 |
| practical_high_urban | 50.8131 | 58.3077 | -7.4946 | 44.7565 | 54.3131 |
| practical_mid_urban | 39.1833 | 43.8048 | -4.6214 | 34.2788 | 41.1549 |
| practical_low_urban | 31.7571 | 34.3821 | -2.6250 | 27.0451 | 31.2940 |

六practical等权平均：full 46.0792%，CE 52.1806%，差-6.1014个百分点。同包七视图不增加独立样本或模型seed数。

## 训练与独立复算

| 配置 | 实际 epoch | batch 更新 | 各优化器更新合计 | 冻结 epoch | 训练阶段秒数 |
|---|---:|---:|---:|---:|---:|
| full | 100 | 9900 | 29700 | 100 | 795.795 |
| ce | 100 | 9900 | 9900 | 100 | 397.460 |

两侧均为E100、batch64、SGD1e-4、momentum0、恒定LR并取末轮。full保留项目SPAWC2023 short-stem适配及原训练目标；CE仅身份CE。完整方法含三个优化器更新，总更新数与CE不同；相同物理训练曝光不等于相同前向或计算成本。两行均物理源曝光630000，full前向曝光1260000、CE630000。

已先验证两行全部七视图168000包prediction闭合、ID顺序和类别范围，再在独立只读分析进程连接truth。14个总体CM、Accuracy、Macro-F1与原scorer一致；392个RX×日期分层计数与CM闭合。完整200条epoch记录和两份完整stdout均解析，未见Traceback、OOM、非有限loss标记。full的负总loss属于其目标组合的数值，不能仅因负号判为技术失败。

## 当前解释

本seed的完整RIEI在clean和全部六practical的Accuracy均低于自身CE。这是需要保留的负结果，不触发改权重、减少CE预算或选择性重跑；其余两个seed按冻结计划继续。单seed不能推出RIEI整体无效、原论文不成立或三seed稳定退化，最终家族结论待三组配对齐全。

原始证据：local_artifacts/phase1_external_dg_20261019/riei_seed01_frozen_pair_audit.json。包含完整曲线、源冻结、日期/RX/TX与联合分层结果；原始日志、prediction和checkpoint不提交Git。
