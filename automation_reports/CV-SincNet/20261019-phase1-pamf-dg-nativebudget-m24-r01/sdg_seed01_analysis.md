# SDG首个模型种子的冻结配对结果

核验时间：2026-10-19T12:26:50+08:00。run：20261019-phase1-pamf-dg-nativebudget-m24-r01。model seed：2026092701。

状态：两行七视图独立复算VERIFIED；SDG家族三种子尚未齐全，不计算跨种子SD或显著性。固定ManySig F0：5源RX、7目标RX，L/U/V=6300/56700/27000；每视图168000包。

## 结果

| 视图 | full Accuracy（%） | CE Accuracy（%） | full−CE（百分点） | full Macro-F1（%） | CE Macro-F1（%） |
|---|---:|---:|---:|---:|---:|
| clean | 73.2315 | 65.9226 | +7.3089 | 72.9424 | 65.7824 |
| practical_high | 65.3768 | 64.1821 | +1.1946 | 64.7761 | 60.4933 |
| practical_mid | 61.9857 | 61.2321 | +0.7536 | 61.2360 | 58.2531 |
| practical_low_suburban | 59.4875 | 58.9958 | +0.4917 | 58.6496 | 56.5113 |
| practical_high_urban | 62.0470 | 60.5149 | +1.5321 | 61.2191 | 57.7880 |
| practical_mid_urban | 47.6935 | 45.8351 | +1.8583 | 46.3501 | 45.4914 |
| practical_low_urban | 37.9935 | 36.5202 | +1.4732 | 36.5868 | 36.7957 |

六practical等权均值：full 55.7640%，CE 54.5467%，差+1.2173个百分点。

## 训练与核验

| 项目 | full | CE |
|---|---:|---:|
| 实际训练epoch | 200 | 200 |
| 优化器更新 | 39400 | 39400 |
| 按源V最小NLL选中epoch | 179 | 39 |
| 训练阶段用时（秒） | 2449.643 | 1459.996 |

两行完整解析E1至E200全部结构化训练记录及完整stdout；未发现Traceback、CUDA OOM或非有限loss标记。日志含原作者增强的NumPy弃用警告和same padding警告，已保留，不将其当训练失败。

检查了默认头联合训练、scratch来源、无目标选模和实际checkpoint的源NLL冻结依据；七视图预测完整性、物理ID顺序和类别范围先通过，再连接truth，逐视图重算混淆矩阵、Accuracy和Macro-F1，与独立scorer一致。

此处full−CE是SDG原生增强与对比目标组合的训练流程净效果，不能单独归因于某一个loss。当前一个seed的clean及六场景Accuracy为正差，但low urban的Macro-F1略低；不据此宣称所有指标全面领先，也不预判后续两个seed。

源已见／未见日期、逐RX、每TX召回／精确率及全部200轮记录在本地证据local_artifacts/phase1_external_dg_20261019/sdg_seed01_frozen_pair_audit.json；不将原始日志、预测或权重提交Git。所有结果不回流改参数、重排或选择性重跑。
