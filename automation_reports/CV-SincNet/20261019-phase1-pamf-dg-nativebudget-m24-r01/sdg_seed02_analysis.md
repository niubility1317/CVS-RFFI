# SDG第二模型种子的冻结配对结果

核验时间：2026-10-19T12:47:27+08:00。run：20261019-phase1-pamf-dg-nativebudget-m24-r01。model seed：2026092702。

状态：两行七视图独立复算VERIFIED。目前SDG完整配对2/3，第三seed继续训练，不将本表写为三seed结果。固定ManySig F0：5源RX、7目标RX，L/U/V=6300/56700/27000；每视图168000物理包。

| 视图 | full Accuracy（%） | CE Accuracy（%） | full−CE（百分点） | full Macro-F1（%） | CE Macro-F1（%） |
|---|---:|---:|---:|---:|---:|
| clean | 73.1554 | 64.3923 | +8.7631 | 72.9560 | 64.3375 |
| practical_high | 62.0220 | 56.2173 | +5.8048 | 61.5641 | 58.5140 |
| practical_mid | 59.1101 | 53.4827 | +5.6274 | 58.4696 | 55.6324 |
| practical_low_suburban | 56.8512 | 51.3381 | +5.5131 | 56.0720 | 53.4230 |
| practical_high_urban | 59.1220 | 52.9440 | +6.1780 | 58.5079 | 55.0679 |
| practical_mid_urban | 45.8625 | 40.3143 | +5.5482 | 44.9832 | 41.8412 |
| practical_low_urban | 37.0024 | 32.2655 | +4.7369 | 35.9772 | 33.2831 |

六practical等权均值：full 53.3284%，CE 47.7603%，差+5.5681个百分点。

## 训练和来源

| 项目 | full | CE |
|---|---:|---:|
| 实际训练epoch | 200 | 200 |
| 优化器更新 | 39400 | 39400 |
| 按源V最小NLL选中epoch | 187 | 32 |
| 训练阶段用时（秒） | 2406.586 | 1458.396 |

已完整解析两行E1至E200全部结构化记录及完整stdout，未发现Traceback、CUDA OOM或非有限loss标记。已检查scratch、无目标训练／选模、默认头联合训练和实际checkpoint的源NLL冻结依据。完整预测覆盖、类别范围与物理ID顺序先通过，再读取truth，七视图重算混淆矩阵、Accuracy及Macro-F1均与独立scorer一致。

full−CE代表SDG原生增强与对比目标组合的净效果，不能归因到单项loss。两个已完成seed均有clean正差，不替代第三seed、其他接收机组合和ManyRx证据。各行均训练满E200但选中轮数不同；实际训练代价与冻结checkpoint位置分别报告。结果不反馈参数、选模或选择性重跑。

逐RX／日期／TX及全部训练记录保存在local_artifacts/phase1_external_dg_20261019/sdg_seed02_frozen_pair_audit.json；原始训练日志、权重和预测不提交Git。
