# ADV3B02-ECRS-V1实验数据包

本目录是run `phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_direct8_20260902_r2`的可复算汇总，不包含体积较大的原始stdout、checkpoint或诊断张量。汇总由`tools/analyze_adv3b02_ecrs_v1_run.py`从只读快照生成，所有文本采用UTF-8。

## 快照边界

- 快照时间：2026-09-05 23:40（Asia/Hong_Kong）。
- 结构化训练记录：1500条；R1–R6各200条，R7为194条，R8为106条。
- R1–R6：已完成E200、clean与三类LEO最终评测，并有source-only诊断张量。
- R7：快照时仍在运行，因此本数据包不把其末行当作最终性能。
- R8：E106发生AMP/BCELoss兼容性异常，保留checkpoint但没有最终clean/LEO性能。
- 每个rung的CSV、JSONL和连续epoch序列已交叉核对；异常见`anomalies.csv`。

## 文件说明

|文件|内容|
|---|---|
|`run_summary.csv`|每个rung的状态、epoch闭合、末轮/最佳source-val、最终clean主指标与跳过批次|
|`epoch_metrics_full.csv`|1500条逐epoch原始结构化指标，前置增加`rung`和ECRS阶段|
|`stage_summary.csv`|按E1–40、E41–90、E91–200三阶段汇总训练、验证和耗时|
|`training_diary.csv`|阶段边界与关键epoch读数及相邻里程碑变化|
|`evaluations.csv`|`FINAL-BEST`与`FINAL-PRIMARY`的clean、分组clean和三类LEO结果|
|`receiver_results.csv`|最终clean按target receiver和seen/unseen day分解|
|`diagnostics_summary.csv`|R1–R6响应拟合方向误差、协方差、曲面幅值和分块可辨识性|
|`probe_results.csv`|source-only诊断payload上的确定性group-split最近质心probe|
|`anomalies.csv`|快照截断、反向批次跳过及R8确定性异常|
|`summary.json`|生成方法、计数、文件清单和解释边界|

## Probe定义与边界

`probe_results.csv`使用`physical_sample_id`的SHA1确定性80/20分组切分，使同一物理样本的clean/LEO视图落在同一侧；仅用训练侧均值/标准差标准化，再以最近质心分类。它是source-only描述性诊断，不是注册实验、target query结果或可用于晋级的主指标。`majority_pct`是同一测试集的多数类基线；negative-control行只来自首个诊断批次，测试样本很少，必须谨慎解释。

## 复算命令

在包含原始快照的授权环境中运行：

```text
conda run -n ssr-gpu python tools/analyze_adv3b02_ecrs_v1_run.py --snapshot <snapshot_dir> --output docs/experiments/results/phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_direct8_20260902_r2
```

脚本不会修改输入快照，也不会读取或使用Phase2 query truth。若使用更新后的R7快照复算，必须同时更新正式报告中的快照时间、epoch总数和状态。
