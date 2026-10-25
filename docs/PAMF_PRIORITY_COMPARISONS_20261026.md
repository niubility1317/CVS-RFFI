# PAMF 重点对比实验（2026-10-26）

当前主报告按用户指定的五类问题排序。**PAMF 使用红色、PAMF+DURA 使用橙色**；其余方法采用低饱和配色。当前半监督方法仍是普通 PL，方法显示名不带版本后缀。

- [8 页 IEEE 彩色重点报告：9 张图、19 个面板、8 张表](reports/pamf_priority_comparisons_20261026/report.pdf)
- [Excel：35 个工作表、51 个可编辑图表](reports/pamf_priority_comparisons_20261026/weekly.xlsx)
- [可编辑正文](reports/pamf_priority_comparisons_20261026/report.md)
- [IEEE 源码](reports/pamf_priority_comparisons_20261026/ieee/main.tex)
- [主图 PDF、SVG 和 PNG](reports/pamf_priority_comparisons_20261026/figures/)
- [逐图数据和配对变化](reports/pamf_priority_comparisons_20261026/tables/)
- [数值、工作簿及交付核验](reports/pamf_priority_comparisons_20261026/verification.json)

## 五类主实验

| 优先实验 | 图表与学习条件 |
|---|---|
| 监督与半监督比较 | 无增强：监督／普通 PL 双柱，仅 CLEAN；有增强：监督／普通 PL 双柱，CLEAN 与综合星地分两个面板 |
| 接收机划分、标注比例、源接收机数量 | 全部使用增强半监督；每个设置的 CLEAN／综合星地双柱均纳入单 PAMF 和 PAMF+DURA |
| 主实验七场景 | 增强半监督，CLEAN 及六种 practical 逐景，六个主比较项完整保留 |
| 1 至 6 源接收机 | 增强半监督，CLEAN 与综合星地分别用折线；5 源复用同种子的主实验模型 |
| DURA 泛用性 | 增强半监督，每种方法未接入／接入恢复器双柱；包含 RIEI、DRIFT、IB-RFF、SDG、PAMF |

“综合星地”仅平均六种 practical，不含 CLEAN；先在每个模型种子内平均，再统计三个种子的均值与样本标准差。PAMF+DURA 的执行顺序仍为先 DURA、再 PAMF，且两种 PAMF 展示项共享识别器。

## PAMF 补回与缺项

无增强监督 PAMF 已从完整冻结包的既有监督参照补回。其原有辅助槽配方如实保留，不能把它与当前有增强 PAMF 主配方的全部分差归因于增强。**当前普通 PL 的匹配无增强 PAMF 仍缺项**，用户已选择先交付已有结果；图中留缺，不填零、不用历史复杂半监督结果代替。

其他重点图均有对应 PAMF。DURA 泛用性由此前四个外部方法扩展为包含 PAMF 的五种识别器；原来的四方法完整面板降为补充资料。

## 5 源接收机的可比口径

主实验原有 5 源、7 目标，其余自然源数量设置使用共同的 6 目标。本版 5 源仍使用主实验的相同模型和种子，只从已经冻结的逐接收机计数中汇总共同目标 `[0,2,5,7,9,10]`，每景 108000 包。因此 1 至 6 折线固定共同目标集合；主比较原 7 目标、每景 126000 包的成绩保持不变。

该汇总没有重新预测、读取逐样本真值或重新训练。训练量仍随源接收机数自然改变，不将曲线称为固定样本预算下纯接收机多样性的因果效应。

## 文件与保留范围

Excel 前 9 页为重点实验，后 26 页为补充，保留原 24 个工作表的结果，并增加原两套 11 设置 DURA 全面板。每个重点页有实验说明、对应数据划分、可编辑图、来源数值、三个模型种子的结果和复制列。原 26 份独立图、r20 PDF 与所有旧包保留。

新包为 `exports/reports/20261026-PAMF-priority-comparisons-r22.zip`。r22 统一补充图的 Macro-F1、模型数量和时延坐标单位，统计值与 r21 相同。纯标准库复核会从冻结计数、种子统计和保存的工作簿复算：`python -X utf8 -S reproduce/verify_priority.py --root .`。

本次新增训练、预测、独立评分与 DURA 训练均为 0。原数据、权重和其他任务未改动。
