# PAMF 与四种对比方法实验报告（2026-10-25）

> 当前使用 [2026-10-26 重点对比版](PAMF_PRIORITY_COMPARISONS_20261026.md)：按用户五类问题排序，配对监督／半监督、补入 5 源折线、PAMF 纳入 DURA 泛用性，并保留全部补充工作表。下文为 r20 归档说明，原文件与数据保留。

当前报告先区分 **监督学习／普通 PL 半监督**，每组再分训练信道增强、推理恢复链和测试人群。各组均列 RIEI、DRIFT、IB-RFF、SDG、PAMF、DURA+PAMF；前五项直接识别，外部接入 DURA 只在泛用性部分报告。PAMF 当前半监督方案暂用普通 PL。

- [16 页 IEEE 双栏报告：23 幅图、20 张表](reports/pamf_comparison_main_and_dura_20261025/report.pdf)
- [可编辑报告正文](reports/pamf_comparison_main_and_dura_20261025/report.md)
- [IEEE LaTeX 源码](reports/pamf_comparison_main_and_dura_20261025/ieee/main.tex)
- [Excel 周报：24 个工作表、24 个可编辑竖柱图](reports/pamf_comparison_main_and_dura_20261025/weekly.xlsx)
- [26 份独立矢量图和预览](reports/pamf_comparison_main_and_dura_20261025/figures/)
- [按学习方式分开的结果表与逐图源数据](reports/pamf_comparison_main_and_dura_20261025/tables/)
- [实验分组规则](PAMF_COMPARISON_REPORTING_RULES_20261025.md)
- [当前方法定义](PAMF_CURRENT_METHOD_20261025.md)
- [独立数值、工作簿与版式核验](reports/pamf_comparison_main_and_dura_20261025/verification.json)

## 阅读结构

两种学习方式分别包含标准对比、接收机重划分、标注比例、源接收机数量、十类规模、无增强 CLEAN 控制、有增强逐景、DURA 泛用性、PAMF 恢复前后对照与时延。无增强仅报告目标接收机 CLEAN；有增强报告 CLEAN 和全部六种 practical。

Excel 每页都保留实验目的、固定与变化因素、学习方式、恢复器数据权限、数据划分、可编辑图表和可复制数值。DURA 使用前后双柱，另列准确率百分点变化。监督组的 PAMF 与恢复器均不使用无标签源训练池；半监督组允许普通 PL 和匹配的全源 IQ 恢复器。

监督组，标准划分、目标跨接收机六景准确率（%）：

| 方法 | RIEI | DRIFT | IB-RFF | SDG | PAMF | DURA+PAMF |
|---|---|---|---|---|---|---|
| 监督 | 50.08±1.60 | 55.22±5.28 | 53.10±1.83 | 65.16±2.03 | 71.29±0.90 | 77.36±0.55 |

普通 PL 半监督组，相同测试定义：

| 方法 | RIEI | DRIFT | IB-RFF | SDG | PAMF | DURA+PAMF |
|---|---|---|---|---|---|---|
| 普通 PL 半监督 | 49.95±1.55 | 56.76±2.29 | 52.57±2.17 | 65.30±1.14 | 71.05±0.91 | 78.52±0.12 |

上表仅帮助定位，两组在正式图表与 Excel 中分别展示。不同方法保留原生预算和有标签目标差异；监督与半监督完整系统的差异还包含恢复器源 IQ 使用范围，不能全部归因于伪标签。

全部仍为 594 个逻辑识别器，每组 297 个；主比较共 330 个独立识别器、396 个模型与推理链展示项。PAMF 与 DURA+PAMF 共享识别器。保留 12320 行汇总、36960 行逐种子指标；当前工作簿独立读回核对 48032 个来源数值和 972 个划分数值，未发现公式错误。

新包为 `exports/reports/20261025-PAMF-PL-comparison-r20.zip`，包含报告、Excel、26 份矢量图、逐图数据、冻结统计和纯标准库复核器。旧 r19 及更早包不覆盖。本次新增训练、预测、真值评分与 DURA 训练均为 0；当前 PAMF 自身消融缺项在报告末尾如实列出，没有自动发布补实验。

结构研究另见[仅分类损失监督训练的分支消融报告](PAMF_BRANCH_ABLATION_REPORT_20261025.md)。
