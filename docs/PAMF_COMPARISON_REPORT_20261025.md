# PAMF 与四种对比方法实验报告（2026-10-25）

当前报告采用 **IEEEtran 双栏论文风格**，中文正文、Times New Roman 英文、编号表格及矢量图。Excel 同样采用黑白三线表、宋体中文与 Times New Roman 英文/数字。主对比固定为 **RIEI、DRIFT、IB-RFF、SDG、PAMF、DURA+PAMF**：前五项直接识别，仅最后一项接入恢复器。PAMF 当前暂用普通伪标签。

- [8 页 IEEE 分组报告](reports/pamf_comparison_main_and_dura_20261025/report.pdf)
- [可编辑报告正文](reports/pamf_comparison_main_and_dura_20261025/report.md)
- [IEEE LaTeX 源码](reports/pamf_comparison_main_and_dura_20261025/ieee/main.tex)
- [Excel 周报数据：10 个工作表](reports/pamf_comparison_main_and_dura_20261025/weekly.xlsx)
- [分组规则](PAMF_COMPARISON_REPORTING_RULES_20261025.md)
- [当前 PAMF 方法定义](PAMF_CURRENT_METHOD_20261025.md)
- [图表与逐模型分组表](reports/pamf_comparison_main_and_dura_20261025/tables/)
- [数值、Excel 与版式核验](reports/pamf_comparison_main_and_dura_20261025/verification.json)

## 实验部分与测试范围

| 实验部分 | 报告范围 |
|---|---|
| 主方法对比 | 外部方法直接识别，PAMF 与 DURA+PAMF 分列；当前固定有信道增强训练组 |
| 无信道增强控制 | 仅目标接收机测试集 CLEAN；不报告 practical |
| 有信道增强控制 | CLEAN 和全部六种 practical，按监督/普通伪标签分行 |
| DURA 泛用性 | 固定同一外部识别器，单独比较接入前后；不混入主方法成绩 |
| PAMF 自身对照 | 监督与普通伪标签分开；固定恢复器数据权限后解释伪标签收益 |

当前整理 594 个已完成逻辑模型，逐模型分组见 `experiment_groups.csv`。主比较使用 165 个独立识别器、198 个模型与推理链展示项；PAMF 与 DURA+PAMF 共享 33 个识别器。额外 33 个 PAMF 监督模型属于自身控制。

标准划分跨接收机六景准确率：RIEI 49.9512%、DRIFT 56.7554%、IB-RFF 52.5705%、SDG 65.3010%、PAMF 71.0507%、DURA+PAMF 78.5161%。各方法均来自固定有增强＋普通伪标签训练组；四种外部方法未接入 DURA。

当前统计导出保留 12320 行汇总和 36960 行逐种子指标。无增强 practical 与无增强源接收机跨日诊断不纳入当前报告；原冻结资料及旧包保留。Excel 包含可复制的均值±标准差及独立数值列，全部 36864 个来源数值单元格读回一致，公式未发现错误。

新包为 `exports/reports/20261025-PAMF-PL-comparison-r19.zip`，包含 IEEE 源码、9 份矢量图、Excel、统计表和纯标准库核验器。本次新增训练、预测、真值评分、DURA 训练均为 0；旧 r18 及更早包不覆盖。自身消融缺项见报告第 XIII 节，本次没有发布补实验。

网络结构另见[仅分类损失监督训练的分支消融详细报告](PAMF_BRANCH_ABLATION_REPORT_20261025.md)。
