# PAMF 网络分支消融：IEEE 版与两类综合测试（2026-10-26）

仅分类损失的监督训练，PAMF直接识别。保留跨日跨接收机主分析，新增跨接收机与跨日跨接收机两类互斥测试群体的综合统计。完整网络与五个变体各3个模型种子，共18模型，复用既有冻结结果。

- [四页 IEEE 双栏报告](reports/pamf_branch_ieee_two_population_20261026/report.pdf)
- [可编辑报告正文](reports/pamf_branch_ieee_two_population_20261026/report.md)
- [综合图 PDF](reports/pamf_branch_ieee_two_population_20261026/figures/composite_testing.pdf)
- [综合图可编辑 SVG](reports/pamf_branch_ieee_two_population_20261026/figures/composite_testing.svg)
- [IEEE LaTeX 源码](reports/pamf_branch_ieee_two_population_20261026/ieee/main.tex)
- [综合与分群统计表](reports/pamf_branch_ieee_two_population_20261026/tables/)

两类每景分别为126000、42000包，合并后168000包，权重为75%和25%。综合准确率按实际样本统计；宏平均F1由两类混淆矩阵合并后重算。每种子、每景先合并，再计算六景均值，最后统计三个种子的均值与标准差。不是简单平均三个群体的百分数。

完整PAMF：综合clean准确率82.67±1.26%，综合六景准确率70.91±0.46%，综合六景宏平均F1为70.65±0.41%。综合六景相对变化如下，仍为“变体减完整网络”，负值下降、正值提高。

| 结构变体 | 综合六景准确率变化（百分点） |
|---|---|
| 去除显式时间坐标 | -0.55±0.95 |
| 去除路径描述支路 | -0.45±0.61 |
| 去除功放相关支路 | -6.82±1.01 |
| 去除参考响应支路 | -0.51±0.03 |
| 打乱参考模板相位 | -0.24±0.11 |

五项结构仍为去显式时间坐标、去路径描述支路、去功放相关支路、去参考响应支路、打乱参考模板相位。综合统计中跨接收机群体占75%，因此综合结果与单一分群结果可能方向不同，二者均保留。

不展示其他推理链或效率内容。本次新增训练、预测、逐包真值访问均为0。新不可覆盖包为`exports/reports/20261026-PAMF-branch-IEEE-r23.zip`，r21与更早包保留。解包可用纯Python标准库从筛选后的冻结混淆矩阵复算。
