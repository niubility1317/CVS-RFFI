# IEEE 实验章节（仅实验部分）

交付主体是 `experiments.tex`，可直接插入正式论文正文。它只包含实验设置、对比/消融结果、源数量与标签效率、计算成本；没有摘要、引言、方法章节、独立讨论、结论或附录。

`main.tex` 仅作为 IEEEtran journal 双栏预览入口，包含本章节及参考文献，不含标题页或虚构作者。正文图表自动编号，真实论文中会跟随已有章节编号。

## 并入论文

1. 将 `experiments.tex`、`cric_dura.tex`、`tables/`、`images/` 复制到正文工程，合并 `references.bib` 条目。
2. 在正文中使用 `\input{experiments}`。
3. 预导言区需要 `amsmath, amssymb, graphicx, booktabs, array, cite, url`，并定义：`\newcommand{\pmv}[2]{\ensuremath{#1\,\pm\,#2}}`。`hyperref` 可按全文已有设置保留。

## 预览与数值来源

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

四幅PDF矢量图、输入CSV及生成表格的 `build_tables.py` 一并放入ZIP。`python build_tables.py` 可以从冻结评分重建外部方法表格，不训练、不重新评分。新增 `cric_dura.tex` 纳入已有 F0 完整 CRIC＋DURA 三F种子／单G七景成绩，并将未测的跨配置、机制和时延设计明确标为 prospective。复算器为仓库 `experiments/cric_dura_comparison/build.py`，不访问IQ、模型或truth；其小型逐seed、汇总和配对差CSV位于source_data。完整原始分层结果与训练明细沿用已完成实验交付包。

图形、CSV、PDF编译产物不提交Git；章节、表格、生成器、引用及IEEEtran未修改发行文件进入Git。IEEEtran类/样式保留原版权与许可头。格式参考：https://template-selector.ieee.org/ 。

论文使用边界：5源/7目标RX；ManyRx16/16；监督和PL都使用拼接星地增强；训练标签与V标签分列；RIEI/DRIFT E100属于实现预算；未来结果仅D3；模拟残余信道不等于真实在轨。所有负结果与零PL接受保持。

2026-10-21 current entry: main.tex includes current_experiments.tex and cric_dura.tex. Independent ResNet18/50 CE/generic legacy studies are excluded from this current experimental chapter; experiments.tex is retained as historical source. Final T1 is13L37U and historical old no-As33L17U results are explicitly separated. New finalT1/DURA cells remain pending until scored.

最终T1 F0 三seed七景子矩阵已闭合，tables/final_t1_f0.tex与source_data/final_t1_f0_*.csv为实际评分复算。其余17契约扩展、机制、时延尚在运行。旧no-As33L17U表保留独立历史标签，不能混为最终T1。

2026-10-21 timing update: final T1+DURA unified timing-r02 complete (500 case-rounds,140000 GPU wall calls,10000 transfer calls). Tables/final_t1_latency.tex contains all B1 comparator calls, same-weight G/GF execution routes and default batch measurements. B1 4.541187ms GPU/P95 4.725135ms/CPU-transfer median4.611311ms; earlier4.531972ms retains historical F50 semantics. Other contract/mechanism experiments remain running.

2026-10-22 source-count completion: sourcecount_results.tex reports all360 external models/2520 view cells,93,240 checked confusion matrices,6,440 matched audit statistics,10 vector-figure panels and Full/CE/PL paired gains. Natural/fixed and six-target vs seven-target F0 stay separate. PAMF extended contract/mechanism rows continue.
