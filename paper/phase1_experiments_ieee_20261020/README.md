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
