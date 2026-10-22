# IEEE 实验章节（仅实验部分）

当前预览入口是 `main.tex`，通过 `current_experiments.tex` 引入外部源数量结果和 `cric_dura.tex`。`experiments.tex` 是保留的历史版本。没有摘要、引言、方法、独立讨论或结论章节，也没有虚构作者。

并入正文时复制当前入口引用的 `.tex`、`tables/`、`images/` 和 `references.bib`。导言区使用 `amsmath, amssymb, graphicx, booktabs, array, cite, url`，以及 `\newcommand{\pmv}[2]{\ensuremath{#1\,\pm\,#2}}`。IEEEtran 发行文件保留原许可。

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

2026-10-23 本次更新的 13 页 PDF 已编译并目检。以下面板完成：外部四方法源数量 360 模型/2520 视图单元、最终 T1 F0 三 F seed/单固定 G、统一时延 500 case-round、21 个 G 学生的机制准确率，以及三核心 A/B/C 的 8 组合 × 3 F seed 完整准确率与交互分析。三核心结果见 `core_mechanisms_results.tex` 和 `images/core/` 四组 PDF/PNG 图；其全部分层冻结评分、训练分段成本、24 份源物理诊断和标准库复算器包含在新的 r06 进度包中。

三核心 A 为 CRIC，B 为成对随机预测一致性，C 为完整原生选择性半监督；分组不按收益调整。主表保留 A/C 的非正条件贡献。源物理干预和几何分析只作敏感性证据，不声明完整硬件解耦。C=0 与 G-L 才是全系统 L-only，搭配 G-LU 仍使用无标签数据。三个 F seed 条件于固定 G，不是三次独立完整系统重复。

细粒度 F、严格监督物理结构、扩展数据契约、外部方法 DURA 读出、PAMF PL/G-L 公平性补充及专用机制时延仍未全部闭合。已发布设置与未完成结果均明确区分，r06 是阶段交付，不能称为全部实验最终结果。

旧 no-As 33L/17U F 和最终 T1 的 13L/37LU 配方分列；旧 4.531972 ms 不改标为最终 T1。统一时延的最终 T1 B1 为 4.541187 ms，五轮计时不是五个模型 seed，共驻留显存不冒称单模型占用。外部源数量固定六目标 RX/144000 条观测，不混入 F0 七目标 RX/168000 的五源结果；没有新增五源扫描。所有负结果、零 PL 接受率和非单调曲线保留。

复算只读冻结评分与成本，不读取 IQ、权重或目标真值。统计可用 Python 标准库重建，重新生成图需要 matplotlib；包内已提供矢量图。复算命令见导出包 `reproduce/README.txt`。大体积冻结输入、权重和私有运行状态不提交 Git；公开小表、分析代码、论文源及矢量图按项目约定交付。
