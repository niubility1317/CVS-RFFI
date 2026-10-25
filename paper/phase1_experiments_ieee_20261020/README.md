> 2026-10-25 当前状态：PAMF 已暂改为普通伪标签。此目录原 `main.pdf` 是先前外推方案的历史 IEEE 草稿，不能改名当作当前稿。当前先使用[普通伪标签与四方法对比报告](../../docs/PAMF_COMPARISON_REPORT_20261025.md)；旧源码、图片和不可覆盖包保留。完整自身消融尚需按新方法匹配，未自动发布训练。

# 历史 IEEE 实验草稿：已退出的外推伪标签方案

入口 `main.tex` / `main.pdf`，只包含实验部分。`current_experiments.tex` 描述数据/方法/配方，接入 `pamf_repl_results.tex` 的完整结果；正文结果在 appendices 前。该历史版本 5 页，四外部只 Full D，其他 A/B/C/D 在补充图表保留。

该历史稿的 PAMF 比较使用 PAMF-REPL；匹配监督和普通 PL 单列，严格 CE 是独立结构层。旧 T1 及旧专用结果源文件是历史记录，当前主文件不导入、不改名。

27 张独立图的 PDF/SVG/预览在 `images/repl/`；英文图注见 `repl_figure_captions_en.md`。完整 4800 dpi PNG/TIFF、逐图源 CSV、28 页中文指南和纯标准库复算输入在新包 `exports/phase1_external_dg/20261025-PAMF-REPL-complete-r17.zip`。

复现编译：pdflatex main.tex；bibtex main；pdflatex main.tex 两遍。包中标准库统计复算见其 README，无需 LaTeX/Torch。

旧正确 r14/r15 和 `exports/unified_figures/20261024-practical-residual-paper-r03` 保留。旧错误 2046 项清理已闭合，本次没有再次删除或隔离。
