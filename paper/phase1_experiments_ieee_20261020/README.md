> 2026-10-26 当前入口：[IEEE 彩色重点实验报告与 35 页 Excel](../../docs/PAMF_PRIORITY_COMPARISONS_20261026.md)。主图按用户五类问题排序，PAMF 以红色突出、PAMF+DURA 以橙色突出；5 源沿用同种子主模型的共同 6 目标统计。无增强普通 PL PAMF 缺项已明确标注，用户选择先交付已有结果；未发布新增训练。旧 r20 报告与全部原图继续保留。

> 2026-10-25 当前状态：PAMF 暂用普通伪标签。当前使用[16 页 IEEE 双栏报告、23 幅图及 24 个工作表的 Excel 周报](../../docs/PAMF_COMPARISON_REPORT_20261025.md)，其 [LaTeX 入口](../../docs/reports/pamf_comparison_main_and_dura_20261025/ieee/main.tex)独立保存。监督与普通 PL 半监督分别成章成图，所有方法明确学习方式；主表外部方法不接 DURA，PAMF 与 DURA+PAMF 分列；无增强只报目标 CLEAN，有增强报 CLEAN 与六种 practical，DURA 泛用性按学习方式分别使用恢复前后双柱。此目录原 `main.pdf` 仍为已退出方案的历史草稿，旧源码、图片和包保留。完整自身消融尚需按新方法匹配，未自动发布训练。

# 历史 IEEE 实验草稿：已退出的外推伪标签方案

入口 `main.tex` / `main.pdf`，只包含实验部分。`current_experiments.tex` 描述数据/方法/配方，接入 `pamf_repl_results.tex` 的完整结果；正文结果在 appendices 前。该历史版本 5 页，四外部只 Full D，其他 A/B/C/D 在补充图表保留。

该历史稿的 PAMF 比较使用 PAMF-REPL；匹配监督和普通 PL 单列，严格 CE 是独立结构层。旧 T1 及旧专用结果源文件是历史记录，当前主文件不导入、不改名。

27 张独立图的 PDF/SVG/预览在 `images/repl/`；英文图注见 `repl_figure_captions_en.md`。完整 4800 dpi PNG/TIFF、逐图源 CSV、28 页中文指南和纯标准库复算输入在新包 `exports/phase1_external_dg/20261025-PAMF-REPL-complete-r17.zip`。

复现编译：pdflatex main.tex；bibtex main；pdflatex main.tex 两遍。包中标准库统计复算见其 README，无需 LaTeX/Torch。

旧正确 r14/r15 和 `exports/unified_figures/20261024-practical-residual-paper-r03` 保留。旧错误 2046 项清理已闭合，本次没有再次删除或隔离。
