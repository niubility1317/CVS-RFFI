# IEEE 双栏实验报告

`main.tex` 使用 IEEEtran journal、10 pt、双栏，英文为 Times New Roman，中文为宋体。图表使用独立矢量 PDF。本文为中文实验报告，未虚构作者单位、投稿状态或文献。

使用已安装 IEEEtran、xeCJK 的原生 XeLaTeX：

```text
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

完整结果在附录之前；无增强目标域 CLEAN、有增强 CLEAN＋practical、DURA 泛用性、PAMF 自身对照分别成节。机器可读表与标准库核验位于上一层目录。
