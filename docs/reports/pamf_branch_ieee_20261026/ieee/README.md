# IEEE 双栏版

main.tex 采用 IEEEtran journal、10 pt、双栏，英文为 Times New Roman，中文为宋体；两张黑白三线表、三幅矢量图。此文件是实验报告，不表示正式发表或投稿。

在本目录使用已安装相应字体与 IEEEtran/xeCJK 的 XeLaTeX 编译两次：

```text
xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
```
