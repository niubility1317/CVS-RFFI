# IEEE 双栏报告：含两类综合测试

main.tex采用IEEEtran journal、10pt、双栏；英文Times New Roman，中文宋体。包含三张三线表、四幅矢量图。实验报告未声明正式发表或投稿。

在本目录使用已安装相应字体、IEEEtran和xeCJK的XeLaTeX编译两次：

```text
xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex
```
