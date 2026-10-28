# 重建方法

使用 Python 与 matplotlib、numpy、pandas 重建逐图 PDF / SVG / PNG：

```text
python reproduce/render_figures.py --root . --force
```

`--force` 仅覆盖本图表包内同名图，不触及训练产物。需要重建独立说明和合并图集时，再安装 reportlab、pypdf；文档默认使用 Windows SimHei 字体：

```text
python reproduce/package_documents.py --root .
python reproduce/validate_exports.py --root .
```

检查导出还需要 Pillow。所有操作仅读取本包内冻结统计；不加载模型或原始 IQ，不访问真值。
