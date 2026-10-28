# 重建说明

在本包根目录使用 Python：

```text
python reproduce/render_figures.py --root . --force
python reproduce/package_documents.py --root .
python reproduce/render_efficiency_tables.py --root .
python reproduce/validate_exports.py --root .
```

依赖 matplotlib、numpy、pandas、Pillow、reportlab、pypdf；中文文档默认使用 Windows SimHei。图表只读取本包冻结目录，效率表只读取本包 CSV。`--force` 仅覆盖本图表包同名图。图集页脚和索引使用新版图号，文件标识保留旧 Fig 追溯。独立效率表另在 tables/；交付时已附在说明 PDF 末尾。
