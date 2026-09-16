# 项目资料管理

本入口整合2026年8月的资料治理数据库和2026年9月的实验登记体系。整理采用稳定路径加分类索引：代码/报告/实验产物已有相互引用，保留有用资料的原路径；只对已确认的缓存和冗余生成文件执行回收。

## 查询

在项目根目录，使用已验证的`ssr-gpu` Python：

```text
python -X utf8 tools/project_files.py find smoke --limit 20
python -X utf8 tools/project_files.py find DAOT --category report_or_reference_keep
python -X utf8 tools/experiment_registry.py search DAOT
python -X utf8 tools/project_files.py legacy status
```

`find`读取[current.json](current.json)指向的本机SQLite，返回匹配总数与路径；它是盘点时快照，当前存在性需核实，并结合[处理清单](20260916/cleanup_manifest.json)。独立克隆可查版本化`files.csv.gz`，或用scan生成自己的SQLite；本机旧库和绝对路径不能假设在克隆存在。

```text
python -X utf8 tools/project_files.py scan --database local_artifacts/project_file_management/NEW_DATE/inventory.sqlite --report-dir docs/project_governance/NEW_DATE
```

扫描跳过Git内部和reparse目标，不越界扫描或远端操作；新日期避免覆盖历史盘点。扫描后核对errors/boundaries再更新current.json。

## 当前整理资料

- [本次结论与保留理由](20260916/REPORT.md)
- [全量目录统计](20260916/directory_map.csv)、[盘点摘要及扫描边界](20260916/inventory_summary.json)、[完整文件表](20260916/files.csv.gz)
- [代码归属及镜像差异](20260916/code_ownership.csv)、[Git工作树状态](20260916/git_worktrees.json)
- [报告与参考资料目录](20260916/reports.csv.gz)、[冒烟/测试资料目录](20260916/smoke_and_tests.csv.gz)
- [清理与恢复映射](20260916/cleanup_manifest.json)
- [实验登记体系](../../experiment_registry/README.md)

## 历史资料系统

2026年8月扫描入口位于本机`code/snapshots/project_governance_20260813_wt/docs/project_governance/latest.json`，SQLite位于`local_artifacts/project_governance/PGOV_20260818T062450Z/governance.sqlite`。旧扫描记有619283个asset记录；这是历史记录数，不是当前文件数或独立实验数。桥接命令只开放status/find/experiment/repo/review查询；旧CLI在本机status输出COMPLETE且返回码为1，判断应同时读JSON和实际文件。

## 归档与恢复

当前恢复包位于`local_artifacts/project_file_management/20260916/`。`removed_caches.zip`保存已移除缓存的相对路径；只恢复需要的单个文件，先检查原路径是否已有新文件，禁止直接覆盖新结果。`artifacts.jsonl.gz`和`facts.jsonl.gz`是上一轮被替代的生成索引，清单记录原路径；gzip解压可恢复。正式实验索引仍为`experiment_registry/`。

清单里的大小为逻辑字节，Windows压缩、稀疏文件和分配单元会影响实际磁盘空间；不把逻辑减少量当作实测磁盘释放量。
