# 项目资料整理结果（2026-09-16）

## 结果

本机全量盘点947,833份文件、173.35GiB逻辑大小；不含Git内部、reparse目标及本次扫描自己的输出。扫描错误3项，跳过边界3902项，详情见inventory_summary.json。文件表反映清理前状态，结合cleanup_manifest.json使用，不代表运行状态。

已建立统一导航、可查询SQLite、版本化压缩CSV及代码/报告/冒烟资料分目录。旧资料治理库保留并接入只读桥接；实验检索使用既有experiment_registry。整理不启动实验、不修改N607。

## 已实际处理

- 移除674个可重建Python/pytest缓存文件，原逻辑大小17,807,168字节；已校验恢复包7,482,329字节。
- 两个已被替代的生成索引从未压缩副本转为gzip恢复副本，压缩前后完整SHA-256一致，原文件移除后再次检查不存在。
- 扣除恢复包后净减少945,426,454逻辑字节，即901.63MiB。未测量NTFS实际空闲空间变化。
- 清理仅限普通源码区域中不属于Git工作树的缓存，以及本次明确列出的旧生成索引；路径逐项解析，跳过Git边界及reparse，不递归删除目录。

## 有用资料与重复文件判断

- 62个Git工作树全部保留：31个有未提交内容，31个干净但仍有独立分支来源。未确认合并与依赖前不删除。
- 18263个Python来源检查：504个与发布面字节相同，属于职责镜像；58个存在差异，作为独立变体保留；17701个未找到同路径Git镜像，保留原位置。数量包含本地复现/展开目录，不等于17701个独立研发模块。
- 报告/参考资料75,613个路径纳入reports.csv.gz。正式报告、论文、协议、失败诊断、数据和权重保持原路径。
- 冒烟/测试相关116,659个路径纳入smoke_and_tests.csv.gz，其中历史pytest fixture分类33,770个。源码继续保留用于回归；输出可能包含模型、数据和唯一证据，不能依据fixture分类直接删除。
- `tmp_*`、旧发布包、解压目录及实验副本没有足够逐项依赖证据时保留，均在全量路径表可检索。该结论不等于它们永久不可清理，也不宣称已经消除所有重复文件。

## 新文件放置

可复用源码进入code/tools/tests；一次性脚本注明关联报告；报告进入docs或所属run；新实验先用experiment-management登记名称、说明、数据/seed、命令与输出位置。未来资料整理按project-file-management技能执行，避免在根目录继续堆放无说明副本。

## 验证与恢复

完整清单见cleanup_manifest.json；每个被移除缓存有路径和SHA-256，zip逐项读回一致；两个gzip完整解压流校验一致。恢复时只解压所需文件到清单原路径，不覆盖后来新生成文件。Git原有暂存/未暂存内容在交付时独立核对。

分类统计是管理用途，不用于推断科学结果有效性；历史实验是否完成仍查对应记录与实时证据。没有对历史数据重新验证，没有读取truth执行训练或选模。

## 分类计数

| 分类 | 文件数 | 逻辑字节 |
|---|---:|---:|
| config_or_structured_evidence | 287801 | 36798716433 |
| source_keep | 205570 | 10339051444 |
| log_or_evidence_keep | 171965 | 15803872723 |
| report_or_reference_keep | 75613 | 2481675009 |
| test_or_smoke_source | 71244 | 714490443 |
| governance_metadata | 47692 | 12069668257 |
| historical_test_fixture | 33770 | 468713480 |
| regenerable_python_cache | 20080 | 358772434 |
| protected_data_or_prediction | 17710 | 16277733887 |
| other_review | 9745 | 28888714535 |
| oneoff_source_keep | 2871 | 386803861 |
| protected_checkpoint | 2577 | 16035409127 |
| archive_or_release | 956 | 45501082746 |
| regenerable_pytest_cache | 239 | 4294288 |

## 盘点未覆盖范围

根目录`.pytest_cache`、`code/.pytest_cache`、`tmpwhf_cnj1`返回Windows拒绝访问，已记录并跳过，没有改ACL或尝试删除。3902个边界记录包括Git内部、reparse点与本次扫描输出；因此“全量盘点”指可访问的普通文件，不包含这些边界的内容。

## 工具验证

`conda run --no-capture-output -n ssr-gpu python -X utf8 -m pytest tests/test_project_files.py -q -p no:cacheprovider`通过3项测试，覆盖扫描自身输出/Git排除、分类保护和分类查询。
