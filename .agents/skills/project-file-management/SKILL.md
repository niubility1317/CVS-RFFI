---
name: project-file-management
description: Organize and locate this project's code, reports, reference materials, smoke tests, archives, and duplicate files. Use for project asset management and cleanup; use experiment-management for individual experiment registration and retrieval.
---

# 项目资料管理

先读[统一导航](../../../PROJECT_NAVIGATION.md)和[资料管理说明](../../../docs/project_governance/README.md)。查询优先用`tools/project_files.py find`；实验名称、配置、种子和结果查询使用相邻`experiment-management`技能。旧资料库通过`project_files.py legacy`查询，不把旧清理许可继承到新任务。

## 整理决策

- 先确认当前Git工作树、未提交修改、实际运行引用和文件用途。SQLite/CSV是时间点快照；处理前检查文件当前状态。
- 活动代码、发布镜像和历史工作树即使字节相同也可能职责不同。先记归属；未知分支、未提交修改、未合并代码不按重复副本删除。
- 报告、数据、checkpoint、预测、日志、评分和失败证据保留原路径。只有本次明确范围授权允许删除/覆盖。不要从文件名含`tmp`、`test`、`smoke`或低分推断无用。
- 冒烟源码按可复用测试/一次性诊断归类；冒烟输出按正式报告引用、唯一复现证据、可重建fixture分别判断。只有可重建且无活动引用的缓存可在整理授权内回收；含权重/数据的fixture仍需辨别真实来源。
- 压缩或去重前记录原路径、保留副本、用途和恢复方式；归档内容完整读回校验后才移除明确授权的原副本。独立记录成功/跳过/失败；部分完成时先核对现状再重试。
- Windows路径必须解析到已授权根目录，跳过reparse/junction和Git内部目录；不用跨shell递归删除。不操作N607，除非本次另有相应授权。

## 持续维护

新代码进入所属模块；一次性工具给出用途并关联报告。报告放`docs/`或对应run目录；原始产物保留存储位置，用路径引用。新实验仍先登记`experiment.json/report.md/events.jsonl`。

批量整理后更新`docs/project_governance/current.json`、日期化盘点/处理清单和统一导航；只在需要刷新全量清单时运行scan。删除/压缩清单是快照的增量变更记录。元数据、技能和工具进入Git；SQLite、恢复包和大体积产物留在`local_artifacts/`，以路径引用。
