# 项目统一导航

## 从哪里找

| 需求 | 入口 |
|---|---|
| 实验名称、说明、状态证据、数据/seed、配置、模型与输出 | [实验总索引](experiment_registry/README.md) |
| 代码、报告、冒烟、临时文件、归档的路径与用途 | [资料管理](docs/project_governance/README.md) |
| 数据协议、方法权限、truth-last边界 | [项目协议](docs/PROJECT_PROTOCOL.md)；Git克隆见[协议镜像](docs/PROJECT_PROTOCOL.md) |
| 当前执行流程、控制面责任 | [职责映射](tools/optimizer_control_manifest.md) |
| 启动实验 | [实验管理技能](.agents/skills/experiment-management/SKILL.md)、[CVS执行技能](.agents/skills/cvs-experiment-workflow/SKILL.md) |
| 资料整理、去重与归档 | [资料管理技能](.agents/skills/project-file-management/SKILL.md) |

## 存储职责

- `code/`、`tools/`、`tests/`、`configs/`：当前本地源码、工具、测试、配置。`github_publish/CVS-RFFI-repo/`是Git发布承载面，镜像不按冗余文件删除。
- `code/snapshots/`及其他Git工作树：历史/并行开发变体，先看分支与未提交修改；目录名字不代表可删除。
- `automation_reports/CV-SincNet/<run_id>/`：实验登记与报告；`experiment_registry/`负责快速定位。
- `docs/`、`analysis/`、`paper/`、`paper_reproduction/`、`RFFI少样本学习/`：项目文档、分析、论文和复现资料，按资料清单查询。
- `local_artifacts/`、`remote_artifacts/`、`logs/`、`runs/`、`outputs/`：本地产物/远端证据副本；不凭日期或名字推断有效性。
- `releases/`、`release_archives/`、`release_artifacts/`、`release_packages/`、`runner_staging/`：发布和恢复材料，删除前确认保留副本及调用路径。
- `.codex_tmp/`、`tmp/`、`analysis_tmp/`和根目录`tmp_*`：待按用途检索的一次性工具与证据，不能整目录清空。

新对话先查入口与命中记录，无需重新递归扫描整个工作区。
