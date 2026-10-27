# 项目统一导航

## 从哪里找

| 需求 | 入口 |
|---|---|
| 实验名称、说明、状态证据、数据/seed、配置、模型与输出 | [实验总索引](experiment_registry/README.md) |
| 代码、报告、冒烟、临时文件、归档的路径与用途 | [资料管理](docs/project_governance/README.md) |
| 数据协议、方法权限、truth-last边界 | [项目协议](项目.md)；Git克隆见[协议镜像](docs/PROJECT_PROTOCOL.md) |
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

历史大日志已在 2026-10-12 完成无损压缩，原路径恢复映射见[本机与 N607 清理报告](docs/project_governance/20261012_cleanup/REPORT.md)。

2026-09-01 前无参考意义的旧产物清理及索引精简见[本机与 N607 永久清理报告](docs/project_governance/20261012_retirement/REPORT.md)。日期不是唯一筛选条件；已删除压缩件以新清单为准。

后续旧产物专项清点（无新增删除）见[归档去重与缓存候选](docs/project_governance/20261012_remaining_audit/REPORT.md)。

## 当前 PAMF 方法定义（2026-10-27，最新）

[当前方法与名称](docs/PAMF_METHOD_DEFINITION_20261027.md)：默认三视图SIG-TIME；完整方法PAMF采用Tri-SIG-TIME SSL。PAMF-CE仅有标签纯CE，PAMF-Sup保持原监督机制，PAMF-PL只增加0.95普通伪标签。后续补充由用户指定，本次无新实验。

## 历史 PAMF 与外部方法对比（2026-10-25）

[14 页 PAMF 网络分支消融报告](docs/PAMF_BRANCH_ABLATION_REPORT_20261025.md)：仅分类损失监督训练，9 种结构、27 个模型；含逐景结果、同种子配对、参数、实测时延及论文可用结论。

当前半监督方案暂用普通伪标签，见[当前方法](docs/PAMF_CURRENT_METHOD_20261025.md)和[2026-10-26 重点实验报告及 Excel](docs/PAMF_PRIORITY_COMPARISONS_20261026.md)。主比较配对监督与半监督，其余重点图统一增强半监督；所有适用图纳入 PAMF 与 PAMF+DURA，DURA 泛用性含 PAMF。每个重点实验分别展示跨接收机与跨日且跨接收机图/表。源数量完整 1 至 6，5 源复用主模型并汇总共同 6 目标。PDF 18 张图，Excel 35 页、98 个可编辑图；旧图表保留。

## 历史 practical residual 论文资料

使用 `exports/unified_figures/20261024-practical-residual-paper-r03`。旧错误增强外部比较及混合包已隔离，原链接需结合[回收与恢复说明](docs/project_governance/20261024_comparison_retirement/REPORT.md)定位；当前完整冻结结果为 practical-residual r14。
