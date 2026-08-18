# 项目Agent治理查询使用说明

## 1.用途

治理查询层用于快速定位资产、实验和Git归属，避免每个任务重新扫描本地与N607。它是导航工具，不是新的实验审批、发布或性能晋级门。

当前指针：`docs/project_governance/latest.json`。

## 2.任务开始

先查看当前基线：

```text
conda run -n ssr-gpu python tools/project_governance_inventory.py status --latest docs/project_governance/latest.json --json
```

然后只执行与当前任务相关的一个或多个查询：

```text
conda run -n ssr-gpu python tools/project_governance_inventory.py find <asset_id或绝对路径> --latest docs/project_governance/latest.json --json
conda run -n ssr-gpu python tools/project_governance_inventory.py experiment <run_id> --latest docs/project_governance/latest.json --json
conda run -n ssr-gpu python tools/project_governance_inventory.py repo <绝对路径> --latest docs/project_governance/latest.json --json
conda run -n ssr-gpu python tools/project_governance_inventory.py review --latest docs/project_governance/latest.json --limit 20 --json
```

## 3.解释边界

- `status`返回`3`且收据为`COMPLETE`时，表示基线可用但含保守警告；读取JSON中的`scan_error_counts`，不要把它误判为索引失败。
- 查询结果是`PGOV_20260818T062450Z`扫描时点的证据，不代表进程、GPU、磁盘或N607连接的实时状态。
- 只有当前任务确实依赖已变化的实时状态时，才按既有规则进行短时只读核验。
- `NOT_FOUND`表示基线中未找到，不能据此声称服务器实时不存在。
- `AMBIGUOUS`表示存在多条归属，Agent必须保留歧义并缩小查询，不能自动选择仓库或实验。
- `SCAN_ERROR`必须保留为错误证据，不能解释为资产不存在。

## 4.按任务使用

- 修改代码前：用`repo`确认正确仓库和工作树，再检查实时Git状态。
- 设计或启动实验前：用`experiment`查重和检查run ID碰撞；实时资源preflight仍按最小实验工作流执行。
- 分析结果时：用`experiment`定位同一run的report、prediction、score和checkpoint，不跨run拼接指标。
- 整理资料时：用`find`查看位置、实验关联、Git归属和保留级别。
- 讨论清理时：用`review`形成精确建议，但任何删除仍需用户对具体条目或批次明确批准。

## 5.任务结束

记录本次新run ID和新增关键路径，等待后续增量登记层处理。普通任务不得自动触发完整扫描。

只有治理根、scope或schema改变，大规模目录迁移，用户明确要求，或人工周期基线确有必要时，才建立新的完整扫描。

## 6.禁止事项

- 不因普通任务重新读取GB级完整CSV/JSON或重扫61万级资产。
- 不把治理警告升级为Exclusive Minimal Experiment Workflow之外的新gate。
- 不把`REVIEW_REQUIRED`、`ORPHAN_REVIEW`、`SCAN_ERROR`、零字节或非Git状态推断为可删除。
- 不通过查询命令连接N607、移动资料、覆盖输出、启动/停止任务或修改审批状态。
