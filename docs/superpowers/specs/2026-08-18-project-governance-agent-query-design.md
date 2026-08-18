# 项目治理视图Agent查询层设计

- 状态：设计已批准；第一阶段代码与真实索引已完成，等待完整回归验证
- 日期：2026-08-18
- 依赖基线：`PGOV_20260818T062450Z`
- 设计目标：让项目Agent按任务局部查询治理视图，减少重复扫描、重复SSH、重复实验和错误工作树操作

## 1.背景与问题

项目已经完成本地与N607的只读治理扫描，形成资产、实验、Git归属、保留级别和删除审批视图。完整清单体积较大，适合作为不可变审计基线，但不适合每个Agent任务都重新加载或全文搜索。

当前缺少的是一个面向日常任务的轻量入口。Agent应能够在开始开发、实验、分析或资料整理前，用少量查询回答以下问题：

- 当前有效治理扫描是哪一次，是否完整闭合？
- 一个路径、资产或run ID位于本地还是N607，属于哪个实验？
- 一个代码路径属于哪个Git仓库、工作树和提交？
- 是否已有相同run ID或关联实验，能否避免重复运行和输出碰撞？
- 一个资料的保留级别是什么，是否存在人工复核限制？

查询层只提供导航和证据摘要，不修改原始资产，也不升级为新的实验审批、发布或性能晋级门。

## 2.目标与非目标

### 2.1目标

1.用小型指针快速定位最新有效扫描和本地查询库。
2.把外置完整清单转换成可索引、只读的SQLite基线。
3.在现有命令入口下提供状态、资产、实验、Git归属和复核查询。
4.让Agent默认只读取与当前任务有关的记录，不重新遍历61万级资产。
5.保留原始CSV、JSON、收据和报告作为治理事实来源，SQLite只是可重建索引。
6.为后续增量登记新run和新资产预留清晰边界。

### 2.2非目标

- 不移动、重命名、覆盖、压缩、同步或删除本地/N607资产。
- 不执行N607实时扫描、SSH、SCP或管理员操作。
- 不自动修改实验状态、保留级别或删除审批状态。
- 不把治理查询加入Exclusive Minimal Experiment Workflow之外的新gate。
- 不引入数据库服务、Web服务、后台守护进程或联网依赖。
- 不建立额外seal、签名、receipt链、逐成员hash或发布许可。
- 不在第一阶段实现自动清理、自动归档或自动实验启动。

## 3.方案比较与选择

### 3.1方案A：直接读取CSV/JSON

优点是零转换、实现最少；缺点是每次查询都可能扫描GB级文件，Agent延迟高且容易反复读取。该方案仅保留为故障回退，不作为默认路径。

### 3.2方案B：SQLite只读索引（选择）

使用Python标准库`sqlite3`流式导入完整表，建立路径、run ID、仓库和状态索引。查询无需额外服务，也不要求引入新依赖，适合Windows本地Agent和Git工作树。

SQLite库属于外置可再生成产物，不提交到Git。Git只保存小型最新指针、设计、代码和测试。

### 3.3方案C：DuckDB/Parquet或可视化平台

该方案适合大规模分析和人工看板，但会新增依赖、格式和维护面。现阶段没有必要；若后续确有跨扫描趋势分析需求，再独立评估，不能阻塞本查询层。

## 4.总体架构

```text
不可变完整CSV/JSON＋scan_receipt.json
                  │
                  │ 流式、可重建导入
                  ▼
      governance.sqlite（外置、只读基线）
                  ▲
                  │ latest.json定位并校验scan_id
                  │
       Agent只读CLI：status/find/experiment/repo/review
                  │
                  ▼
     当前任务所需的少量文本或JSON结果
```

组件边界如下：

1.`PointerLoader`：读取并验证`latest.json`，解析收据和SQLite位置。
2.`IndexBuilder`：从一次已闭合扫描流式构建SQLite，使用临时文件完成后原子发布。
3.`QueryStore`：以只读模式打开SQLite，执行参数化查询。
4.`QueryCommands`：提供稳定的人类可读和JSON输出，不包含写资产接口。
5.`AgentUsagePolicy`：规定Agent何时查询、何时允许实时核验及如何解释结果。

## 5.文件与入口

计划新增或修改的Git文件：

```text
docs/project_governance/latest.json
docs/project_governance/agent-usage.md
tools/project_governance/query_index.py
tools/project_governance/cli.py
tests/test_project_governance_query.py
AGENTS.md
```

计划生成但不提交的外置索引：

```text
E:/type10-7/local_artifacts/project_governance/<scan_id>/governance.sqlite
```

继续使用现有入口：

```text
python tools/project_governance_inventory.py <command>
```

## 6.`latest.json`契约

`latest.json`保持小型、可读、无凭据，至少包含：

```json
{
  "schema_version": 1,
  "scan_id": "PGOV_20260818T062450Z",
  "receipt_path": "E:/type10-7/code/snapshots/project_governance_20260813_wt/docs/project_governance/PGOV_20260818T062450Z/scan_receipt.json",
  "external_root": "E:/type10-7/local_artifacts/project_governance/PGOV_20260818T062450Z",
  "sqlite_path": "E:/type10-7/local_artifacts/project_governance/PGOV_20260818T062450Z/governance.sqlite",
  "created_at_utc": "2026-08-18T06:46:10Z",
  "implementation_git_head": "791c775ec19f3aee78f67cc544fd6f0a287de71a"
}
```

约束：

- 路径必须是绝对路径且位于批准的治理输出根内。
- `scan_id`必须与收据和SQLite元数据一致。
- 只有终态完整、收据可读的扫描才能成为latest。
- 更新指针不改变旧扫描，旧SQLite和收据仍按scan ID保留。
- 指针不复制完整收据，不形成新的hash或receipt链。

## 7.SQLite数据模型

SQLite只保存查询所需的原字段，不创造新的科学事实。第一阶段至少包含：

### 7.1`metadata`

记录`schema_version`、`scan_id`、收据路径、导入时间、源文件路径和行数。查询入口先用它与`latest.json`交叉核对。

### 7.2`assets`

保存资产身份、位置、root、相对路径、类型、大小、mtime、访问状态、实验关联、证据角色和保留级别。主要索引：

- 唯一键：`asset_id`
- `location, root_id, relative_path`
- `experiment_id`
- `retention_class`

### 7.3`experiments`

保存实验身份、run ID、状态、报告路径、Git提交、预期/已观察artifact摘要和闭合缺口。主要索引：

- `experiment_id`
- `run_id`
- `experiment_state`

### 7.4`git_ownership`

保存资产、仓库根、工作树根、分支、HEAD、Git状态和归属类型。主要索引：

- `asset_id`
- `repository_root`
- `worktree_root`

### 7.5`retention`

保存资产保留级别、建议动作和判定原因。主要索引：

- `asset_id`
- `retention_class`

### 7.6`deletion_candidates`

原样保存治理扫描已经产生的候选记录。查询层不得生成授权状态，不得把其他表中的`REVIEW_REQUIRED`、`ORPHAN_REVIEW`、`SCAN_ERROR`或零字节资产转换为删除候选。

## 8.构建与一致性规则

1.只接受收据终态完整的扫描作为输入。
2.按CSV逐行流式导入，不把完整清单一次载入内存。
3.先写同目录临时SQLite，完成建表、索引和行数核对后再原子改名。
4.构建失败时不更新`latest.json`，也不覆盖已有可用SQLite。
5.构建完成后用SQLite内部`metadata`核对scan ID和各表行数。
6.SQLite是可再生成缓存；任何冲突都以原始完整表和收据为准。
7.查询时使用SQLite只读URI模式，禁止隐式创建空数据库。
8.不读取或存储文件正文、模型内容、数据样本、SSH密钥和服务器凭据。

## 9.命令设计

### 9.0`build-index`

离线管理命令，从一个已闭合扫描的收据和外置CSV流式构建SQLite。它要求显式提供`--receipt`、`--external-root`和`--database`，只写新的外置数据库临时文件及最终数据库，不更新原始清单、不连接N607，也不自动修改`latest.json`。日常Agent不调用此命令。

### 9.1`status`

显示latest scan ID、收据终态、SQLite可读性、基线时间、主要表行数和保守警告。默认不访问N607。

### 9.2`find <query>`

按精确`asset_id`、规范化绝对路径或受限路径前缀查询资产。默认最多返回20条；结果过多时要求缩小查询，不进行无界输出。

### 9.3`experiment <run_id>`

汇总对应实验状态、本地/N607资产数量、报告、Git提交、prediction、score、checkpoint和闭合缺口。不存在时明确返回`NOT_FOUND`，不能据此声称服务器实时不存在。

### 9.4`repo <path>`

返回最具体的仓库根、工作树、分支、HEAD和路径Git状态。若同一路径存在多条冲突归属，返回`AMBIGUOUS`，不自动选仓库。

### 9.5`review`

按位置、保留级别、实验状态或Git归属列出需要人工复核的记录，默认只给计数和有限样本。删除候选单独展示，并持续标明“需要用户明确批准”。

### 9.6输出格式与退出码

- 默认输出简洁文本，供人阅读。
- `--json`输出稳定JSON，供Agent解析。
- `0`：查询成功，包括明确的空结果。
- `2`：输入、指针、收据或数据库不一致。
- `3`：基线可读但状态陈旧或存在保守警告。
- `4`：命令在查询前被安全校验拒绝。

命令不得因查询结果为`REVIEW_REQUIRED`或存在实验缺口而返回“禁止研究”的新gate；这些是导航信息。

## 10.Agent工作流

### 10.1任务开始

1.读取`latest.json`和`status --json`。
2.根据任务类型执行一个或多个局部查询。
3.只有治理视图无法回答且目标状态可能已变化时，才进行现有规则允许的实时只读核验。

### 10.2按任务使用

- 代码修改：先用`repo`选择正确仓库/工作树，再检查Git状态。
- 实验设计或启动：先用`experiment`查重和检查run ID碰撞；它不替代最小实验工作流中的实时资源preflight。
- 结果分析：用`experiment`定位同一run的report、prediction、score和checkpoint，避免跨run拼接指标。
- 资料整理：用`find`查看来源、实验关联、Git归属和保留级别。
- 清理讨论：用`review`形成精确候选建议；任何删除仍需用户按条目或批次明确批准。

### 10.3任务结束

第一阶段只记录新增路径和run ID的待增量清单，不修改不可变基线。第二阶段再增加显式`register-run`或delta数据库；该功能需单独设计和批准，且不得静默改变历史扫描。

## 11.全量扫描与增量边界

正常Agent任务不触发全量扫描。只有以下情况才建议新建完整基线：

- 本地或N607治理根、scope或schema发生变化。
- 发生大规模目录迁移或服务器资产结构调整。
- 用户明确要求刷新完整基线。
- 周期性人工基线确有管理价值。

候选、adapter、超参数、epoch、checkpoint、报告修改、代码重构或普通新run不触发全量重扫。新run的日常登记应由后续增量层解决。

## 12.安全与失败处理

- 所有第一阶段命令只读；不提供删除、移动、覆盖、同步、启动或停止接口。
- `latest.json`缺失、越界、schema不符或scan ID不一致时失败关闭。
- SQLite缺失时提示先构建索引，不自动触发完整扫描或N607连接。
- SQLite损坏时回退到明确的`INDEX_UNAVAILABLE`，保留原始表，不覆盖现场。
- 查询结果中的`SCAN_ERROR`保持错误证据，不能解释为资产不存在。
- 索引时间晚于扫描不代表资产实时存在；涉及运行中进程、GPU、磁盘和远端连接时必须使用现有实时核验规则。
- 用户既有脏工作树、DOCX和未跟踪QA目录不被索引构建器修改或暂存。

## 13.验证策略

1.临时fixture验证完整表流式导入、表行数和索引结果。
2.验证构建失败不会覆盖旧SQLite或更新latest指针。
3.验证指针路径越界、scan ID不一致、收据非终态和空数据库均被拒绝。
4.验证SQLite以只读模式打开，查询不会改变数据库mtime或创建旁路文件。
5.验证五个命令的文本和JSON输出、限制数量、空结果和歧义状态。
6.验证`REVIEW_REQUIRED`、`ORPHAN_REVIEW`、`SCAN_ERROR`和零字节资产不会成为授权删除。
7.验证实验查询保持同run证据，不跨run合并prediction、score或checkpoint。
8.验证repo查询在多仓库/多工作树冲突时返回`AMBIGUOUS`。
9.验证CLI源码不包含SSH、SCP、删除、移动、进程终止或服务器写入调用。
10.在真实基线上完成一次只读构建和定点查询，记录耗时、索引大小和查询延迟，但不把性能数字设为实验阻断gate。

## 14.实施分期

### 第一阶段：只读查询基线

- 实现SQLite构建器、latest指针和五个查询命令。
- 增加测试和`agent-usage.md`。
- 在`AGENTS.md`加入最小启动规则。
- 用当前已完成扫描构建一次外置SQLite。

### 第二阶段：增量登记

- 为新run和少量新增资产设计显式delta层。
- 查询时由delta覆盖同身份基线记录，同时保留来源scan ID。
- 不原地改写不可变基线。

### 第三阶段：方法与性能工作接入

- 用治理查询快速定位合法同row实验、活动路线和证据缺口。
- 直接回到候选实现、最小矩阵和真实性能验证。
- 不扩展成通用资产管理平台或新的发布控制面。

## 15.验收标准

- Agent启动只需读取小型指针、收据摘要和局部查询结果。
- 常用路径、run ID和Git归属查询不再扫描完整GB级文件。
- 查询结果能精确回指原scan ID、asset ID和源记录。
- 选择工作树、实验查重和artifact定位都有明确、可解析结果。
- 所有查询保持本地只读，N607连接次数为0。
- 原始完整表、收据和报告保持不变。
- 删除、移动和覆盖数量均为0；删除候选仍需用户明确批准。
- 查询层不成为白名单外的新实验gate。

## 16.待用户确认的实施边界

用户复核本书面设计后，下一步只制定并执行第一阶段实施计划。第二阶段增量登记和第三阶段研究接入不在同一批次中自动展开。
