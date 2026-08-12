# CVS项目本地与N607资产治理设计

- 状态：用户已批准方案A，等待书面设计复核
- 日期：2026-08-13
- Git分支：`codex/project-governance-20260813`
- 基线提交：`b2e0f30c96b4586cff625a63b3a3a976c2f8e896`

## 1.目标

本设计为`E:\type10-7`与N607项目根`/home/szu2070436088/2510044040/CV-SincNet`建立统一、可复现、证据优先的资产治理面。第一阶段只读取元数据并生成索引，不移动、不覆盖、不删除任何原始资产，也不启动、停止或修改实验。

治理完成后，使用资产和实验索引确定真正的活动路线、证据缺口及Git承载关系，再另立方法与性能优化设计。资产治理本身不产生或改写任何科学性能结论。

## 2.范围

### 2.1纳入范围

- 本地根级文件和目录。
- `automation_reports\CV-SincNet`、`code\snapshots`、`local_artifacts`、`remote_artifacts`、`runs`、`logs`、`outputs`、`server_log_backups`、`runner_staging`与Git发布仓库。
- N607项目根、`automation_reports`、`runs`、`logs`、`releases`、`remote_artifacts`、`snapshots`、`code`与其他已发现顶层承载面。
- Git仓库、linked worktree、非Git证据目录、未跟踪文件和异常命名资产。
- 实验报告、运行目录、日志、预测、score、manifest、receipt、归档和代码版本之间的可验证关系。

### 2.2不在第一阶段范围内

- 移动、重命名、覆盖、压缩、解压、复制或删除原始资产。
- 清理缓存、空文件、异常名称、重复归档或旧worktree。
- 全量读取或哈希数据集、checkpoint、权重和大型运行树。
- 修改N607代码、配置、权限、服务或目录结构。
- 启动、停止、重启或调优实验。
- 选择、修改或晋级Phase1、Phase2、Phase3方法。

### 2.3资产粒度与扫描深度

“资产总表”管理的是可治理单元，不等于枚举每个数据样本：

- 本地与N607项目根的全部直接子项必须逐项登记。
- `runs`、`logs`、`releases`、`automation_reports`、`snapshots`和artifact承载面的每个直接子项作为一个治理单元；单元内部只发现深度不超过3层的报告、manifest、receipt、metrics、prediction目录摘要和Git版本线索。
- 数据集、checkpoint、权重和大型run payload目录作为受保护聚合资产登记，不枚举其中每个样本或张量文件。
- Git仓库优先使用`git ls-files`和Git状态建立代码归属，不用不受控的全树递归扫描替代Git证据。
- 超出默认深度但被报告、manifest或receipt明确引用的路径可精确补扫；补扫不得扩展到该路径的无关兄弟目录。

## 3.不可变安全规则

1.任何删除必须由用户对精确条目或精确批次重新明确批准；“整理项目”或批准本设计不构成删除授权。
2.第一阶段不提供自动删除执行器，只生成待审批清单。
3.本地与N607分别审批；不得用一次本地批准推定远端批准，反之亦然。
4.不得把空文件、异常名称、旧时间戳、未跟踪状态或非Git状态单独作为删除依据。
5.数据集、checkpoint、权重、正式报告、日志、metrics、predictions、receipts、manifests和run输出默认保留。
6.N607仅使用普通账号和短连接只读命令；不使用管理员账号，不留下SSH会话、端口转发或复用连接。
7.若发现运行中任务，服务器侧自动进入监控模式，不修改任何关联资产。
8.现有脏工作树和未跟踪内容均视为用户资产；本任务只在独立worktree中显式暂存自身文件。

## 4.总体架构

治理系统分为七个边界清晰的组件：

1.`LocalCollector`：读取本地指定根和承载面的目录项、文件类型、大小、mtime、链接属性及访问错误。
2.`N607Collector`：在本地完成预检后，通过短连接把只读Python脚本流式送入N607；脚本只输出JSON元数据，不在服务器落文件。
3.`Normalizer`：把Windows与POSIX路径转换为`location + root_id + relative_path`，保留原始显示名称和可逆转义名称。
4.`ExperimentIndexer`：按`run_id`、报告状态、PID/进程证据、artifact闭环和终态标记建立实验索引；不得只按mtime推断活动状态。
5.`GitOwnershipMapper`：记录仓库、分支、HEAD、worktree、tracked/untracked/ignored状态和最近可验证提交；非Git资产明确标为证据资产而非错误。
6.`RetentionClassifier`：依据资产类型、实验关系、可再生成性、依赖和证据价值给出保留级别；不执行任何动作。
7.`ReportEmitter`：生成CSV、JSON、Markdown总览、扫描receipt和待审批删除表，并在Git中保存小型治理产物。

组件之间只传递规范化记录，不让分类器直接访问或修改原始资产。采集器也不包含移动或删除接口。

## 5.数据模型

### 5.1资产记录

每项资产至少包含：

```text
asset_id
scan_id
location                 LOCAL | N607
root_id
relative_path
display_name
escaped_name
asset_kind               file | directory | symlink | junction | other
size_bytes
mtime_utc
access_status             OK | SCAN_ERROR
hash_status               SHA256 | METADATA_ONLY | NOT_HASHED_SIZE_LIMIT | ERROR
sha256
experiment_id
git_ownership
evidence_role
retention_class
recommended_action
decision_reason
```

`asset_id`由`location`、`root_id`和规范化相对路径稳定生成。首轮不把mtime或大小放入身份键，避免内容变化导致同一路径被误认为不同资产。

### 5.2实验记录

每个实验至少包含：

```text
experiment_id
run_id
phase
method_or_candidate
report_path
local_artifact_paths
n607_artifact_paths
git_commit
process_evidence
prediction_count
score_count
expected_artifacts
observed_artifacts
experiment_state
closure_gaps
```

有合法`run_id`时，`experiment_id`由规范化`run_id`生成；缺少`run_id`时使用`ORPHAN:<location>:<root_id>:<relative_path>`形成稳定占位身份，直到证据支持合并。名称相似、mtime接近或目录前缀相同不能自动合并实验。

实验状态固定为：

- `ACTIVE_LIVE`：存在与run根、CWD和cmdline绑定的实时进程证据。
- `OPEN_INCOMPLETE`：报告或artifact未闭环，但没有充分实时进程证据。
- `COMPLETE_EVIDENCE`：报告与预期artifact完整，终态可验证。
- `HISTORICAL_ARCHIVE`：已完成、已索引且不再是当前活动路线。
- `ORPHAN_REVIEW`：存在产物但缺少足够的报告、版本或run绑定。
- `SCAN_ERROR`：信息不足源于读取失败，不能解释为实验不存在或失败。

### 5.3Git归属

Git归属固定为：

- `TRACKED_GIT`
- `UNTRACKED_IN_GIT_WORKTREE`
- `IGNORED_REGENERABLE`
- `NON_GIT_EVIDENCE`
- `REMOTE_NON_GIT`
- `MIRROR_PENDING`
- `GIT_STATE_ERROR`

### 5.4保留级别

保留级别固定为：

- `KEEP_IMMUTABLE`：正式证据、数据、checkpoint、预测、score、receipt、manifest或不可替代源文件。
- `KEEP_ACTIVE`：活动或未闭环路线相关资产。
- `KEEP_UNTIL_PUBLISHED`：当前论文、发布或复核仍依赖的资产。
- `HISTORICAL_ARCHIVE`：历史证据，保留但可降低在线可见度。
- `REGENERABLE_CACHE`：已证明可再生成的缓存；仍不自动删除。
- `REVIEW_REQUIRED`：来源、依赖或价值不足以自动判断。
- `DELETE_CANDIDATE`：通过依赖和证据检查后进入审批表；仍保持原位。

`HISTORICAL_ARCHIVE`只用于已有可验证终态，且未被当前目标文档、活动报告或活动Git分支引用的实验。不能仅因时间较早而归入历史归档。

## 6.扫描与关联流程

1.生成不可覆盖的`scan_id`，记录时间、操作者、代码提交和扫描范围。
2.读取本地根和指定承载面的深度受控元数据，不跟随符号链接或junction。
3.执行N607只读预检；成功后以流式`python3 -`获取深度受控元数据。
4.每个SSH命令结束后检查本机`ssh.exe`及N607/桥接TCP22连接均为0。
5.规范化路径并保留原始名称、转义名称和采集错误。
6.读取小型报告、manifest、receipt和Git元数据，建立实验与资产关系。
7.优先用`run_id`、明确路径、commit、manifest和receipt关联；名称相似或mtime接近只能产生低置信候选关系。
8.运行保留分类器；任何证据冲突自动降级为`REVIEW_REQUIRED`。
9.输出完整性统计、未闭环实验、非Git证据、异常名称和待审批删除候选。
10.把小型治理产物显式暂存到独立Git分支；不暂存原始运行产物。

## 7.哈希与资源策略

- 文档、脚本、配置、manifest和receipt等小型控制文件默认计算SHA256。
- 数据集、checkpoint、权重、run树和大型归档首轮使用`METADATA_ONLY`。
- 只有疑似重复、准备进入`DELETE_CANDIDATE`或需要证明本地/N607一致性的对象才进行补充SHA256。
- 补充哈希必须按明确文件清单执行，不递归扫描整个数据集或运行根。
- N607扫描不得启动高负载训练、全盘`du`、全盘hash或跨数据集递归搜索。

## 8.待审批删除表

删除候选表至少包含：

```text
candidate_id
location
absolute_path
asset_kind
size_bytes
reason
evidence
dependencies
recoverability
estimated_space_reclaim
approval_state
approved_scope
execution_state
```

初始`approval_state`统一为`AWAITING_USER_APPROVAL`，`execution_state`统一为`NOT_AUTHORIZED`。候选表允许给出“保留”“归档”“删除”的建议，但第一阶段不得改变这两个字段。

进入`DELETE_CANDIDATE`必须同时满足：不属于默认保护类型；没有活动进程、活动报告、manifest、receipt、Git worktree或实验索引依赖；来源和用途已经查明；能够证明可再生成或与保留副本内容一致；可恢复性和风险已经记录。任一条件缺失时只能标为`REVIEW_REQUIRED`。

未来若用户批准删除，必须另开执行批次并遵循：重新验证精确路径、确认没有活动进程或依赖、记录hash/备份或不可恢复性、展示最终dry-run、取得当前批次明确批准、执行最小范围动作、验证结果并写回审计记录。通配符、目录上级授权或“全部清理”不能替代精确批次。

## 9.输出结构

计划中的正式输出位于独立Git工作树：

```text
docs/project_governance/<scan_id>/report.md
docs/project_governance/<scan_id>/asset_inventory_local.csv
docs/project_governance/<scan_id>/asset_inventory_n607.csv
docs/project_governance/<scan_id>/experiment_index.csv
docs/project_governance/<scan_id>/git_ownership.csv
docs/project_governance/<scan_id>/retention_decisions.csv
docs/project_governance/<scan_id>/deletion_candidates.csv
docs/project_governance/<scan_id>/scan_receipt.json
```

单个治理文件不超过10MiB且一次扫描提交总量不超过50MiB时，完整JSON可与CSV同时保存。超过任一门槛时，只在Git中提交统计、分片索引和receipt；不可变完整清单保存到`E:\type10-7\local_artifacts\project_governance\<scan_id>\`，并在Git receipt中记录每个文件的SHA256、字节数和绝对路径。不得为满足Git体积而丢弃错误记录或删除候选证据。

## 10.错误处理

- 读取失败写入`SCAN_ERROR`，不得静默跳过。
- SSH失败时终止当前N607扫描；旧清单可作历史参考，但必须标记`STALE_NOT_CURRENT`。
- 发现活动任务时继续只读采集必要状态，不读取会影响作业的超大文件，不做任何远端变更。
- 路径编码异常时同时保存显示名称、转义名称和父目录；不尝试自动重命名。
- 报告和artifact冲突时保留双方证据并标为`REVIEW_REQUIRED`。
- Git仓库脏时只读取状态；本任务提交只能在独立worktree中显式列出文件。
- 任何分类器异常都默认提升保留级别，不得默认进入删除候选。

## 11.验证策略

实现阶段必须具备以下验证：

1.使用临时fixture验证普通文件、空文件、异常名称、符号链接、junction、访问错误和大文件元数据策略。
2.验证相同mtime或相同大小不会自动判定为重复文件。
3.验证缺报告、缺Git或零字节证据只能进入`REVIEW_REQUIRED`，不能直接进入已授权删除。
4.验证采集器没有移动、覆盖、删除或远端写入接口。
5.验证实验状态不依赖mtime单独判定，且`ACTIVE_LIVE`需要进程绑定证据。
6.验证待审批删除表的所有行均为`AWAITING_USER_APPROVAL / NOT_AUTHORIZED`。
7.执行本地只读fixture扫描、N607短连接只读smoke、SSH断连检查、`git diff --check`和敏感路径扫描。
8.只提交本任务代码、测试、设计和小型治理产物；显式排除数据、权重、原始大日志和用户未跟踪内容。

## 12.验收标准

- 覆盖已发现的本地166个根级资产和N607的185个根级资产，后续数量变化必须由新的`scan_id`解释。
- 覆盖本地主要报告/快照承载面和N607的`runs`、`logs`、`releases`等主要承载面。
- 每个已发现实验都具有明确实验状态，或被显式标为`ORPHAN_REVIEW/SCAN_ERROR`。
- 每项资产都具有Git归属或非Git证据说明。
- 所有分类决策都能回指证据和规则。
- 生成精确待审批删除清单，但本阶段实际删除、移动和覆盖数量均为0。
- N607无远端落盘变更，所有SSH连接在检查后完全退出。
- Git提交只包含本任务明确文件，不含现有未跟踪资产。

## 13.后续阶段边界

完成本设计对应的治理实现和用户复核后，另行创建“方法与性能优化”设计。该设计以`COMPLETE_EVIDENCE`实验、明确活动路线、合法同row指标和当前`项目.md`为输入，不把缓存数量、目录整齐度、技术闭环或历史Oracle结果当作方法晋级证据。
