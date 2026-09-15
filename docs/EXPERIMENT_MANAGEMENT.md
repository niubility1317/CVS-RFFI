# 实验管理与快速定位

版本：2026-09-16。适用于CVS主方法、外部论文对比、联邦方法、消融、诊断、seed扫描及独立评估。科学权限仍由本机`项目.md`或克隆中的[PROJECT_PROTOCOL.md](PROJECT_PROTOCOL.md)定义；本页落实既有最小预登记，不定义额外审批。

## 从哪里开始

- [实验总索引](../experiment_registry/README.md)：方法/用途入口、近期记录。
- [完整目录CSV](../experiment_registry/catalog.csv)：适合筛选和浏览。
- [覆盖与缺项](../experiment_registry/coverage.json)：哪些目录扫过、哪些缺失、哪些字段未读取。
- [登记skill](../.agents/skills/experiment-management/SKILL.md)：以后每次启动及跨对话查找的入口。
- 每次实验记录：`automation_reports/CV-SincNet/<run_id>/report.md`、`experiment.json`、`events.jsonl`。

`experiment.json`保存矩阵与引用，原始config保存算法设置，实际运行生成的resolved config证明哪些设置生效。报告解释问题、证据和结论。索引由这些记录生成，不手工维护第二份矩阵。

## 常用命令

在`E:/type10-7`使用已验证的ssr-gpu解释器；以下PowerShell只调用原生Python。独立克隆省略`--root`即可使用脚本所在仓库，也可显式传入工作区。

```powershell
& 'C:/Users/lh594/.conda/envs/ssr-gpu/python.exe' -X utf8 tools/experiment_registry.py search 'daot fasttrust' --limit 10
& 'C:/Users/lh594/.conda/envs/ssr-gpu/python.exe' -X utf8 tools/experiment_registry.py search 'core90' --seed 392005
& 'C:/Users/lh594/.conda/envs/ssr-gpu/python.exe' -X utf8 tools/experiment_registry.py search 'mopc'
& 'C:/Users/lh594/.conda/envs/ssr-gpu/python.exe' -X utf8 tools/experiment_registry.py show 'legacy:automation_reports/CV-SincNet/response_games_prepare_20260914' --section facts --limit 30
```

关键词按AND匹配。`search`显示总命中数与展示数；`show`必须使用唯一ID，同名组不会悄悄合并。`--kind comparison`只筛明确登记为对比实验的新记录；历史对比先按方法关键词或路径标签查找。历史`--seed`表示某个有出处的元数据字段含此seed，不保证整组每一行都用它；通过facts的source/scope/row回到实际行。

`show --section facts`可加`--field model_seed --row SIM_s392005`精确查一个字段/行；`--section artifacts --category checkpoint`可查权重位置。显示达到limit后即停止，`has_more=true`表示还有记录，`detail_file`给出完整压缩细目位置；不把展示上限当总数。

新实验的最短登记路径：

```powershell
# 输出路径必须不存在；填写模板中的真实值后再登记。
& 'C:/Users/lh594/.conda/envs/ssr-gpu/python.exe' -X utf8 tools/experiment_registry.py template --output .codex_tmp/new_experiment_spec.json
& 'C:/Users/lh594/.conda/envs/ssr-gpu/python.exe' -X utf8 tools/experiment_registry.py validate .codex_tmp/new_experiment_spec.json --launch-ready
& 'C:/Users/lh594/.conda/envs/ssr-gpu/python.exe' -X utf8 tools/experiment_registry.py new --spec .codex_tmp/new_experiment_spec.json
# 根据实际启动后的独立读回记录，RUN_ID和路径替换为本次值。
& 'C:/Users/lh594/.conda/envs/ssr-gpu/python.exe' -X utf8 tools/experiment_registry.py record RUN_ID --status RUNNING --evidence launch_readback.json --note 'PID/CWD/argv/GPU及log增长已核实'
```

工具不会启动实验。`new`拒绝已有run目录，避免改写历史；已有run补登记应在原目录按模板创建缺少的experiment.json并保留旧report，然后`build --managed-only`。已经启动的配置保持原始记录；变化另记说明/新版本，真正的新运行使用新run ID。

`record`支持`--row-id`。行事件不把全run状态改为完成；汇总run状态由launch owner根据所有预登记行的证据明确更新。它记录声明与出处，不替代证据读取。中途手工更新spec后用`build --managed-only`更新检索；全历史整理才用`build`。

## 名称、层级与存储

|层级|含义|示例/规则|
|---|---|---|
|group_id|长期方法/研究问题|`phase1-daot-response-game`；跨seed和修复保持可检索|
|run_id|一次发布/执行尝试|`20260916-phase1-daot-wisig-m3-r01`；可沿用已有不可变ID|
|display_name|人能理解的名称|“DAOT响应博弈三seed源域扫描”|
|row_id|独立矩阵行|`eg-s392005-rx7-14-k5`，每行独占输出|
|aliases/tags|旧名和检索词|中文名称、论文缩写、旧run名；不替代真实配置|
|parent_run_ids|训练/模型继承关系|必须另列checkpoint来源与契约核对；不能凭父ID授予复用|
|replaces_run_id|技术修复后替代|保留失败run，明确替代关系|
|comparison_group_id|可比较条件分组|固定数据契约、划分与预算口径后由人声明；同名不自动视为可比|

```text
automation_reports/CV-SincNet/<run_id>/
  experiment.json             元数据、数据契约引用、逐行矩阵和artifact计划
  report.md                   问题、说明、结果、证据边界、交接
  events.jsonl                状态事件、时间、证据路径、可选row_id
  configs/                    可选：本次配置快照；也可引用Git中的实体配置
  evidence/                   可选：启动读回、解析后的轻量核验
  results/                    可选：指标汇总；大文件可引用外部存储
<remote project>/runs/<run_id>/<row_id>/   checkpoint、resolved config、prediction
<remote project>/logs/<run_id>/<row_id>/   stdout/stderr、metrics、曲线
```

只在实际需要时建立可选子目录。历史路径保持原样；已存在的launcher布局通过execution/row字段准确登记，无需为命名迁移产物。数据集和共享capsule单独保存，登记只引用资产身份和路径；不能把大文件复制进Git。机器换路径时更新locator或增加副本位置，保持run身份和原路径证据。

## 模板字段如何填写

|字段|应保存的事实|
|---|---|
|description/kind/stage|研究问题、基线、唯一改变因素、实际启用组件、属于CVS/对比/联邦/诊断哪类|
|authorization/permissions|本次授权出处、source/target标签访问规则、外部方法例外、正式/诊断声明范围|
|code|完整commit、checkout、解释器/环境、CWD；执行代码和配置属于哪个版本|
|data|dataset/version、raw/equalized、物理ID/role/split/标签映射引用、RX/day/TX、训练比例、K/support/query、LEO配置、capsule_id/split_id/VALIDATED_ONCE证据|
|checkpoint|scratch或完整来源，含resume/teacher/EMA/蒸馏/原型/统计继承、训练数据契约核对和source-only选模规则|
|rows[].seeds|model、split、data、augmentation、support、evaluation逐个给出；null说明不适用/尚不确定的原因|
|rows[]|row_id、方法、对照用途、实体配置与resolved config、数据覆盖项、K/scene、优化器/LR/epochs/fl_rounds/步数预算、精确命令、输出与日志路径|
|execution|host、launch owner、GPU策略、run/log/本地回收root、启动命令、技术停止规则；PID/GPU读回记录在证据中|
|expected_artifacts/metrics_plan|计划保留的checkpoint、曲线、固定prediction、独立scorer结果、每row与适用RX/day/TX/scene/K/seed指标、Macro-F1、跨seed均值/SD、成本|
|events/report|状态、证据时间、失败与缺项、替代关系、协议有效性、结论和下一步|

配置可通过ref引用已有manifest，不重复抄大数组。字段记录的是本次实际配置，不用全局默认补写历史；改变model seed不等于改变数据划分。外部方法按原论文权限运行时记录真实访问条件及新类LEO配置；不把CVS专用query限制强加给已获显式例外的方法，也不能把对比方法权限用于CVS。

## 状态与证据

`PLANNED → LOCAL_VERIFIED → LANDED → QUEUED/RUNNING → TRAINING_COMPLETE → PREDICTIONS_COMPLETE → ARTIFACTS_COMPLETE → ANALYZED`仅表示可能的生命周期，不要求每种实验经过所有状态。源域训练后的冻结/选择等待单独记录为`SOURCE_TRAINING_COMPLETE_AWAITING_SOURCE_REVIEW`；失败、停止、替代、partial和UNKNOWN独立保留。

外部动作仍按独立证据报告VERIFIED/FAILED/UNKNOWN。ARTIFACTS_COMPLETE仅说明预登记产物闭合，不能自动证明协议有效、目标benchmark获胜或实验可比较。未来训练代码不能读取含评分结果的总索引来调参、选模或选择性重跑。

## 历史整理的边界

历史目录常同时包含多个run、row、修复、评分和配置，故自动整理单位叫`legacy_evidence_group`。不能把证据组数当成实验数、seed数或已完成数，也不能凭目录名把两份镜像合并。记录原路径、artifact类别、方法/用途提示、带文件/行号/JSON位置的元数据及更新时间。

早期已整理的日志CSV逐行保留为`legacy_log_record`，包含原目录出处、source_path、方法族、训练比例与seed；同一run的多个日志/备份仍各有记录。N607目录快照作为`remote_directory_locator`，保留host、读取时间、精确路径及直属子目录/文件名，状态始终UNKNOWN。两者均不冒充已复核的独立实验。

有权限且需要补远端目录时，先按[N607操作](workflows/n607.md)执行只读preflight，再运行`tools/collect_experiment_inventory_n607.py --output experiment_registry/snapshots/<新文件名>.json`，随后`build --managed-only`。该采集器只读取目录，短连接结束即退出，不修改远端、解压归档、打开权重或读取实验数据。超时先核对连接层，不重复提交任务。

历史facts保留来源，不把跨配置的seed、LR和指标拼成一行。`reported_statuses`可能包含计划、失败、替代和完成的不同时间点；默认`HISTORICAL_UNVERIFIED`。正式比较需回到实际config、日志和评分证据重新核对需要的部分。

覆盖清单列出本地扫描范围与远端快照、文件数、缺目录、读取错误与排除项。元数据细目按记录压缩，查询时只读取命中组。归档先登记路径，不自动解压；模型、IQ、prediction和原始日志只登记位置，不为建目录读取其内容。远端深层内容、其他磁盘、隔离worktree或仅存在于对话中的实验不能宣称已全部恢复，定位需要时按权限补充。仅整理记录不授权GPU实验、清理、移动或远端修改。
