# Phase2运行时隔离与query信息边界完善追踪

日期：2026-07-15

目标：在现有`LEO_weak-only`与逐样本全注册类决策协议上，把“字段自声明”升级为可执行、可验证的运行时隔离契约；同时把容易混淆的`class_count`字段明确为“真实query批次类别计数”，并把预测与带标签评分拆成权限隔离的两阶段。

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|----|----------------|-------------|--------------|--------|--------------|-------|
| PRI-01 | 用户本轮要求 | 将`phase2_query_class_count_access`规范化为`phase2_query_true_batch_class_count_access`，明确不禁止读取已注册类别总数 | `AGENTS.md`、根`项目.md`、Git协议镜像、生产schema/validator | implemented | validator与predictor-contract负测覆盖deprecated字段和合法`registered_class_count` | 旧字段仅作为deprecated输入识别，不足以单独证明新协议合规 |
| PRI-02 | query决策边界 | Phase2决策包不得包含query truth、old/new/unknown role、真实批次类别计数、每类quota、标签分块或排序提示 | formal plan、sealed inference request、benchmark predictor | partial | `phase2_runtime_contract.py`已提供顶层allowlist与递归禁止键/路径负测 | legacy benchmark尚未替换，故不得标为implemented |
| PRI-03 | 预测/评分隔离 | 预测必须先产出不可变prediction artifact；独立post-prediction scorer再连接truth/role计算指标 | predictor、scorer、runner、summarizer | blocked | 当前benchmark在同进程读取truth并产出评分；已在runner/candidate-lock发布入口阻断 | scorer输出不得反向影响adapter、门限、回滚、候选选择或prediction |
| PRI-04 | clean dataset不可达 | Phase2进程只能访问sealed LEO cache/inference包allowlist根，不得接收或打开ManySig/ManyTx、raw PKL、clean路径或cache build spec | sealed package builder/loader、runner、candidate lock | blocked | 审计确认lock含build spec、完整项目根可达且无OS级隔离 | build spec只允许在Phase1离线打包阶段使用，不能进入Phase2包 |
| PRI-05 | clean cache/API/control-flow不可达 | Phase2生产入口使用专用strict loader；拒绝legacy raw/clean loader参数、环境变量、sidecar及非allowlist模块入口 | strict loader、CLI、runtime manifest | partial | predictor-contract已拒绝raw路径/build spec/truth sidecar；strict生产入口尚未落地 | 不以`clean_sample_access=false`自述替代实际隔离 |
| PRI-06 | 9项硬字段证据化 | 每个plan row、candidate lock、prediction manifest和formal result都记录4项clean不可达字段及5项query决策字段，并绑定验证证据 | config、plan、lock、predictor、scorer、summarizer | partial | 已统一为3基础+4clean+5query共12项，并要求8项runtime evidence | 字段缺失、冲突或证据哈希缺失均fail closed；尚未接通正式artifact链 |
| PRI-07 | pre-open fail closed | 在打开任何Phase2 payload前验证sealed manifest、成员allowlist、LEO provenance、scenario、satellite seed、SHA256和禁止路径 | strict loader、runner、tests | pending | 统一契约可在materialization前运行；同fd package loader尚未实现 | 验证失败不得打开payload或生成新诊断 |
| PRI-08 | 历史artifact分级 | 区分`PROTOCOL_VALID`、`UNVERIFIED_UNDER_CURRENT_PROTOCOL`和确认泄漏后的`PROTOCOL_INVALID_*` | protocol docs、report/summarizer | implemented | `classify_legacy_phase2_record`负测区分缺证据与确认clean访问 | 缺字段阻断晋升，但不自动等价于已证明实际泄漏 |
| PRI-09 | v14修复闭环 | 修复candidate-lock契约漂移，并把上述隔离契约接入effective8 v14正式计划；target matrix保持阻断直到全部验证通过 | v14 config/plan/lock/runner/report | blocked | plan写入`LOCAL_PROTOCOL_REPAIR_REQUIRED`且`launch_authority=false`；runner和candidate-lock CLI fail-closed | 本轮不启动N607实验 |
| PRI-10 | Git与文档同步 | 根协议、Git镜像、追踪表、代码、测试和报告保持一致并形成单独Git提交 | 根`AGENTS.md`、根`项目.md`、repo docs/report、Git | in_progress | 根三份tool镜像SHA已与Git承载面一致；待最终diff、测试与commit | 保留现有无关工作树改动，不纳入本次提交 |

## 反向审计门槛

- 任何`implemented`项必须具有可到达的生产代码路径和至少一项负向验证。
- Phase2 predictor进程输入中不得出现truth/role/query-per-class count/quota字段。
- formal scorer只能读取已经密封并哈希固定的prediction artifact；不得导入训练、适配、门限选择或预测入口。
- Phase2 sealed package不得包含原始数据路径、cache build spec、clean/raw成员或可调用legacy loader的运行参数。
- 本地验证完成前不进行SCP，不启动或恢复N607 target matrix。
