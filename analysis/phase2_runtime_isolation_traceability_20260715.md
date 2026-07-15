# Phase2运行时隔离与query信息边界完善追踪

日期：2026-07-15

目标：在现有`LEO_weak-only`与逐样本全注册类决策协议上，把“字段自声明”升级为可执行、可验证的运行时隔离契约；同时把容易混淆的`class_count`字段明确为“真实query批次类别计数”，并把预测与带标签评分拆成权限隔离的两阶段。

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|----|----------------|-------------|--------------|--------|--------------|-------|
| PRI-01 | 用户本轮要求 | 将`phase2_query_class_count_access`规范化为`phase2_query_true_batch_class_count_access`，明确不禁止读取已注册类别总数 | `AGENTS.md`、根`项目.md`、Git协议镜像、生产schema/validator | implemented | validator与predictor-contract负测覆盖deprecated字段和合法`registered_class_count` | 旧字段仅作为deprecated输入识别，不足以单独证明新协议合规 |
| PRI-02 | query决策边界 | Phase2决策包不得包含query truth、old/new/unknown role、真实批次类别计数、每类quota、标签分块或排序提示 | formal plan、sealed inference request、benchmark predictor | implemented | 双根bundle使用每次密封随机HMAC token；request exact schema、禁止键/值负测及truth-free生产predictor均已接通 | 正式plan仍须复用该唯一request builder，不允许回退旧runner |
| PRI-03 | 预测/评分隔离 | 预测必须先产出sealed/tamper-evident prediction artifact；独立post-prediction scorer再连接truth/role计算指标 | predictor、scorer、runner、summarizer | implemented | 单一防API覆盖`.cvspred`容器绑定payload→manifest→seal；独立scorer先验artifact/seal SHA后才连接truth sidecar，并强制3个场景顺序及相同query-token集合 | scorer输出不得反向影响adapter、门限、回滚、候选选择或prediction；不宣称宿主同UID绝对不可变 |
| PRI-04 | clean dataset不可达 | Phase2进程只能访问sealed LEO cache/inference包allowlist根，不得接收或打开ManySig/ManyTx、raw PKL、clean路径或cache build spec | sealed package builder/loader、runner、candidate lock | partial | 7文件最小runtime closure、固定system root allowlist、`/runtime/code`只读挂载、无网络bwrap策略及sandbox外父级strace tracer均已本地测试；predictor不接收trace FD且trace路径不挂载，open ledger只保留已绑定Python成功`execve`后的预测阶段 | adapter/head/TTA provenance与固定inode/不同UID snapshot尚未闭合；N607真实等价隔离前formal=false |
| PRI-05 | clean cache/API/control-flow不可达 | Phase2生产入口使用专用strict loader；拒绝legacy raw/clean loader参数、环境变量、sidecar及非allowlist模块入口 | strict loader、CLI、runtime manifest | implemented | strict生产入口仅导入7文件审查闭包；AST exact import closure拒绝dataset、training、legacy和动态导入；固定argv不接受loader/scorer参数 | 旧`adv3b02_supervised_da_runner.py`与并发Landlock比较路线不自动成为本严格runtime的可替代入口 |
| PRI-06 | 9项硬字段证据化 | 每个plan row、candidate lock、prediction manifest和formal result都记录4项clean不可达字段及5项query决策字段，并绑定验证证据 | config、plan、lock、predictor、scorer、summarizer | partial | 已统一为3基础+4clean+5query共12项；9项pre-run bundle由外部seal锚、closure、控制代码、实际可执行文件和物理分根交叉绑定，4项post-run证据仅在open ledger PASS后生成 | 本地artifact链已闭合，但正式qKNN plan/candidate lock尚未接入且无N607实际post-run证据 |
| PRI-07 | pre-open fail closed | 在打开任何Phase2 payload前验证sealed manifest、成员allowlist、LEO provenance、scenario、satellite seed、SHA256和禁止路径 | strict loader、runner、tests | implemented | request先做exact 12字段校验；同fd package loader、exact NPZ allowlist、路径/symlink/篡改负测、外部seal SHA、pre-run bundle及closure在预测前后复验 | 本地fake subprocess仅验证控制流，不替代Linux OS隔离事实 |
| PRI-08 | 历史artifact分级 | 区分`PROTOCOL_VALID`、`UNVERIFIED_UNDER_CURRENT_PROTOCOL`和确认泄漏后的`PROTOCOL_INVALID_*` | protocol docs、report/summarizer | implemented | `classify_legacy_phase2_record`负测区分缺证据与确认clean访问 | 缺字段阻断晋升，但不自动等价于已证明实际泄漏 |
| PRI-09 | v14修复闭环 | 修复candidate-lock契约漂移，并把上述隔离契约接入effective8 v14正式计划；target matrix保持阻断直到全部验证通过 | v14 config/plan/lock/runner/report | blocked | plan写入`LOCAL_PROTOCOL_REPAIR_REQUIRED`且`launch_authority=false`；严格runner和candidate-lock CLI保持fail-closed | 本严格qKNN管线未启动；并发运行的ADV3B02 Stage2-B Landlock比较是另一条artifact链，不得替代本项闭环 |
| PRI-10 | Git与文档同步 | 根协议、Git镜像、追踪表、代码、测试和报告保持一致并形成单独Git提交 | 根`AGENTS.md`、根`项目.md`、repo docs/report、Git | in_progress | 根三份tool镜像SHA已与Git承载面一致；待最终diff、测试与commit | 保留现有无关工作树改动，不纳入本次提交 |

## 反向审计门槛

- 任何`implemented`项必须具有可到达的生产代码路径和至少一项负向验证。
- Phase2 predictor进程输入中不得出现truth/role/query-per-class count/quota字段。
- formal scorer只能读取已经密封并哈希固定的prediction artifact；不得导入训练、适配、门限选择或预测入口。
- Phase2 sealed package不得包含原始数据路径、cache build spec、clean/raw成员或可调用legacy loader的运行参数。
- 本严格qKNN管线完成本地验证并取得可用的N607等价隔离执行路径前不进行SCP，不启动或恢复其target matrix。

## 2026-07-15当前实现计数

- `implemented`：6项（PRI-01、PRI-02、PRI-03、PRI-05、PRI-07、PRI-08）。
- `partial`：2项（PRI-04、PRI-06）。
- `blocked`：1项（PRI-09）。
- `in_progress`：1项（PRI-10）。
- 当前聚焦回归为117项全部通过；唯一告警族为TorchScript弃用提示，不影响本次诊断合同测试。
- 最高剩余风险：adapter/head/TTA生成provenance尚未绑定外部candidate/plan trust root，普通同UID目录只读bind不能排除瞬时替换后恢复，N607也不能直接使用当前bwrap user namespace路径。isolated runner因此固定返回`LOCAL_DIAGNOSTIC_PASS`且`formal_launch_authority=false`；仅发布带`protocol_valid_claim_allowed=false`的diagnostic wrapper，13字段post-run合同不单独作为正式PASS文件落盘。必须先关闭上述blocker并取得真实Linux后验open ledger，才可标记`PROTOCOL_VALID`。并发ADV3B02 Stage2-B正式运行不得被误计为本严格qKNN管线已落地。
