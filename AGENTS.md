## 本对话 F 后续只测 clean（2026-10-19，用户最新要求）

- 用户明确本对话后续只做 clean 测试，取消 practical 场景；覆盖此前 clean＋六种 practical 的测试默认。训练增强以每个 run 实际已固定配置为准，不因修改测试范围追溯改写。r04 实际训练含 sat CE0.68，完整模型保留，仅测试范围改为 clean。
- 已取消的 practical 预测与缓存保留，停止操作仅限经PID/完整命令核实的本 run 调度和预测进程。默认头冻结预测、独立 truth-last 与本机最多3个GPU worker保持。

## 本对话本机 F 实验并行（2026-10-19，用户最新要求）

- 用户针对本对话本机显存余量明确要求多实验并行。当前 F＋跨 RX／弱 TX／joint160 四组研究在本机 GPU0 最多并行 3 个训练或预测 worker，独立 CPU scorer 最多 1 个；覆盖本对话此前单 worker 串行默认，不改变其他对话或远端 owner 的资源约定。
- 只切换本 run 排队控制器并接管其健康子进程；已完成行复用，不重复训练，不热改模型、预算、种子或科学配方。源域扩种子决定仍在任何新目标访问前冻结，逐行全视图预测后独立 truth-last 评分。并行后记录实际吞吐、系统内存与显存，不以占用率代替加速证据。

## 本对话 F 路线改用本机（2026-10-19，用户最新要求）

- 后续本对话的 F＋跨 RX 训练机制研发在本机完成，使用已验证的本地 Phase1 环境与原物理划分缓存。单 GPU0 串行、最多一个本任务训练或预测 worker；覆盖本对话此前 N607 GPU2/3 发布默认，不改动其他 owner 的远端健康任务。
- 本机使用独立、不可覆盖的 run/output root 和 Git commit 固定代码；保留源域权限、默认头联合训练、源域冻结、逐行自动测试及独立 truth-last 评分。N607 登录、远端发布和传输检查对新本机 run 不适用。数据、权重、原始日志与本机配置仍不提交；提交方法实现、登记和公开汇总结果。

- 后台控制器与各训练/预测子进程采用无控制台启动，使用已配置 venv 共享的 `base_python` 直接入口，避免重定向器产生可被关闭的控制台；解释器依赖、数值设置和科学配方保持。保留实际进程与日志证据，不把预约文件或后台启动返回值当成运行成功。

# Project Instructions

## Phase1 默认分类头约束（2026-10-13，用户最新要求）

- 本轮及后续 Phase1 特征提取优化使用网络既有默认分类头，随骨干按原流程联合训练；禁止冻结骨干后单独训练、重训或替换分类头，也不另训身份/接收机诊断分类头。
- 训练冻结后直接使用 checkpoint 内默认分类头预测和独立 truth-last 评分。源域诊断使用默认头或不拟合参数的特征分析；不得把另训头作为发布、扩种子或性能目标门槛。
- 新机制进入实际部署特征，分类头结构和原联合训练方式保持。已完成的历史换头诊断保留并明确标注，后续正式性能比较只用默认头；不删除、覆盖或重跑旧产物。


## E20基线与新探索预算（2026-10-12，当前用户覆盖）

- 用户要求后续探索纳入 Fishr、MixStyle 等机制，覆盖浅层和中层特征，不能只修改最终特征空间。按独立消融检验；Fishr 必须明确梯度对象与域分组，MixStyle 必须记录实际层位和身份保护配对，未验证前不宣称有效。
- 后续探索对比基线改为三种子纯CE＋旧门控＋cos的固定E20。允许从头训练，也允许从对应seed的E20续训；不再默认从E200开始。原84% clean三种子均值目标保持。
- 从头训练总计20个epoch。用户明确允许训练阶段、学习率日程、半监督及新机制启用安排不同于原E20基线；具体方案发布前固定并披露，不把基线配方当成必须照搬的限制。
- E20续训只追加20个epoch，累计40个epoch，覆盖此前“追加40”的预算。保留同父权重、同增量预算的纯CE续训对照；从头E20与E20＋20不冒充相同总训练成本。
- 基线保持纯CE、L_s clean、label_smoothing=0。研究变体按用户后续明确要求利用U_s进行半监督，U_s TX真值始终隐藏；训练不用星地信道增强。数据角色和物理ID约定不变，源V只作健康及机制诊断，不追逐微小涨分。
- 三种子2026092701/02/03，逐row固定训练预算后自动clean预测与独立truth-last评分；每GPU最多3个总实验进程，计入其他owner。不得热改、停止或重启健康旧任务。
- E20来自已暴露测试基准的epoch压缩研究，现由用户明确指定为探索基线。报告如实保留该选择背景；新机制不依据目标分数调参、重排或选择性重跑，不声称新的独立盲测。
- E20实际来源、只读核验和后续辅助网络处理见`docs/CVS_E20_EXPLORATION_PLAN_20261012.md`。本条覆盖旧“固定E200第二阶段”及旧预算约定，历史实验不追溯改写。

## 多解耦探索默认从固定纯CE父基线开始第二阶段（2026-10-11）

- 用户要求后续多解耦方案复用第一次纯CE基线训练，再开展独立第二阶段训练，以减少重复训练成本。保留身份网络内部门控、参考响应结构、cosine分类头与cosine学习率日程；纯CE下不重新开启旧伪标签门控、半监督或星地信道增强。
- 每个model seed使用与本次实际物理数据角色完全一致、来源已核实、按源域规则或固定轮次冻结的父权重。当前已核实来源及下一版方案见`docs/CVS_MULTI_DISENTANGLE_STAGE2_20261011.md`；同名“旧门控+cos”实验不能替代实际配置核验。
- 各变体从同一seed对应的固定父权重独立分叉，不串接上轮按测试成绩挑出的变体。改变数据契约或出现继承污染时，依原checkpoint禁令处理；本条不放宽数据权限。
- 第二阶段必须明确是恢复原优化轨迹，还是重置优化器和学习率的微调；先严格加载父模型，再建立辅助网络参考快照。保留同预算纯CE第二阶段对照，分别报告父训练成本、增量成本和累计预算。未明确的新预算不擅自缩减。
- clean测试准确率84%以上是本路线的最终验证目标，不是已达成结果，也不得作为反复测试调参、选模或选择性重跑的反馈。训练及方案选择仍只使用合法源域证据；冻结后按既定自动测试流程报告实际三seed结果和缺项。
- 本约定仅适用于后续新方案，不停止、重启或热修改当前健康实验。

## 后续探索实验默认纯CE基线（2026-10-10）

- 用户为加快训练，要求后续探索实验基线只使用纯CE，关闭半监督和星地信道增强。新探索基线保留当前身份网络结构及分类头，只以合法有标签源域L_s的clean IQ计算身份交叉熵；label_smoothing=0。只使用L_s不等于解封U_s标签或扩大有标签数据比例，V仅验证，数据契约保持。
- 基线关闭U_s训练、伪标签、EMA教师及教师前向/更新、伪标签置信度/邻近批次/强增强一致门控、无标签熵损失和130+70半监督阶段；关闭卫星视图生成、clean+satellite拼接及其CE/一致性损失。基线不默认附加其他随机增强、多视图前向、GRL、正交、互信息、原型、域损失或L/T/R辅助网络。
- 保留身份网络中的时频/PA/参考响应结构、分类头和cosine学习率日程；优化器与weight decay属于优化设置，按本次配置显式记录。纯CE指损失目标，不要求改成线性分类器或删除身份网络内部融合。新的待研究机制只加入对应变体，和同训练预算的纯CE基线对照。
- 新探索launcher必须实际关闭无用教师、无标签迭代、增强生成和辅助前向，不能仅把loss权重设零后仍执行昂贵分支。日志及resolved config核对实际执行状态。未指定的新epoch/seed预算不自行缩减；速度收益需实测，不预先承诺倍数。
- 此项是探索实验训练默认的明确例外，覆盖历史默认半监督/星地拼接配方；当前健康任务和历史实验保持原配置，不重新训练。训练关闭信道增强不改变测试：仍按既有约定自动测试clean及六种practical，prediction固定后独立truth-last评分。


## 逐行训练后自动测试与六种practical默认（2026-10-10）

- 用户要求以后每次训练完成自动启动测试。新实验的launcher/队列须实际串联训练、合法源域选模/冻结、测试预测、独立truth-last评分、报告和登记，不只写提醒或停在SOURCE_TRAINED，也不再次等待测试指令。
- 固定对比矩阵中，每个row完成并冻结后即进入测试队列；有资源即执行，不因其他row尚未训练完成而等待整批结束。若预登记要求跨row源域候选选择，先完成必要的source-only选择并冻结选中候选，再自动测试；不得借此对未选候选访问target。
- Phase1默认测试保留clean对照，星地环境只测六种practical：`practical_high`、`practical_mid`、`practical_low_suburban`、`practical_high_urban`、`practical_mid_urban`、`practical_low_urban`。不再默认追加`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`。本条覆盖旧的Phase1最终测试场景默认，不改变训练增强或Phase2固定received IQ协议。
- 每row全部预登记视图的prediction固定后，独立scorer才可连接truth。逐row结果不得影响其他健康训练、源域选模、参数或候选排序；完成后汇总实际seed数、配对差与缺项，不把不完整seed汇总冒充完整矩阵。
- 同一row只有一个测试launch owner；恢复时先核实进程和prediction，已完成测试复用原产物，不重复启动。资源容量、checkpoint来源、数据角色与输出保护继续遵守原规则。
- 本约定适用于后续新发布实验。当前健康运行任务保持已冻结计划；不热修改、停止、重启或追溯重测。用户明确只训练、只源诊断、只测clean或指定其他场景时按明确范围执行。


## 实验发布默认收尾（2026-10-01）

- 用户要求每次发布实验最后默认进行测试集测试。新发布实验在既有预登记中写明测试数据、物理ID、视图/场景、逐行矩阵、指标和输出；默认完成训练（如适用）、合法源域选模/冻结、测试集预测、独立truth-last评分、同row报告及登记/Git交付，不以仅训练完成或源V指标作为最终结果。
- 已有授权范围内且输入合法时，冻结后直接继续预登记测试，不再次索要测试许可。技术失败或缺输入时保留产物、说明缺项及准确状态；未经测试/评分不得宣称实验全部完成。
- 源域候选筛选矩阵先按固定源规则选择并冻结，只测试预登记选中的候选；固定对比基准按已登记矩阵测试。测试结果不得回流调参、候选重排、选模或选择性重跑。
- 用户明确只训练/只做源域诊断时按该范围执行；本次明确只测clean的实验不追加LEO/卫星测试。默认收尾不改变科学协议、数据角色或当前矩阵，不授权停止、重启、热修改健康任务，也不追溯重跑已完成的历史实验。

## Checkpoint复用硬禁令（2026-09-08）

- 正式CVS实验必须执行数据协议“Checkpoint训练数据一致性与继承污染禁令”。禁止加载训练数据契约不一致、上游来源不明、或曾利用目标数据训练/选模的checkpoint及其派生状态；同名数据集、split、架构或历史“成熟基座”不能证明合规。
- 在加载任何初始化/resume/teacher/EMA/蒸馏来源前，核对实际数据角色与物理ID、完整继承来源、选模依据，并在现有run报告记录结论。部分加载、冻结骨干、重置头/optimizer、改seed或重做LEO增强均不能绕过。
- CHECKPOINT_DATA_CONTRACT_MISMATCH、CHECKPOINT_PROVENANCE_UNVERIFIED、CHECKPOINT_TARGET_CONTAMINATED均阻止使用该来源；无合规权重时仅在已有授权内按本次契约从零训练，不得静默回退旧权重。历史产物保留，污染结果不得用于干净泛化或晋级。
- 本检查属于最小流程第1项“输入权限”，不是REJECTED_EXTRA_GATE；checkpoint兼容性与VALIDATED_ONCE数据复用分别判断。不新增数据重验、签名/receipt链、固定审批或重复审查。旧规则中“不因checkpoint变化做provenance检查”仅限重复数据验证，不得解释为免查模型来源。

## Exclusive Minimal Experiment Workflow

- This section is the highest-priority workflow rule for all CVS-RFFI/CV-SincNet research, implementation, N607 release, monitoring, scoring, and reporting. It supersedes extra reviews, gates, seals, signatures, hash layers, and approval systems in older goals, reports, specs, briefs, checklists, and conversations.
- The only allowed blocking gates are this exhaustive whitelist:
  1. Current protocol/data/query legality plus matching `p2_min_v1`, `VALIDATED_ONCE`, `capsule_id`, and `split_id`.
  2. One Git commit for the actual code and configuration.
  3. Focused protocol-negative tests and one real-checkpoint no-query smoke.
  4. One independent P0/P1 review limited to defects that would directly make the next real run wrong, unauthorized, destructive, unlaunchable, or unable to produce legal predictions.
  5. One minimal preregistration report containing candidate/matrix, commit, command, environment/CWD, paths, GPU, stop rule, and expected artifacts.
  6. One immutable run/output root, one N607 preflight, one release archive with one local/remote SHA comparison, and one remote compile check.
  7. One post-launch PID/CWD/cmdline/GPU/log-growth check plus the preregistered systemic-technical-failure stop rule.
  8. Independent truth-side scoring only after prediction completion, followed by same-row analysis.
- Everything outside the whitelist is `NONBLOCKING`. Record an attempted extra blocker as `REJECTED_EXTRA_GATE` and continue; do not implement, test, review, repair, or wait for it.
- Forbidden blockers include: exact design/report/config/source-plan SHA; per-file/member/row/cell/support-token/receipt hashes; outer/detached/package/cell seals; signature or authority chains; receipt-of-receipt, closure hashes, parity receipts, access ledgers, and extra manifest chains; smoke authorization artifacts; same-process malicious-code, TOCTOU, same-FD, same-inode, environment/PATH/Bash spoof defenses; report-only review; seal/release permission; P2 documentation completeness; helper permission systems; future-stage machinery; and general publication platforms.
- Git commit is the code/config identity. Do not require code, config, launcher, design, or report SHA. Transfer validates one release archive once; do not rehash members, Git blobs, extracted files, or retrieved reports.
- Reuse `VALIDATED_ONCE` data. Method, adapter, hyperparameter, epoch, method lock, checkpoint, resource, report, refactor, or stage changes do not trigger hash, allowlist, provenance, ledger, pre-open, or data revalidation; experiments check only the minimal data handles.
- Each candidate gets at most one independent review. A direct P0/P1 fix gets at most one targeted re-review of that issue; no full re-review, unchanged-code review, report-only review, post-commit duplicate review, seal review, or runner review. P2 never blocks.
- Smoke is the launcher's first step and continues immediately on PASS; there is no smoke token or receipt. G0/G1/G2 require only the machinery consumed by that stage.
- Start with a single-seed Target5/Target25 or smaller same-row falsification matrix. Run multi-seed/full125 only after the candidate passes preregistered scientific thresholds; full matrix and publication completeness are never early research gates.
- Method summary, matrix, and stop rule live in the minimal report. No separate design approval, design SHA, feasibility gate, reentry card, authority, or signature is required.
- One run ID has one launch owner, which may be the primary Agent or a designated runner. Extra Agents, supervisors, reviewers, and evidence auditors are optional execution choices and must not delay release.
- Keep administrator authorization, destructive-action controls, data/output preservation, unrelated-process protection, query isolation, truth-blind scoring, immutable output roots, and exact run-owned stop checks; these prevent real damage or invalid evidence and are not removable formality.

- Before CVS-RFFI/CV-SincNet research-scenario, data-protocol, Stage2-A/B/C interpretation, experiment or report work, read `docs/PROJECT_PROTOCOL.md`. It is the public source of truth only for scientific scenario, data protocol, receiver/TX sets, Phase1/Phase2 boundary, Stage2 permissions and claim semantics. Active targets, method routes, matrices and resource budgets belong in a separate goal document.
- Project-related changes must use a Git-backed workflow. Before editing, run `git status -sb`; after editing, inspect `git diff`/`git status -sb`, run the narrowest useful verification, and commit the intended change unless the user explicitly says not to commit.
- Update this file only for workflow/safety changes and update `docs/PROJECT_PROTOCOL.md` before changing scientific/data-protocol/Stage2/claim semantics. Ordinary code/config/script changes do not require README/docs/report updates before experiment; user-facing documentation may be completed after evidence returns and never becomes a release gate.
- For tool-using or long-running tasks, send concise, evidence-based progress commentary before the first tool call, at meaningful phase changes, after reconnect or context-compaction recovery, whenever a blocker appears, and at least once every 60 seconds while work is active. Report observable actions, findings, and next steps; do not expose private chain-of-thought or dump raw logs. Commentary may be omitted only for short answer-only turns with no tools.
- Do not commit datasets, trained weights, checkpoints, private runtime details, logs, generated experiment outputs, or local machine state. Keep `.gitignore` aligned with this boundary.
- Do not claim deployment success from clean-view or diagnostic-only evidence. Any reported result must be tied to a concrete run, split, K-shot setting, satellite/LEO view, and same-row metric context.
- Every Phase1 experiment that completes training must evaluate the selected final checkpoint on the declared clean test split and on each `leo_weak` family scenario: `leo_clear_weak`, `leo_low_elev_weak`, and `leo_rain_weak`. Training completion alone is not experiment completion. The run may advance to `ARTIFACTS_COMPLETE` or `ANALYZED` only after the checkpoint identity, evaluation configuration, clean metrics, per-LEO-scenario metrics, and corresponding logs are preserved; do not replace per-scenario results with only an aggregate mean. If any required evaluation cannot run or fails technically, preserve the training outputs, record the exact missing/failed evaluation, and do not mark the experiment complete.
- Phase2 follows `protocol_schema=p2_min_v1`: one randomly selected allowed LEO weak observation per physical sample; K independent support samples; scenario and support/query physical-ID separation; query-only testing; no clean/source runtime access except immutable int8 multi-sample Phase1 knowledge jointly sealed with the checkpoint; and independent all-registered-class decisions without query truth/role, true batch counts, quota or global reassignment.
- After a builder returns matching `capsule_id`, `split_id` and `phase2_data_status=VALIDATED_ONCE`, reuse the data across methods. Revalidate only for received-IQ bytes, physical IDs, receiver/TX sets, scenario, K, support/query split or schema changes. Candidate, adapter, hyperparameter, epoch, prototype rule, method lock, model state and resource changes do not trigger data revalidation; one-time provenance/hash/allowlist/runtime checks belong to the validator appendix.

## Administrator Account Authorization Boundary

- Administrator and privileged remote accounts are denied by default. Use an ordinary non-privileged account unless the user explicitly authorizes administrator-account use for a concrete, bounded task in the current request.
- Administrator authorization is task-scoped and expires when that task ends. It does not carry into later turns, automation, monitoring, subagents, retries, or adjacent maintenance.
- Authorization to log in as an administrator does not authorize state changes. Configuration edits, writes, deletion, movement, permission/ownership changes, package or service operations, process termination, network/storage changes, reboot, shutdown, power-cycle, firmware, BMC, iDRAC, and other persistent or risky actions require separate explicit user authorization naming the action and target.
- If administrator use or mutation scope is ambiguous, stop and ask. Never infer authorization from a prior password entry, available credentials, troubleshooting urgency, or a general request to inspect or fix a system.
- After explicit authorization, keep commands minimally scoped, inspect workloads and targets first, prefer read-only checks and dry-runs, preserve recovery paths, record exact commands/results, and never publish private account identifiers, credentials, keys, or other local runtime details.

## Windows Shell and Command Hygiene

- Prefer PowerShell 7 for complex Windows commands. Use `pwsh -NoLogo -NoProfile -Command "<command>"` when shell behavior, quoting, or UTF-8 output matters.
- Do not run Conda wrapper commands in parallel or in tight bursts. If a Conda temp activation file is locked or missing, treat it as command-wrapper noise first, rerun serially, and only then classify a project validation failure.
- Do not paste Bash here-doc syntax into PowerShell. Use a PowerShell here-string, a temporary script, `python -c`, or UTF-8 stdin to `ssh ... python3 -` / LF-only stdin to `ssh ... bash -s`.
- For Markdown, JSON, and non-ASCII text, use explicit UTF-8 writes, BOM-aware reads, and Python JSON I/O when payloads are large or encoding-sensitive.
- Do not assume all PowerShell versions support the same parameters or .NET helpers. Check support before relying on `Tee-Object -Encoding`, `Test-Connection -TimeoutSeconds`, `[IO.Path]::GetRelativePath`, or compression APIs that may require explicit `Add-Type`.
- After SSH/SCP timeouts or malformed quoting, verify local SSH clients and TCP connections have exited, then use read-only remote process/log evidence before relaunching or declaring failure.


## CVS基础网络优化优先级（2026-10-01）

- 用户明确：主要目的是提升识别性能，参数轻量是次要条件。后续基础网络研发以合法源域性能与跨接收机稳定性为主要依据，参数/计算/状态成本作为次要比较，不能为了更少参数默认接受较低性能分数。
- 参数轻量仍是软约束；不以单纯堆参数替代结构分析，也不把参数减少本身当作性能目标完成。性能实际提升必须由冻结后的独立测试证明，所有负结果保留。
- 本轮健康源训练不热改或重启。原0.2个百分点内成本优先的source_selection保留为历史记录；新确认在任何新target访问前使用独立performance_selection，按四seed源V/最差源RX性能分数最高优先，性能完全并列后才比较成本。目标测试结果不得回流选模、结构、超参数或重跑。
