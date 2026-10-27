## PAMF REPL 时间打乱对比优先（2026-10-27，用户最新要求）

- 用户指定本轮优先完成三视图＋原版 REPL＋固定时间打乱的整组训练和七景正式评分，然后才启动单视图同方法匹配组；不再按普通 PL 运行本轮。独立 run 为 `20261027-phase1-pamf-repl-shuffled-views-m16-r01`，每组八个配对种子；原默认单视图结构约定仍保留。
- 原单视图普通 PL 论文 run `20261027-phase1-pamf-single-view-paper-m120-r01` 已按用户要求暂停，本 owner 的 dispatcher／八 worker 精确停止，已存产物保留，不自动恢复。其他 owner 健康任务保持。此次仅改变本轮优先级与明确方法，数据和 truth-last 权限保持；配置见 `docs/PAMF_REPL_SHUFFLED_VIEWS_20261027.md`。

## PAMF 单视图默认与论文命名（2026-10-27，用户最新要求）

- 后续未限定的 PAMF 结构默认单视图 SIG-TIME（完整六类识别器 196001 参数），保留时频、PA、参考响应和原联合分类头；半监督保持普通 PL0.95/权重1，无 EMA 或复杂门控。该默认覆盖下方历史 T1／三视图默认；历史权重和成绩保持原身份。
- 论文主比较只替换为匹配单视图，完整有标签骨架保留；网络分支消融分别跑纯 CE 监督和 CE＋普通 PL，两组无 CRIC/KL/EMA，预算／标签权限匹配。当前矩阵和复用边界见 `docs/PAMF_SINGLE_VIEW_PAPER_20261027.md` 与 `configs/pamf_current_method.json`。
- 新展示名使用 Main split、Receiver split 1/2、1%/5% labeled、m source receivers 等语义名称；原 F0/F1/F2、F01–F21、T1 仅留追溯。不得移动历史输出目录或将不同历史划分静默改标。
- 当前论文复现 run 为 `20261027-phase1-pamf-single-view-paper-m120-r01`，108 新 F、12 精确复用；新 G／恢复缓存为0。保留其他 owner 的健康任务，不重复其已完成单视图对照。

## PAMF最终识别器固定为T1续训E50（2026-10-21，用户最新要求）

- 用户指定“T1：去除As独立KL，原E30续训20至E50”为PAMF最终识别方法，累计日程明确为 **13L＋37U**；37U阶段仍联合使用L/U。标识 `pamf_t1_e50`，定义见 `configs/pamf_final_t1_e50.json`，完整机制见 `docs/PAMF_FINAL_T1_E50.md`。这覆盖本对话此前原F50／33L＋17U作为当前识别器基底的约定；该特定T1配方继续保留旧U门控与自身EMA，不静默替换为普通PL。
- 当前已完成默认单模型固定 `20261020-phase1-T1-AsKL0-e50-s2026092701-m2-r01/E30PLUS20-s2026092701/source/final_ssdg.pth` 的FINAL50学生；原同seed FINAL30学生/自身EMA续训，新增20轮/4440步，累计E50/11100。保留SIG_TIME_X3、CENTER_25K/CENTER_UNITS、ordinary/raw RDrop、As两次真实标签CE而As KL=0、LS0.01、ORD128浅中层CROSS_RX、原隐藏U门控/EMA/熵项及原默认头联合训练；不混入SCRATCH50的33L＋17U、T6、KD或另训头。
- DURA v1.0继续固定为恢复器G，PAMF-T1-E50是识别器F。此前DURA＋原F50的4.531972ms与七景指标保持原配对语义，不自动改标成DURA＋T1-E50。新组合需独立冻结测试，不因本次版本指定而启动新实验。
- 原T1三seed续训复现由其既有owner按原矩阵完成，保留健康进程，不重训或按目标结果替换默认单模型。本次只固定既有方法并分析，原权重和历史对照均保留，不扩展现有证据到其他数据契约或Phase2。

## 默认恢复器固定为 DURA v1.0（2026-10-21，用户最新要求）

- 用户在完整七景与实际时延结果公布后明确固定 **DURA v1.0**，中文名“双阶段展开恢复网络”，部署标识 `dura_v1`。后续未限定的“恢复器／恢复网络／当前恢复版本”均指本版；定义见 `experiments/pamf_stage2_inference/frozen_version.json`，机制见 `docs/DURA_V1.md`。
- 固定 R69 `20261021-phase1-restoration-R65-four-route-s2026092701-m4-r69/stage2/native_map_prior/final.pt`，源 E20/4940、266951 参数；固定已验证执行 commit `47c264286ca689f060a3f36aafb763fe209a6f06` 和 `InferenceEngine(reuse=True, graph=True, native=False)`，FP32/TF32off，两 prior／四 CG／27 FFT-IFFT。原权重不可覆盖，不静默回退旧恢复器、native 或改变精度。
- 验证 run 为 `20261021-phase1-restoration-stage2-graph-s2026092701-r01`，原 F50 seed2026092701/defaulthead，168000×7、独立 truth-last；B1 同卡 G+F 41.220936→4.531972 ms。七景无下降，high urban 两条错变对；low urban 68.889881%，70% 与其他五景 5pp 长期目标未全部达成。
- 新网络、权重、求解预算或运行路径另记版本，明确比较，用户指定后才更换本默认。R17、R65 `prune_after10` 与两阶段 native 保留为历史对照；原 R17 科学容忍对照、TX 无标签／F-free 训练约束及逐版七景测试保持。其他 owner 的健康任务按原计划继续；本次固定不重训、不重测，不扩展现有证据到其他数据契约或 Phase2。

## 本对话恢复器改为压缩与性能联合优化（2026-10-20，用户最新要求）

- 用户明确将后续研发改为前面最优 R17 恢复路线的压缩精简，并要求“压缩的同时也要优化”。完整恢复器的参数、计算、存储、实测延迟与正式识别性能共同报告；每场景最多下降 1 个百分点是容忍上限，不是性能优化目标。不得只凭网络更小、源域分数或教师拟合误差宣布成功。
- 固定压缩对照为已在目标访问前源域选定的原 R17 `unroll_unshared` E20/4940，完整参数 4665073，原 F50 seed2026092701/defaulthead 不变；不从新目标结果重新挑父模型。每个新版均完成原 168000 物理样本 × clean、三郊区和三 urban 正式测试，每场景较原 R17 最多少 1680 条正确预测，不能以均值掩盖超限。优化争取实际正收益，原 low urban≥70% 和其他五环境距 raw clean≤5pp 继续单列。
- 保持本对话恢复器训练的 TX 无标签和 F-free 约束：利用合法 L_s/U_s IQ、模拟配对及来源合规的原 R17 IQ 教师，不读 TX 真值、训练好的 F 权重/输出或 target 来优化恢复器；F 仅在冻结后的评测与无 query smoke 使用。其他 owner 的 F 蒸馏压缩模型只作注明权限差异的参考，不静默继承。
- 盘点和压缩包含 coarse、action estimator、prior 及物理求解成本，不把删除 optimizer 当作结构精简，也不以权重共享推定推理加速。已运行 R37 两版正式测试保持健康完成并归档；后续新增网络研发沿联合压缩路线，不热改/停止其他 owner。具体验收、原版计数和下一步研究见 `docs/goals/PAMF_UNSUPERVISED_COMPRESSION_20261020.md`。

## 星地恢复器新版逐版正式测试（2026-10-20，用户最新要求）

- 用户明确：后续每个恢复网络新版本都必须进行正式测试，不能仅凭源域收益判断有效，也不得因源域无提升而取消该版本的正式测试。从当前 R32 的 `pair_aug_control`、`observable_response` 两版开始执行；源域只用于合法冻结 checkpoint 和机制诊断，正式测试矩阵在目标访问前固定。
- 每版连接同一冻结 F50 原默认头，覆盖 clean、三种 suburban 和三种 urban practical，共七个视图；逐版全部预测固定后独立 truth-last 评分，报告实际测试收益、clean 损伤、逐 TX/RX/day、净纠错及目标差距。测试结果不回流选模、调参或选择性重跑；源域提升不能替代测试收益或 TX 细节真实性证明。
- 当前健康 R32 保持原训练和调度，不热改。独立 R36 已固定两版正式矩阵；R32 在训练完成后源评分接口出错，后续由新 R37 复用权重和完整源预测、修复评分并执行相同两版正式矩阵，复用原队列已完成且可比较的七视图结果，其余补齐。未发布的条件补测 R33 被本次范围更新替代，保留登记。历史已完成源域负结果不因此自动追溯补测；同版只有一个测试 owner，不重复启动。

## 训练默认 N607、恢复器多 GPU 验证（2026-10-19，用户最新要求）

- 用户最新要求训练默认使用 N607，并允许本对话使用多个 GPU 快速验证研发；本对话此前的本机单 GPU 训练默认由本条覆盖。除非用户另行指定，本机用于实现、测试与登记，正式训练在 N607 进行。
- 具体 GPU、并发数与恢复器矩阵由当前 run 预登记固定，启动时核实实际空闲容量；每卡最多一个本 run 训练／预测 worker，总进程容量计入其他 owner。保留现有健康任务和不可覆盖输出，不因提高优先级停止或修改其他任务。
- 恢复器有效性须直接接到来源与数据契约已核实的训练好 F 基座和原默认头，冻结识别器验证同一输入对应的真实 TX、净纠错、clean 保持及逐 practical 表现。波形外观、NMSE 或生成质量单项不能替代身份保真证据。

# Project Instructions

## CVS名称与历史E200种子扩展（2026-10-12）

- 后续未加限定的“CVS”默认指残差融合架构 `residual_fusion`。原架构 `native` 仅在必须辨认历史记录时明确标注，不再默认作为新增对比项。
- 用户指定历史E200对比扩展：保留原划分和配方，除原CVS外，其余13种网络各新增4个模型种子；共52次从零训练。原CVS不新增种子。训练后逐row自动clean预测与独立truth-last评分，原有历史数据保留，完成后合并8种子统计。

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
