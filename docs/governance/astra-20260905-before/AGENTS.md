# Project Instructions

- Before CVS-RFFI/CV-SincNet research-scenario, data-protocol, Stage2-A/B/C interpretation, experiment or report work, read `项目.md` in UTF-8 after this file. `项目.md` is the source of truth only for scientific scenario, data protocol, receiver/TX sets, Phase1/Phase2 boundary, Stage2 permissions and claim semantics. Active performance targets, method routes, matrices and resource budgets belong in a separate goal document; workflow, automation, N607, Git and safety rules remain in `AGENTS.md`.
- Phase2 uses `protocol_schema=p2_min_v1`. Every clean/raw physical IQ record may produce exactly one randomly selected allowed `leo_*_weak` observation before Phase2; K-shot means K independent physical samples; scenario sets and support/query physical IDs are disjoint. Post-reception equalization/FFT/mathematical views may only read the fixed received IQ and do not add K. Query and its views are test-only and cannot update state. Phase2 cannot access clean/source samples or unapproved derived state; the only exception is immutable int8 multi-sample aggregated Phase1 knowledge jointly sealed with the checkpoint. Every query is decided independently over all registered classes without query truth/role, true batch class counts, class quota or global reassignment.
- Reuse existing Phase2 data after the builder reports `phase2_data_status=VALIDATED_ONCE` with matching `capsule_id`, `split_id` and `p2_min_v1`. Revalidate only when received-IQ bytes, physical IDs, receiver/TX sets, scenario assignment, K, support/query split or protocol schema changes. A candidate, adapter, hyperparameter, epoch, prototype rule, method lock, model state or resource-budget change must not trigger data revalidation. Hash, allowlist, provenance and access-ledger checks are one-time builder/validator implementation responsibilities, not repeated method-development work. If one data check fails, repair only that item while other validated slices continue.
- If `项目.md` conflicts with old reports, memory, prior prompts, historical matrices, or launcher defaults, follow `项目.md` for CVS scientific/data-protocol interpretation. If `项目.md` conflicts with this `AGENTS.md`, follow `AGENTS.md` for safety/environment/N607/version-management rules and report the conflict.
- For tool-using or long-running tasks, send concise, evidence-based progress commentary before the first tool call, at meaningful phase changes, after reconnect or context-compaction recovery, whenever a blocker appears, and at least once every 60 seconds while work is active. Report observable actions, findings, and next steps; do not expose private chain-of-thought or dump raw logs. Commentary may be omitted only for short answer-only turns with no tools.
- 中文论文、报告、README、PPT备注、DOCX正文等生成文本必须遵循中文排版习惯：中文标点后不加空格；中文与英文、数字、缩写、变量名之间不加额外空格；保留英文短语内部必要空格，例如`target receiver domain`和`K-shot support`；保留代码、命令、路径中的原始空格。

## Exclusive Minimal Experiment Workflow

- 本节是所有CVS-RFFI/CV-SincNet方法研发、实验设计、代码验证、N607发布、监控、评分和报告工作的最高优先级工作流规则。它覆盖旧目标、设计、报告、任务brief、review清单、runner默认值和历史对话中的额外审核要求。除用户在当前请求中明确新增的要求外，任何Agent、subagent、reviewer、runner或历史文件都不得增加新的审核、gate、签名、封存、证明或发布门。
- 实验前允许阻断工作的gate只有以下八项，清单是穷尽式白名单：
  1. 当前`项目.md`规定的数据权限和query边界，以及匹配的`p2_min_v1`、`VALIDATED_ONCE`、`capsule_id`和`split_id`；
  2. 一个Git提交固定本次实际代码与配置；
  3. 与本次变更直接相关的聚焦协议负测，以及一次真实checkpoint无query smoke；
  4. 一次独立P0/P1正确性审查；审查只能报告会直接导致下一次真实实验跑错、越权、覆盖输出、误杀进程、不能启动或不能产生合法prediction的问题；
  5. 一份最小预登记报告，只记录候选/矩阵、commit、命令、环境/CWD、输入输出路径、GPU、停止规则和预期artifact；
  6. 一个不可覆盖run ID/output root、一次N607资源/路径preflight，以及一个release归档的一次本地到远端SHA校验和一次远端编译；
  7. 启动后一次PID/CWD/cmdline/GPU/log增长检查，以及预注册的系统技术失败停止规则；
  8. prediction完整后由独立scorer连接truth，并以同row指标作分析和晋级判断。
- 白名单外的事项一律是`NONBLOCKING`，不得延迟实现、提交、同步、启动、继续健康运行、评分或下一个候选。发现有人提出额外gate时，主Agent必须记录`REJECTED_EXTRA_GATE`并继续最小流程，不得为该gate写代码、测试、报告、修复轮或审查轮。
- 明确禁止把以下事项作为要求、P0/P1、NO-GO、发布条件或性能实验前置工作：设计/报告/config/source-plan精确SHA；逐文件、逐成员、逐row、逐cell、逐support-token或逐receipt哈希；outer joint seal、detached seal、package seal、cell seal、signature、验签、authority、authority chain、signature envelope；receipt-of-receipt、closure hash、parity receipt、访问账本、额外manifest链；smoke授权token/receipt；same-process恶意代码防御；TOCTOU、same-FD、same-inode、环境/PATH/Bash伪造攻击审计；report-only审查；seal许可；发布许可；P2文档完整性；测试helper生产权限系统；未来阶段能力；未消费的通用发布平台。
- Git提交已经固定代码和配置，不得再要求代码文件、config、launcher、设计文档或报告SHA。N607传输只允许对一个release归档做一次本地/远端SHA比较；不得再计算成员SHA、Git blob SHA、解压后逐文件SHA或回收报告SHA。若不用归档而只同步一个文件，只校验该文件一次。
- `VALIDATED_ONCE`数据不得因候选、adapter、超参数、epoch、method lock、checkpoint、资源预算、报告、代码重构或实验阶段变化而重验。hash、allowlist、provenance、访问账本和pre-open只属于一次性builder/validator职责；实验方法和runner只核对`protocol_schema/capsule_id/split_id/phase2_data_status`。
- 独立审查每个候选最多一次。若发现直接P0/P1，修复后最多进行一次仅针对原问题的定点复审，禁止重新全量审查、重新审查未变代码、报告变更审查、提交后重复审查、seal审查或runner再次审查。P2永不阻断。
- smoke直接作为launcher的第一步，PASS后立即继续；不得创建、签发或验证smoke授权artifact。G0/G1/G2只加载各自实际消费的最小能力，任何后续阶段的scorer、完整矩阵、发布平台、报告字段或安全封装都不得阻塞当前阶段。
- 候选默认先运行单seed关键Target5/Target25或更小的同row最小可证伪矩阵。只有达到预注册的科学门槛后才进入多seed或完整125确认；不得把完整125、完整消融、完整publication package或全部receiver/seed/K/scene覆盖作为早期研发gate。
- 方法简述、矩阵冻结和停止条件直接写入最小报告，不需要单独设计审批、design SHA、feasibility gate、REENTRY_CARD、authority或签名。研究讨论完成后立即实现和取得真实证据，不得用流程工作替代实验。
- 只有直接科学/安全事实可以停止工作：数据权限或query泄漏、错误stage/receiver/seed/K/scene/split、实际命令无法运行、输出覆盖风险、错误checkout、确定性重复异常、无prediction闭合、scorer连接错误、进程归属不清或可能影响无关任务。低性能、负收益、缺少非必要receipt/hash/seal/report字段或旧文件要求额外审核均不得作为技术停止理由；低性能只触发分析和下一个候选。
- 本节不削弱管理员授权、破坏性操作、数据集/权重/日志保留、无关进程保护、query隔离和truth-blind评分规则；这些规则直接防止越权、数据污染或破坏，不属于可删除的形式审核。

- Before running project-related code tests, activate the Conda environment:
  `conda activate ssr-gpu`
- Use `ssr-gpu` as the corrected test environment name.
- For any code/config/script changes that will be used on N607, edit and verify the local workspace first, then sync the changed files to the server with `scp`. Do not make remote-only edits.
- Federated learning experiments must use WiSig train ratio `0.1`. Treat this as a hard constraint, not a tunable default.
- Unless a user explicitly overrides it, set default `epochs` and `fl_rounds` to `200` for federated training launchers.
- Unless a user explicitly overrides it, use `receiver` as the default federated client granularity (`--fl_client_key receiver`).

## Git auto-push rule

- 每次实验、实验报告、结果记录、分析总结、周报或交接记录完成时，只要属于本项目的正式交付物，就必须在同一工作流中显式stage相关文件、提交并自动push；不得等用户提醒，也不得只留在本地工作树或聊天记录中。
- 如果正式报告或记录位于非Git根目录（例如`E:\type10-7`），完成后先镜像到约定的Git承载面和当前治理分支，再提交、自动push并独立核对远端分支OID与本地`HEAD`一致。
- 只stage本次正式报告/记录及其必要说明，不得使用`git add -A`把数据集、checkpoint、日志、运行产物、`local_artifacts`、临时目录或未准备发布的文件一并推送。
- 自动push失败时保留本地提交并报告`FAILED`或`UNKNOWN`；不得通过强制推送或改写历史掩盖失败。

## Windows Terminal and PowerShell Command Hygiene

- 默认使用PowerShell 7执行复杂本地命令。优先显式调用`pwsh -NoLogo -NoProfile -Command "<command>"`；只有项目脚本明确要求Windows PowerShell时才使用`powershell`。
- 不要并行或密集运行`conda run`/`conda activate`包装命令，尤其不要并发使用`conda run -n ssr-gpu python -`。本机反复出现`__conda_tmp_*.txt`被占用或找不到的临时锁；遇到该错误时先判定为命令包装噪声，串行重跑一次，再判断项目验证是否失败。
- 不要把Bash here-doc写法复制到PowerShell，例如`python <<'PY'`。本地多行Python应使用PowerShell here-string管道、临时脚本或`python -c`；涉及中文路径时先解析`FullName`并使用UTF-8安全读写。
- 远端多行脚本优先通过UTF-8字节送入`ssh ... python3 -`，或用LF-only内容送入`ssh ... bash -s`。避免在本地PowerShell中嵌套远端`$(...)`、here-doc、复杂引号或会被本地提前展开的变量；远端计数、JSON解析和日志聚合优先拉取小文件后在本地解析。
- 写Markdown、JSON、中文报告或状态文件时，不要依赖PowerShell默认文本编码、`Out-File`隐式编码或大JSON的`ConvertFrom-Json`管道。优先使用显式UTF-8写入、BOM-aware读取和Python JSON I/O；N607返回的`.out`、报告或状态片段可能是UTF-16LE，解析marker前必须先检测编码。
- 不要假设所有PowerShell版本都支持`Tee-Object -Encoding`、`Test-Connection -TimeoutSeconds`、`[IO.Path]::GetRelativePath`或已加载`ZipArchiveMode`。不确定时先捕获输出再`Set-Content -Encoding UTF8`，用`.NET`的`Ping.Send(...,2000)`做短超时ping，手写相对路径，并显式加载`System.IO.Compression`与`System.IO.Compression.FileSystem`。
- 避免写会产生空管道段的PowerShell one-liner，例如`foreach { ... } | Format-*`。先把结果收集到变量，再把变量送入后续管道或格式化命令。
- SSH/SCP超时、引号错误或输出乱码后，不要把本地命令失败直接当作远端实验失败或成功。先检查并清理本地`ssh.exe`与TCP22残留，再用只读远端进程、日志、run/log根目录和启动artifact验证是否已经landed，确认后再决定是否重试或重启。
- 不要把`E:\type10-7`根目录当作Git仓库。根目录Markdown或控制面规则改动必须先报告根目录非Git状态，并按版本管理规则镜像到约定Git承载面、创建交接记录或明确说明仍未版本化。

## N607 SSH Automation

- Use the plain `N607` direct SSH target first for N607 access by default. If direct `N607` access fails, use the verified lab-computer bridge as the fallback route instead of trying ad-hoc SSH routes.
- Before any task that needs SSH or SCP access to N607, start with the local read-only direct preflight:
  `powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`
- The preflight must verify, without changing server state: direct `N607` SSH config and identity, server time, project-root visibility, and GPU visibility. If direct preflight fails because the direct TCP/SSH path is unavailable while local config and identities are otherwise valid, fall back to the lab bridge below. If identity, key, or target ambiguity is the problem, stop and report diagnostics instead of trying interactive passwords or ad-hoc SSH routes.
- Use short-lived SSH commands such as `ssh -o BatchMode=yes N607 "<command>"`. Do not keep persistent SSH shells open after the needed check or operation is complete.
- Every SSH/SCP connection must actively disconnect as soon as its bounded task finishes. Do not leave interactive shells, background SSH clients, `tmux` relay sessions, `ControlMaster`/multiplex master connections, port forwards, or long-running monitor sessions open. A normally completed bounded SSH/SCP command needs no separate process/TCP audit.
- Check local `ssh.exe` and TCP22 state only after timeout, interruption, malformed quoting, missing completion evidence or another concrete orphan risk. Close only the exact stale client; if ownership is unclear, stop instead of touching unrelated connections.
- To avoid SSH reliability loss, do not keep long-lived SSH sessions, idle interactive shells, port forwards, or multiplex master connections open to N607. Monitoring must use short, discrete SSH commands and then disconnect.
- For file syncs to N607, use direct SCP such as `scp <local> N607:<remote>`, after local verification and any required snapshot/report updates.
- The verified lab bridge is `administrator@172.31.105.18`, using local key `C:/Users/lh594/.ssh/id_ed25519_lab_bridge_172_31_105_18`. Use it only as a fallback after the direct attempt fails. Keep the N607 private key local; do not copy N607 keys, datasets, checkpoints, or server credentials onto the lab computer.
- Preferred bridge command pattern:
  `ssh -i C:/Users/lh594/.ssh/id_ed25519_n607 -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -o "ProxyCommand=ssh -i C:/Users/lh594/.ssh/id_ed25519_lab_bridge_172_31_105_18 -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -W %h:%p administrator@172.31.105.18" szu2070436088@172.31.111.215 "<command>"`
- When a bridged SSH/SCP task ends, close the lab-computer-to-N607 SSH leg first, then close the local-host-to-lab-computer SSH leg. With the `ProxyCommand -W` pattern, this means letting the bounded N607 remote command/channel exit first and then letting the proxy connection to `172.31.105.18` exit. After timeouts or interruptions, check and clear local `ssh.exe` processes and `ESTABLISHED` connections to both `172.31.111.215:22` and `172.31.105.18:22` before continuing.
- Treat any other non-direct relay host as outside the default N607 workflow. Do not route through any unverified relay, run experiments on it, copy datasets/checkpoints to it, or store server credentials on it unless the user explicitly asks.

### N607 Administrator Account Authorization Boundary

- The N607 administrator account `szu2310433034`, SSH alias `N607-admin`, and its dedicated private key are denied by default. Use the ordinary `N607` account for all work unless the user explicitly authorizes administrator-account use for a concrete, bounded task in the current request.
- Administrator authorization is task-scoped and expires when that bounded task ends. Do not carry it into later turns, follow-up tasks, automation, monitoring, subagents, retries, or adjacent maintenance merely because the account or key is available or was authorized previously.
- Authorization to log in with the administrator account does not authorize state changes. Configuration edits, file writes, deletion, movement, ownership or permission changes, package installation/removal, service changes, process termination, mount/storage operations, firewall/network changes, reboot, shutdown, power-cycle, firmware/BMC/iDRAC actions, and any other risky or persistent mutation each require separate explicit user authorization that identifies the intended action and target.
- When administrator access or a proposed administrator action is ambiguous, stop and ask the user. Do not infer authorization from troubleshooting context, urgency, a prior password entry, availability of `sudo`, or a general request to “fix” or “check” the server.
- Even after explicit authorization, keep administrator commands minimally scoped, inspect targets and active workloads first, prefer read-only checks and dry-runs, preserve backups and recovery paths, record exact commands and results, and never expose or copy administrator credentials or private keys.

## Server Maintenance and Safety

- Remote commands must be minimally scoped and read-only by default. Prefer safe checks such as `hostname`, `date`, `pwd`, targeted `test -f` / `test -d`, `nvidia-smi`, targeted `ps` / `pgrep`, `tail -n`, `sed -n`, and bounded `find <specific-dir> -maxdepth ...`.
- Avoid high-impact commands unless the user explicitly requests them and the target has been verified. This includes `rm -rf`, broad `mv` / `cp` / `chmod` / `chown`, `kill` / `pkill` / `killall`, `reboot`, `shutdown`, package installs, service changes, recursive scans of `/` or dataset roots, cleanup of logs/checkpoints/metrics, and anything that can interrupt jobs or consume major CPU/GPU/disk/network resources.
- Before any state-changing remote action, inspect active processes/GPU/disk context, confirm exact target paths, prefer dry-runs when available, record the exact command, and update the local experiment report when the action is experiment-related.
- If remote jobs are active and the user has not explicitly asked to intervene, switch to monitor-only behavior. Do not launch, kill, restart, patch remote files, or clean outputs merely because remote access is available. The only standing exception is a pre-registered systemic-technical-failure stop executed by the sole runner for the exact run ID it owns: the runner must first verify the run-root/CWD/cmdline/PID-parent-child binding, stop only that run's dispatch and processes, preserve all partial artifacts, and record the evidence. This exception never authorizes intervention in unrelated workloads.
- Never stop a formal experiment because interim accuracy or another performance metric looks poor. Health stops are restricted to protocol/safety violations, execution faults, missing prediction closure, deterministic exception fingerprints, or non-progress evidence defined before launch.
- For N607 experiment packing, the user allows up to two concurrent training experiments per GPU by default. If one training process is already active on each GPU, it is acceptable to launch one additional experiment per GPU after preflight, occupancy recording, and report update; do not exceed two per GPU unless the user explicitly overrides it.
- Preserve datasets, checkpoints, logs, metrics, reports, and run outputs. Do not delete or overwrite them unless the user explicitly requests it and the scope is unambiguous.

## Experiment Execution and Reporting

- One run ID has exactly one launch owner. The owner may be the primary Agent or one designated runner; creating a subagent, supervisor, second reviewer, release approver or evidence auditor is optional and must never delay the run. No other Agent may duplicate-launch or change the frozen method/matrix.
- Use immutable, non-overwriting run IDs and the minimal states `LOCAL_VERIFIED -> LANDED -> RUNNING -> ARTIFACTS_COMPLETE -> ANALYZED`. A technical stop is `STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`.
- Every recurring experiment monitor must treat a verified training interruption caused by a system technical error as the start of a repair-and-republish workflow, not as terminal completion. After precisely binding and stopping only the failed run tree and preserving all partial artifacts, the monitor must record the failure fingerprint and root cause, reproduce it locally, patch only in a Git-backed local worktree, add a regression test that fails before the fix, run the focused verification and one scoped P0/P1 review, commit and push with remote-OID readback, build a new immutable release, allocate a fresh incremented run ID/output root, run the required smoke and N607 preflight, relaunch the frozen experiment, verify PID/CWD/cmdline/GPU/log binding, and update the monitor to follow the replacement run. Keep the original matrix, protocol, seed set, training budget and scientific selection rules unchanged unless the technical fix directly requires a recorded compatibility adjustment.
- Never hot-patch or restart a failed run in place, reuse its output root, edit only on N607, or erase its partial artifacts. Poor performance, an uncertain SSH timeout, a possibly landed submit, and a completed negative result are not repair-and-republish triggers; reconcile them read-only first. If the same deterministic fingerprint recurs after one locally reproduced and verified repair, or the root cause cannot be reproduced safely, preserve the new failed run, stop blind relaunch loops and notify the user with the evidence. This standing rule authorizes repair and fresh republication only for the failed experiment lineage; it never authorizes administrator use, unrelated-process intervention, destructive cleanup or protocol/query-boundary expansion.
- Every Phase1 experiment that completes training must evaluate the selected final checkpoint on the declared clean test split and on each `leo_weak` family scenario: `leo_clear_weak`, `leo_low_elev_weak`, and `leo_rain_weak`. Training completion alone is not experiment completion. The run may advance to `ARTIFACTS_COMPLETE` or `ANALYZED` only after the checkpoint identity, evaluation configuration, clean metrics, per-LEO-scenario metrics, and corresponding logs are preserved; do not replace per-scenario results with only an aggregate mean. If any required evaluation cannot run or fails technically, preserve the training outputs, record the exact missing/failed evaluation, and do not mark the experiment complete.
- Before N607 launch, execute only the eight-item whitelist in `Exclusive Minimal Experiment Workflow`. Passing the whitelist requires immediate release; no additional checklist, approval, seal, hash, receipt, report field, future-stage feature or subagent handoff is allowed.
- The minimal runner handoff, when a separate runner is used, contains only: run ID, Git commit, candidate/matrix, exact command, environment/CWD, input/output/log paths, GPU, expected artifacts, direct technical stop rule and whether a fresh new run is allowed. Do not include file hashes beyond the single release-archive transfer SHA; do not require signature, authority, receipt chains or review envelopes.
- Immediately after detached launch, verify the main PID, exact CWD/cmdline/run-root binding, GPU mapping and log growth once. For a multi-row run, later short checks need only counts, active workers, GPU state, log progress and deterministic exception fingerprints. A one-shot G0 gets no multi-row telemetry machinery.
- Stop dispatching and terminate only the exact run-owned process tree for a protocol/safety violation, output collision, wrong checkout, missing prediction closure, launcher-wide fault or the same deterministic pre-prediction exception in at least two rows. Never stop for poor interim metrics. Resolve exact PIDs/CWD/cmdlines before termination, preserve partial artifacts and never use broad `pkill`.
- Every N607 experiment uses one report at `E:\type10-7\automation_reports\CV-SincNet\<run-id>\report.md`. Before launch it contains only the minimal fields in whitelist item5. After completion, append final status, same-row results, anomalies, interpretation and next-candidate decision. Missing narrative, tables, hashes, receipts, audits or publication fields never blocks launch or scoring.
- Finished-result interpretation must keep each candidate/run row together with receiver/TX split, K-shot, seed, old/seen-new/unknown metrics, floor/forgetting and verdict where those metrics exist. Do not combine unrelated single-metric maxima into a fictional best run.
- Collaboration and model routing are execution choices, not gates. Use extra Agents only when they reduce wall time without adding approvals. The primary Agent owns protocol interpretation, scientific integration, performance analysis and final promotion; a runner owns only the exact frozen launch and evidence retrieval.

### Four-state DA and registration metric naming

- Every future experiment/report that jointly studies domain adaptation and new-class registration must use the four explicit states `DA0_REG0`（域适应前/新类注册前）, `DA1_REG0`（域适应后/新类注册前）, `DA0_REG1`（域适应前/新类注册后）, and `DA1_REG1`（域适应后/新类注册后）. Do not use ambiguous standalone `before/after`, `B/C`, or “适应后” labels without both DA and registration state.
- Report the DA effect before registration as `DA1_REG0 - DA0_REG0`, the DA effect after registration as `DA1_REG1 - DA0_REG1`, the registration effect without DA as `DA0_REG1 - DA0_REG0`, and the registration effect with DA as `DA1_REG1 - DA1_REG0`. For metrics defined in all four states, report the joint interaction as the corresponding difference-in-differences.
- New-class accuracy and old/new harmonic metrics are defined only for `REG1`. In `REG0`, report them as `N/A` rather than zero and do not score an unregistered class as if it were registered or unknown-rejection capable. Old-class accuracy, old-class floor, resource state and latency may be reported in all four states.
- This naming requirement must reuse the same frozen inputs, caches and predictions where possible; it is a reporting and causal-comparison requirement, not authorization to enlarge the frozen experiment matrix or add release gates.

## Conversation History Lookup

- New conversations cannot automatically browse full prior chat history. For project-related history, use the local index tool before relying on memory alone.
- Build or refresh the project-scoped index with:
  `conda activate ssr-gpu; python tools/conversation_index.py build`
- Search only `E:\type10-7` related historical conversations with:
  `conda activate ssr-gpu; python tools/conversation_index.py search "<keywords>"`
- The generated index is stored under `E:\type10-7\conversation_index\` and includes source paths back to the Codex rollout summary or session JSONL when available.
- Do not treat the index as a replacement for experiment reports. For N607 experiment design, launch, monitor, or completion analysis, still create or update the required report under `E:\type10-7\automation_reports\CV-SincNet\...`.

## Exploration Retrospective Cadence

- Retrospectives are short nonblocking research notes, not a launch gate. Perform one when it helps choose the next mechanism, without stopping an already ready candidate or requiring a fixed three-round cadence, conversation-index rebuild, separate review or approval.
- A retrospective may summarize lessons, rejected routes and remaining hypotheses in the active report. It must not introduce new hashes, seals, receipts, matrices, reviewers, reentry cards or release conditions.

## Version Management

- Every project-related change must enter a Git-backed workflow. Before editing code, config, scripts, prompts, matrices, reports, or project Markdown, locate the relevant Git repository and run `git status -sb` or report that the target tree is not a Git repository. After editing, inspect `git diff` / `git status -sb`, run the narrowest useful verification, and record the changed files plus verification result in the relevant report or Markdown handoff.
- Do not leave intended project changes only in chat or in an untracked working directory. If the edited target is inside a Git repository, stage and commit the intended change with a concise message unless the user explicitly says not to commit, then immediately push the current branch and independently verify the remote branch resolves to the new `HEAD`. If the branch has no upstream, use `git push --set-upstream origin HEAD`. If the edited target is not inside a Git repository, stop before treating the change as versioned and either initialize/choose a Git repository with user-visible scope or mirror the change into the agreed Git-backed release workspace/branch.
- The project release repository installs the `.git/hooks/post-commit` hook by copying `scripts/auto_push_after_commit.sh` through `scripts/install_auto_push_hook.sh`; automatic push is the default for every commit in this repository and its worktrees. After each commit, verify `git status -sb` has no ahead/behind count and independently compare the remote branch OID with `HEAD`. A push failure leaves the commit intact and must be reported as `FAILED` or `UNKNOWN`; never force-push or stage unrelated dirty/untracked files.
- For GitHub-facing publication or repository-structure changes, use a branch/PR flow by default. Do not force-push, rewrite shared history, or overwrite unrelated remote content unless the user explicitly asks and the exact scope is confirmed.
- Update `AGENTS.md` only for workflow/safety changes and update `项目.md` before a scientific/data-protocol/Stage2/claim-semantic change. Ordinary code/config/script changes do not require README/docs/report synchronization before experiment; user-facing documentation may be updated after evidence returns and never becomes a release gate.
- Keep all code/config/script edits local first, then sync to N607 with `scp` only after local verification.
- Before launching or changing a server experiment, record the local version state in the report:
  - changed files and purpose of each change;
  - relevant command outputs from local checks;
  - Git commit and `git status`/diff summary;
  - remote destination paths used for sync.
- If the target directory is not a Git repository, use an existing Git-backed project worktree or mirror before syncing. Do not create an additional timestamped snapshot, hash inventory or approval gate when the same content is already fixed by the required Git commit.
- Record the release archive local-to-remote mapping in the run report. Do not create an additional sync manifest, receipt or hash inventory unless the user explicitly asks for one.
- Never overwrite unrelated local or remote changes. If a file has unowned edits, inspect and preserve them; ask before destructive operations.
- Do not delete datasets, checkpoints, logs, metrics, reports, or run outputs as part of version cleanup unless the user explicitly requests it.
