# Project Asset Governance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不移动、不覆盖、不删除任何本地或N607原始资产的前提下，为`E:\type10-7`和`/home/szu2070436088/2510044040/CV-SincNet`生成可复现的资产总表、活跃实验索引、Git归属、保留级别和待审批删除清单。

**Architecture:** 在`tools/project_governance`中实现纯只读采集器、规范化器、实验索引器、Git归属映射器和保留分类器；只有输出器可在全新的`scan_id`目录写治理产物。N607采集通过普通账号、短连接和stdin流式Python脚本返回NDJSON，不在服务器落盘。CLI只负责组合组件，不包含移动、覆盖、删除、实验启停或Git自动提交能力。

**Tech Stack:** Python标准库、pytest、Git只读命令、Windows Git Bash、OpenSSH、N607的Python3、现有`tools/n607_ssh_preflight.ps1`。

## Global Constraints

- 所有终端命令的外层必须是`C:\Program Files\Git\bin\bash.exe`且`login=false`；首次调用验证`MSYSTEM=MINGW64`。禁止执行`pwsh`或`pwsh.exe`，包括子进程。
- 项目测试必须串行使用`ssr-gpu`环境。每次验证先运行`conda run -n ssr-gpu python -c "import os,sys; print(sys.executable); print(os.environ.get('CONDA_DEFAULT_ENV'))"`并确认环境名为`ssr-gpu`；不得并发调用Conda包装器。
- 仅现有N607预检脚本可按项目规则从Git Bash窄调用`powershell.exe`；不得新增PowerShell实现。调用前已完成`powershell-failure-catalog.md`审阅。
- 扫描根、数据、checkpoint、日志、报告、run输出、Git工作树和N607内容均为只读对象。采集器不得包含写入、移动、重命名、覆盖、删除、权限修改或进程控制接口。
- 输出器只能写入调用方显式提供且尚不存在的`scan_id`目录；任何已存在路径都必须失败关闭，不得覆盖。
- 删除候选只生成审批记录。所有候选固定为`AWAITING_USER_APPROVAL / NOT_AUTHORIZED`；本计划不实现删除执行器。
- N607只使用普通账号，默认直连`N607`，直连路径不可用时才按现有规则使用已验证实验室桥接。不得使用`N607-admin`。
- 每个SSH调用都必须有超时并主动退出；超时、输出可读或退出码0本身均不是断连成功证据。调用后必须验证采集子进程结束，并确认到N607及桥接机TCP22连接为0。
- 发现活动远端任务后只记录绑定证据并继续保守只读采集；不得启动、停止、重启、修补或清理任何任务。
- 必须保留当前工作树中不属于本任务的所有`.docx_qa_cvs_ntn*/`目录、`tools/build_cvs_ntn_scenario_docx.py`（若存在）及实施期间新出现的非本任务文件，不得暂存、修改或删除。
- 本地根`E:\type10-7`不是Git仓库。实现、测试、配置和小型治理产物只进入独立工作树`E:\type10-7\code\snapshots\project_governance_20260813_wt`的分支`codex/project-governance-20260813`。
- 正式扫描前先提交实现代码。正式receipt必须记录实际实现提交；扫描产物在验证后形成单独提交。

---

## File Structure

- Create: `tools/project_governance/__init__.py`——公开稳定类型和版本号。
- Create: `tools/project_governance/models.py`——枚举、不可变记录和序列化。
- Create: `tools/project_governance/config.py`——配置解析、路径边界和配置校验。
- Create: `tools/project_governance/paths.py`——路径规范化、可逆显示和稳定身份。
- Create: `tools/project_governance/collect_local.py`——深度受控的本地只读元数据采集。
- Create: `tools/project_governance/collect_git.py`——候选Git仓库发现和归属映射。
- Create: `tools/project_governance/index_experiments.py`——报告、manifest、receipt、artifact和实时进程证据关联。
- Create: `tools/project_governance/classify_retention.py`——保留级别和待审批候选生成。
- Create: `tools/project_governance/collect_n607.py`——N607预检、只读NDJSON采集和断连验证。
- Create: `tools/project_governance/emit.py`——CSV、JSON、Markdown、receipt和体积分流。
- Create: `tools/project_governance/cli.py`——无副作用参数解析和组件编排。
- Create: `tools/project_governance_inventory.py`——唯一命令行入口。
- Create: `configs/project_governance_inventory_v1.json`——固定扫描根、承载面、深度和哈希策略。
- Create: `tests/test_project_governance_models.py`——枚举、稳定ID、路径与序列化测试。
- Create: `tests/test_project_governance_local.py`——本地采集、链接、错误和哈希策略测试。
- Create: `tests/test_project_governance_git.py`——tracked、untracked、ignored和非Git归属测试。
- Create: `tests/test_project_governance_experiments.py`——实验关联和状态机测试。
- Create: `tests/test_project_governance_retention.py`——保留优先级和审批不变量测试。
- Create: `tests/test_project_governance_n607.py`——远端payload、路由、超时和断连测试。
- Create: `tests/test_project_governance_emit_cli.py`——输出、阈值、CLI和零变更集成测试。
- Modify: `docs/project_governance/worklog/task_plan.md`——记录实施任务和门禁。
- Modify: `docs/project_governance/worklog/findings.md`——记录发现、错误和分类依据。
- Modify: `docs/project_governance/worklog/progress.md`——记录验证、正式扫描和提交。

## Task 1: Freeze Configuration and Domain Contracts

**Files:**
- Create: `tools/project_governance/__init__.py`
- Create: `tools/project_governance/models.py`
- Create: `tools/project_governance/config.py`
- Create: `configs/project_governance_inventory_v1.json`
- Test: `tests/test_project_governance_models.py`

- [ ] **Step 1: Write failing enum and record tests**

Add tests that instantiate every fixed enum and assert stable JSON values. The core test must include:

```python
from tools.project_governance.models import (
    ApprovalState,
    AssetKind,
    ExecutionState,
    ExperimentState,
    GitOwnership,
    Location,
    RetentionClass,
)


def test_fixed_governance_vocabularies_are_stable():
    assert [item.value for item in Location] == ["LOCAL", "N607"]
    assert ExperimentState.ACTIVE_LIVE.value == "ACTIVE_LIVE"
    assert GitOwnership.NON_GIT_EVIDENCE.value == "NON_GIT_EVIDENCE"
    assert RetentionClass.DELETE_CANDIDATE.value == "DELETE_CANDIDATE"
    assert ApprovalState.AWAITING_USER_APPROVAL.value == "AWAITING_USER_APPROVAL"
    assert ExecutionState.NOT_AUTHORIZED.value == "NOT_AUTHORIZED"
    assert AssetKind.JUNCTION.value == "junction"
```

Run:

```bash
conda run -n ssr-gpu python -m pytest tests/test_project_governance_models.py::test_fixed_governance_vocabularies_are_stable -q
```

Expected first result: FAIL with `ModuleNotFoundError: No module named 'tools.project_governance'`.

- [ ] **Step 2: Implement fixed enums and immutable records**

Use `str, Enum` values exactly matching the approved design. Define frozen dataclasses for:

```python
@dataclass(frozen=True)
class AssetRecord:
    asset_id: str
    scan_id: str
    location: Location
    root_id: str
    relative_path: str
    display_name: str
    escaped_name: str
    asset_kind: AssetKind
    size_bytes: int | None
    mtime_utc: str | None
    access_status: AccessStatus
    hash_status: HashStatus
    sha256: str | None
    experiment_id: str | None = None
    git_ownership: GitOwnership | None = None
    evidence_role: str | None = None
    retention_class: RetentionClass | None = None
    recommended_action: str = "REVIEW"
    decision_reason: str = "UNCLASSIFIED"
```

Also define `ScopeResult`, `GitOwnershipRecord`, `ExperimentRecord`, `RetentionDecision`, `DeletionCandidate` and `ScanBundle`. Optional evidence fields must use`None`; zero is reserved for a measured zero count.

- [ ] **Step 3: Add validated versioned configuration**

The committed JSON must fix these roots and carrier surfaces:

```json
{
  "schema_version": 1,
  "local": {
    "root_id": "TYPE10_7",
    "root": "E:/type10-7",
    "carrier_surfaces": [
      "automation_reports/CV-SincNet",
      "code/snapshots",
      "local_artifacts",
      "remote_artifacts",
      "runs",
      "logs",
      "outputs",
      "server_log_backups",
      "runner_staging",
      "github_publish/CVS-RFFI-repo"
    ]
  },
  "n607": {
    "root_id": "N607_CVS_SINCNET",
    "root": "/home/szu2070436088/2510044040/CV-SincNet",
    "carrier_surfaces": [
      "automation_reports",
      "runs",
      "logs",
      "releases",
      "remote_artifacts",
      "snapshots",
      "code"
    ]
  },
  "discovery": {
    "control_evidence_max_depth": 3,
    "hash_max_bytes": 10485760,
    "text_read_max_bytes": 2097152
  },
  "output": {
    "git_file_max_bytes": 10485760,
    "git_scan_max_bytes": 52428800
  }
}
```

`load_config()` must reject unknown schema versions, absolute carrier entries, `..` components, non-positive limits and roots that do not match the requested location. Missing optional carrier surfaces are recorded as`NOT_PRESENT` rather than silently dropped.

- [ ] **Step 4: Verify contract tests**

Run:

```bash
conda run -n ssr-gpu python -m pytest tests/test_project_governance_models.py -q
```

Expected: PASS.

## Task 2: Normalize Paths and Collect Local Metadata Read-Only

**Files:**
- Create: `tools/project_governance/paths.py`
- Create: `tools/project_governance/collect_local.py`
- Modify: `tests/test_project_governance_models.py`
- Create: `tests/test_project_governance_local.py`

- [ ] **Step 1: Write failing identity and boundary tests**

Cover Windows separator normalization, Unicode/abnormal names, same-size and same-mtime identity independence, case-insensitive local identity, case-sensitive N607 identity, root escape rejection and reversible escaped display:

```python
def test_asset_id_uses_location_root_and_path_not_metadata():
    first = stable_asset_id(Location.LOCAL, "TYPE10_7", "runs/A")
    second = stable_asset_id(Location.LOCAL, "TYPE10_7", "runs\\A")
    assert first == second
    assert first == stable_asset_id(Location.LOCAL, "TYPE10_7", "RUNS/a")
    assert first != stable_asset_id(Location.N607, "TYPE10_7", "runs/A")
```

`normalize_relative_path()` must return forward-slash NFC text, reject absolute input and reject any normalized path escaping the configured root.

- [ ] **Step 2: Write failing fixture scan tests**

Build a temporary tree containing a normal file, zero-byte file, abnormal Unicode name, small manifest, large fake checkpoint, symlink, mocked junction and a directory whose`os.scandir` raises`PermissionError`. Snapshot names, sizes and mtimes before and after collection and assert exact equality.

Required assertions:

```python
assert zero.retention_class is None
assert zero.recommended_action == "REVIEW"
assert checkpoint.hash_status is HashStatus.METADATA_ONLY
assert manifest.hash_status is HashStatus.SHA256
assert denied.access_status is AccessStatus.SCAN_ERROR
assert not any(row.relative_path.startswith("link/") for row in records)
```

- [ ] **Step 3: Implement bounded collector**

`LocalCollector.collect()` must:

1. enumerate every direct child of the configured local root;
2. enumerate every direct child of each present carrier surface;
3. descend from each carrier unit at most three levels only to discover allowlisted control evidence and prediction/score directory summaries;
4. use`os.scandir` and`DirEntry.stat(follow_symlinks=False)`;
5. detect symbolic links and Windows reparse/junction points without following them;
6. deduplicate on`(location, root_id, normalized_relative_path)` while merging coverage tags;
7. emit a`SCAN_ERROR` record instead of skipping access failures.

Do not call`Path.rglob()`on project, dataset, run or checkpoint roots. Do not calculate directory recursive sizes.

Hash only small control evidence such as Markdown, JSON, YAML, TOML, Python, shell scripts, manifests and receipts. Protected payload suffixes including`.pt`,`.pth`,`.ckpt`,`.npy`,`.npz`,`.pkl`,`.h5`,`.mat`,`.tar`,`.zip`and`.7z`must remain`METADATA_ONLY`; eligible control files above10MiB use`NOT_HASHED_SIZE_LIMIT`.

- [ ] **Step 4: Prove collector has no destructive API**

Add an AST-based test over`tools/project_governance/collect_local.py`that fails on calls ending in`unlink`, `remove`, `rmdir`, `rmtree`, `rename`, `replace`, `chmod`, `chown`or`kill`. This is a regression guard, not the sole safety proof.

- [ ] **Step 5: Verify local collector tests**

Run:

```bash
conda run -n ssr-gpu python -m pytest tests/test_project_governance_models.py tests/test_project_governance_local.py -q
```

Expected: PASS with fixture contents unchanged.

## Task 3: Map Git Ownership Without Modifying Worktrees

**Files:**
- Create: `tools/project_governance/collect_git.py`
- Create: `tests/test_project_governance_git.py`

- [ ] **Step 1: Write failing temporary-repository tests**

Create a temporary repository with one committed file, one untracked file and one ignored cache. Assert:

```python
assert ownership["tracked.txt"].ownership is GitOwnership.TRACKED_GIT
assert ownership["draft.txt"].ownership is GitOwnership.UNTRACKED_IN_GIT_WORKTREE
assert ownership["cache.tmp"].ownership is GitOwnership.IGNORED_REGENERABLE
assert outside.ownership is GitOwnership.NON_GIT_EVIDENCE
```

Capture`git status --porcelain=v2 -z`before and after mapping and assert byte-for-byte equality.

- [ ] **Step 2: Implement repository discovery and batched ownership checks**

Repository candidates come only from configured seeds and already indexed directories containing a`.git`file or directory. Expand linked worktrees with`git worktree list --porcelain`; do not recursively search the entire project for`.git`.

For every repository record, collect:

- resolved worktree root;
- common Git directory;
- branch or detached state;
- full HEAD commit;
- porcelain-v2 dirty summary;
- linked-worktree list;
- command error evidence.

For indexed assets under a repository, classify exact path batches with read-only`git ls-files --stage -z -- <paths>`and`git check-ignore -z --stdin`. Assets outside any known worktree become`NON_GIT_EVIDENCE`; N607 assets become`REMOTE_NON_GIT`; command failure becomes`GIT_STATE_ERROR`and may never downgrade retention.

- [ ] **Step 3: Prevent broad Git enumeration**

Add a fake command runner test asserting no command contains`git add`, `git commit`, `git clean`, `git reset`, `git checkout`, `git restore`, `git gc`or an unbounded`git ls-files --others`call. Every path query must be scoped to indexed paths.

- [ ] **Step 4: Verify Git mapper tests**

Run:

```bash
conda run -n ssr-gpu python -m pytest tests/test_project_governance_git.py -q
```

Expected: PASS and unchanged temporary repository status.

## Task 4: Build Evidence-Bound Experiment Index

**Files:**
- Create: `tools/project_governance/index_experiments.py`
- Create: `tests/test_project_governance_experiments.py`

- [ ] **Step 1: Write failing association tests**

Fixtures must cover:

- a report with explicit`run_id`, commit and expected artifacts;
- a matching run directory with predictions and scores;
- a live process whose CWD and cmdline bind the same run root;
- a similar name with the same mtime but no explicit binding;
- an orphan artifact;
- a report/manifest conflict;
- an unreadable evidence file.

Assert that similar names and timestamps do not merge automatically:

```python
assert index["RUN_A"].experiment_state is ExperimentState.ACTIVE_LIVE
assert orphan.experiment_id.startswith("ORPHAN:")
assert same_mtime_candidate.experiment_id != index["RUN_A"].experiment_id
assert conflict.experiment_state is ExperimentState.ORPHAN_REVIEW
assert "CONFLICTING_RUN_ID" in conflict.closure_gaps
```

- [ ] **Step 2: Implement bounded evidence parsing**

Read only indexed allowlisted report, JSON manifest, receipt, metrics summary and small text files up to2MiB. Do not import or deserialize pickle, NumPy, PyTorch or checkpoint content. Extract claims as`EvidenceClaim(source_asset_id, field, value, confidence, parse_status)`and preserve conflicting claims.

Association precedence is fixed:

1. exact normalized`run_id`;
2. explicit absolute or root-relative artifact path;
3. exact Git commit plus manifest/receipt binding;
4. low-confidence name similarity for review only.

Low-confidence candidates must remain separate and be recorded in`closure_gaps`; they cannot produce an automatic merge.

- [ ] **Step 3: Implement conservative state machine**

Apply states in this order:

```text
SCAN_ERROR          unreadable required evidence prevents classification
ACTIVE_LIVE         process PID + CWD/cmdline + exact run-root binding
COMPLETE_EVIDENCE   explicit terminal report + all declared expected artifacts observed
HISTORICAL_ARCHIVE  COMPLETE_EVIDENCE plus explicit archive marker and no active reference
OPEN_INCOMPLETE     recognized report/run with missing closure evidence and no live binding
ORPHAN_REVIEW       artifact exists without sufficient report/version/run binding, or evidence conflicts
```

`ACTIVE_LIVE`must never be inferred frommtime、directory name、PID file alone or GPU utilization alone. Performance metrics are copied only as opaque evidence references; the governance index does not rank or promote methods.

- [ ] **Step 4: Verify experiment index tests**

Run:

```bash
conda run -n ssr-gpu python -m pytest tests/test_project_governance_experiments.py -q
```

Expected: PASS.

## Task 5: Classify Retention and Generate Approval-Only Candidates

**Files:**
- Create: `tools/project_governance/classify_retention.py`
- Create: `tests/test_project_governance_retention.py`

- [ ] **Step 1: Write failing precedence and safety tests**

Parameterized cases must prove:

- dataset、checkpoint、正式报告、日志、metrics、prediction、score、receipt、manifest和run输出默认`KEEP_IMMUTABLE`；
- `ACTIVE_LIVE`和`OPEN_INCOMPLETE`关联资产为`KEEP_ACTIVE`；
- 当前论文、发布分支或复核引用为`KEEP_UNTIL_PUBLISHED`；
- 只有显式终态和archive标记才允许`HISTORICAL_ARCHIVE`；
- 已记录生成器、源依赖和重建命令的缓存才允许`REGENERABLE_CACHE`；
- 零字节、旧mtime、异常名称、untracked或non-Git单独出现时只能`REVIEW_REQUIRED`；
- 任一读取、Git或证据冲突都提升到`REVIEW_REQUIRED`或更高保留级别。

- [ ] **Step 2: Implement ordered, explainable classifier**

Each decision returns a rule code and evidence asset IDs. Use this priority:

```python
RETENTION_PRIORITY = (
    "ERROR_OR_CONFLICT",
    "PROTECTED_EVIDENCE",
    "ACTIVE_OR_OPEN_EXPERIMENT",
    "CURRENT_PUBLICATION_DEPENDENCY",
    "VERIFIED_HISTORICAL_ARCHIVE",
    "PROVEN_REGENERABLE_CACHE",
    "FULLY_PROVEN_DELETE_CANDIDATE",
    "INSUFFICIENT_EVIDENCE",
)
```

A`DELETE_CANDIDATE`requires all of:

- not a protected evidence type;
- no active process, experiment, Git worktree, report, manifest, receipt or current-document dependency;
- provenance and purpose known;
- either deterministic regeneration evidence or a byte-identical retained canonical copy withSHA256;
- recoverability and estimated reclaim recorded;
- no scan, parse or Git error.

Any missing predicate yields`REVIEW_REQUIRED`.

- [ ] **Step 3: Lock approval and execution states**

`build_deletion_candidates()`must be the only constructor for deletion rows and must hard-code:

```python
approval_state=ApprovalState.AWAITING_USER_APPROVAL
execution_state=ExecutionState.NOT_AUTHORIZED
approved_scope=None
```

The package must not expose`delete`, `cleanup`, `execute_candidate`or equivalent mutation functions. Add an AST scan across the entire package for destructive filesystem and process calls; allow only fresh output creation inside`emit.py`.

- [ ] **Step 4: Verify retention tests**

Run:

```bash
conda run -n ssr-gpu python -m pytest tests/test_project_governance_retention.py -q
```

Expected: PASS; every deletion row is awaiting approval and not authorized.

## Task 6: Implement N607 Read-Only Streaming Collector

**Files:**
- Create: `tools/project_governance/collect_n607.py`
- Create: `tests/test_project_governance_n607.py`

- [ ] **Step 1: Write failing remote-payload safety tests**

Generate the remote Python payload without opening a connection. Parse it with`ast.parse`and reject calls or attributes matching`open`in write/append/create mode,`write_text`, `write_bytes`, `unlink`, `remove`, `rmdir`, `rmtree`, `mkdir`, `makedirs`, `rename`, `replace`, `chmod`, `chown`, `kill`, `system`, `Popen`or`subprocess.run`.

Assert the payload only imports an allowlist of read-only standard-library modules and emits one JSON object per stdout line.

- [ ] **Step 2: Implement remote metadata protocol**

The payload receives the frozen scan config in its source and performs only:

-`os.scandir`and`os.lstat`with no link following;
- boundedSHA256 for eligible small control files;
- bounded reads of allowlisted small evidence;
- project-root-filtered`/proc`CWD/cmdline/PPID inspection for live bindings;
- hostname and server timestamp collection;
- stdout NDJSON emission.

It must not run remote`find`, `du`, recursive hash, archive, package, service, process-control or shell commands. Every record carries`schema_version`, `scan_id`, `record_type`and normalized path fields. Errors become`SCAN_ERROR`records.

- [ ] **Step 3: Implement direct-first short connection and fallback gate**

Use an injected command runner in tests. Production sequence:

1. run the existing local read-only preflight;
2. if direct identity/config is valid, stream payload to`ssh -T -o BatchMode=yes -o ConnectTimeout=10 N607 python3 -`with a45-second timeout;
3. only when the preflight classifies direct TCP/SSH path unavailable and identities valid, use the exact approved bridge pattern;
4. never prompt for a password and never try an ad-hoc host;
5. parse stdout incrementally as UTF-8NDJSON and preserve stderr tail without treating it as success evidence.

The preflight child invocation from Git Bash is fixed to:

```bash
GOV_PREFLIGHT_WIN="$(cygpath -w "$PWD/tools/n607_ssh_preflight.ps1")"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "$GOV_PREFLIGHT_WIN"
```

No other PowerShell command is authorized by this plan.

- [ ] **Step 4: Implement disconnect verification without automatic termination**

After every attempt, require:

- the exact local SSH child and any proxy child have exited;
-`netstat.exe -ano`shows no`ESTABLISHED`or`SYN_SENT`connection to`172.31.111.215:22`or`172.31.105.18:22`owned by the attempt;
- the receipt records route, child PID, exit evidence and disconnect result.

If a connection remains, classify the outcome`UNKNOWN`, stop further SSH work and report the exact PID/endpoint. Do not call`taskkill`, `kill`or retry automatically.

- [ ] **Step 5: Test route and failure semantics with fakes**

Cover direct success, direct path unavailable with authorized bridge fallback, identity ambiguity, timeout, malformed NDJSON, nonzero exit, lingering connection and active remote training. Only direct success or valid fallback plus clean disconnect can be`VERIFIED`; all other outcomes are`FAILED`or`UNKNOWN`as applicable.

- [ ] **Step 6: Verify N607 unit tests without connecting**

Run:

```bash
conda run -n ssr-gpu python -m pytest tests/test_project_governance_n607.py -q
```

Expected: PASS and zero network calls because all runners are fakes.

## Task 7: Emit Deterministic Reports and Immutable Receipts

**Files:**
- Create: `tools/project_governance/emit.py`
- Create: `tests/test_project_governance_emit_cli.py`

- [ ] **Step 1: Write failing small-output tests**

Given a deterministic`ScanBundle`, assert creation of:

```text
report.md
asset_inventory_local.csv
asset_inventory_n607.csv
experiment_index.csv
git_ownership.csv
retention_decisions.csv
deletion_candidates.csv
asset_inventory_full.json
scan_receipt.json
```

Rows must be sorted stably, CSV usesUTF-8withBOM for Chinese spreadsheet compatibility, JSON usesUTF-8withoutBOM, timestamps areUTCISO8601 and the receipt is written last.

- [ ] **Step 2: Implement exclusive fresh-output writes**

`ReportEmitter`must resolve both output roots, require that`<root>/<scan_id>`does not exist, create it once and open every file in exclusive`x`mode. It must not remove partial output on failure; absence of`scan_receipt.json`marks an incomplete emission.

The receipt includes:

- schema and scan IDs;
- local and N607 roots/scopes;
- implementation Git HEAD and tracked-diff state;
- collector versions;
- per-fileSHA256 and bytes;
- record counts and all`SCAN_ERROR`counts;
- N607 route/preflight/disconnect evidence;
-`source_asset_mutations: 0`、`moves: 0`、`overwrites: 0`、`deletions: 0`；
- deletion rows awaiting approval and authorized row count0.

- [ ] **Step 3: Implement size threshold and external artifact routing**

Serialize before writing. When every governance file is at most10MiB and the total is at most50MiB, keep complete CSV/JSON in Git output. Otherwise:

- write complete tables to the fresh external directory`E:\type10-7\local_artifacts\project_governance\<scan_id>`；
- write deterministic CSV shards no larger than10MiB plus summaries under Git output;
- record every external absolute path, size andSHA256 in the Git receipt;
- preserve all error and deletion-candidate evidence;
- fail instead of silently truncating.

- [ ] **Step 4: Render an evidence-first report**

`report.md`must contain scope and freshness, inventory counts, live/open/complete/orphan/error experiments, Git ownership, retention distribution, exact deletion-approval table, coverage gaps, N607 connection outcome and an explicit statement that actual move/overwrite/delete count is0. It must not include performance promotion claims.

- [ ] **Step 5: Verify emitter tests**

Run:

```bash
conda run -n ssr-gpu python -m pytest tests/test_project_governance_emit_cli.py -k emitter -q
```

Expected: PASS; a second write to the same scan ID fails with`FileExistsError`and leaves the first output unchanged.

## Task 8: Add Safe CLI and End-to-End Fixture Scan

**Files:**
- Create: `tools/project_governance/cli.py`
- Create: `tools/project_governance_inventory.py`
- Modify: `tests/test_project_governance_emit_cli.py`

- [ ] **Step 1: Write failing parser and no-network default tests**

The CLI requires the`scan`subcommand, explicit`--scan-id`, `--output-root`, `--external-output-root`and`--operator`. N607 is disabled unless`--include-n607`is present. Assert that arguments named`--delete`, `--cleanup`, `--move`, `--overwrite`, `--kill`or`--admin`are rejected.

- [ ] **Step 2: Implement orchestration with dependency injection**

The entrypoint sequence is:

```text
validate config and immutable output paths
collect local records
map Git ownership
optionally preflight and collect N607
build experiment index
classify retention and deletion candidates
emit report and receipt
print one-line JSON outcome
```

Return codes are fixed:

-`0`: all requested scopes and disconnect checks`VERIFIED`；
-`2`: scan completed with recorded local scope errors or remote`FAILED`；
-`3`: remote outcome`UNKNOWN`or disconnect cannot be proven；
-`4`: configuration/output safety gate rejected before scanning。

The CLI never stages, commits or pushes Git changes.

- [ ] **Step 3: Add fixture-only end-to-end test**

Use a temporary local root, fake Git runner and fake N607 collector. Assert expected files, stable IDs, experiment joins, approval states and receipt hashes. Capture the fixture tree before and after; the only new paths may be under the separately supplied output roots.

- [ ] **Step 4: Add production command preview**

`--print-plan`must validate and print roots, scopes, depth, hash limits, expected output directories and whether N607 would be contacted, then exit without scanning or creating files. It must never print secrets or private-key contents.

- [ ] **Step 5: Verify CLI and full focused suite**

Run serially:

```bash
conda run -n ssr-gpu python -m pytest tests/test_project_governance_models.py tests/test_project_governance_local.py tests/test_project_governance_git.py tests/test_project_governance_experiments.py tests/test_project_governance_retention.py tests/test_project_governance_n607.py tests/test_project_governance_emit_cli.py -q
conda run -n ssr-gpu python tools/project_governance_inventory.py scan --config configs/project_governance_inventory_v1.json --scan-id PLAN_PREVIEW --output-root "$PWD/docs/project_governance" --external-output-root "E:/type10-7/local_artifacts/project_governance" --operator codex --print-plan
```

Expected: all focused tests PASS; preview creates no output and makes no network connection.

## Task 9: Review and Commit the Implementation Before Real Scanning

**Files:**
- Modify: `docs/project_governance/worklog/task_plan.md`
- Modify: `docs/project_governance/worklog/findings.md`
- Modify: `docs/project_governance/worklog/progress.md`
- Review: all files created inTasks1–8

- [ ] **Step 1: Run full static safety review**

Run targeted searches:

```bash
rg -n "\b(pwsh|rm|del|rmdir|unlink|remove|rmtree|rename|replace|chmod|chown|kill|pkill|taskkill|shutdown|reboot)\b" tools/project_governance tools/project_governance_inventory.py tests/test_project_governance_*.py
rg -n "(N607-admin|szu2310433034|ControlMaster|ControlPersist|StrictHostKeyChecking=no)" tools/project_governance tools/project_governance_inventory.py configs/project_governance_inventory_v1.json
git diff --check
```

Review every match. Allowed mentions are negative tests, explicit safety errors and fresh-output implementation; there must be no destructive execution path or administrator route.

- [ ] **Step 2: Run focused and existing regression tests**

Run serially:

```bash
conda run -n ssr-gpu python -m pytest tests/test_project_governance_models.py tests/test_project_governance_local.py tests/test_project_governance_git.py tests/test_project_governance_experiments.py tests/test_project_governance_retention.py tests/test_project_governance_n607.py tests/test_project_governance_emit_cli.py tests/test_n607_training_inventory.py -q
conda run -n ssr-gpu python -m compileall -q tools/project_governance tools/project_governance_inventory.py
```

Expected: all tests PASS and compilation exits0.

- [ ] **Step 3: Review tracked diff and preserve unrelated assets**

Run:

```bash
git status -sb
git diff --stat
git diff -- tools/project_governance tools/project_governance_inventory.py configs/project_governance_inventory_v1.json tests/test_project_governance_*.py docs/project_governance/worklog
```

Confirm the three unrelated paths named inGlobal Constraints remain untracked and unchanged.

- [ ] **Step 4: Explicitly stage only implementation-owned files**

Use one explicit`git add --`list containing only the files enumerated inthis plan. Then run:

```bash
git diff --cached --name-only
git diff --cached --check
```

The cached list must not contain any`.docx_qa_cvs_ntn*/`path、`tools/build_cvs_ntn_scenario_docx.py`or other non-task path.

- [ ] **Step 5: Commit implementation**

Commit message:

```text
feat: add read-only project asset governance inventory
```

Record the full commit ID in`docs/project_governance/worklog/progress.md`. If recording it requires a follow-up documentation commit, use`docs: record governance implementation verification`and keep both hashes in the later scan receipt.

## Task 10: Run the Formal Local and N607 Read-Only Inventory

**Files:**
- Create at runtime: `docs/project_governance/<scan_id>/...`
- Create only if thresholds require: `E:\type10-7\local_artifacts\project_governance\<scan_id>\...`
- Modify: `docs/project_governance/worklog/findings.md`
- Modify: `docs/project_governance/worklog/progress.md`

- [ ] **Step 1: Verify the committed implementation and preview scope**

From the governance worktree in exact Git Bash:

```bash
GOV_SCAN_ID="PGOV_$(date -u +%Y%m%dT%H%M%SZ)"
GOV_GIT_OUTPUT="$PWD/docs/project_governance"
GOV_EXTERNAL_OUTPUT="E:/type10-7/local_artifacts/project_governance"
conda run -n ssr-gpu python tools/project_governance_inventory.py scan --config configs/project_governance_inventory_v1.json --scan-id "$GOV_SCAN_ID" --output-root "$GOV_GIT_OUTPUT" --external-output-root "$GOV_EXTERNAL_OUTPUT" --operator codex --include-n607 --print-plan
```

Expected: preview lists exactly the approved roots/surfaces, says N607 will be contacted, creates no files and reports no deletion execution capability.

- [ ] **Step 2: Run the required direct N607 preflight**

Invoke only the existing reviewed script:

```bash
GOV_PREFLIGHT_WIN="$(cygpath -w "$PWD/tools/n607_ssh_preflight.ps1")"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "$GOV_PREFLIGHT_WIN"
```

Require verified direct SSH config/identity, server time, project-root visibility and GPU visibility. If identity/key/target is ambiguous, stop. If only direct TCP/SSH path is unavailable while identity/config remains valid, allow the collector's approved bridge fallback. Do not prompt for passwords.

- [ ] **Step 3: Execute one immutable formal scan**

Run:

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 conda run -n ssr-gpu python tools/project_governance_inventory.py scan --config configs/project_governance_inventory_v1.json --scan-id "$GOV_SCAN_ID" --output-root "$GOV_GIT_OUTPUT" --external-output-root "$GOV_EXTERNAL_OUTPUT" --operator codex --include-n607
```

Do not rerun with the same scan ID. If it fails, preserve partial output, diagnose locally and use a new immutable scan ID after the concrete defect is fixed.

- [ ] **Step 4: Verify coverage and zero mutation**

Programmatically validate the receipt and tables. Acceptance requires:

- local root-level observed count is at least the approved baseline166, or any difference is explained by the current timestamped scan;
- N607 root-level observed count is at least the approved baseline185, or any difference is explained;
- configured carrier surfaces each have`VERIFIED`, `NOT_PRESENT`or explicit`SCAN_ERROR`coverage;
- every asset has Git ownership or non-Git evidence;
- every recognized experiment has a fixed state;
- all deletion rows are`AWAITING_USER_APPROVAL / NOT_AUTHORIZED`；
- source mutation、move、overwrite和delete counters are all0；
- N607 route and disconnect state are`VERIFIED`；
- no SSH or bridge TCP22 connection remains.

Baseline counts are comparison points, not hard-coded completeness substitutes. New assets must not be omitted to force equality.

- [ ] **Step 5: Review deletion candidates without executing them**

Inspect every candidate's exact path、evidence、dependencies、recoverability and estimated reclaim. Any uncertain row is downgraded to`REVIEW_REQUIRED`by fixing classifier input or logic and generating a new scan ID. Do not ask for broad deletion approval; present the precise table to the user after the inventory commit.

- [ ] **Step 6: Commit only small validated governance outputs**

Explicitly stage the new`docs/project_governance/<scan_id>`directory and updated worklogs. Before committing:

```bash
git diff --cached --name-only
git diff --cached --check
git status -sb
```

Confirm unrelated untracked assets remain excluded. Commit message:

```text
docs: add local and N607 asset governance inventory
```

## Task 11: User Review Gate Before Any Reorganization or Method Optimization

**Files:**
- Review: `docs/project_governance/<scan_id>/report.md`
- Review: `docs/project_governance/<scan_id>/deletion_candidates.csv`
- Review: `docs/project_governance/<scan_id>/scan_receipt.json`

- [ ] **Step 1: Deliver the governance outcome**

Report the branch、implementation commit、inventory commit、scan ID、local/N607 freshness、coverage、active experiments、Git ownership gaps、retention distribution and exact deletion-candidate count. State explicitly that no original asset was moved、overwritten or deleted.

- [ ] **Step 2: Request precise deletion decisions only if candidates exist**

Group candidates into small, exact batches but retain one row per path. The user must approve exact candidate IDs and location; local approval cannot authorize N607 deletion and vice versa. Until then all rows remain in place.

- [ ] **Step 3: Keep later work outside this implementation**

Any approved deletion requires a new, separately reviewed execution plan with fresh path/process/dependency verification and dry-run. Method/performance optimization also requires a separate design based on current`项目.md`and`COMPLETE_EVIDENCE`same-row results. Neither action is implied by completing this plan.

## Self-Review

- Spec coverage: covers approved local/N607 scope、bounded scan depth、stable identity、Git ownership、experiment state、retention classes、approval-only candidates、size routing、receipt、SSH route/disconnect and zero-mutation acceptance.
- Safety boundary: no task authorizes deletion、movement、overwrite、remote write、admin access、experiment launch/stop or automatic Git mutation.
- Completeness scan: no unresolved implementation choice remains. Runtime`<scan_id>`is a documented output schema token; the executable command derives an immutable value.
- Type consistency: enum names、record names、file names、state precedence、return codes and output names are fixed across tasks.
- Version order: implementation is committed before the formal scan; the formal scan receipt binds that commit; governance artifacts form a separate commit.
- User gate: deletion remains unauthorized even if a row is classified`DELETE_CANDIDATE`；later deletion and method optimization each require a new explicit user decision.
