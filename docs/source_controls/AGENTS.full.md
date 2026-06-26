# Project Instructions

- Before any CVS-RFFI / CV-SincNet research-scenario, data-protocol, experiment-matrix, optimization-route, automation-control, paper/report-framing, or Stage2-A/B/C interpretation work, read `项目.md` in UTF-8 after this file. Treat `项目.md` as the source of truth for CVS scientific scenario, data protocol, receiver/TX split, weak-label/semi-supervised DG framing, single-satellite-receiver deployment protocol, Stage2-A/B/C boundaries, metric claims, automation semantics, and allowed research narrative.
- If future optimization, modification, monitor/optimizer/runner behavior, launcher design, matrix generation, report writing, or paper framing changes the CVS scenario, labeled/unlabeled data definition, `rho_label` grid, receiver split, old/new/unknown TX split, K-shot grid, Stage2-A/B/C semantics, satellite/LEO deployment-primary view, or success criteria, update and verify `项目.md` first before changing prompts, contracts, scripts, matrices, reports, automation state, registries, validators, or N607 launch plans.
- CVS automation must not launch, mark PASS, register as deployment evidence, or write a paper/report success claim for any candidate whose data split, support/query permission, target receiver definition, old/new/unknown TX semantics, K-shot setting, satellite/LEO view, or metric claim conflicts with `项目.md`. Such candidates must be blocked as protocol repair or marked diagnostic-only until `项目.md` is explicitly revised.
- If `项目.md` conflicts with old reports, memory, prior prompts, historical matrices, or launcher defaults, follow `项目.md` for CVS scientific/data-protocol interpretation. If `项目.md` conflicts with this `AGENTS.md`, follow `AGENTS.md` for safety/environment/N607/version-management rules and report the conflict.
- 中文论文、报告、README、PPT备注、DOCX正文等生成文本必须遵循中文排版习惯：中文标点后不加空格；中文与英文、数字、缩写、变量名之间不加额外空格；保留英文短语内部必要空格，例如`target receiver domain`和`K-shot support`；保留代码、命令、路径中的原始空格。

- Before running project-related code tests, activate the Conda environment:
  `conda activate ssr-gpu`
- Use `ssr-gpu` as the corrected test environment name.
- For any code/config/script changes that will be used on N607, edit and verify the local workspace first, then sync the changed files to the server with `scp`. Do not make remote-only edits.
- Federated learning experiments must use WiSig train ratio `0.1`. Treat this as a hard constraint, not a tunable default.
- Unless a user explicitly overrides it, set default `epochs` and `fl_rounds` to `200` for federated training launchers.
- Unless a user explicitly overrides it, use `receiver` as the default federated client granularity (`--fl_client_key receiver`).

## N607 SSH Automation

- Use the plain `N607` direct SSH target first for N607 access by default. If direct `N607` access fails, use the verified lab-computer bridge as the fallback route instead of trying ad-hoc SSH routes.
- Before any task that needs SSH or SCP access to N607, start with the local read-only direct preflight:
  `powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`
- The preflight must verify, without changing server state: direct `N607` SSH config and identity, server time, project-root visibility, and GPU visibility. If direct preflight fails because the direct TCP/SSH path is unavailable while local config and identities are otherwise valid, fall back to the lab bridge below. If identity, key, or target ambiguity is the problem, stop and report diagnostics instead of trying interactive passwords or ad-hoc SSH routes.
- Use short-lived SSH commands such as `ssh -o BatchMode=yes N607 "<command>"`. Do not keep persistent SSH shells open after the needed check or operation is complete.
- Every SSH/SCP connection must actively disconnect as soon as its bounded task finishes. Do not leave interactive shells, background SSH clients, `tmux` relay sessions, `ControlMaster`/multiplex master connections, port forwards, or long-running monitor sessions open.
- After every SSH/SCP task, ensure the local SSH client has actually exited. On Windows, if there is any sign of a lingering `ssh.exe` or an `ESTABLISHED` TCP connection to N607 port 22, identify the exact command/PID, close the stale local client, and verify that no N607 SSH connection remains before continuing. If it cannot be closed, stop and report the blocker instead of running more SSH/SCP work.
- Do not rely on command completion text alone as proof of disconnect; local process/connection state must be checked whenever a command timed out, was interrupted, produced malformed quoting, or otherwise may have left an orphaned SSH client.
- To avoid SSH reliability loss, do not keep long-lived SSH sessions, idle interactive shells, port forwards, or multiplex master connections open to N607. Monitoring must use short, discrete SSH commands and then disconnect.
- For file syncs to N607, use direct SCP such as `scp <local> N607:<remote>`, after local verification and any required snapshot/report updates.
- The verified lab bridge is `administrator@172.31.105.18`, using local key `C:/Users/lh594/.ssh/id_ed25519_lab_bridge_172_31_105_18`. Use it only as a fallback after the direct attempt fails. Keep the N607 private key local; do not copy N607 keys, datasets, checkpoints, or server credentials onto the lab computer.
- Preferred bridge command pattern:
  `ssh -i C:/Users/lh594/.ssh/id_ed25519_n607 -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -o "ProxyCommand=ssh -i C:/Users/lh594/.ssh/id_ed25519_lab_bridge_172_31_105_18 -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -W %h:%p administrator@172.31.105.18" szu2070436088@172.31.111.215 "<command>"`
- When a bridged SSH/SCP task ends, close the lab-computer-to-N607 SSH leg first, then close the local-host-to-lab-computer SSH leg. With the `ProxyCommand -W` pattern, this means letting the bounded N607 remote command/channel exit first and then letting the proxy connection to `172.31.105.18` exit. After timeouts or interruptions, check and clear local `ssh.exe` processes and `ESTABLISHED` connections to both `172.31.111.215:22` and `172.31.105.18:22` before continuing.
- Treat any other non-direct relay host as outside the default N607 workflow. Do not route through any unverified relay, run experiments on it, copy datasets/checkpoints to it, or store server credentials on it unless the user explicitly asks.

## Server Maintenance and Safety

- Remote commands must be minimally scoped and read-only by default. Prefer safe checks such as `hostname`, `date`, `pwd`, targeted `test -f` / `test -d`, `nvidia-smi`, targeted `ps` / `pgrep`, `tail -n`, `sed -n`, and bounded `find <specific-dir> -maxdepth ...`.
- Avoid high-impact commands unless the user explicitly requests them and the target has been verified. This includes `rm -rf`, broad `mv` / `cp` / `chmod` / `chown`, `kill` / `pkill` / `killall`, `reboot`, `shutdown`, package installs, service changes, recursive scans of `/` or dataset roots, cleanup of logs/checkpoints/metrics, and anything that can interrupt jobs or consume major CPU/GPU/disk/network resources.
- Before any state-changing remote action, inspect active processes/GPU/disk context, confirm exact target paths, prefer dry-runs when available, record the exact command, and update the local experiment report when the action is experiment-related.
- If remote jobs are active and the user has not explicitly asked to intervene, switch to monitor-only behavior. Do not launch, kill, restart, patch remote files, or clean outputs merely because remote access is available.
- For N607 experiment packing, the user allows up to two concurrent training experiments per GPU by default. If one training process is already active on each GPU, it is acceptable to launch one additional experiment per GPU after preflight, occupancy recording, and report update; do not exceed two per GPU unless the user explicitly overrides it.
- Preserve datasets, checkpoints, logs, metrics, reports, and run outputs. Do not delete or overwrite them unless the user explicitly requests it and the scope is unambiguous.

## Experiment Reporting

- Every time an experiment is designed and run on N607, create or update a local report before handing off or ending the turn.
- Store reports under `E:\type10-7\automation_reports\CV-SincNet\<timestamp-or-run-id>\report.md`.
- The report must be useful for post-run analysis. Include at minimum:
  - experiment ID or run name, timestamp, operator/agent, and objective;
  - hypothesis and comparison target;
  - local files changed, verification commands, and sync destination on N607;
  - exact server command, Conda/Python environment, working directory, log path, PID, GPU allocation, and expected output files;
  - key configuration values, dataset split/scenario, seed, metrics to watch, and early-stop or success criteria;
  - known risks, assumptions, and what to inspect when the run finishes.
- When an experiment finishes, update the same report with final status, best epoch/checkpoint reference, detailed result tables, anomalies, interpretation, and recommended next experiment.
- Finished-run reports must include a detailed per-candidate or per-experiment result table. Each row must keep the metrics from the same candidate/run together, including candidate ID, mechanism/category, receiver/TX split, K-shot, seed, old/seen-new/unknown metrics, coverage/rollback/defer fields where available, loss/adapter summary, and final verdict.
- Do not report standalone maxima or minima as if they describe one experiment. If max/min values are useful, attach the candidate/run ID and the full same-row metric context, or present them in a separate distribution table clearly marked as marginal statistics. The main interpretation must be based on joint rows or explicitly named joint-ranking criteria, not on unrelated single-metric extrema from different rows.
- Use Markdown tables for result summaries, comparisons against prior runs, best/joint-ranking candidates, failure modes, and next-step decisions whenever the data are tabular.
- Do not rely only on chat history for experiment context; persist the analysis trail in the report file.

## Conversation History Lookup

- New conversations cannot automatically browse full prior chat history. For project-related history, use the local index tool before relying on memory alone.
- Build or refresh the project-scoped index with:
  `conda activate ssr-gpu; python tools/conversation_index.py build`
- Search only `E:\type10-7` related historical conversations with:
  `conda activate ssr-gpu; python tools/conversation_index.py search "<keywords>"`
- The generated index is stored under `E:\type10-7\conversation_index\` and includes source paths back to the Codex rollout summary or session JSONL when available.
- Do not treat the index as a replacement for experiment reports. For N607 experiment design, launch, monitor, or completion analysis, still create or update the required report under `E:\type10-7\automation_reports\CV-SincNet\...`.

## Version Management

- Every project-related change must enter a Git-backed workflow. Before editing code, config, scripts, prompts, matrices, reports, or project Markdown, locate the relevant Git repository and run `git status -sb` or report that the target tree is not a Git repository. After editing, inspect `git diff` / `git status -sb`, run the narrowest useful verification, and record the changed files plus verification result in the relevant report or Markdown handoff.
- Do not leave intended project changes only in chat or in an untracked working directory. If the edited target is inside a Git repository, stage and commit the intended change with a concise message unless the user explicitly says not to commit. If the edited target is not inside a Git repository, stop before treating the change as versioned and either initialize/choose a Git repository with user-visible scope or mirror the change into the agreed Git-backed release workspace/branch.
- For GitHub-facing publication or repository-structure changes, use a branch/PR flow by default. Do not force-push, rewrite shared history, or overwrite unrelated remote content unless the user explicitly asks and the exact scope is confirmed.
- Project-relevant Markdown must be kept in sync with each code/config/script/protocol change. Update `AGENTS.md` for workflow, safety, environment, Git, N607, or collaboration rules; update `项目.md` before any CVS scientific/data-protocol/Stage2/metric-claim change; update README/docs/reports when user-facing usage, experiment interpretation, or publication scope changes.
- Keep all code/config/script edits local first, then sync to N607 with `scp` only after local verification.
- Before launching or changing a server experiment, record the local version state in the report:
  - changed files and purpose of each change;
  - relevant command outputs from local checks;
  - file hashes or `git status`/diff summary when a git repository is available;
  - remote destination paths used for sync.
- If the target directory is not a git repository, create a timestamped local snapshot for changed code/config/script files under `E:\type10-7\code\snapshots\<timestamp-or-run-id>\` before syncing.
- Keep `E:\type10-7\code\SYNC_MANIFEST.txt` or the run report updated with the exact local-to-remote file mapping for each sync.
- Never overwrite unrelated local or remote changes. If a file has unowned edits, inspect and preserve them; ask before destructive operations.
- Do not delete datasets, checkpoints, logs, metrics, reports, or run outputs as part of version cleanup unless the user explicitly requests it.

