# Progress: Mitigating receiver impact DA

## 2026-07-10

- Read `AGENTS.md` and `项目.md` in UTF-8 and preserved the boundary that this work is paper reproduction, not CVS Stage2 deployment evidence.
- Read the selected workflow skills.
- Confirmed the Git-backed workspace is `E:/type10-7/github_publish/CVS-RFFI-repo` and recorded unrelated modified/untracked files.
- Created this task-specific planning set because the root planning files belong to unrelated prior work.
- Logged the first subagent-spawn API error; no agent or file change resulted from it.
- Spawned three independent read-only audits for paper specification, implementation, and complete experiment artifacts.
- Extracted and read all 11 PDF pages, with focused review of Sections II, IV, V, Algorithm 1, and Tables II-IV.
- Inspected the authors' public trainer and its linked Pytorch-Template source.
- Recorded confirmed differences in normalization, pseudo-state lifecycle, batch pairing, source initialization, BatchNorm forward count, weighted CE reduction, and MINE moving-average handling.
- Ran direct N607 preflight, process/GPU inventory, and a bounded ManySig shape/DC-power audit. No job was launched and no remote file was changed.
- Verified after SSH use that no local `ssh.exe` or N607 TCP/22 connection remained.
- Error: full PDF extraction initially failed because the Windows console used GBK and could not encode a ligature. Resolution: reran with `PYTHONIOENCODING=utf-8`.
- Error: a combined local inspection command failed when an `rg` scan encountered an access-denied temporary path. Resolution: narrowed searches to explicit project directories/files.
- Error: first task-report mirror used `Copy-Item -LiteralPath` with a wildcard, which does not expand. Resolution: enumerated files with `Get-ChildItem` and copied the exact file objects.
- Supervisory review issued NO-GO until runtime claim profiles became fail-closed; algorithm/gradient/BN/MINE changes had no additional blocker.
- Added mutually checked claim profiles: `paper_equations_bounded`, `released_trainer_semantics_bounded`, `diagnostic_extension`, and `oracle_diagnostic`. Mixed, truncated, quota/floor, and target-label-selected runs now downgrade automatically.
- Local repair verification now passes 52 focused/adjacent tests and the paper-config dry-run.
- Synced the nine locally verified runtime files to N607; remote SHA256, `py_compile`, `bash -n`, and paper dry-run all passed.
- Launched eight Proposed-only validation runs at 11:22:34 CST, one per GPU. Startup process/GPU checks passed and no error marker was present.
- Error: the first startup-health command embedded a PowerShell-expanded remote shell loop and failed before inspection. Resolution: verified no stale SSH process, then split process/GPU/error checks into three bounded SSH commands.
- All eight first-matrix runs completed by 11:46 CST. Pulled all result JSON and complete `.out` logs; verified that no related process, local `ssh.exe`, or established N607 TCP/22 connection remained.
- Parsed every epoch and final row. Strict five-task mean is 62.37% versus 96.14% in the paper; the corrected cross-day row is 85.73%, while receiver-pair rows remain 30.20%-75.40%.
- Added a linked-template ResNet1D architecture hypothesis and paper Table III component switches. The architecture and every ablation fail closed to `diagnostic_only`.
- Local verification now passes 62 focused/adjacent tests, model shape/parameter smoke (`3,839,303` parameters), Python compile, launcher syntax, and CLI option checks.
- Spawned two additional independent reviewers for the component-ablation semantics and architecture/launcher before N607 sync.
- Architecture/launcher reviewer issued GO for diagnostic-only execution and identified one provenance gap. Added explicit CLI/result fields and fail-closed checks for `lr`, `m`, `tau`, `mu`, and `lambda`; reran all 54 tests and syntax checks successfully.
- Algorithm reviewer issued NO-GO because four Table III combinations used the wrong source CE scale and the all-disabled control leaked target BN state. Repaired the seven-combination coefficient map, made the all-disabled target audit eval/no-grad, preserved composite claim reasons, and added seven-combination plus BN-state tests.
- The same algorithm reviewer completed a second pass and issued GO with no Critical/Important finding. Full local verification now passes 62 tests.
- Committed the round-2 implementation, tests, launcher, and mirrored report as Git commit `63af9ab` before N607 sync.
- Direct preflight at 12:04 CST passed; all eight GPUs were idle, no related process was active, and `/home` had 7.6 TB free.
- Synced the five committed runtime files; all remote SHA256 values matched, and remote `py_compile`, launcher `bash -n`, and CLI option checks passed.
- Launched the eight-run architecture/Table III localization matrix at 12:05:58 CST, one process per GPU. The 12:06 startup check found every PID alive, active GPU load, and no error marker.
- At the 12:11 five-minute health gate, all eight PIDs remained active with no error marker; no result file had completed yet.
- Parsed full first-matrix confusion matrices. The hard tasks contain balanced class swaps/cycles, explaining why marginal class weighting cannot correct the collapse.

## 2026-07-13

- 重新读取`AGENTS.md`和`项目.md`，确认本任务仍是WiSig闭集UDA论文复现诊断，不是CVS Stage2或部署成功证据。
- 完整解析round2的8组JSON/`.out`；每组语义一致、20+20epoch完整、无运行时错误marker，且全部为`completed_diagnostic_only`。
- 记录template五任务均值69.9439%、standard五任务均值62.3742%、论文均值96.1440%，并完成8个同run结果及论文差值表。
- 记录组件定位：DA-only68.5667%对76.36%，DA+CW40.9875%对77.02%，CPL+CW22.2750%对77.11%。
- 新增三波20项Proposed-only多seed计划：seed20260711/20260712，standard/template两个profile，wave规模8/8/4。
- 初版launcher监督审查为NO-GO；修正版加入run/log/manifest防覆盖、跨wave并发门控、每GPU最多2个计算进程门控、启动锁和20项完整性matrix后，独立复审为GO。
- 根目录不是Git仓库；Git承载面为`E:/type10-7/github_publish/CVS-RFFI-repo`。当前无关`local_artifacts`保持未跟踪且不纳入本任务stage范围。
- 在`ssr-gpu`串行运行`test_wisig_random_split.py`、`test_wisig_fewshot_payload.py`和`test_mitigating_receiver_impact_da.py`，62/62通过。首次仅运行50项主测试时，8个`tmp_path`用例受用户临时目录权限阻断；改用工作区独立`--basetemp`并关闭pytest缓存后通过，临时目录已清理。
- `py_compile`覆盖`paper_reproduction/mitigating_receiver_impact_da/*.py`，新launcher的`bash -n`和`git diff --check`均通过。
- 新launcher本地SHA256为`df9de171051e97d23787aaf91bbc62a11193cdb9eb9922d37ea9085c990bc1de`；远端仅允许同步到同相对路径，不允许启动任何wave。
- 限定stage审计确认暂存区仅含新launcher和5份本任务Git镜像报告；无关`local_artifacts`未纳入。生成pre-sync提交`a3b7a331b60d6da08667aa8840c771cafb9532b4`。
- 11:20 CST直接N607 preflight通过。实时inventory和GPU UUID/PID映射确认GPU0-7各仅1个compute进程，全部属于无关`phase1_dgleo_jointp0_leoweak8r2_20260713`；本实验精确进程匹配为0。
- 仅SCP新launcher到远端同路径；远端SHA256与本地`df9de171051e97d23787aaf91bbc62a11193cdb9eb9922d37ea9085c990bc1de`一致，远端`bash -n`通过，未产生任何wave run/log目录。
- 未启动任何wave；所有SSH/SCP后本地`ssh.exe`和N607/bridge TCP22连接均为0。
