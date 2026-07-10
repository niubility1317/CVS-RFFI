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
