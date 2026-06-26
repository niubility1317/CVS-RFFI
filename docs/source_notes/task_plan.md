# Task Plan: Baseline paper/code audit and training efficiency

## Goal
Comprehensively audit `E:/type10-7/baselines` papers and implementations, identify mismatches between paper descriptions and code, then design and implement focused fixes for baseline training efficiency and post-training test evaluation, including satellite-ground channel evaluation.

## Current Phase
Phase 6

## Phases

### Phase 1: Scope & Inventory
- [x] Read project instructions and existing planning files.
- [x] Inventory baseline papers, baseline method folders, shared training/evaluation utilities, logs, and previous runs.
- [x] Map each paper to its corresponding implementation folder and entry points.
- **Status:** complete

### Phase 2: Paper Extraction & Method Summaries
- [x] Extract enough text from each baseline PDF to capture model, loss, training schedule, data split, and evaluation protocol.
- [x] Record concise per-paper expectations in `findings.md`.
- **Status:** complete

### Phase 3: Code Review Against Papers
- [x] Review each baseline implementation against its paper expectations.
- [x] Prioritize behavioral mismatches, missing evaluation paths, reproducibility risks, and performance hot spots.
- [x] Record actionable issues with file/function references.
- **Status:** complete

### Phase 4: Design Fixes
- [x] Propose implementation design for efficiency and evaluation changes.
- [x] Get user approval before editing training behavior.
- **Status:** complete

### Phase 5: Implementation
- [x] Optimize TIFS2025 slow training path after root-cause evidence.
- [x] Ensure all comparison methods run test-set evaluation after training.
- [x] Ensure satellite-ground channel evaluation is included consistently.
- [x] Add focused tests or dry-run checks around changed behavior.
- **Status:** complete

### Phase 6: Verification & Delivery
- [x] Run project tests through `conda activate ssr-gpu`.
- [x] Run parser/dry-run checks for baseline launch paths.
- [x] Update `findings.md` and `progress.md`.
- [ ] Provide concise final findings, changed files, and remaining risks.
- **Status:** in_progress

## Key Questions
1. Which PDFs correspond to `tifs2025_channel_receiver_rffi`, `receiver_agnostic_rffi`, `riei`, `drift`, and `cvcnn`?
2. Does each code path implement the paper's stated model, losses, and training stages?
3. Why is `tifs2025` training much slower than other baselines?
4. Where should post-training clean/test and satellite-ground evaluations be centralized so all baselines share the same behavior?
5. Which checks can be verified locally without launching full GPU training?

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Continue from existing planning files but replace the active plan. | Existing files tracked previous SGV/SSDG work and no longer matched the current request. |
| Treat `baselines` as the primary scope. | User specifically asked for baseline-folder papers and generated comparison-method code. |
| Do paper/code audit before edits. | Efficiency and evaluation fixes should target root causes rather than guesses. |
| Use `ssr-gpu` for project-related tests. | Required by `AGENTS.md`. |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| `git status --short` failed because `E:/type10-7` is not a Git repository. | 1 | Track edits through planning files and final file references. |
| Initial `python -m baselines... --help` failed because `dataset_wisig` lives under `code/`. | 1 | Added project `code/` to `baselines.common` import path initialization. |
| Existing `tests.test_cvs_rffi_launcher` failed with empty stdout while checking `scripts/run_cvs_rffi_staged_8gpu.sh`. | 1 | Treated as pre-existing/unrelated to baseline trainer changes; targeted baseline tests pass. |

## Active Addendum: CVS-RFFI baseline-matched comparison run

### Goal
Run the same receiver-curriculum comparison settings used for the baseline experiments on CVS-RFFI, using the strongest available backbone configuration/checkpoint `BEX02_fishr002_mixed_e170`.

### Phases
- [x] Read the experiment-group description and identify the baseline comparison matrix.
- [x] Locate the existing CVS-RFFI launcher/checkpoint path and confirm how to pass `BEX02_fishr002_mixed_e170`.
- [x] Dry-run or inspect the generated queue for matching receiver-curriculum settings.
- [ ] Launch the CVS-RFFI experiment queue under the required `ssr-gpu` Conda environment.
- [x] Record process IDs, logs, run directory, and any immediate failures.

## Active Addendum: 2026-06-21 CVS 自动化瘦身与多子 agent 监督优化

### Goal
基于此前 N607/CV-SincNet 自动化运行情况，全面审计 monitor、optimizer、runner、validator、report/state/registry 的不足，删除或降级累赘逻辑，并落地低风险本地优化，且不削弱 `AGENTS.md` 与 `项目.md` 的协议/安全边界。

### Phases
- [x] 读取 `AGENTS.md`、`项目.md`、相关技能、记忆与本地 conversation index。
- [x] 派出只读子 agent：控制面历史缺陷、代码表面改动点、科研协议语义监督。
- [x] 审计当前控制面/runner/validator/state/report 代码与测试。
- [x] 选择低风险、可本地验证的精简优化并编辑本地文件。
- [x] 运行本地验证；不做 N607 SSH/SCP/launch，除非用户另行要求。
- [x] 汇总不足、删除项、优化项、验证结果和剩余风险。

### Guardrails
- 不改变 `项目.md` 定义的 CVS 场景、Stage2-A/B/C、单一 `r_sat`、`Y_old/Y_new/Y_unknown` 互斥、K-shot 网格、satellite/LEO 主视图。
- 不用日志/metrics/checkpoints 决定 monitor gate；monitor gate 保持 process/CWD/cmdline/GPU/lane scoped。
- 不把 validator PASS 当成 launchable；route duplication、多角色审查、协议字段缺失仍是实质 gate。
- 不远程编辑；本轮默认不 launch、不 kill、不清理远端输出。

## Active Addendum: Paper-exact baseline reproduction on CVS-RFFI config

### Goal
Run reproduction experiments that match each comparison paper's own training strategy, loss setup, data protocol, and evaluation protocol as closely as the local implementations allow, under the CVS-RFFI/WiSig data configuration. Do not treat prior "comparison" runs as sufficient unless their commands and artifacts prove paper-setting parity.

### Phases
- [x] Refresh project conversation index and read current N607/experiment rules.
- [ ] Inventory comparison methods, local paper-setting summaries, launcher entry points, and existing CVS-RFFI comparison runs.
- [ ] Create/update a traceability table mapping each paper requirement to exact CLI/config/code evidence.
- [ ] Dispatch independent review agents for paper-setting parity and omission checks.
- [ ] Build a reproduction matrix with run IDs, GPU allocation, seeds, data ratio, epochs/rounds, losses, and metrics.
- [ ] Locally verify scripts/configs under `ssr-gpu` before any N607 sync or launch.
- [ ] Run N607 preflight, inspect current GPU/process occupancy, and only launch within the two-training-runs-per-GPU rule.
- [x] Record startup health, PID/GPU/log paths, and follow-up inspection steps in an experiment report.

### Guardrails
- Federated runs must use WiSig train ratio `0.1`, default `epochs=200`, `fl_rounds=200`, and `--fl_client_key receiver` unless explicitly overridden by the user.
- For centralized paper baselines, record any unavoidable dataset/protocol difference from the original paper as a traceability note instead of silently calling it exact.
- No remote-only edits; sync only locally verified files with direct `scp`.

### Launch Status
- [x] N607 preflight passed.
- [x] Local snapshot created under `code/snapshots/20260601_152529_paper_exact_baselines_r010/`.
- [x] Remote pre-sync backup created under `/home/szu2070436088/2510044040/CV-SincNet/snapshots/20260601_152529_paper_exact_baselines_r010_pre_sync`.
- [x] Verified/synced baseline package and launcher to N607.
- [x] Launched four-method queue under `/home/szu2070436088/2510044040/CV-SincNet/{runs,logs}/wisig_baselines_paper_exact_seed1337_ratio010`.
- [x] Startup health verified for PIDs `664267`, `664285`, `664293`, `664300`.

### Correction: Original Paper Dataset Protocol Required
- [x] User clarified that the target is original-paper original-dataset protocol reproduction, not CVS-RFFI/WiSig data-config reproduction.
- [x] Marked `20260601_152529_paper_exact_baselines_r010` as non-target auxiliary evidence only.
- [x] Extracted original dataset/protocol requirements per paper and corrected the target after user clarified non-WiSig papers should align to WiSig original settings.
- [x] Checked local/N607 dataset availability: local `ManySig.pkl` is missing; N607 has WiSig `ManySig.pkl`.
- [x] Built a new WiSig original-protocol matrix in `analysis/cvsrffi_original_paper_protocol_traceability_20260601.md`.
- [x] Implemented and locally verified the first executable protocol: DRIFT/WiSig Day1 receiver-disjoint `drift_day1`.
- [x] Run N607 preflight, sync verified local changes, and launch only if occupancy gate is clean.
- [ ] Monitor completion and parse `paper_eval_window` metrics from the WiSig original-protocol queue.

### Current Status
Running on N607: `wisig_original_protocol_drift_day1_seed1337` on GPUs 4-7 with PIDs `685180`, `685194`, `685207`, and `685214`. Startup health passed. This queue is exact for the DRIFT WiSig Day1 protocol and aligns non-WiSig comparison methods to that WiSig original setting.

### Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| `git -C E:\type10-7\code diff --stat` failed because the code tree is not a git repository. | Session catchup version-state check. | Record current analysis as file-based; use explicit path reads and existing snapshots/manifests instead of git state. |

## Active Addendum: StyleBank/ProtoBank Design-vs-Implementation Audit

### Goal
Comprehensively compare `C:/Users/lh594/Downloads/fed_pvs_cprffi_final_design.md` with the current local federated implementation, especially StyleBank, ProtoBank/ProtoEvidenceBank, and their actual integration into federated training. Determine what is consistent, what is only partially implemented, what is diagnostic/eval-only, and what is missing before claiming design parity.

### Phases
- [x] Extract StyleBank/ProtoBank requirements and training flow from the final design report.
- [x] Inspect local StyleBank, style packet/extractor/sampler, conditioned receiver-DG, ProtoEvidenceBank, reliability fusion, FedProto, and trainer integration code.
- [x] Check current launcher/CLI/tests to see which modules are actually reachable in normal FL training.
- [x] Compare design intent vs implementation in a structured table.
- [x] Write persistent findings/report and answer with concise but concrete conclusions.

### Current Status
Completed. Detailed audit written to `E:\type10-7\analysis\federated_log_forensics\stylebank_protobank_design_gap_audit.md`. No training-code changes were made in this audit pass.

### Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| `python` is not visible inside the local WSL `bash.exe`. | Non-dry-run launch with `--python python`. | Passed the active `ssr-gpu` Python as a WSL-style executable path. |
| `WISIG_PKL not found: /mnt/e/type10-7/Dataset_WigSig/ManySig.pkl`. | Non-dry-run FULL launch. | Training did not start; user or environment must provide the actual `ManySig.pkl` path via `WISIG_PKL` or `--wisig-pkl`. |

## Active Addendum: StyleBank/ProtoBank Design-Parity Implementation

### Goal
Fix the mismatches found in the audit so federated training defaults to the design-report path: StyleBank collection plus remote style-conditioned multi-style batches with explicit `d_style`, maturity-gated DG losses, and ProtoEvidenceBank/FedProto separation with conservative prototype evidence diagnostics.

### Phases
- [x] Recover prior context and confirm `E:\type10-7\code` is not a git repository.
- [x] Add failing tests for default federated StyleBank enablement, style-batch construction, and ProtoBank CLI/plumbing.
- [x] Implement default federated StyleBank training path without requiring custom test-only `style_batch_fn`.
- [x] Wire ProtoEvidenceBank diagnostics and conservative eval fusion entry points while keeping FedProto baseline separate.
- [x] Update launchers/docs/reports and run focused verification under `ssr-gpu`.

### Current Status
Completed locally and synced to N607 after remote backup. Detailed implementation report: `E:\type10-7\analysis\federated_log_forensics\stylebank_protobank_design_parity_implementation.md`. No N607 training launch has been performed in this pass.

### Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| `git -C E:\type10-7\code diff --stat` failed because `code` is not a git repository. | Session catchup version-state check. | Record changes in planning/report files and use explicit file snapshots if syncing later. |
| Remote backup command failed with `unexpected EOF while looking for matching '"'`. | First PowerShell-wrapped SSH backup command. | Re-ran with remote shell variable assignment inside a single-quoted SSH command; backup succeeded before scp. |
| `ssr-gpu` was not present on N607. | Remote environment check before verification. | Used the existing project env `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python` for py_compile and unittest. |

## Active Addendum: Reusable Workflow Packaging Audit

### Goal
Review the recent E:\type10-7/Codex work history, identify repeated manual workflows worth packaging, create only high-confidence missing assets, and add a specific guard for repeated omissions when implementing design reports.

### Phases
- [x] Extract the instruction text from the supplied image.
- [x] Refresh/search the local conversation index with `ssr-gpu`.
- [x] Search Codex memory and existing rollout summaries for repeated workflow patterns.
- [x] Inventory existing skills, custom-agent locations, and the active `cv-sincnet` automation.
- [x] Create or extend only high-confidence missing assets.
- [x] Summarize extracted command text, candidates, created/extended assets, skipped items, and evidence gaps.

### Current Status
Completed. Created new skills `C:\Users\lh594\.codex\skills\cv-sincnet-n607-automation\SKILL.md` and `C:\Users\lh594\.codex\skills\design-report-traceability\SKILL.md`; extended automation `cv-sincnet` with conversation-index lookup and formal federated default `--fl_client_key receiver`. No remote experiment or project code change was launched.

### Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| Root `git status` failed because `E:\type10-7` is not a git repository. | Version-state check. | Recorded changes through explicit file paths and validation commands. |
| Skill validation failed under default Python because `yaml` was missing. | `python quick_validate.py ...` | Re-ran with `conda activate ssr-gpu`; validation passed. |
| `design-report-traceability` validation initially failed on Windows GBK decoding because the skill description contained Chinese trigger phrases. | First `quick_validate.py` run under `ssr-gpu`. | Rewrote the skill description/body as ASCII while preserving the Chinese-intent trigger semantics; validation passed. |

## Active Addendum: CVS-RFFI star-ground augmentation-method comparison experiments

### Goal
Design and implement a focused SSH-run experiment matrix inside CVS-RFFI to compare two satellite-ground channel augmentation methods.

### Phases
- [x] Re-read local planning context and prior evidence.
- [x] Connect to remote host and confirm the CV-SincNet workspace exists.
- [x] Inspect remote dataset, Conda environment, launcher availability, and prior logs.
- [x] Present and refine the experiment design to use CVS-RFFI only, not baseline as the comparison method.
- [x] Implement the launcher locally first, then sync it to the remote server.
- [x] Dry-run on the remote server with the `CVS-RFFI` Conda Python.
- [x] Launch the approved CORE queue and record logs/PIDs.
- [x] Add the local-first-then-sync rule to project instructions and automation.

### Current Status
CORE queue is running on N607. Local launcher: `E:/type10-7/code/scripts/run_cvs_rffi_sat_aug_compare_5gpu.sh`. Remote launcher: `/home/szu2070436088/2510044040/CV-SincNet/scripts/run_cvs_rffi_sat_aug_compare_5gpu.sh`. Logs are under `/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_rffi_sat_aug_compare`.

### Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| Local `bash -n` invoked missing WSL. | Syntax check on Windows. | Used remote `bash -n` after syncing local script. |
| Remote `train.py` rejected new `--exp_group sat_aug_method_compare`. | First CORE launch at `20260525_182003`. | Changed the local script to use existing enum `s3_rxrobust_no_dac`, synced with `scp`, dry-ran again, then relaunched. |
| SSH launch command timed out locally. | Background `nohup` launch at `20260525_182115`. | Verified the remote queue and GPU usage; training processes are running. |
| Initial "strong view" implementation was only an auxiliary sat-CE approximation, not strict baseline-style concatenation. | User asked whether it was exactly the same as baseline. | Added separate local module `concat_sat_channel_aug.py`, wired `--use_concat_sat_channel_aug` into `train.py`, synced to N607, and launched strict queue under `cvs_rffi_concat_sat_compare`. |
| Local unit test cannot import `torch` in `ssr-gpu`. | `python -m unittest tests/test_concat_sat_channel_aug.py`. | Kept local `py_compile` verification and ran the tensor unit test successfully in the remote `CVS-RFFI` environment. |

## Active Addendum: Spaceborne FL-DG-FSL Prototype Research Synthesis

### Goal
Synthesize a broad research direction for spaceborne RFFI that coordinates federated learning, domain generalization, few-shot learning, and multi-prototype classification, while staying compatible with the existing CVS-RFFI/CV-SincNet codebase and prior Fed-PVS / collaborative prototype notes.

## Active Addendum: Fed-PVS-CPRFFI Final Design Integration

### Goal
Analyze `C:/Users/lh594/Downloads/fed_pvs_cprffi_final_design.md` as the latest FL + DG + few-shot + collaborative multi-prototype design, then map it organically onto the existing CVS-RFFI/CV-SincNet codebase without prematurely rewriting the core model.

### Phases
- [x] Read current project instructions and planning context.
- [x] Confirm the uploaded final design report exists and inspect its section structure.
- [x] Inspect existing CVS-RFFI training, federated trainer, satellite augmentation, prototype/FJMP, SGC/SSDG, and evaluation surfaces.
- [x] Extract the design's core mechanisms and compare them to current code capabilities.
- [x] Produce a detailed integration analysis with staged implementation options, risks, diagnostics, and first-code-change recommendations.

### Current Status
Complete for analysis. Detailed integration document written to `E:/type10-7/docs/fed_pvs_cprffi_cvs_integration_analysis.md`. No code implementation, tests, N607 sync, or N607 experiment launch was performed in this turn.

### Phases
- [x] Read current planning context and project instructions.
- [x] Read the local Fed-PVS-RFFI research plan and federated collaborative prototype fusion report.
- [x] Survey recent and foundational literature across RFFI, federated learning, federated domain generalization, few-shot/meta/prototype learning, satellite/space-air-ground RF settings, and privacy/security.
- [x] Inspect N607 experiment families and extract the most relevant metrics for FL/DG/few-shot/prototype/satellite evidence.
- [x] Produce an integrated architecture and staged experiment roadmap.
- [x] Persist the research synthesis under `docs/` and summarize the most important recommendations.

### Current Status
Synthesis document written under `docs/`. Ready for final handoff.

### Working Thesis
The four themes should not be treated as parallel modules. In this project, federated learning supplies the deployment and data-governance frame; StyleBank/physical virtual styles repair the local multi-domain condition needed by DG; few-shot learning handles new receiver/ground-station/satellite domains after deployment; and multi-prototype heads should become conservative, reliability-weighted collaborative evidence rather than a strong replacement classifier.

## Active Addendum: Fed-PVS-CPRFFI Strategy Loophole Audit

### Goal
Stress-test the proposed Fed-PVS-CPRFFI integration strategy, identify loopholes that could make it fail or overclaim, and revise it into a gated strategy that is compatible with the current CVS-RFFI codebase and verified N607 evidence.

### Phases
- [x] Re-read the current integration strategy and planning evidence.
- [x] Inspect local federated training, prototype, DG, satellite augmentation, and aggregation surfaces for hidden assumptions.
- [x] Query N607 logs/metrics for the relevant FL-DG, FedProto, satellite augmentation, and concat-sat evidence.
- [x] Enumerate concrete loopholes and proper fixes.
- [x] Write a revised gated strategy and confidence boundary.
- [x] Implement Phase -1 hygiene fixes and StyleBank V0 diagnostics.
- [x] Add `d_style` plumbing and tests so constructed style domains can feed model/domain losses separately from raw receiver/day labels.
- [x] Add ProtoEvidenceBank/reliability-fusion eval-only support and harm/rescue tests.
- [x] Snapshot and sync the changed local files to N607, then run remote verification.

### Current Status
Phase -1/Phase 1 implementation complete. New local modules under `E:/type10-7/code/federated/` implement StylePacket/RFStyleExtractor/FederatedStyleBank/VirtualDomainSampler/StyleConditionedReceiverDG/ProtoEvidenceBank/reliability fusion. `FederatedTrainer` now supports optional no-op StyleBank diagnostics, optional `style_batch_fn` `d_style` plumbing, and local-only aggregation exclusions. The satellite comparison launcher now strips the base `--use_sat_consistency` flag from concat-sat dry/run commands to avoid the previous mutually exclusive flag failure.

### Confidence Boundary
The strategy cannot be 100% empirically guaranteed before controlled experiments. The current confidence claim is narrower: the revised V2 protocol is a stricter, falsifiable, lower-loophole strategy than the original direct implementation plan.

### Verification
- Local compile: `conda run -n ssr-gpu python -m py_compile ...` passed for touched implementation and test files.
- Local unit tests: `conda run -n ssr-gpu python -m unittest tests.test_fed_pvs_style_bank tests.test_fed_pvs_proto_fusion tests.test_federated_d_style_plumbing tests.test_federated_aggregation tests.test_federated_train_integration tests.test_cvs_rffi_sat_aug_launcher -v` passed 12 tests and skipped 2 bash/WSL-dependent launcher tests.
- Remote N607 unit tests: `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m unittest ... -v` passed all 14 tests, including the bash launcher dry-run checks.
- Local snapshot before N607 sync: `E:/type10-7/code/snapshots/20260526_000700_fed_pvs_phase1_impl/`.

### Post-Implementation Audit Update
- [x] Compact `d_style` labels so remote StyleBank ids cannot exceed model domain-head dimensions.
- [x] Add domain CE target-range guards for constructed domains.
- [x] Make StyleBank vector schema stable under packet stat-key drift.
- [x] Fix StyleBank trim ordering to prefer higher-count/newer centroids.
- [x] Add launcher-test subprocess timeouts to avoid local Windows bash hangs.

Fresh verification after these fixes: local compile passed; local unittest ran 16 tests with 15 passed and 1 local bash/WSL skip; N607 compile and unittest passed 16/16.

## Active Addendum: FL82 federated validation experiment

### Goal
Design, record, automate, and run an N607 federated experiment whose explicit targets are: validate federated learning effectiveness, pursue clean strict `test_unseen_day_unseen_rx` accuracy >= 82%, run clean named tests every federated round, and include satellite-channel evaluation every round. The formal FL constraint is train ratio `0.1` with default epoch/round count `200`. The satellite target explicitly includes clear_leo split floors: `test_unseen_day_seen_rx >=84.30%`, `test_seen_day_unseen_rx >=60.10%`, and `test_unseen_day_unseen_rx >=53.78%`.

### Phases
- [x] Inspect current federated and satellite evaluation surfaces.
- [x] Create a local FL82 launcher and persistent N607 experiment report.
- [x] Verify locally, snapshot changed code/script files, and update the sync manifest.
- [x] Sync the launcher and missing local dependencies to N607, then run remote compile/dry-run checks.
- [x] Launch the N607 CORE queue on GPUs 3,4,5,7 and verify the jobs entered training.
- [x] Write the FL82 goal into the `cv-sincnet` hourly automation.
- [x] Add `[SAT-TEST-SPLIT]` logging and the `SAT_BASELINE` launcher plan using baseline-style clean+sat supervised view expansion.
- [x] Verify/snapshot/sync the SAT_BASELINE update, launch `FL82_07`..`FL82_09` on GPUs 0,1,2, and confirm round-1 split metrics print.
- [x] Update the `cv-sincnet` automation with the clean strict and clear_leo split floors.
- [x] Correct FL82 launcher defaults to train ratio `0.1` and epoch/round `200`, rename future rows to `r010`, verify locally/remotely, sync to N607, and update automation/reporting.
- [ ] Monitor convergence, update the report with best strict UDU/satellite metrics, and decide the next experiment.

### Current Status
Running on N607, but the active `0.2/220` CORE and SAT_BASELINE runs are now historical/debug-only under the corrected constraint. Future formal launches must use `r010`, `--wisig_train_ratio 0.1`, `--epochs 200`, and `--fl_rounds 200`. Active CORE scheduler log: `/home/szu2070436088/2510044040/CV-SincNet/logs/fl82_fed_validation/scheduler_CORE_20260526_005707.log`. Active SAT_BASELINE scheduler log: `/home/szu2070436088/2510044040/CV-SincNet/logs/fl82_fed_validation/scheduler_SAT_BASELINE_20260526_014110.log`. Local report: `E:/type10-7/automation_reports/CV-SincNet/20260526_004220_fl82_fed_validation/report.md`.

### Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| `ModuleNotFoundError: baseline_origin_sat_view`. | First N607 launch. | Synced local `baseline_origin_sat_view.py` and verified remote compile. |
| `ModuleNotFoundError: cvsrffi`. | Second N607 launch. | Synced local `cvsrffi/*.py`, updated snapshot/manifest/report, and verified remote compile. |
| `train.py: error: unrecognized arguments: --exp_desc ...`. | Third N607 launch. | Removed unsupported `--exp_desc` from launcher; descriptions remain in launcher logs. |
| Local SSH launch wrapper timed out after fourth launch. | Fourth N607 launch. | Verified remotely that launcher and all four training jobs are running; did not relaunch. |

## Active Addendum: Federated Learning Log Forensics and FL82 Diagnosis

### Goal
Comprehensively analyze local N607 log backups for federated-learning experiments, reconstruct each experiment's actual configuration, then deeply diagnose why the latest `fl82_fed_validation` federated runs underperform the centralized `BEX02_fishr002_mixed_e170` CVS-RFFI baseline. If key diagnostics cannot be recovered from existing logs, add focused training-log output locally and verify it before any N607 sync.

### Phases
- [ ] Inventory federated log/run artifacts from the local exhaustive N607 backup.
- [ ] Extract per-experiment configuration, launch command, train ratio, FL client split, objective/loss flags, satellite eval settings, and available metric curves.
- [ ] Build a focused `fl82_fed_validation` table with best/latest clean strict UDU plus named satellite split metrics.
- [ ] Compare FL82 with earlier federated evidence and centralized CVS-RFFI baseline evidence.
- [ ] Identify root causes and confidence boundaries for poor federated DG performance.
- [ ] Patch missing diagnostic logging only where logs do not contain required evidence.
- [ ] Verify parser/report outputs and any code changes under the required local Conda environment.

### Current Status
In progress. Source backup: `E:/type10-7/server_log_backups/N607/20260526_101853/exhaustive_log_files`.

### Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
## 2026-06-13 CVS-RFFI Ground-to-Space FSL Audit

Status: in_progress

Goal: 对当前 CVS-RFFI 地面训练 + 星上部署小样本适应工作做证据驱动科研审计，输出 `AUDIT_CVS_RFFI_GROUND_TO_SPACE_FSL.md`、`metrics_summary.csv`、`evidence_map.csv`、`missing_experiments.md`、`improvement_plan.md`。

Phases:

1. Scope and evidence inventory - in_progress
2. Five-agent audit collection - pending
3. Metrics extraction and cross-check - pending
4. Missing experiments and root-cause analysis - pending
5. Final report writing and artifact verification - pending

Constraints:

- Do not run long training.
- Prefer existing code, configs, logs, reports, CSV/JSON, and scripts.
- Any unsupported claim must be marked as missing evidence / unable to judge.
- N607 access must follow AGENTS.md preflight and short-lived cleanup rules.

## 2026-06-14 Inference-Stage Few-Shot Remediation and 10h N607 Batch

Status: launched; remote batch running

Goal: Focus only on deployment/inference-stage few-shot learning gaps from the audit, design and launch a bounded N607 batch that targets open-set SFE, new-class recognition, old-class target alignment, rollback, and satellite-channel support/query evidence. Ground DG training is not the main resource target in this batch.

Phases:
1. Re-read AGENTS/skills and recover planning context - completed
2. Dispatch read-only subagents for SFE, FTRC, satellite support/query, metrics, and N607 launch risk - completed
3. Create traceability and experiment report before code/script edits - completed
4. Implement a local-first 10h inference-FSL launcher and minimal reporting helpers - completed
5. Verify locally under ssr-gpu / bash syntax / dry-run - completed
6. Run N607 preflight, capacity gate, sync verified files, and remote verify - completed
7. Launch only safe slots, record PID/GPU/logs, startup-health check, and cleanup SSH - completed
8. Wait for FTRC completion, collect final remote artifacts, and write post-run analysis - pending

Constraints:
- Do not start ground retraining.
- Remote launch must not exceed two training/compute processes per GPU.
- Prefer existing SFE/FTRC code paths; only add minimal launcher/reporting glue unless subagent evidence proves code changes are necessary.
- Every launched candidate must map to an audit gap and a measurable metric.

Current remote state:
- Scheduler PID: `3303518`.
- Remote logs: `/home/szu2070436088/2510044040/CV-SincNet/logs/spaceborne_fsl_inference_10h_20260614/`.
- Remote runs: `/home/szu2070436088/2510044040/CV-SincNet/runs/spaceborne_fsl_inference_10h_20260614/`.
- Startup health: passed at 2026-06-14 01:38-01:41 CST. SFE clean/satellite bundles completed; 12 metrics JSON and 12 score tables exist. FTRC candidates are running; LoRA K20 remains deferred until GPU7 capacity frees.
- Early SFE finding: clean best `mahal_t070_mh6` has raw/deployed full 61.79%, old 90.00%, new 3.00%, unknown reject 98.00%, FAR 2.00%, AUROC 79.62%, FPR95 36.67%; satellite-target best raw `mahal_t070_mh6` has full 55.36%, old 90.00%, new 14.00%, unknown reject 69.00%, FAR 31.00%, AUROC 75.99%, FPR95 50.00%, and rollback triggered.
