# Phase1 P0 Closed-loop Traceability

Source: `automation_reports/CV-SincNet/phase1_dgleo_v2fix8_20260708/report.md`, section `2026-07-10完成结果与终局分析`.

Protocol: Phase1 source-only ManySig weak-label/semi-supervised DG. No target receiver, true unknown, ManyTx, Stage2 threshold fitting, or deployment-success claim.

|ID|Source section|Requirement|Target files|Status|Verification|Notes|
|---|---|---|---|---|---|---|
|P0-U01|无标签、local component与损失预算|Make U_s direct geometry reachable without hidden TX labels; replace confidence-only fallback with geometry-first `trusted_core/ambiguous_tail/outside_reject`; log class counts and reason codes.|`code/SSDG/train_ssdg.py`; `code/cvsrffi/losses.py`; focused tests|verified|Receiver-aware U_s component test; pseudo/component mismatch rejection; all-valid routing and clean/sat paired-core assertions|`rho_label<=0.1` is enforced before and after grouping. Only L_s receiver/day components route U_s; ambiguous/outside reject pseudo CE, entropy and direct-metric routing but remain available to domain/ADV/paired invariance losses.|
|P0-L02|无标签、local component与损失预算|Replace diagnostic leave-one-domain counts with real receiver/day local components tied to an invariant class core, explicit radii, density gating, and exportable structure.|`code/cvsrffi/losses.py`; `code/cvsrffi/phase2_prototypes.py`; `code/SSDG/train_ssdg.py`; tests|verified|No-leave-domain gradient test; local accept/density losses; source-val component calibration; launcher dry-run|Clean and satellite local components are optimized separately; no pooled global acceptance ball.|
|P0-G03|无标签、local component与损失预算|Turn `B_os_eff` from a passive gate into active closed-loop control with bounded loss scaling and gradient-conflict handling; log pre/post budget and intervention.|`code/SSDG/train_ssdg.py`; `code/cvsrffi/phase1_v2_control.py`; tests|verified|Real-gradient budget test; closed-only head gradient exclusion; zero-gradient loss cannot fake budget; open-priority conflict projection test|Budget uses the shared closed/open z_id path, requires a positive target and is rebound to the selected checkpoint's same-epoch evidence.|
|P0-S04|泛化与星地压力/Open-set代理|Preserve clean-sat pair correspondence, optimize clean and satellite geometry separately plus paired invariance, and log per-view p95/p99/tail/proxy/bridge/overflow/ratio metrics.|`code/SSDG/train_ssdg.py`; `code/cvsrffi/losses.py`; launcher; tests|verified|Multiview direct-metric gradient test; multiview source-component test; CLI/dry-run|`concat_sa` is retained. TX CE, domain/ADV, U_s consistency, direct geometry and source local geometry all consume satellite views.|
|P0-T05|Open-set代理结果|Replace stochastic single-batch best-tail comparison with robust/fixed-reference windows and make rollback/cooldown an actual training action, not export-only telemetry.|`code/cvsrffi/phase1_v2_control.py`; `code/SSDG/train_ssdg.py`; tests|verified|Fixed source-val tail protocol; single-risk fail-closed test; real best-to-final p99 artifact; strict full-state rollback|Any p95/p99/CVaR/proxy breach blocks the epoch. Best-to-final p99 is recomputed on fixed source-val data; rollback requires model/EMA/optimizer/AMP/prototype/CPU+CUDA+sat RNG/guard/tail state.|
|P0-E06|证据完整性|Require a versioned `endpoint_accept_v1` artifact with threshold source, calibration split, reason codes, component/radius fields, and entry-parity evidence; remove hard-coded exported PASS.|`code/cvsrffi/phase1_v2_control.py`; `code/cvsrffi/phase2_prototypes.py`; `code/cvsrffi/prototype_bank.py`; `code/cvsrffi/hard_gate.py`; `code/export_spaceborne_features.py`; evaluator/tests|verified|Identity/shape/center/radius tamper tests; real three-entry decision parity; strict offline endpoint test|Boundary binds z_id dimension, checkpoint SHA, strict load, class-to-TX/logit order and head contract. Runtime/offline require actual identity; tail review cannot be auto-accepted. Dynamic DM remains non-exportable.|
|P0-O07|证据完整性|Make inactive/skipped/missing telemetry explicit and prevent misleading `BEST-JOINT E000`, safe-path, and NaN status from appearing as completed artifacts.|`code/SSDG/train_ssdg.py`; tests; report|verified|Terminal-state tests; stale-output refusal; selected-checkpoint evidence binding; nonzero failure exits|STOPPED/FAILED/NON_PROMOTABLE/NO_CHECKPOINT/INCOMPLETE are distinct from COMPLETE. Missing P0 mechanism activity or endpoint identity returns a nonzero code and promotion_ready=false.|
|P0-M08|协议边界/模型选择|Prevent held-out receiver/day and satellite-test results from selecting a Phase1 checkpoint.|`code/SSDG/train_ssdg.py`; launcher; tests|verified|Source-val-only fail-closed parser/trainer test; disabled-protocol rejection; source-val satellite H-mean test; frozen evaluation path|P0 launcher selects by source-val clean/satellite-floor H-mean. Training-epoch held-out evaluation is unconditionally disabled; receiver/day/satellite held-out views run once after checkpoint freeze.|
|P0-B09|证据粒度|Fix epoch telemetry that previously appended only the last batch and make tail aggregation conservative across batches.|`code/SSDG/train_ssdg.py`; tests|verified|Source indentation regression assertion; epoch max-batch tail aggregation field|This fixes a major prior evidence bug. Exact all-sample epoch quantiles remain a P1 refinement.|

## Reverse Audit

- pending: 0
- implemented: 0
- verified: 9
- deferred: 0
- rejected: 0
- blocked: 0

Three independent final read-only audits covered endpoint identity/parity, U_s routing/shared gradients, and tail/checkpoint control. Each returned `no remaining P0` after the final fixes.

## Verification

- `ssr-gpu` syntax validation passed for all changed Phase1 modules.
- P0 focused suite: `93 passed`.
- Broader SSDG/Phase1 regression suite: `35 passed`.
- No N607 sync, launch, or result claim was made in this change.

## Residual P1

- Training-log diagnostics still use conservative max-of-batch p95/p99/CVaR; promotion/tail safety no longer depends on them and uses the fixed source-val full collection.
- Add atomic `.pt/.json` export commit plus whole-file cross-digests; current boundary hash already protects runtime centers, radii, gate thresholds, calibration evidence and entry parity.
- Persist receiver-aware local components across batches with EMA/memory IDs and remove duplicate source-episode counting in diagnostic rates.
- Measure post-AdamW z_id parameter-update budget in addition to the implemented shared-path pre-update gradient budget.
- Run the P0 mechanism matrix and require joint Phase1 evidence before promotion; code-level verification does not prove proxy metrics or DG accuracy improved.
