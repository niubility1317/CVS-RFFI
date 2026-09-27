# Focused integrated IR-EG review

Scope: supplied `acceptance/final_review_package.diff`, current integration sources, specification sections 2–10, and the previous mathematical findings. Reviewed cached/reference execution, native loss assembly, transaction rollback, BR lifecycle, runtime reachability, and diagnostic isolation. Source snapshot reviewed while the integration and math owners were still applying their final fixes. No tests were rerun and no production code was changed by this reviewer.

## Verdict

No P0/P1 failure found. The scoped follow-up confirms all three prior findings are addressed: finite scalar validation, frozen head/None semantics, and formal launcher import provenance. One narrowly verified P2 legacy-configuration compatibility regression remains open, described below. The follow-up did not rerun tests or broaden the review.

## Addressed: Cached corrector finite scalar validation

Current fix locations: `code/cvsrffi/xuc_fusion/ir_solver.py:20–23,207,259,279`. Reference behavior: `code/cvsrffi/game_tracking/solvers.py:112`.

The original defect was a missing scalar-finiteness check despite checking gradients; a NaN scalar can have finite derivatives. The follow-up confirms `_validate_loss` now checks scalar/differentiable/finite semantics at predictor capture (`ir_solver.py:207`) and cached corrector assembly (`ir_solver.py:259`). A dedicated `ResponseNumericalFailure` class limits EG fallback to response-solve numerical failures (`ir_solver.py:243–246,279`); nonfinite native/corrector objectives instead propagate `NonFiniteStep` to the outer rollback. A focused regression test now covers the nonfinite scalar with finite derivatives. This finding is addressed in source; no test was rerun by this reviewer.

## Previous findings

- **Frozen head layout: fixed.** `ir_types.py:71–76` now retains all head tensors and all nonhead optimizer slots. `IREGSolver` selects trainable derivative targets and maps frozen head leaf gradients back to None. The metric excludes absent gradients, leaving their full-vector response zero. Added tests cover partial/all freezing and the metric's frozen coordinate.
- **Native import provenance: addressed.** `audit_runtime_imports(expected_code_root)` checks critical native modules against an independent supplied root. Both `train_response_games.py:29` and `train_native_dr_eg.py:26` now call it with their own `ROOT/'code'` before invoking training. Required native modules are explicitly imported before the audit. The result is retained in resolved arguments. `audit_contract` also requires the independent root.

## Open P2: Copied legacy EG configuration is rejected after adding IR defaults

Location: `code/scripts/train_response_games.py:21–22`; schema expansion: `code/cvsrffi/xuc_fusion/response_config.py:13–15`.

The launcher reconstructs a row with `make_row` and then requires raw dictionary equality with the supplied row. `make_row` now inserts nine IR fields into every method's response settings, including legacy EG. The copied `configs/separate_controls/PURE_EG_s392005.json` lacks these fields, so its otherwise unchanged row fails before even the non-executing validation result. A read-only canonicalization probe confirmed `canonical_equal=False`, exactly nine added response keys (`implementation`, `ir_start_epoch`, `ir_kappa`, `ir_cg_max_iterations`, `ir_cg_rtol`, `ir_origin_scales`, `br_refresh_interval`, `ir_curvature`, `ir_feature_anchor`), and no changed nonresponse fields. This breaks the copied prepared legacy controls, not the newly generated IR rows. Migrate the copied configs explicitly or normalize only absent, irrelevant defaults for legacy methods before comparison; continue rejecting actual configuration drift. No training or test execution was needed to reproduce the comparison failure.

## Scoped cuDNN policy follow-up

`runtime.py:133–145` sets `cudnn.deterministic=True` and `cudnn.benchmark=False` unconditionally for every comparison method using this package's `train` entry point, then records the policy. It is therefore shared by the baseline and IR paths rather than selectively applied to IR. `docs/cuda_kernel_diagnostic.json` reports unchanged materialized inputs, a nonzero reference/reference discrepancy under uncontrolled cuDNN, and zero reference/reference and reference/cached parameter differences under the controlled policy. The report explicitly does not claim global deterministic algorithms, which remain disabled because of the native median-with-indices operation. This reviewer checked the source and recorded diagnostic artifact, not a fresh CUDA execution; the parent's native CUDA acceptance remains the authority for final results.

## Specification compliance

The reviewed loss extraction replaces the two registered CE leaves inside the original labeled/RC4 assembly order rather than subtracting an old loss from a total. Detached features are confined to head solving; cached final assembly retains native post-GRL inputs, including the U zero-gradient edge. The response is stopped numerically and the formal optimizer step starts from restored origin parameters and optimizer state. Reference mode installs response values only after discarding the capture packet and its context callbacks.

The outer transaction includes model/optimizer/RNG/buffers, model flags, context state, normalizer/route history, teacher, prototype state and serialized solver/BR history. External commit exceptions restore the origin transaction. BR updates history after every accepted final raw field, distinguishes None from zero, refreshes for activity/signature changes, and uses the actual mixed predictor clip coefficient. The full numerical domain head remains available when response coordinates are frozen.

The IR diagnostics inspected use disposable head parameters and isolated model/optimizer/RNG state, keep source training fit records separate from the V monitoring panel, and do not feed a controller. Clip-attribution steps are isolated from the formal step. GGN weighting, dropout replay and whitened two-step CG retain the previously reviewed mathematical semantics. The scoped fixes address the prior mathematical/integration findings; the legacy row compatibility issue remains an entry-point defect.

## Quality and evidence limits

The test suite contains substantive cached/reference state comparisons, U identity/adversarial ownership checks, external-commit failure injection, same-step fallback, BR history/refresh/resume checks, and native FusionObjective + DAOT + FastTrust integration. These cover meaningful behavior rather than merely matching implementation shapes. Parent-reported CPU acceptance was 97 passing tests; CUDA acceptance was still in progress when this review began. This reviewer did not execute or independently reclassify those runs.

The native resume fixture in the supplied snapshot resumes E1 to E2, before the E21 response activation. Active BR resume equivalence is exercised by the toy integration fixture; a native E21+ resume is not established by those two tests alone. This is a precise evidence boundary, not a new blanket test gate.

No formal ManySig data run, historical checkpoint probe, target evaluation, scientific convergence run, or real-data efficiency claim was reviewed or performed. The prepared matrix explicitly leaves the existing physical-role manifest binding and post-E200 convergence continuation policy unresolved. Synthetic source acceptance must not be described as having validated those production-data or scientific requirements. Existing validated physical data should be reused under the project's protocol, not revalidated merely because this solver changed.


## Final implementation evidence (controller readback)

The remaining legacy-config P2 was fixed by validate_prepared_row: only missing known IR defaults on non-IR rows are normalized; canonical native drift/unknown keys still fail. Four compatibility tests pass, including old PURE_EG CLI validation with no output directory. Final full acceptance: CPU 110/110 and actual CUDA 110/110, all zero skips/failures. Legacy regression 106/106. All actionable findings above are addressed; no broad review cycle was repeated.
