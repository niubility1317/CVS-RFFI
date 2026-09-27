# Native IR-EG / BR integration report

Implemented in the isolated `experiments/ir_eg_v1/code` package. No historical snapshot, dataset, checkpoint, remote job, or existing run was modified. Parent agent owns the integrated Git delivery.

## Reachable native path

`response_config.make_row` accepts `IR_EG` and `BR_IR_EG`; `response_solver.create_response_solver` registers `IREGSolver`. The existing `runtime.train` and `response_runtime.response_step` select it. Both copied launchers perform an independent import-origin audit against their own `ROOT/code`, including the model, native SSDG objective, normalizer, solver, and loaded IR modules. A local import-only probe verified 16 loaded modules.

The response schema uses `implementation`, `ir_start_epoch`, `ir_kappa`, `ir_cg_max_iterations`, `ir_cg_rtol`, `ir_origin_scales`, `br_refresh_interval`, `ir_curvature`, and `ir_feature_anchor`. Unknown fields and bool-as-integer values are rejected; v1 fixes E21 activation, predictor ratio one, FP32 AdamW, no AMP, and no unsupported optimizer variants. E1–20 use original complete EG. Non-BR kappa zero calls original `GameSolver.step` directly without head capture, response leaves, metric construction, or CG.

## Native objective and state semantics

Labeled objective assembly is extracted as `assemble_labeled_terms`. SSDG RC4 exposes a pure `assemble_rc4` closure. `FusionObjective` records native additions in their original order and replaces only registered L-clean and U-strong adversarial CE leaves. It does not reconstruct the objective using old-total plus new-minus-old losses. DAOT, native identity components, satellite CE, normalization divisors, and optional X/U interaction additions remain present.

The packet keeps post-GRL graph inputs separately from detached head solve inputs. New head leaves carry the response values. Formal gradients map those independent leaves to the original head slots; frozen head coordinates retain `grad=None`. The U adversarial branch has an actual zero-GRL edge, while the native U identity loss continues to backpropagate.

Reference mode executes three full field forwards and two complete gradient fields. Cached mode executes two full field forwards and two complete gradient fields. Ordinary BR steps retain the origin forward and current head backward, but use the previous accepted final raw nonhead field; they execute one complete gradient field. Autograd connectivity determines current None-versus-active ownership without running the full origin backward. Every accepted step refreshes the history; interval eight controls current full-field prediction, not history age.

`TrainingState` now additionally restores module training flags and requires-grad flags. The IR outer transaction covers optimizer/model/registered buffers/RNG, explicit native context pending state, solver/history, EMA, normalizer/route history, and the legacy prototype bank. The runtime supplies native external commits inside that transaction; no duplicate commit follows. Failures after formal AdamW or midway through external commits restore the previous accepted state. Only a response-solve numerical failure uses same-step EG; invalid/nonfinite scalar objectives and structural failures abort and roll back.

## Validation evidence

- Initial reference, cached, and transaction tests: 19 passed.
- Packet and BR history/resume tests: 13 passed initially, subsequently extended with illegal head-gradient ownership and serialized checkpoint coverage.
- Complete native FusionObjective + DAOT + FastTrust reference/cached fixture passed CPU loss, raw gradients, actual parameter displacements, AdamW moments, EMA, prototypes, normalization state, routing, and RNG comparisons. It reaches both native H/P identity branches.
- The 10-step toy comparison checks the specified FP32 tolerances, exact RNG, and actual model forward counts (six reference versus four cached calls for its two-call field).
- Failure injection covers origin, virtual step, capture, CG entry, corrector, formal update, post-external commit, and partial external commit. A separate test proves that a NaN scalar with finite derivatives cannot be committed by the cached path.
- Active BR serialized continuation is bitwise equal to uninterrupted execution in the toy integration fixture. R=1 matches IR at kappa zero and 0.25; BR kappa zero is distinct from full EG after stale nonhead prediction begins.
- Full native entrypoint pause/resume is exercised across E1→E2, comparing numerical state under the already prescribed multistep FP32 tolerance and exact solver clocks/RNG. This test verifies native runtime checkpoint wiring, while active BR continuation is covered separately; it does not claim a long native E80 resume experiment.
- Latest targeted CPU validation after scalar-finiteness and native-resume fixes: 19 passed (`test_ir_transaction.py`, `test_ir_native_resume.py`, `test_ir_br_resume.py`, `test_ir_packet.py`). Parent acceptance owns the final whole-package CPU/CUDA result.
- Serialized active BR CUDA continuation uses the production CPU checkpoint deserializer, preserving CPU AdamW step scalars while optimizer moments and history map to CUDA parameters. Three focused CUDA resume tests pass with exact state/device comparisons.
- Legacy non-IR prepared rows receive only the nine missing known IR defaults during read-only canonical comparison. Saved baseline files remain unchanged; altered native fields and unknown response keys still fail validation.

## CUDA equivalence diagnosis

The first CUDA acceptance exposed a native reference/cached displacement mismatch despite gradients meeting tolerance. The independent reference/reference control also failed under nondeterministic cuDNN, with a larger maximum parameter difference (2.62e-4). Initial parameters and materialized L/U/satellite inputs were exactly equal, and RNG was restored.

With `torch.backends.cudnn.deterministic=True` and `benchmark=False`, both reference/reference and reference/cached maximum parameter differences became exactly zero in the diagnostic. This common kernel policy is applied to every comparison method in the new package runtime and to native acceptance fixtures. Tolerances were not widened. The runtime writes `kernel_policy.json`; numerical evidence is in `cuda_kernel_diagnostic.json`. Global `torch.use_deterministic_algorithms(True)` is intentionally not enabled because the unchanged native DAOT detached median-with-indices operation rejects that switch on CUDA.

## Delivery boundaries

The package is prepared and locally validated. No remote training, scientific convergence claim, performance promotion, target scoring, or checkpoint reuse is performed here. Local acceptance fixtures are synthetic source-role data. Detailed CUDA and complete regression outcomes are recorded by the parent acceptance process.
