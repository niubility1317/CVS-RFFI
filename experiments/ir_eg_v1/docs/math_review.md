# Independent mathematical review: Tasks 1–4

Reviewed the supplied `acceptance/math_review_package.txt`, the current four IR modules, their focused tests, and the clip-coefficient export. Governing requirements: `IR_EG_DESIGN_SPEC.md`, sections 2–4. This is a source review; already-passing tests were not rerun. No production code was edited.

## Verdict

No P0/P1 mathematical defect found in the supported, fully trainable four-tensor head path. Two P2 contract defects remain. The central R formula, native-order parameter packing, isolated dropout replay, weighted matrix-free GGN, whitened scatter/gather, and finite two-iteration CG semantics are consistent with the specification.

## Actionable findings

### P2 — Frozen head tensors are removed from the function's parameter tuple

Location: `code/cvsrffi/xuc_fusion/ir_types.py:75`; downstream assumption: `code/cvsrffi/xuc_fusion/ir_head.py:66`.

`build_layout` removes every `requires_grad=False` head tensor from `phi_indices`. However, `head_logits` requires the complete `(w1, b1, w2, b2)` tuple. For example, freezing the first linear bias before building the layout leaves three phi tensors, so otherwise valid native head replay fails instead of retaining the bias as a fixed value and giving its coordinate zero response. Specification section 3.3 explicitly removes frozen coordinates from the response activity subspace, not from the head's numerical forward representation. Preserve the complete native forward parameter tuple and maintain a separate response/trainability mask (or explicitly reject partial freezing at construction and narrow the stated supported contract). Add a focused partial-freeze case covering both replay and zero response on the frozen coordinate.

### P2 — Default import audit can approve a mismatched native model import

Location: `code/cvsrffi/xuc_fusion/ir_types.py:137–142`.

The default expected-import map contains only `ir_types` and `ir_head`, with its root inferred from the loaded `ir_types.__file__`. Neither the model's actual defining module nor the native objective/solver dependencies are checked. Consequently, importing `model_dual_cvsincnet` from another same-named tree and then passing its model to `audit_contract` can still return `gradient_ownership_verified=True` and successful `imports`; the optional caller override is not required and no production caller of this audit was found. This misses Task 1 and specification section 1.2's concrete same-name-module hazard. Require an execution-package-root expectation from the entry point and check the actual critical native imports against it; test a foreign native-model path, not merely an explicitly wrong path for the audit helper itself.

## Specification compliance

- Head ownership is based on object identity and `adv_head`, not `dom_head`; head order survives optimizer group reordering. The freeze caveat above remains.
- `audit_contract` differentiates actual task tensors and compares the native phi gradient against the registered, separately normalized L/U CE field. It does not multiply head gradients by GRL.
- Dropout capture replays the native random draw on ones under `fork_rng`, preserves call separation, and uses non-inplace ReLU with a fixed mask for differentiation. It does not infer masks from zero activations or freeze the later ReLU pattern.
- R uses original per-parameter state, actual group betas/epsilon/lr, the recorded clip coefficient once, and checks the real predictor second moment and local step. None gradients and zero learning rates are excluded from the active metric. The solver restricts predictor ratio to one, consistent with this interface.
- GGN applies each sample weight once, uses detached features, and detaches the output-space cotangent before VJP. No dense parameter-space matrix or extra decay curvature is constructed.
- CG solves the whitened active system, uses FP64 inner products, returns whitened u, and leaves the second sqrt(R) multiplication to the caller. Two iterations without tolerance satisfaction return `truncated_usable`; zero RHS and nonpositive/nonfinite operator failures have distinct behavior.

## Task quality

The focused tests substantively cover numerical replay, weighted reductions, parameter-local AdamW steps, fixed-moment finite differences, explicit small GGN comparisons, nonlinear GGN versus Hessian, and CG status semantics. They are not merely shape or smoke checks. The two gaps above are absent from the current tests. Full native transaction, cached/reference equivalence, reporting, and remote execution are outside this review's scope.
