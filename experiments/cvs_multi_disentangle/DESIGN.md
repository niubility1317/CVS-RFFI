# Multi-disentanglement implementation contract

This package implements the approved architecture: one existing identity backbone is constrained by multiple independent training-only disentanglement networks. It does not replace the proposal with one nuisance encoder and several output heads. The unified nuisance network is an explicit control arm only. This document describes mechanisms and implementation boundaries; it is not evidence of completed training, test accuracy improvement, or physical parameter recovery.

The authoritative executable settings are `design.py::RECIPE` and `design.py::config`. Run: `20261009-phase1-multi-disentangle-manysig-m32-r01`. No historical checkpoint is inherited. All rows initialize from scratch under the current source-data contract.

## Identity backbone and real E/G boundary

The identity model remains `reference_response`, including its existing time/frequency/PA representation, relative known-excitation response, and cosine classifier. Normal identity forward and deployment do not call any auxiliary network.

`model.intermediate(identity, x)` captures three real intermediate tensors from that model:

| Component | Dimensions | Exact boundary |
|---|---:|---|
| Native time/frequency fused representation | 160 | Input to the existing head's `base_norm` |
| Native physical representation | 160 | Input to `pa_norm`; already includes the original PA delta addition |
| Reference observables | 29 | Output of the existing fixed `ExcitationResponse` |
| Total `h` | 349 | Concatenation of the above existing intermediates |

`identity_from_intermediate(identity, h)` applies the original learned LayerNorms, bounded PA fusion gain, reference-response projection and reference gain. It returns the original 160-dimensional identity representation. `classify_intermediate` applies the original cosine classifier. The concatenation is an internal replay boundary, not additional identity information or a new classifier. The original model path is called when collecting the intermediates; call-local hooks are removed even on failure. The small downstream operations are replayed for feature interventions.

With zero feature intervention, this is the original identity model, not an approximate distilled substitute. Exact equivalence must also hold after the native runtime integration, EMA copying and checkpoint loading; a standalone synthetic check alone does not establish all integration properties.

## Independent networks

Each main network owns its view encoder, state estimator, bounded conditional operator and optional parameter-regression head. There is no learned encoder shared between the main networks. Fixed analytic observables may be computed by more than one network.

| Network | View and encoder | Supervision target |
|---|---|---|
| `D_L` / `linear` | 29 relative reference-response observables; independent 29→64→8 MLP | Known added receive-side linear perturbation and its intermediate-feature change |
| `D_T` / `temporal` | Ordered four 20-sample cycles from samples 80:160, both quadratures normalized by window RMS; independent temporal convolution encoder | Known added constant/linear/quadratic phase perturbation and its feature change |
| `D_R` / `receiver` | 29 reference observables plus eight bounded power/correlation descriptors; independent 37→64→8 MLP | Residual statistically matched source TX×RX group changes after detached available L/T predictions, not individual packet counterfactuals |
| `D_LT` / `interaction` | Reference and ordered temporal views, 189 values; its own 189→64→8 MLP | Ordered factorial interaction residual |

The state is estimated from an observed pair:

`q = encoder(view1) - encoder(view0)`.

Thus this first implementation learns a local **pair-conditioned action**, not a single-packet estimate of an absolute physical channel or receiver parameter. This is appropriate for training-only supervision and must remain explicit in result interpretation. Auxiliary state estimates are neural outputs. Known intervention coefficients enter regression targets only; they are not substituted for those outputs in the action-model dataflow.

The action uses independent rank-8 maps per network:

`raw_delta = U [tanh(a(q)) * V h] + B tanh(b(q))`.

Both coefficient maps are bias-free. A smooth norm bound enforces `||delta|| < 0.25 sqrt(||h||² + 1e-8)`. Identical paired views imply `q=0` and an exact zero action. No TX classifier, new identity feature concatenation, GRL, feature orthogonality or mutual-information objective is introduced into these networks.

At the registered dimensions the parameter counts are:

| Auxiliary configuration | Trainable parameters |
|---|---:|
| Linear only | 10,976 |
| Temporal only | 17,560 |
| Receiver only | 11,456 |
| Three independent main networks | 39,992 |
| Additional interaction network | 21,184 |
| Main networks plus interaction | 61,176 |
| Unified control | 41,552 |

The unified control has one combined-view 197→160→8 encoder, one operator and a task-conditioned state scale. It learns the same L/T/R task types and has approximately comparable capacity: 3.9% more auxiliary parameters than the three-main-network configuration. It is not identical in architecture or compute, so measured cost remains necessary.

## Exact controlled observations

All perturbations are mild operations on already received source IQ. They are not simulations that uniquely recover the transmitter or receiver chain. The input is `[B,2,256]` real IQ; operators use complex arithmetic internally.

For independent coefficients uniformly sampled in `[-1,1]`, linear parameters are four real values `p0...p3`:

`A_L(x)[n] = x[n] + 0.06 (p0+j p1) x[n-1] + 0.06 (p2+j p3) x[n-3]`.

Values before the packet are zero. There is no circular wrap and no post-intervention RMS normalization.

Temporal parameters are three real values `t0...t2`, with sample coordinate `u` evenly spaced over `[-1,1]`:

`A_T(x)[n] = x[n] exp(j [0.08 t0 + 0.10 t1 u[n] + 0.06 t2 u[n]^2])`.

The four observations are exactly:

- `x00 = x`
- `x10 = A_L(x)`
- `x01 = A_T(x)`
- `x11 = A_T(A_L(x))`

The order is part of the contract. The operators generally do not commute. Zero coefficients preserve the input exactly. An explicitly supplied private CPU `torch.Generator` supplies every intervention draw; its outputs are transferred to the IQ device/dtype. The physics functions do not consume the native CPU or CUDA training RNG streams.

Detached current-run EMA intermediate anchors provide targets `h00`, `h10`, `h01`, `h11`. Main targets are `h10-h00` and `h01-h00`. The interaction target is:

`delta_LT = h11 - h10 - h01 + h00`.

The interaction network independently encodes the L-only and T-only paired observations, then uses their elementwise state product. If either factor is a no-op, its state product and action are exactly zero. This gives the interaction a separately observable target rather than letting a free joint decoder allocate arbitrary residuals to it. The residual includes effects of the selected reference point, operator order and nonlinear encoder; it is not a unique physical interaction parameter.

## Source RX statistics and remaining confounding

`D_R` uses only labeled source training packets. Source validation is not used to update its model, queues or group statistics. Matching uses the registered quality bins at 0.25 and 0.75, relative-response RMS threshold 1.0, and absolute relative-CFO threshold 0.1. Groups require at least two observations, retain at most 888 steps of history, and provide at most eight receiver relations per auxiliary update.

The runtime groups by source TX, RX and matching stratum. Every RX aggregate stores a fixed 197-value observed view: 37 receiver descriptors followed by 160 ordered temporal values. This lets the same aggregate provide inputs to available L/T networks without retaining individual source packets. The independent R encoder still reads its 37-value receiver view; this does not change the listed trainable parameter counts.

Action fitting uses group means across RX, not exact sample pairs. For the `multi`, `multi_interaction` and `unified` arms, available L/T networks also predict actions between the matched group views. Their detached predictions define the R fit target:

`target_R = (mean_h_target_RX - mean_h_source_RX) - stop_gradient(predicted_L + predicted_T)`.

The receiver-only control has no L/T networks, so this subtraction is zero and R explains the total matched group change. The main configurations therefore implement an explicit residual decomposition; the receiver-only ablation has a different allocation of responsibility by design. This is a model-dependent residual after the learned L/T actions, not a unique recovery of RX hardware. Transferring controlled-action models to real group changes can be imperfect and must be checked in source diagnostics.

Historical effective group count is capped at 32 before incorporating a new batch, to limit the effect of drifting EMA anchors. Candidate pair selection is deterministic with recent groups first and bounded to 32 candidate pairs in the registered recipe. Identity-side RX supervision compares statistical cross-TX relations using the **total** predicted L+T+R action, with reliability measured against the total group change. It does not force each individual packet onto another receiver's group prototype as if it were an observed counterfactual. Only groups represented by the current labeled batch provide identity gradients; historical sufficient statistics remain detached. The same shared receiver action model serves multiple source TXs.

`D_LT` is fitted only to the exact same-packet controlled factorial observations. No LT target or unobserved factorial RX counterfactual is invented for real source RX groups; the statistical total action above is L+T+R.

Matching reduces selected observable differences but cannot eliminate all source channel-distribution confounding. `D_R` therefore models conditional cross-RX residual variation. It does not claim to identify pure RX hardware, and a source TX×RX relation must not be interpreted as a synchronized multi-receiver recording. Missing eligible groups are skipped and counted, not fabricated. Any stored group state is source-training-only and is absent from deployment inference.

## Optimization and baseline preservation

The original baseline retains its batch-neighbor pseudo-label gate, EMA, cosine schedule, native basic clean/satellite concatenation and original identity objective. Training is 200 epochs × 222 native updates, with the fixed epoch-200 checkpoint evaluated for every registered control. No target result may select a row, checkpoint, perturbation strength or loss weight.

Epochs 1–20 provide the identity cold start without auxiliary updates. Thereafter auxiliary work runs every four native steps with at most 32 labeled source packets. Auxiliary networks have a separate optimizer and schedule, initial learning rate 0.0002 and minimum 0.000001. Their initialization is enclosed in a native-RNG-preserving fork in the runtime. The native identity optimizer and EMA retain their original parameter membership; auxiliary state is separately saved.

The update phases have different gradient permissions:

1. Fit each action model to detached current-run EMA anchors and supervise the L/T parameter estimates. Identity/EMA parameters receive no gradient in this phase.
2. Hold the fitted auxiliary models fixed, detach their proposed deltas and train the identity model through its actual E/G path. The identity objective combines real changed IQ and prescribed feature interventions, rather than relying exclusively on synthetic features.
3. Apply only statistical identity-relation supervision for the RX group path.

Registered identity auxiliary weight is 0.05, paired consistency weight 0.03, and receiver identity weight 0.01. Active mechanisms share the auxiliary budget rather than multiplying total pressure merely by adding branches. Runtime telemetry must expose actual branch execution and losses, not just a configured enable flag. Predictive reliability or missing-group handling must never become an unconstrained learned gate that can suppress all auxiliary supervision.

Concretely, fitting averages active normalized action MSE terms and, for L/T, parameter MSE. The action target's detached global mean-square value, lower-bounded by `1e-4`, normalizes action MSE. The detached reliability is `1/(1+normalized_action_error)`; there is no learned reliability head. For each L/T path, the identity term is `0.05 * reliability * (0.5*(CE_actual_view+CE_predicted_feature) + 0.03*cosine_consistency)`. The consistency target is the unmodified packet's current-run EMA identity representation. The interaction path applies the sum of predicted L, T and LT residuals and compares with the actual ordered `x11` observation; its fit target remains only the factorial residual.

The RX identity term is `0.01 * reliability * (cos(z_a,z_b)-cos(z_a_moved,z_b_moved))²`, averaged over valid distinct-TX relations sharing RX pair and matching stratum. The moved group means use the detached total L+T+R prediction where L/T are present, or R alone in the receiver-only control. Reliability compares this total prediction with the observed total group-mean change. There is no individual RX-paired CE. Identity auxiliary terms are summed and divided by the number of registered active paths, including paths with no eligible RX relation; absence does not strengthen other paths. Extra identity forwards temporarily use evaluation mode while retaining gradients, then restore every module's original training flag, so auxiliary forwards do not update the native running buffers or invoke training dropout. Their RNG effects are enclosed in a CPU/CUDA fork.

All source EMA anchors descend solely from this row's current scratch training. There is no imported teacher, checkpoint, target calibration or query fitting. Strict inference uses the unchanged identity model and classifier; all auxiliary networks and statistical queues are training-only.

## Fixed comparison matrix and scope decisions

The matrix has eight arms × four model seeds `2026092701`–`2026092704`, hence 32 scratch rows:

| Arm | Purpose |
|---|---|
| `legacy_cosine` | Exact original baseline |
| `fixed_interventions` | Known L/T observations and exact detached EMA changes without a learned auxiliary network |
| `unified` | One type-conditioned network for L/T/R tasks |
| `linear` | Independent L constraint only |
| `temporal` | Independent T constraint only |
| `receiver` | Independent RX-statistical constraint only |
| `multi` | Three independent L/T/R networks |
| `multi_interaction` | Three independent main networks plus independent factorial interaction network |

The first matrix implements the recommended main mechanism and its explicit optional interaction extension. The later suggestion of a second physical-plus-residual network for the same factor is not silently substituted for these networks and is not included in this first matrix. It remains a separate future hypothesis.

The fixed-intervention control covers L/T only. Real RX observations do not provide exact packet counterfactuals, so constructing a supposedly exact RX delta would violate the proposed statistical distinction. Consequently fixed versus multi is not a perfectly identical-task comparison; the unified control and receiver single arm provide the additional necessary context.

The additional views and replay operations are extra training exposure and computation even when the native update count is unchanged. Reports must distinguish native steps, auxiliary updates, actual packet/view forwards, replayed feature interventions, RX relations, measured runtime and peak memory. Parameter count alone is not a cost measurement. Missing measured quantities remain N/A rather than being inferred from nominal batch size.

## Required evidence and claim limits

Relevant verification includes real E/G equivalence, finite zero-input gradients, private RNG isolation, correct factorial order, neural-estimator action gradients, independent branch parameters, separated optimizer gradients, source-only statistical grouping, native runtime execution, strict checkpoint/EMA roundtrip and identity-only batch-independent deployment. These checks establish implementation properties, not recognition improvement.

The final experiment must complete source training, fixed lawful freezing, preregistered test prediction and independent truth-last scoring. Existing test results must not influence this implementation or matrix. No successful launch, completed integration, full testing or improved recognition performance is asserted by this design document.
