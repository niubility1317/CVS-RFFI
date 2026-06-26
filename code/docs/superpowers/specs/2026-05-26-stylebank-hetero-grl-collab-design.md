# StyleBank Heterogeneous GRL Collaborative Design

## Goal

Implement the paper-inspired receiver-agnostic collaborative RFFI idea inside the current federated CVS-RFFI stack: each FL receiver client builds local clean plus StyleBank-transferred receiver-style views, trains transmitter classification while adversarially suppressing constructed style/receiver labels, and evaluates virtual collaborative inference by fusing clean and style-view predictions.

## Scope

This is method-equivalent adaptation, not strict reproduction of the paper's multi-hardware deployment. The paper uses multiple physical receivers for collaboration; this implementation uses StyleBank-conditioned virtual receivers during training and evaluation. Satellite-channel robustness remains in the explicit satellite path, not in StyleBank, unless a flag explicitly enables the older experimental style-satellite view.

## Requirements

1. StyleBank training must remain opt-in. Existing non-StyleBank FL behavior must not change.
2. Local StyleBank batches must represent constructed receiver heterogeneity: clean view plus one or more remote receiver-style views.
3. Constructed style labels must be explicit and usable by GRL and Fishr as `d_style`.
4. Receiver-agnostic GRL must avoid double-counting the same adversarial head as both receiver loss and generic adversarial loss.
5. Fishr must activate only when enough constructed style domains exist.
6. Virtual collaborative inference must evaluate clean prediction plus StyleBank-transferred predictions and report base/fused accuracy, rescue, harm, and net gain.
7. Collaborative fusion must support soft mean and adaptive weighting. Adaptive weighting must use confidence and optional style reliability while remaining base-anchored enough to diagnose harm.
8. CLI/config snapshots/logs/metrics must expose the new collaborative evaluation switches and outputs.
9. The FL82 launcher and documentation must include a paper-inspired StyleBank heterogeneous collaborative experiment variant with hard formal defaults: WiSig ratio 0.1, epochs 200, FL rounds 200, client key receiver.
10. Tests must prove the fusion math, parser/config reachability, trainer evaluation reachability, and StyleBank/GRL/Fishr training semantics.

## Architecture

`federated.reliability_fusion` owns pure probability fusion utilities:
- `collaborative_probability_fusion()` accepts one base posterior and N auxiliary posteriors.
- `collaborative_reliability_from_probabilities()` computes per-sample confidence weights from entropy.
- `harm_rescue_report()` remains the common diagnostic.

`FederatedTrainer` owns StyleBank-aware evaluation because it already owns the model, test loaders, StyleBank, and style transform. It will:
- sample remote StyleBank centroids after each round,
- transform each evaluation batch into virtual receiver views,
- run the model on clean and style views,
- fuse probabilities according to config,
- store `global_style_collab_fusion` in logs and summary.

Training keeps the existing `StyleDomainBatch` path. The only semantic tightening is that default StyleBank batches should use constructed sequential `d_style` labels for clean/style views, while metadata keeps raw target receiver labels for diagnostics.

## Data Flow

Training:

```text
client batch x,y,d_raw
  -> optional clean augmentation
  -> StyleBank samples remote receiver styles
  -> style transform creates x_style_1..K
  -> StyleDomainBatch(x=[clean,style...], y=[y,y...], d_style=[0,1..K])
  -> model domain_labels=d_style
  -> CE(tx) + lambda_rx_adv * GRL_CE(d_style) + lambda_fishr * Fishr(logits,y,d_style)
```

Evaluation:

```text
test x,y
  -> clean logits -> p0
  -> StyleBank styles -> x_style_1..K -> p1..pK
  -> soft/adaptive fusion -> p_fused
  -> harm/rescue/net_gain per split and aggregate
```

## Error Handling

Collaborative eval returns `{enabled:false, reason:...}` instead of failing when StyleBank is disabled, empty, or the style transform is missing. Shape mismatches in pure fusion utilities raise `ValueError` because they indicate a code bug.

## Testing

1. Add probability-fusion tests for soft mean and adaptive weighting.
2. Add trainer tests that seed a StyleBank, run one federated round, and assert `style_collab_fusion` is present with harm/rescue keys.
3. Add training-path tests proving constructed `d_style` labels are sequential clean/style view labels and GRL activates on them.
4. Add integration tests for CLI flags, config snapshot keys, metrics CSV keys, docs, and launcher tokens.

## Non-Goals

- No remote launch in this implementation step.
- No claim that the new method reaches 82%/60% until a formal N607 run fully trains and logs are fully parsed.
- No satellite views inside StyleBank by default.
