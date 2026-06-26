# Federated Style Transfer Settings for CVS-RFFI

## Diagnosis

The failed StyleBank run was not only a bad hyperparameter point. Four implementation/configuration problems made the signal ill-conditioned:

1. StyleBank was enabled by default, so non-StyleBank controls could be silently contaminated.
2. The legacy `--use_fl_style_bank_stats` flag could not reliably act as an opt-in alias once the canonical flag was present.
3. In `receiver_agnostic_bex02`, models without an explicit `rx_logits` head reused `adv_dom_logits` for `loss_rx_adv`, then also counted the same logits again as `loss_adv`.
4. CVS satellite classification/consistency losses were disabled whenever a StyleBank `d_style` batch existed, so the intended satellite objective did not train in StyleBank runs.

The style-transfer schedule was also too aggressive for receiver-client FL: replay from round 2, DG from round 3, full replay probability, two remote views, and large RF perturbation bounds. With 10% WiSig data and receiver clients, this expands each local batch before the classifier has learned a stable TX decision surface.

## Default Policy

Style transfer must be opt-in. Plain FL and named non-StyleBank controls should run with:

```bash
--no_use_fed_style_bank --no_use_fl_style_bank_stats
```

or simply omit both flags under the current defaults.

StyleBank should be treated as a late, weak, diagnostic augmentation first. Do not use it as the primary satellite robustness mechanism until clean FL and baseline-view satellite training are stable.

## Conservative StyleBank Setting

Recommended first opt-in setting for FL82-style receiver-client experiments:

```bash
--use_fl_style_bank_stats \
--fl_style_replay_start_round 20 \
--fl_style_phys_start_round 20 \
--fl_style_dg_start_round 40 \
--fl_style_dg_min_domains 2 \
--fl_style_max_views 1 \
--fl_style_replay_prob 0.25 \
--fl_style_phys_jitter_scale 0.25 \
--fl_style_phys_max_gain_delta 0.05 \
--fl_style_phys_max_noise_std 0.01 \
--fl_style_phys_max_cfo_hz 5000 \
--fl_style_phys_max_sro_ppm 25 \
--fl_style_phys_max_iq_gain_db 0.5 \
--fl_style_phys_max_iq_phase_deg 0.5 \
--fl_style_phys_max_phase_noise_std 0.0005 \
--fl_style_phys_min_awgn_snr_db 20 \
--fl_style_phys_p_lowpass 0.2 \
--fl_style_phys_p_multipath 0.2 \
--fl_style_phys_max_multipath_taps 3 \
--fishr_min_domains 2
```

Keep `--use_fed_style_sat_view` off at first. Satellite channel training should stay in the explicit satellite path (`baseline_view` or `cvs_consistency`) rather than being mixed into StyleBank batches, because satellite views are not receiver-style domains.

## Why These Values

- `start_round=20` lets the global classifier and StyleBank collect non-trivial style packets before replay.
- `dg_start_round=40` delays GRL/Fishr on constructed style domains until replay has stopped being purely bootstrap noise.
- `max_views=1` limits local-batch expansion to clean plus one remote-style view. This keeps the class distribution stable and avoids the 3x-4x local objective shock seen in the failed run.
- `replay_prob=0.25` makes style transfer stochastic regularization, not the dominant local batch.
- Small RF perturbation bounds preserve transmitter identity while nudging receiver/channel statistics.
- `fishr_min_domains=2` matches clean plus one style view. If `max_views=2`, use `fishr_min_domains=3`; if `max_views=3`, use `fishr_min_domains=4`.
- Keep `fl_local_epochs=1` for StyleBank probes until the loss curves prove stable. Local3 amplified the collapse in the failed run.

## Recommended Experiment Order

1. Clean receiver-client RA baseline without StyleBank:
   `fedprox + receiver_agnostic_bex02 + cvs_consistency`, local epoch 1.
2. Satellite target baseline without StyleBank:
   `fedprox + receiver_agnostic_bex02 + baseline_view`, all-five satellite scenarios, local epoch 1.
3. Conservative StyleBank probe:
   same as step 2 plus the conservative StyleBank block above.
4. If step 3 is stable through round 80, only then test stronger variants:
   increase `replay_prob` to `0.5`, or increase `max_views` to `2` with `fishr_min_domains=3`. Do not increase both in the same run.

## Paper-Inspired Virtual Collaborative Inference

The receiver-agnostic collaborative RFFI paper trains a transmitter classifier while adversarially removing receiver identity, then performs collaborative inference by fusing probability vectors from multiple receivers. In this FL adaptation, StyleBank provides virtual heterogeneous receivers:

```text
clean x -> p0
StyleBank receiver style 1 -> p1
StyleBank receiver style 2 -> p2
fusion(p0, p1, p2) -> transmitter prediction
```

Enable this diagnostic with:

```bash
--use_style_collab_eval \
--style_collab_views 2 \
--style_collab_fusion adaptive \
--style_collab_base_weight 1.0 \
--style_collab_max_aux_weight 0.75
```

`soft` fusion is the paper-style unweighted mean of clean and virtual receiver probabilities. `adaptive` fusion weights each virtual view by prediction confidence and StyleBank reliability. Use `adaptive` first because StyleBank views are generated approximations, not independent real receivers. The logs report `global_style_collab_fusion` and the metrics CSV includes `style_collab_rescue`, `style_collab_harm`, `style_collab_net_gain`, `style_collab_base_tx_acc`, and `style_collab_fused_tx_acc`.

This is an approximation of collaborative inference, not strict reproduction of physical multi-receiver fusion. A positive signal is fused accuracy improving with low harm after StyleBank has collected mature centroids.

## Metrics to Watch

Stop treating a StyleBank run as promising if any of these appear:

- `train_acc` approaches 0 for multiple rounds.
- `loss_rx_adv` grows monotonically while `loss_cls` also rises.
- `diag_sat_cls_active` or `diag_sat_cons_active` is 0 when the run is supposed to train satellite losses.
- `diag_fishr_active` is 0 after `fl_style_dg_start_round` in a Fishr-enabled StyleBank run.
- `style_batch_views` exceeds the planned clean-plus-style count.
- Clean strict `test_unseen_day_unseen_rx` drops sharply before satellite metrics improve.

The target remains dual-axis: clean strict `test_unseen_day_unseen_rx >= 82%` and clear-LEO unseen-day unseen-RX satellite accuracy at or above 60%.
