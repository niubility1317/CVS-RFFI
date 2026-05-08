# Findings

This file stores extracted evidence for the CVS-RFFI version/log analysis.

## Source Map
- Root code is newest active integration: `train.py` includes SGC presets, SSDG-SSL presets, slimming presets, satellite evaluation hooks, and current docs.
- `type10-4` contains 4.23/4.24/4.26/4.27 code snapshots plus full ablation/silming logs and prior parsed outputs.
- `type10-7/CV-SincNet` contains the SAT ablation run set with 40 satellite-channel robustness logs.
- `unkown` contains a more experimental satellite-module upgrade (`sat_hybrid_*`, classic DG losses, baselines) and a report, but no matching completed training logs in this root workspace.
- Root `logs/` from 2026-05-06 are not usable as model evidence: most are `python: command not found`, dry-run lines, empty logs, or short launcher records.

## Experiment Metrics
- Parsed 183 log files, 110 with train epochs, 6 empty.
- Current best primary OOD among parsed logs: `SAT37_r19_fishr`, score 87.95, strict UDU 86.43, overall 90.77, params 1.672M.
- Very close runner-up: `SAT34_r19_groupdro_smooth`, score 87.94, strict UDU 86.44, overall 90.72, params 1.672M.
- Best compact SAT candidate: `SAT07_r25_compact_sat_mixed`, score 87.85, strict UDU 86.27, overall 90.79, params 1.050M.
- Highest SAT scenario average among final-primary checkpoints: `SAT13_r19_mixed_high_weight`, avg 43.91 across clear/low_elev/rain/storm/mixed, but this satellite absolute accuracy remains far below clean OOD.
- Root SGC adapter presets have docs and tests but no completed training evidence in `logs/`.

## Code Route Notes
- The proven backbone route is `DualCVSincNetDisentangle` Lite-B/Lite-D with `branch_ablation=no_dac`, conservative MixStyle, PA auxiliary path retained, DAC path disabled.
- Removing DAC is consistently supported; removing stats/RCN or core time/frequency paths is risky.
- SAT consistency training improves satellite-eval averages only modestly and can trade clean/OOD depending on weight and start epoch.
- SGC adapter is architecturally promising and integrated in current root code, but current logs do not yet validate it.

## Recommendation Evidence
- Primary recommendation should be evidence-first: use R19 Lite-B no-DAC + Fishr or GroupDRO as main clean/OOD route; use R25 Lite-D no-DAC SAT-mixed as deployment/compact candidate.
- SGC adapter should be the next experimental route, not the selected best model, until a full source->augment->adapt chain completes successfully.

## Current SGC Analysis Notes
- Current root files relevant to SGC: `sgc_adapter.py`, `sgc_losses.py`, `sat_channel.py`, `training_controls.py`, `train.py`, `model_dual_cvsincnet.py`, `docs/SGC_EXPERIMENTS.md`, and SGC design/plan docs under `docs/superpowers/`.
- `rg` is unavailable due to access denial in this workspace; PowerShell search is used for current evidence gathering.
- The current root implementation should be treated as primary; historical copies under `type10-*` and `unkown/` are useful only for comparison.
- `sgc_adapter.py` implements four shape-preserving blocks: per-sample RMS normalization, CNN-based normalized CFO/Doppler compensation, FFT-domain soft spectral masking with residual blending, and a depthwise residual channel compensator with learnable `gamma` initialized to 0.
- `sat_channel.py` simulates satellite-ground effects that are relevant to RFFI: elevation/slant-range path loss, LOS/LOO/Rayleigh state, atmospheric fading, Doppler plus CFO, phase noise, optional multipath, mild AGC, AWGN, and IQ imbalance.
- `train.py` wires SGC as an opt-in preset before both backbones, supports source/sgc_augment/sgc_adapt stages, adds satellite augmented views for feature/classification consistency, and can freeze all non-adapter parameters in `sgc_adapt`.
- `sgc_losses.py` defines extra adaptation helpers, but current `train.py` only imports/uses `residual_regularization`; pseudo-label and entropy CLI flags are currently reserved rather than wired into the main loss.
- Root `logs/*sgc*` files are tiny launcher/startup records or failures, not completed training evidence; SGC conclusions here are architectural/code-level rather than empirically validated in this workspace.

## 2026-05-07 5.7 SGC Evidence
- `5.7` code snapshot matches current root `train.py`, `sgc_adapter.py`, and `model_dual_cvsincnet.py`; the useful new evidence is the completed training logs, not a new code branch.
- `5.7/logs` contains 32 complete SGC logs: 11 source, 11 augment, and 10 adapt.
- 5.7 clean/OOD winner is `sgc_baseline_no_adapter source`: Primary 88.22, strict UDU 85.91, overall 90.53, worst-RX 85.07, SAT Avg 38.31.
- Full SGC from source is negative versus no-adapter baseline: `sgc_lite_b_no_dac source` Primary -1.92, strict UDU -1.96, overall -1.89, worst-RX -7.92, SAT Avg -3.34.
- SAT augment itself is effective but costly: no-adapter source to augment changes SAT Avg +6.31, Primary -1.10, strict UDU -1.60, worst-RX -3.78.
- Full SGC augment adds only SAT Avg +1.49 versus no-adapter augment while hurting Primary -1.69, strict UDU -1.49, overall -1.89, worst-RX -5.87.
- `no_amp` is the only Lite-B SGC augment variant with weak positive clean/UDU signal versus no-adapter augment: Primary +0.24, strict UDU +0.63, but worst-RX -4.32 and SAT Avg -0.58.
- Current `sgc_adapt` is not validated: it freezes non-adapter params, trains only adapter params, and logs `LOSS-SAT cls_sat=0.0000 sat_cons=0.0000`; average adapt metrics are below augment.
- Residual is not inert: `residual_only - no_res` gives source SAT Avg +8.52 and augment SAT Avg +1.44, with small Primary improvements, but still hurts worst-RX. Residual should be studied as a conservative gated mechanism after a strong backbone checkpoint.

## 2026-05-08 5.8 Evidence
- `5.8/logs` contains 19 complete seed-1337 logs: 4 B-group model/SAT-label-smoothing variants, 4 C-group ECC variants, 6 D-group domain disentanglement ablations, and 5 E-group SGC follow-ups.
- Best development primary score is `E2_residual_only_std_res001`: Primary 88.24, overall 90.70, strict UDU 86.92, worst-RX 86.99, SAT Avg 41.58.
- Best pre-SGC source by primary is `D1_domain_enhancer_off`: Primary 87.87, overall 90.52, strict UDU 86.45. The automatic SGC source selector chose this checkpoint as SRC-P.
- `D4_domain_no_pa_no_stats` is also strong: Primary 87.65, overall 90.75, strict UDU 85.99, worst-RX 86.13. This challenges the prior assumption that PA auxiliary loss is always necessary.
- ECC did not win: `C2_A1_ecc003_satmain` is only slightly better than B1 on primary (+0.10), and `C4_A2_ecc003_satmain` strongly hurts SAT Avg.
- Residual-only SGC is the useful SGC direction in 5.8: `E2` improves over `E0_no_adapter_continue` by +0.40 primary, +0.45 strict UDU, +0.48 worst-RX, and +0.73 SAT Avg.
- Full SGC remains negative in this setup: `E4_full_sgc_mild_res001` trails E0 by -0.23 primary and -0.56 worst-RX.
- Rigor warning: 5.8 primary rankings and Phase-E source selection use test-derived metrics. They are valid for development analysis but should not be reported as final unbiased test results.

## 2026-05-08 SGC Optimization Evidence
- User clarified that SGC must do more than improve SAT overlay accuracy: it should suppress satellite-ground channel interference and support on-orbit target-domain adaptation with unlabeled target samples.
- Literature direction: Radio Transformer Networks and DeepRx support learnable receiver/channel transformations, but their communication objectives differ from RFFI; SGC needs fingerprint-preserving residual constraints.
- RFFI domain-generalization/source-free adaptation literature supports using unlabeled target receiver/domain samples, but final evaluation data must be separated from calibration/adaptation data to avoid leakage.
- Code change direction: SAT evaluation should default to all named test loaders and report per-split SAT results; residual-only SGC should be expanded with bounded gamma, multiscale residual kernels, channel-stat gating, and residual diagnostics.
