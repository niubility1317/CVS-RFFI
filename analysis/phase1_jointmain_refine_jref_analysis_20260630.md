# Phase1 JREF Completion Analysis 20260630

This note mirrors the completed JREF analysis from the local automation report. JREF belongs to `phase1_jointmain_refine_20260630` and remains a Phase1 source-only ground-training experiment. It does not use target receiver support/query, unknown query thresholding, Stage2-B/C enrollment, or real satellite deployment evidence.

## Evidence

| Evidence | Path/status |
|---|---|
| Remote process | No active `phase1_jointmain_refine_20260630` process found |
| Logs | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_jointmain_refine_20260630/JREF_*.out`, 14 files |
| Metrics | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_jointmain_refine_20260630/*/metrics_epoch.csv`, 14 files |
| Prototypes | Each candidate exported `phase2_zid_prototypes.json/.pt` |
| Completion | 220-epoch candidates have 220 rows; 240-epoch candidates have 240 rows |
| Error scan | Full stdout scan found no `Traceback`, `RuntimeError`, CUDA OOM, `Killed`, `ValueError`, or unrecognized arguments |

SHORT195_S3 reference: overall 90.34, strict UDU 84.14, receiver floor 76.24, satellite floor 75.39.

## Same-Row Ranking

| rank | candidate | lineage | best epoch | overall | strict UDU | receiver floor | sat floor | sat mean | sat strict floor | delta overall vs SHORT | delta sat floor vs SHORT | final overall | final sat floor | fused components | proto radius max | p95 | p99 | overflow | final drop guard |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `JREF_C5_C10_FLOOR_M4_E220` | C10 | 210 | 89.85 | 83.97 | 72.15 | 75.48 | 76.87 | 69.42 | -0.49 | +0.09 | 88.77 | 74.76 | 8 | 8.80 | 52.75 | 75.28 | 0.477 | False |
| 2 | `JREF_C9_MULTICOMP_M2_E220` | multi | 208 | 88.73 | 83.86 | 76.42 | 74.62 | 75.92 | 69.25 | -1.61 | -0.77 | 87.44 | 74.09 | 23 | 8.08 | 52.31 | 77.73 | 0.587 | True |
| 3 | `JREF_C7_C3_SAT_M3_E220` | C3 | 200 | 89.66 | 83.51 | 71.66 | 74.19 | 75.42 | 68.04 | -0.68 | -1.20 | 86.79 | 72.51 | 14 | 9.44 | 56.14 | 77.78 | 0.596 | True |
| 4 | `JREF_C11_TAILSOFT_E220` | tail | 210 | 88.34 | 83.34 | 72.44 | 76.58 | 77.91 | 69.90 | -2.00 | +1.19 | 87.30 | 75.43 | 10 | 10.63 | 55.35 | 78.91 | 0.590 | False |
| 5 | `JREF_C12_TAILMID_E240` | tail | 200 | 88.48 | 82.42 | 72.67 | 76.28 | 77.26 | 69.71 | -1.86 | +0.89 | 87.03 | 76.86 | 10 | 9.83 | 54.72 | 76.83 | 0.493 | False |
| 6 | `JREF_C6_C10_SATSAFE_E240` | C10 | 160 | 88.59 | 82.91 | 76.86 | 71.30 | 72.43 | 64.05 | -1.75 | -4.09 | 87.83 | 73.13 | 8 | 8.38 | 59.40 | 79.06 | 0.549 | False |
| 7 | `JREF_C13_CONSERVE_E240` | conservative | 228 | 88.46 | 82.97 | 71.24 | 75.16 | 76.39 | 69.03 | -1.88 | -0.23 | 87.85 | 75.28 | 12 | 9.47 | 56.89 | 78.05 | 0.516 | False |
| 8 | `JREF_C3_C8_M4_SC34_E240` | C8 | 232 | 87.96 | 81.43 | 73.35 | 74.81 | 76.14 | 68.20 | -2.38 | -0.58 | 86.08 | 73.28 | 6 | 9.83 | 55.91 | 78.24 | 0.395 | True |
| 9 | `JREF_C4_C10_FLOOR_M3_E220` | C10 | 212 | 88.23 | 82.51 | 72.61 | 72.46 | 73.81 | 65.81 | -2.11 | -2.93 | 87.14 | 72.19 | 8 | 8.08 | 54.80 | 78.77 | 0.592 | False |
| 10 | `JREF_C1_C8_M4_SC30_E220` | C8 | 206 | 87.84 | 81.76 | 73.87 | 72.74 | 74.03 | 65.08 | -2.50 | -2.65 | 85.58 | 71.46 | 7 | 9.68 | 51.96 | 77.00 | 0.568 | True |
| 11 | `JREF_C2_C8_M3_SC32_E240` | C8 | 222 | 87.54 | 82.46 | 69.36 | 74.33 | 75.37 | 67.44 | -2.80 | -1.06 | 86.19 | 75.40 | 11 | 7.91 | 51.96 | 75.87 | 0.481 | False |
| 12 | `JREF_C8_C3_SAT_M4_E240` | C3 | 228 | 87.59 | 82.10 | 69.01 | 74.18 | 75.24 | 67.78 | -2.75 | -1.21 | 87.27 | 73.74 | 6 | 8.70 | 56.01 | 76.66 | 0.505 | False |
| 13 | `JREF_C0_C8_M3_SC30_E220` | C8 | 180 | 87.72 | 82.00 | 69.72 | 71.79 | 73.08 | 64.19 | -2.62 | -3.60 | 85.88 | 72.40 | 12 | 10.50 | 59.03 | 81.10 | 0.624 | False |
| 14 | `JREF_C10_MULTICOMP_M25_E220` | multi | 210 | 86.60 | 81.81 | 70.94 | 73.95 | 75.22 | 67.69 | -3.74 | -1.44 | 86.60 | 74.75 | 15 | 10.55 | 53.25 | 76.35 | 0.495 | False |

## Interpretation

JREF does not produce a replacement for SHORT195_S3. The best same-row candidate is `JREF_C5_C10_FLOOR_M4_E220`, but it only beats SHORT195_S3 on satellite floor by 0.09 pp while losing 0.49 pp overall, 0.17 pp strict UDU, and 4.09 pp receiver floor.

`JREF_C9_MULTICOMP_M2_E220` is the strongest mechanism diagnostic. It keeps 23 fused components and reaches receiver floor 76.42, slightly above SHORT195_S3, showing that preserving local domain modes helps weak receivers. Its overall and satellite floor are still lower, so it is not a mainline model.

`JREF_C11` and `JREF_C12` are useful satellite-floor diagnostics. They improve satellite floor to 76.58 and 76.28, but classification and receiver-floor metrics are not strong enough for promotion.

C8-line variants did not recover the prior C8/SHORT195 strengths. Adjusting merge angle and source cap alone is insufficient.

Several candidates peak before final and then degrade after the pseudo phase. Future ranking should use best same-row metrics and separately diagnose late pseudo-stage regression.

## Decision

Do not promote JREF as Phase1 mainline success or Stage2 deployment evidence. Use `JREF_C5` as a satellite-floor repair control, `JREF_C9` as a local-mode receiver-floor diagnostic, and `JREF_C11/C12` as satellite robustness schedule evidence. The next real route should be the V2 unknowncompact/local-mode accept design rather than expanding C8-style JREF tuning.
