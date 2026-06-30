# PHASE1_GPU0_JOINTSAFE36同批完成分析

生成时间：2026-06-30 10:16-10:35 CST  
批次：`phase1_gpu0_jointsafe36_queue_20260629_0930`  
锚点：`PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SHORT195_S3`

## 证据范围

- 已按项目规则读取`AGENTS.md`和`项目.md`；本批只允许解释为Phase1 source-only SSDG地面训练，不是Stage2-B/C、不是在轨部署成功。
- 已读取本地批次`report.md`和`matrix.json`。
- N607只读preflight通过：直接`N607`可达，项目根目录可见，8张RTX3090可见。
- 远端证据：`runs/phase1_gpu0_jointsafe36_queue_20260629_0930/*/metrics_epoch.csv`、`metrics_epoch.jsonl`、`logs/phase1_gpu0_jointsafe36_queue_20260629_0930/*.out`和`scheduler.out`。
- 解析方式：36个候选的`metrics_epoch.csv`全部读取，合计36 x 200个epoch；stdout按完整文件扫描硬错误标记；没有只看tail。
- 调度器标记：36个candidate、36个launched、36个complete；failed=0。
- 硬错误扫描：`Traceback=0`、`RuntimeError=0`、`unrecognized arguments=0`、`CUDA out of memory=0`、`OOM=0`、`Killed=0`。

## 口径边界

- 所有候选均为`Safe-SSDG-CVS-R01`、`K=0`、`target_visibility=source_only_ground_training_no_target_receiver`、`label_set_relation=Y_old_source_only`。
- `phase2_audit_requested=true`但`phase2_audit_active_loss=false`；`lambda_tx_proto/lambda_rx_proto/lambda_mask_aux/lambda_txrx_rect`等审计权重为0。它只能证明Phase2相关模块可导入和遥测存在，不能证明Stage2适配效果。
- `joint_safe_score`权重来自`code/cvsrffi/ssdg_guard.py`：`val_tx`0.20、`overall_tx`0.20、`strict_udu`0.25、`receiver_floor`0.15、`sat_mean_tx`0.10、`sat_strict_mean`0.10。
- 本报告中的“最强”只在本批Phase1 source-only地面训练指标内成立；不得写成真实卫星验证、Stage2-C seen-new识别或部署成功。

## 结论

1. 按best checkpoint，第一名是`EMA_KEEP15_FISHRSOFT_S8`，E172的`best_score=84.9829`。但它到E200回落到`final_score=82.4554`，strict UDU从84.69降到81.91，receiver floor从77.91降到70.97。因此它是“best checkpoint单点最强”，不是最终稳定锚点。
2. `SOFTPSEUDO_190X10_SHORT195_S3`是本批最值得作为后续基线的候选：best rank第2，`best_score=84.9483`，只比第1低0.0346；同时best=final=E200，是最终epoch全批第1。它的优势不是某个单项极值，而是overall 90.34、strict UDU 84.14、receiver floor 76.24、sat mean 76.62、sat strict mean 70.45的联合平衡。
3. `SHORT195_S3`这个sweep整体最稳：4个家族平均`best_score=83.8798`，平均`final_score=82.9183`，总drop guard epoch只有5。相比之下`FISHRSOFT_S8`有最高单点，但平均final只有81.0401且drop guard epoch为23。
4. 家族平均上，`SOFTPSEUDO_190X10`最好：平均`best_score=83.7006`、平均`final_score=82.4332`，也是唯一让锚点在E200继续刷新best的家族。`EMA_KEEP15`有最高单点和较好sat strict，但最终稳定性不如`SOFTPSEUDO_190X10`。
5. 本批没有PAIC guard触发；一共有159个one-epoch drop guard epoch。drop guard不是运行失败，而是说明许多候选后段有明显单epoch保护性回落，不能只看最终文件或单项最大值。

## 家族聚合

| family | n | mean best_score | max best_score | mean final_score | best candidate | mean strict | mean floor | mean sat_strict | drop_guard_epochs |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `SOFTPSEUDO_190X10` | 9 | 83.7006 | 84.9483 | 82.4332 | `SOFTPSEUDO_190X10_SHORT195_S3` | 83.6789 | 74.9185 | 67.1615 | 32 |
| `EMA_KEEP15` | 9 | 83.5969 | 84.9829 | 82.2969 | `EMA_KEEP15_FISHRSOFT_S8` | 83.2100 | 74.0759 | 68.4815 | 42 |
| `SATSOFT_NO_CONS` | 9 | 83.4053 | 84.3496 | 81.5291 | `SATSOFT_NO_CONS_TAU88_S2` | 83.1182 | 74.1380 | 67.0569 | 51 |
| `GROUPSOFT_190X10` | 9 | 83.2818 | 84.5436 | 81.2470 | `GROUPSOFT_190X10_DOMAINFIRM_S7` | 82.6700 | 73.6713 | 68.3084 | 34 |

## sweep聚合

| sweep | n | mean best_score | max best_score | mean final_score | best candidate | mean strict | mean floor | mean sat_strict | drop_guard_epochs |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `SHORT195_S3` | 4 | 83.8798 | 84.9483 | 82.9183 | `SOFTPSEUDO_190X10_SHORT195_S3` | 83.7329 | 73.8834 | 68.7952 | 5 |
| `DOMAINSOFT_S6` | 4 | 83.7743 | 84.3901 | 82.5820 | `GROUPSOFT_190X10_DOMAINSOFT_S6` | 83.4488 | 75.3000 | 67.4118 | 19 |
| `BASE_S0` | 4 | 83.7657 | 84.2722 | 82.3335 | `EMA_KEEP15_BASE_S0` | 84.2267 | 74.9416 | 66.7227 | 17 |
| `DOMAINFIRM_S7` | 4 | 83.6763 | 84.5436 | 82.7159 | `GROUPSOFT_190X10_DOMAINFIRM_S7` | 82.5996 | 74.9458 | 68.9157 | 12 |
| `TAU88_S2` | 4 | 83.6619 | 84.3496 | 81.1804 | `SATSOFT_NO_CONS_TAU88_S2` | 82.6116 | 74.6437 | 69.4633 | 19 |
| `SEED_S1` | 4 | 83.5581 | 84.0275 | 82.4607 | `SOFTPSEUDO_190X10_SEED_S1` | 83.2979 | 74.8146 | 66.2666 | 19 |
| `FISHRSOFT_S8` | 4 | 83.5300 | 84.9829 | 81.0401 | `EMA_KEEP15_FISHRSOFT_S8` | 83.2554 | 72.9125 | 68.7121 | 23 |
| `SATLOW_S5` | 4 | 82.8713 | 83.5332 | 80.6774 | `SATSOFT_NO_CONS_SATLOW_S5` | 82.7383 | 72.8125 | 67.0608 | 23 |
| `MID188_S4` | 4 | 82.7477 | 84.1110 | 80.9806 | `SATSOFT_NO_CONS_MID188_S4` | 82.6121 | 73.5542 | 66.4207 | 22 |

## 全36行best checkpoint联合排序

`final_pseudo`格式为`correct/selected/total`，来自E200训练遥测。

| rank | candidate | family | sweep | best_epoch | best_score | overall | strict_udu | receiver_floor | sat_mean | sat_strict_mean | final_score | drop_guard_epochs | final_pseudo |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_FISHRSOFT_S8` | `EMA_KEEP15` | `FISHRSOFT_S8` | 172 | 84.9829 | 89.30 | 84.69 | 77.91 | 75.54 | 70.18 | 82.4554 | 6 | 7515/7518/8320 |
| 2 | `PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SHORT195_S3` | `SOFTPSEUDO_190X10` | `SHORT195_S3` | 200 | 84.9483 | 90.34 | 84.14 | 76.24 | 76.62 | 70.45 | 84.9483 | 1 | 7554/7554/8320 |
| 3 | `PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_FISHRSOFT_S8` | `SOFTPSEUDO_190X10` | `FISHRSOFT_S8` | 176 | 84.6576 | 90.50 | 85.23 | 76.13 | 73.79 | 67.41 | 82.0107 | 4 | 7517/7518/8320 |
| 4 | `PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_DOMAINFIRM_S7` | `GROUPSOFT_190X10` | `DOMAINFIRM_S7` | 185 | 84.5436 | 89.30 | 83.32 | 77.68 | 75.15 | 69.47 | 83.4856 | 2 | 7369/7372/8320 |
| 5 | `PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_DOMAINSOFT_S6` | `GROUPSOFT_190X10` | `DOMAINSOFT_S6` | 163 | 84.3901 | 89.58 | 84.55 | 76.89 | 73.77 | 67.16 | 82.5358 | 5 | 7427/7432/8320 |
| 6 | `PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_TAU88_S2` | `SATSOFT_NO_CONS` | `TAU88_S2` | 189 | 84.3496 | 89.46 | 84.33 | 77.51 | 73.82 | 66.75 | 82.2493 | 7 | 7560/7560/8320 |
| 7 | `PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_BASE_S0` | `EMA_KEEP15` | `BASE_S0` | 199 | 84.2722 | 88.92 | 84.69 | 75.65 | 74.87 | 67.48 | 82.7567 | 5 | 7525/7527/8320 |
| 8 | `PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_MID188_S4` | `SATSOFT_NO_CONS` | `MID188_S4` | 188 | 84.1110 | 89.20 | 84.89 | 77.17 | 72.37 | 65.27 | 81.0935 | 8 | 7341/7345/8320 |
| 9 | `PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_SHORT195_S3` | `EMA_KEEP15` | `SHORT195_S3` | 166 | 84.0587 | 89.25 | 84.25 | 76.44 | 73.45 | 66.44 | 82.5421 | 1 | 7490/7493/8320 |
| 10 | `PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SEED_S1` | `SOFTPSEUDO_190X10` | `SEED_S1` | 173 | 84.0275 | 89.54 | 84.33 | 74.33 | 74.15 | 67.48 | 82.5060 | 5 | 7495/7495/8320 |
| 11 | `PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_DOMAINSOFT_S6` | `EMA_KEEP15` | `DOMAINSOFT_S6` | 195 | 83.9084 | 89.13 | 84.39 | 75.16 | 73.29 | 66.42 | 81.9861 | 5 | 7518/7519/8320 |
| 12 | `PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_BASE_S0` | `SATSOFT_NO_CONS` | `BASE_S0` | 149 | 83.8468 | 89.54 | 85.30 | 76.35 | 70.48 | 64.05 | 81.0508 | 6 | 7468/7470/8320 |
| 13 | `PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_DOMAINFIRM_S7` | `SOFTPSEUDO_190X10` | `DOMAINFIRM_S7` | 198 | 83.8005 | 88.30 | 83.58 | 74.64 | 75.07 | 68.45 | 81.8084 | 4 | 7482/7482/8320 |
| 14 | `PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_SEED_S1` | `GROUPSOFT_190X10` | `SEED_S1` | 165 | 83.6325 | 89.34 | 83.14 | 77.93 | 72.13 | 64.01 | 82.9823 | 3 | 7458/7460/8320 |
| 15 | `PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_TAU88_S2` | `EMA_KEEP15` | `TAU88_S2` | 199 | 83.6262 | 87.66 | 82.22 | 73.53 | 77.11 | 71.23 | 82.1926 | 5 | 7392/7398/8320 |
| 16 | `PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_TAU88_S2` | `GROUPSOFT_190X10` | `TAU88_S2` | 194 | 83.5987 | 87.60 | 82.56 | 73.62 | 76.72 | 70.76 | 78.4545 | 4 | 7413/7414/8320 |
| 17 | `PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_BASE_S0` | `SOFTPSEUDO_190X10` | `BASE_S0` | 158 | 83.5444 | 89.22 | 84.86 | 73.86 | 71.90 | 65.59 | 82.8060 | 3 | 7479/7482/8320 |
| 18 | `PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_SATLOW_S5` | `SATSOFT_NO_CONS` | `SATLOW_S5` | 179 | 83.5332 | 88.61 | 83.50 | 73.48 | 74.54 | 67.68 | 81.6190 | 6 | 7405/7407/8320 |
| 19 | `PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_DOMAINSOFT_S6` | `SATSOFT_NO_CONS` | `DOMAINSOFT_S6` | 195 | 83.4778 | 88.28 | 82.50 | 74.12 | 75.23 | 68.57 | 82.5483 | 7 | 7647/7648/8320 |
| 20 | `PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_DOMAINFIRM_S7` | `EMA_KEEP15` | `DOMAINFIRM_S7` | 196 | 83.4737 | 88.38 | 82.34 | 73.18 | 75.62 | 69.52 | 83.3414 | 4 | 7491/7491/8320 |
| 21 | `PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_BASE_S0` | `GROUPSOFT_190X10` | `BASE_S0` | 178 | 83.3994 | 87.99 | 82.06 | 73.91 | 75.70 | 69.78 | 82.7206 | 3 | 7509/7516/8320 |
| 22 | `PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_SEED_S1` | `SATSOFT_NO_CONS` | `SEED_S1` | 184 | 83.3951 | 89.25 | 82.69 | 76.31 | 72.41 | 64.72 | 81.7691 | 6 | 7502/7504/8320 |
| 23 | `PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_DOMAINSOFT_S6` | `SOFTPSEUDO_190X10` | `DOMAINSOFT_S6` | 183 | 83.3211 | 88.43 | 82.35 | 75.03 | 74.06 | 67.49 | 83.2579 | 2 | 7475/7479/8320 |
| 24 | `PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_SHORT195_S3` | `GROUPSOFT_190X10` | `SHORT195_S3` | 171 | 83.3210 | 87.99 | 83.70 | 72.67 | 74.24 | 67.83 | 82.4954 | 1 | 7502/7503/8320 |
| 25 | `PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_SHORT195_S3` | `SATSOFT_NO_CONS` | `SHORT195_S3` | 198 | 83.1912 | 87.52 | 82.84 | 70.18 | 77.18 | 70.47 | 81.6875 | 2 | 7373/7378/8320 |
| 26 | `PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_SEED_S1` | `EMA_KEEP15` | `SEED_S1` | 194 | 83.1775 | 88.98 | 83.03 | 70.69 | 74.47 | 68.86 | 82.5857 | 5 | 7483/7483/8320 |
| 27 | `PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_TAU88_S2` | `SOFTPSEUDO_190X10` | `TAU88_S2` | 181 | 83.0731 | 87.61 | 81.34 | 73.92 | 75.64 | 69.11 | 81.8250 | 3 | 7496/7496/8320 |
| 28 | `PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_SATLOW_S5` | `GROUPSOFT_190X10` | `SATLOW_S5` | 198 | 83.0088 | 88.31 | 81.87 | 71.18 | 76.17 | 69.61 | 80.2074 | 5 | 7418/7423/8320 |
| 29 | `PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SATLOW_S5` | `SOFTPSEUDO_190X10` | `SATLOW_S5` | 142 | 82.9773 | 88.05 | 84.46 | 73.91 | 70.24 | 64.68 | 80.0112 | 5 | 7478/7478/8320 |
| 30 | `PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_MID188_S4` | `SOFTPSEUDO_190X10` | `MID188_S4` | 152 | 82.9552 | 88.04 | 82.80 | 76.20 | 71.47 | 63.79 | 82.7254 | 5 | 7498/7500/8320 |
| 31 | `PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_MID188_S4` | `EMA_KEEP15` | `MID188_S4` | 193 | 82.9061 | 87.17 | 82.16 | 71.45 | 75.32 | 69.94 | 81.9401 | 4 | 7391/7397/8320 |
| 32 | `PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_DOMAINFIRM_S7` | `SATSOFT_NO_CONS` | `DOMAINFIRM_S7` | 195 | 82.8873 | 87.65 | 81.15 | 74.28 | 74.10 | 68.21 | 82.2284 | 2 | 7594/7597/8320 |
| 33 | `PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_FISHRSOFT_S8` | `GROUPSOFT_190X10` | `FISHRSOFT_S8` | 199 | 82.6236 | 87.32 | 82.23 | 69.77 | 75.50 | 69.47 | 80.1782 | 6 | 7279/7287/8320 |
| 34 | `PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_SATLOW_S5` | `EMA_KEEP15` | `SATLOW_S5` | 197 | 81.9660 | 86.28 | 81.11 | 72.67 | 72.50 | 66.27 | 80.8720 | 7 | 7497/7497/8320 |
| 35 | `PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_FISHRSOFT_S8` | `SATSOFT_NO_CONS` | `FISHRSOFT_S8` | 176 | 81.8558 | 87.81 | 80.87 | 67.84 | 74.46 | 67.79 | 79.5164 | 7 | 7504/7507/8320 |
| 36 | `PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_MID188_S4` | `GROUPSOFT_190X10` | `MID188_S4` | 185 | 81.0186 | 84.27 | 80.60 | 69.39 | 72.77 | 66.68 | 78.1633 | 5 | 7514/7515/8320 |

## 锚点SHORT195_S3细节

`SOFTPSEUDO_190X10_SHORT195_S3`从E195 label末端到E200 pseudo末端继续提升，且E200同时是best和final。

| metric | E195 label | E200 final | delta |
|---|---:|---:|---:|
| `protected_val_tx` | 98.3631 | 98.5060 | +0.1429 |
| `protected_overall_tx` | 89.2270 | 90.3402 | +1.1132 |
| `protected_strict_udu` | 82.1217 | 84.1433 | +2.0217 |
| `protected_receiver_floor` | 73.1917 | 76.2417 | +3.0500 |
| `protected_sat_mean_tx` | 76.2333 | 76.6199 | +0.3866 |
| `protected_sat_floor_tx` | 74.9755 | 75.3877 | +0.4123 |
| `protected_sat_strict_mean` | 68.9678 | 70.4500 | +1.4822 |
| `protected_sat_strict_floor` | 67.7850 | 69.2400 | +1.4550 |
| `train_pseudo_correct/selected/total` | 0/0/0 | 7554/7554/8320 | selected precision 100.00%, coverage 90.79% |

E200 satellite分项：

| scenario | aggregate_tx_acc | strict_udu |
|---|---:|---:|
| `leo_clear_weak` | 78.7338 | 72.4167 |
| `leo_low_elev_weak` | 75.7382 | 69.6933 |
| `leo_rain_weak` | 75.3877 | 69.2400 |

E200 named test中，`receiver_floor=76.2417`来自`test_unseen_day_rx_8`；次低为`test_unseen_day_rx_11=76.2500`。这说明当前瓶颈不是普通seen-day receiver测试，而是unseen-day下的个别target-like receiver组合。

## best与final不一致的风险

| best-final gap | candidate | best_epoch | best_score | final_score | strict best->final | floor best->final | sat_strict best->final |
|---:|---|---:|---:|---:|---|---|---|
| 5.1443 | `GROUPSOFT_190X10_TAU88_S2` | 194 | 83.5987 | 78.4545 | 82.56->76.78 | 73.62->60.38 | 70.76->65.62 |
| 3.0175 | `SATSOFT_NO_CONS_MID188_S4` | 188 | 84.1110 | 81.0935 | 84.89->79.42 | 77.17->72.01 | 65.27->64.11 |
| 2.9660 | `SOFTPSEUDO_190X10_SATLOW_S5` | 142 | 82.9773 | 80.0112 | 84.46->79.60 | 73.91->64.34 | 64.68->64.17 |
| 2.8553 | `GROUPSOFT_190X10_MID188_S4` | 185 | 81.0186 | 78.1633 | 80.60->76.70 | 69.39->61.11 | 66.68->65.01 |
| 2.8014 | `GROUPSOFT_190X10_SATLOW_S5` | 198 | 83.0088 | 80.2074 | 81.87->78.86 | 71.18->66.31 | 69.61->65.13 |
| 2.7961 | `SATSOFT_NO_CONS_BASE_S0` | 149 | 83.8468 | 81.0508 | 85.30->79.27 | 76.35->70.09 | 64.05->65.61 |
| 2.6469 | `SOFTPSEUDO_190X10_FISHRSOFT_S8` | 176 | 84.6576 | 82.0107 | 85.23->80.43 | 76.13->71.98 | 67.41->65.58 |
| 2.5276 | `EMA_KEEP15_FISHRSOFT_S8` | 172 | 84.9829 | 82.4554 | 84.69->81.91 | 77.91->70.97 | 70.18->67.80 |

这些候选可以保留best checkpoint作后续诊断，但如果下游需要“训练到E200仍稳定”的地面表征基线，不能优先于`SOFTPSEUDO_190X10_SHORT195_S3`。

## 建议

1. 后续`z_id`特征空间桥接继续以`SOFTPSEUDO_190X10_SHORT195_S3`为主基线，原因是它在全批best几乎并列第一，同时是final第一且E195->E200伪标签阶段有正收益。
2. 可保留`EMA_KEEP15_FISHRSOFT_S8`作为best-checkpoint对照，但报告中必须写明它的E200回落和best-final gap，不应把它作为稳定final基线。
3. 若要追求satellite strict上限，可单独审计`EMA_KEEP15_TAU88_S2`和`GROUPSOFT_190X10_TAU88_S2`，但它们的source strict/floor与final稳定性不足，不能直接替代主线。
4. 下一步若进入Stage2-B/C，必须重新按`项目.md`检查target receiver domain、old/new/unknown TX互斥、K-shot support/query权限和unknown query eval-only边界；本批Phase1不能直接推出Stage2部署成功。
