# AFFINE_JOINT 完整训练机制诊断

状态：COMPLETE_AFFINE_JOINT_TRAINING_DIAGNOSTICS_DERIVED；run：20261001-phase2-d92-affine-joint-support-m2-r01。

全部训练流已读取，仅报告监督训练机制与实际工作量。

| 模式 | train K | 阶段 | 有信息 | 更新 | RMSCE 初→末 | prox 初→末 | total 初→末 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B | 1 | 1440 | 0 | 0 | N/A | N/A | N/A |
| B | 13 | 80 | 80 | 80 | 1.482545→1.467768 | 0.000000→0.006255 | 1.482545→1.474023 |
| B | 14 | 40 | 40 | 40 | 1.469647→1.454849 | 0.000000→0.006182 | 1.469647→1.461031 |
| B | 3 | 80 | 80 | 80 | 1.642257→1.617452 | 0.000000→0.010195 | 1.642257→1.627647 |
| B | 4 | 40 | 40 | 40 | 1.628084→1.606467 | 0.000000→0.009102 | 1.628084→1.615569 |
| B | 6 | 40 | 40 | 40 | 1.564367→1.544892 | 0.000000→0.008124 | 1.564367→1.553016 |
| B | 7 | 80 | 80 | 80 | 1.557970→1.538715 | 0.000000→0.008016 | 1.557970→1.546732 |
| C_seq | 1 | 1152 | 0 | 0 | N/A | N/A | N/A |
| C_seq | 13 | 64 | 64 | 64 | 2.311936→2.306964 | 0.000000→0.002269 | 2.311936→2.309233 |
| C_seq | 14 | 32 | 32 | 32 | 2.300025→2.294901 | 0.000000→0.002345 | 2.300025→2.297246 |
| C_seq | 3 | 64 | 64 | 64 | 2.448191→2.441598 | 0.000000→0.003040 | 2.448191→2.444639 |
| C_seq | 4 | 32 | 32 | 32 | 2.428173→2.422341 | 0.000000→0.002675 | 2.428173→2.425016 |
| C_seq | 6 | 32 | 32 | 32 | 2.389520→2.383348 | 0.000000→0.002840 | 2.389520→2.386187 |
| C_seq | 7 | 64 | 64 | 64 | 2.377212→2.371063 | 0.000000→0.002796 | 2.377212→2.373859 |

实际阶段 3240 个；曲线记录 20564 个；准备 3240 个。

完整分层见 by_*.csv。C 的内层注册残差与后续 adapter 变化分列；全 support 的 Z=0 final C head 未保存，不能补拟合。
梯度分解为 g_Z−Z；近端不除以 N。固定 nuisance、实际 trace、参考测度与缓存费用分别保留。
实际训练成本保留；outer held 评分及其资源不属于本收集器读取范围。

- Complete independently verified summary is the mathematical certificate. This collector derives descriptions without a second optimizer/head audit, fit, parameter selection or extra hashes/receipts.
- All compact training events, scalar fit-stage logs and summary training objectives are scanned. Full training events supplement identifiers, prior/final-problem preparation metadata and measured distribution summaries omitted by compact_event; raw parent traces and outer feature/score arrays are never opened.
- Class CE means pool all folds by physical class counts BEFORE RMS. Old/new CE and correctness are supervised diagnostics; inner-held labels trained the adapter and are not independent validation.
- CE gradient is g_Z-Z; proximal gradient is Z with no division by physical N. V is fixed, U remains original coordinates and Z uses actual retained rank, including empty coordinates.
- Displacement and the 0.5 accepted-path bound are PRE-TANGENT on current training support. They do not measure real score retention or imply a parameter ball.
- Fixed tau/gamma/s0 and q can be compared per inner head. Actual kernel trace and moving reference centers are measured separately. Null old scale and exact tau=0 are not replaced by epsilon.
- Angles are measured in the core. Mixed distances, kernels, scores and reference means are archived. Tangent/kappa and adapted-only distances are absent and stay null; no new forward is run to invent them.
- C inner Z0 score minus M_held describes the jointly fitted residual kernel plus analytic intercept at inherited U_B; final inner minus initial describes subsequent adapter/closed-head change on the same supervised inner problem. The full final C head at Z0 is not archived, so its isolated full-support registration effect is N/A.
- Final full-support head retains actual B prior and residual measurements, but no independent outer score transitions are calculated. These paired inner changes do not isolate causality or establish generalization.
- Accepted STEP repeats its accepted TRIAL; cached GRADIENT/final displays reuse old head forward times. Resource costs come from final stage/preparation counters and verified resource totals, not sums over curve displays.
- Prior preparation/solve/scoring, student original solves, CE adjoints and baseline primal/EDF work remain separate. Reference pair counters are subsets of raw distance work and must not be added again.
- All thirteen actual RHS/intercept counters are retained, including rejected trials. Head/prior C+1 versus adjoint C columns are archive measurements; n squared times RHS dense work units are proxies, not measured FLOPs.
- Analytic intercept b, Schur z/s and combined RHS are described from saved arrays. Complete companion g_b/T/eta/RHS are described only where present; there is no solve, SVD, new forward, distance evaluation or second mathematical certification.
- Outer held scoring, outer_stats and summary outer resources are excluded rather than used as training diagnostics. Per-stage training costs and bytes are retained; any unpassed hardware/package/transfer/GPU resource remains N/A.
- Real/proxy K1 still fits final closed heads but has no supervised adapter objective; Nnew=0 reuses B and creates no C stage. Actual B repeats are charged; exact physical-binding dedup is descriptive only and does not create independent evidence.
- Persistent training numeric bytes, separately measured deployment numeric bytes, preparation/cache bytes and compressed NPZ bytes differ. Overlapping component bytes/times are not additive peak/work measurements. Package, transfer, GPU, source validation and unpassed runtime fields stay N/A.
