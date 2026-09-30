# FCR8 完整训练机制诊断

状态：COMPLETE_FCR8_TRAINING_DIAGNOSTICS_DERIVED；run：20260930-phase2-d92-fcr8-support-m2-r01。

全部训练事件已读取。本文只解释监督训练机制，不报告外层评分，不调整冻结方法。

| 模式 | train K | 阶段 | 有信息 | 有更新 | 零更新有信息 | task 初→末 | keep 初→末 | proximal 初→末 | total 初→末 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 1 | 1400 | 0 | 0 | 0 | N/A | N/A | N/A | N/A |
| B | 3 | 80 | 80 | 80 | 0 | 0.438342→0.390547 | 0.006531→0.003404 | 0.000000→0.017866 | 0.444873→0.411816 |
| B | 4 | 40 | 40 | 40 | 0 | 0.424009→0.384079 | 0.006988→0.003985 | 0.000000→0.015625 | 0.430996→0.403689 |
| B | 6 | 40 | 40 | 40 | 0 | 0.379977→0.346241 | 0.007426→0.004907 | 0.000000→0.013138 | 0.387403→0.364286 |
| B | 7 | 80 | 80 | 80 | 0 | 0.363214→0.331455 | 0.004020→0.002668 | 0.000000→0.012250 | 0.367234→0.346373 |
| B | 13 | 80 | 80 | 80 | 0 | 0.303884→0.282671 | 0.003535→0.002496 | 0.000000→0.008599 | 0.307419→0.293766 |
| B | 14 | 40 | 40 | 40 | 0 | 0.290896→0.270878 | 0.002712→0.001989 | 0.000000→0.007970 | 0.293608→0.280837 |
| C_reset_init | 1 | 1120 | 0 | 0 | 0 | N/A | N/A | N/A | N/A |
| C_reset_init | 3 | 64 | 64 | 64 | 0 | 0.489547→0.464589 | 0.015836→0.010819 | 0.000000→0.011548 | 0.505383→0.486955 |
| C_reset_init | 4 | 32 | 32 | 32 | 0 | 0.477999→0.456998 | 0.021865→0.016234 | 0.000000→0.010474 | 0.499864→0.483705 |
| C_reset_init | 6 | 32 | 32 | 32 | 0 | 0.455638→0.436687 | 0.026599→0.020867 | 0.000000→0.009879 | 0.482238→0.467432 |
| C_reset_init | 7 | 64 | 64 | 64 | 0 | 0.440729→0.421787 | 0.026400→0.020836 | 0.000000→0.009709 | 0.467129→0.452332 |
| C_reset_init | 13 | 64 | 64 | 64 | 0 | 0.399560→0.385297 | 0.037252→0.031544 | 0.000000→0.008101 | 0.436812→0.424942 |
| C_reset_init | 14 | 32 | 32 | 32 | 0 | 0.386537→0.373119 | 0.031320→0.026629 | 0.000000→0.007229 | 0.417857→0.406977 |
| C_seq | 1 | 1120 | 0 | 0 | 0 | N/A | N/A | N/A | N/A |
| C_seq | 3 | 64 | 64 | 64 | 0 | 0.474935→0.459238 | 0.013308→0.009835 | 0.000000→0.007844 | 0.488243→0.476917 |
| C_seq | 4 | 32 | 32 | 32 | 0 | 0.466028→0.452171 | 0.019729→0.015356 | 0.000000→0.007537 | 0.485757→0.475065 |
| C_seq | 6 | 32 | 32 | 32 | 0 | 0.447009→0.433217 | 0.025260→0.020279 | 0.000000→0.007729 | 0.472268→0.461224 |
| C_seq | 7 | 64 | 64 | 64 | 0 | 0.431701→0.417963 | 0.025287→0.020526 | 0.000000→0.007594 | 0.456988→0.446083 |
| C_seq | 13 | 64 | 64 | 64 | 0 | 0.393332→0.382574 | 0.036333→0.031249 | 0.000000→0.006521 | 0.429665→0.420343 |
| C_seq | 14 | 32 | 32 | 32 | 0 | 0.381061→0.370474 | 0.030574→0.026338 | 0.000000→0.006138 | 0.411635→0.402951 |

实际状态 4576 个，完整曲线位置 19308 个；教师折记录 1944 个。
原始 NPZ 文件 25387 个，共 1602162579 B；阶段分层见 archive_by_phase.csv。

teacher q 分布、错误下界及无法区分的并列见 teachers.csv；保持余量/违反、Z 梯度冲突、guard、三项接受条件、拒因与已有实测前向机制见完整 curves.csv/jsonl。

task_vs_keep 使用 g_Z − keep_g_Z − Z，不除以 N。动态秩、H 谱与实测白化见 stages.csv；共享准备计数见 preparations.csv。V 固定，函数坐标没有参数球投影。

teacher q=0 不等于精确错误数；inner held 准确率及 winner change 均为监督训练诊断。accepted_step 与其 trial 是重复展示，不能将所有曲线行相加计费。

- Only complete independently verified training products are analyzed. Parent result traces, outer scores, query, registry and history are never opened or deserialized.
- The prior summary certifies training bindings, full coordinates and optimizer mathematics. This collector derives descriptive metrics without repeating that audit, fitting or selecting parameters.
- All initial, gradient, accepted/rejected trial, accepted-step and final events are consumed. Accepted-step rows repeat their trial objective; resource totals are taken from verified counts, not curve-row sums.
- Information and updated stages differ. Initial forward/backward work remains charged for zero-update information stages; K1 loss fields are N/A.
- Task function-coordinate gradients are g_Z minus keep_g_Z minus Z. The proximal gradient is Z without division by N. total_vs_keep and task_vs_keep are separate; cosine is N/A when either norm is zero.
- Z has 736 times the retained latent rank coordinates, including exact empty rank-zero arrays. U stays in the original 736-by-8 coordinates and V is fixed; no V gradient or parameter-ball diagnostic is fabricated.
- H/W and the full singular-value array retain original NPZ references. Spectrum/rank/whitening are measured preparation fields; unestimated rank and legitimate original tau=None remain N/A.
- Forward mechanism statistics use existing folds only: pre-tangent displacement, block angles, tangent/kappa, joint/adapted distances, kernel changes and held-winner transitions. Zero-bandwidth adapted-distance bypass remains N/A with its measured reason.
- Preparation records are deduplicated by actual preparation context because C_seq and C_reset_init share one preparation. Stage counters and complete run coverage retain every actual measured dynamic counter.
- q=0 includes negative or tied teacher margin. Zero-q with a unique score winner gives a guaranteed wrong-teacher lower bound; exact wrong/tied classification is N/A without held labels.
- B physical bindings may repeat across new-count parents. Actual resources retain repetitions; deduplicated B statistics are separately labeled and do not create independent evidence.
- Inner losses/accuracy and keep constraints are supervised training diagnostics, not outer performance or a guarantee of old-class retention.
- Array paths preserve all coordinates in original NPZ files. No vectors, member hashes or raw source event copies are embedded in derived reports.
- Per-stage outer score time is N/A when unavailable in the pre-scoring FCR_FINAL event; verified summary resource totals retain actual resource measurements.
- Source validation, query prediction transitions and unmeasured deployment/transfer/hardware metrics stay N/A. Held-winner changes refer only to supervised inner training. Stage timing sums are work, not concurrent elapsed; nested archive timings are not added twice.
