# MC-Residual8 完整训练机制诊断

状态：COMPLETE_MC_TRAINING_DIAGNOSTICS_DERIVED；run：20260930-phase2-d92-mc-residual8-support-m2-r01。

全部训练事件已读取。本文只解释监督训练机制，不报告外层评分，不调整冻结方法。

| 模式 | train K | 阶段 | 有信息 | 有更新 | 零更新有信息 | task 初→末 | keep 初→末 | proximal 初→末 | total 初→末 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 1 | 1400 | 0 | 0 | 0 | N/A | N/A | N/A | N/A |
| B | 3 | 80 | 80 | 80 | 0 | 0.438342→0.437440 | 0.006531→0.006442 | 0.000000→0.000549 | 0.444873→0.444430 |
| B | 4 | 40 | 40 | 40 | 0 | 0.424009→0.422740 | 0.006988→0.006894 | 0.000000→0.000798 | 0.430996→0.430433 |
| B | 6 | 40 | 40 | 40 | 0 | 0.379977→0.377224 | 0.007426→0.007077 | 0.000000→0.001945 | 0.387403→0.386246 |
| B | 7 | 80 | 80 | 80 | 0 | 0.363214→0.360908 | 0.004020→0.003899 | 0.000000→0.001607 | 0.367234→0.366414 |
| B | 13 | 80 | 80 | 80 | 0 | 0.303884→0.301420 | 0.003535→0.003432 | 0.000000→0.001344 | 0.307419→0.306196 |
| B | 14 | 40 | 40 | 40 | 0 | 0.290896→0.288630 | 0.002712→0.002631 | 0.000000→0.001260 | 0.293608→0.292521 |
| C_reset_init | 1 | 1120 | 0 | 0 | 0 | N/A | N/A | N/A | N/A |
| C_reset_init | 3 | 64 | 64 | 64 | 0 | 0.489547→0.488512 | 0.013766→0.013433 | 0.000000→0.000815 | 0.503313→0.502760 |
| C_reset_init | 4 | 32 | 32 | 32 | 0 | 0.477999→0.476735 | 0.020297→0.019798 | 0.000000→0.000939 | 0.498296→0.497472 |
| C_reset_init | 6 | 32 | 32 | 32 | 0 | 0.455638→0.453694 | 0.025395→0.024872 | 0.000000→0.001294 | 0.481033→0.479860 |
| C_reset_init | 7 | 64 | 64 | 64 | 0 | 0.440729→0.438937 | 0.024729→0.024102 | 0.000000→0.001081 | 0.465458→0.464120 |
| C_reset_init | 13 | 64 | 64 | 64 | 0 | 0.399560→0.397771 | 0.036045→0.035276 | 0.000000→0.000662 | 0.435605→0.433709 |
| C_reset_init | 14 | 32 | 32 | 32 | 0 | 0.386537→0.384964 | 0.029722→0.029084 | 0.000000→0.000610 | 0.416259→0.414659 |
| C_seq | 1 | 1120 | 0 | 0 | 0 | N/A | N/A | N/A | N/A |
| C_seq | 3 | 64 | 64 | 64 | 0 | 0.489250→0.487889 | 0.013736→0.013338 | 0.000000→0.001035 | 0.502987→0.502262 |
| C_seq | 4 | 32 | 32 | 32 | 0 | 0.477587→0.475826 | 0.020244→0.019640 | 0.000000→0.001227 | 0.497831→0.496693 |
| C_seq | 6 | 32 | 32 | 32 | 0 | 0.454661→0.451697 | 0.025309→0.024673 | 0.000000→0.001518 | 0.479971→0.477887 |
| C_seq | 7 | 64 | 64 | 64 | 0 | 0.440000→0.437433 | 0.024672→0.023927 | 0.000000→0.001269 | 0.464672→0.462629 |
| C_seq | 13 | 64 | 64 | 64 | 0 | 0.398673→0.396077 | 0.035924→0.034960 | 0.000000→0.000702 | 0.434597→0.431739 |
| C_seq | 14 | 32 | 32 | 32 | 0 | 0.385910→0.383713 | 0.029635→0.028850 | 0.000000→0.000644 | 0.415545→0.413207 |

实际状态 4576 个，完整曲线位置 17604 个；教师折记录 1944 个。
原始 NPZ 文件 19644 个，共 2000310343 B；阶段分层见 archive_by_phase.csv。

teacher q 分布、错误下界及无法区分的并列见 teachers.csv；保持余量/违反、梯度冲突、guard/球投影影响、三项接受条件与拒因见完整 curves.csv/jsonl。

teacher q=0 不等于精确错误数；task_vs_keep 使用扣除 keep 与 proximal 后的 task 梯度。accepted_step 与其 trial 是重复展示，不能将所有曲线行相加计费。

- Only complete independently verified training products are analyzed. Parent result traces, outer scores, query, registry and history are never opened or deserialized.
- The prior summary certifies source bindings, full coordinates and optimizer mathematics. This collector derives descriptive metrics without repeating that audit, fitting or selecting parameters.
- All initial, gradient, accepted/rejected trial, accepted-step and final events are consumed. Accepted-step rows repeat their trial objective; resource totals are taken from verified counts, not curve-row sums.
- Information and updated stages differ. Initial forward/backward work remains charged for zero-update information stages; K1 loss fields are N/A.
- Task gradients are total minus keep minus physical proximal gradient. total_vs_keep and task_vs_keep are reported separately; cosine is N/A when either norm is zero.
- q=0 includes negative or tied teacher margin. Zero-q with a unique score winner gives a guaranteed wrong-teacher lower bound; exact wrong/tied classification is N/A without held labels.
- B physical bindings may repeat across new-count parents. Actual resources retain repetitions; deduplicated B statistics are separately labeled and do not create independent evidence.
- Inner losses/accuracy and keep constraints are supervised training diagnostics, not outer performance or a guarantee of old-class retention.
- Array paths preserve all coordinates in original NPZ files. No vectors, member hashes or raw source event copies are embedded in derived reports.
- Per-stage score time is N/A when unavailable in the pre-scoring MC_FIT event; verified summary resource totals retain the actual scoring measurements.
- Source validation, prediction transitions and unmeasured deployment/transfer/hardware metrics stay N/A. Stage timing sums are work, not concurrent elapsed; nested archive timings are not added twice.
