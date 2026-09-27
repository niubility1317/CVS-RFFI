# IR-EG需求追溯与交付状态

基准：`acf0a6407c4a2cd114851e9f60d0cd346e4b2c77`。33项开发要求已验证，4项需要未来真实实验的要求延期；无未解决实现阻塞。

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| R01 | 1.2 | Isolated copy, pinned runtime imports and baseline identity | ir_types.py | verified | test_ir_contract.py | Synthetic development evidence; no scientific training claim |
| R02 | 1.1/7 | Frozen native joint source roles/configuration; launch disabled | ir_experiments.py | verified | test_ir_experiment_manifest.py | Synthetic development evidence; no scientific training claim |
| R03 | 2 | Object identity layout, aliases, frozen and zero-LR coordinates | ir_types.py; ir_metric.py | verified | test_ir_contract.py; test_ir_metric.py | Synthetic development evidence; no scientific training claim |
| R04 | 2 | Separate weighted L/U CE and phi ownership | ir_objective.py | verified | test_ir_contract.py; test_ir_native_joint.py | Synthetic development evidence; no scientific training claim |
| R05 | 3.2 | Raw h0 and real virtual AdamW predictor | ir_solver.py | verified | test_ir_metric.py; test_ir_reference.py | Synthetic development evidence; no scientific training claim |
| R06 | 3.3 | Frozen second-moment R, actual group/state/clip, no duplicate clipping | ir_metric.py | verified | test_ir_metric.py | Synthetic development evidence; no scientific training claim |
| R07 | 3.4 | Matrix-free weighted CE GGN; detached backbone and VJP cotangent | ir_head.py | verified | test_ir_curvature.py | Synthetic development evidence; no scientific training claim |
| R08 | 3.5 | Two-step whitened CG and honest truncated/failure statuses | ir_cg.py | verified | test_ir_cg.py | Synthetic development evidence; no scientific training claim |
| R09 | 4 | Native per-call dropout RNG/mask replay including zero activations | ir_head.py | verified | test_ir_head.py | Synthetic development evidence; no scientific training claim |
| R10 | 5.1 | Three-forward/two-full-backward reference; formal update from origin | ir_solver.py | verified | test_ir_reference.py; test_ir_native_joint.py | Synthetic development evidence; no scientific training claim |
| R11 | 5.2 | Cached native graph and pure leaf replacement without live mutation | ir_objective.py; objective.py; dr_objective.py | verified | test_ir_packet.py; test_ir_cached.py | Synthetic development evidence; no scientific training claim |
| R12 | 5.2 | kappa=0 direct original EG and packet equivalence | ir_solver.py | verified | test_ir_reference.py; test_ir_packet.py | Synthetic development evidence; no scientific training claim |
| R13 | 1.1/3.6 | U direct encoder domain gradient zero with U identity retained; no extra LEO domain CE | dr_objective.py | verified | test_ir_packet.py; test_ir_native_joint.py; test_ir_head.py | Synthetic development evidence; no scientific training claim |
| R14 | 3.1/6 | Origin-used normalizer sequence, frozen route/teacher and origin buffers/RNG | ir_solver.py | verified | test_ir_native_joint.py; test_ir_cached.py | Synthetic development evidence; no scientific training claim |
| R15 | 6 | Outer model/optimizer/ctx/EMA/prototype/normalizer/history transaction | ir_solver.py; game_tracking/state.py | verified | test_ir_transaction.py | Synthetic development evidence; no scientific training claim |
| R16 | 6 | Same-step numerical fallback; finite scalar/gradient/state checks; structural abort | ir_solver.py | verified | test_ir_transaction.py | Synthetic development evidence; no scientific training claim |
| R17 | 7 | E21 fixed response, no TR ramp or DRIC cadence | response_config.py | verified | test_ir_reference.py | Synthetic development evidence; no scientific training claim |
| R18 | 1.2/7 | Strict FP32 AdamW schema and optimizer variants | response_config.py; ir_solver.py | verified | test_ir_metric.py; test_ir_reference.py | Synthetic development evidence; no scientific training claim |
| R19 | 8.1 | BR previous accepted final raw psi field, current h0 and real mixed clipping | br_history.py; ir_solver.py | verified | test_ir_br_history.py | Synthetic development evidence; no scientific training claim |
| R20 | 8.2 | BR periodic/discrete/activity refresh and age-one history | br_history.py | verified | test_ir_br_history.py | Synthetic development evidence; no scientific training claim |
| R21 | 8.3 | R1=IR; kappa0=BR-EG; serialized history and restored stochastic state | br_history.py; resume.py | verified | test_ir_br_resume.py; test_ir_native_resume.py | Synthetic development evidence; no scientific training claim |
| R22 | 9 | Executed flags, actual model/derivative counts, norms, clip and displacement telemetry | ir_solver.py; ir_telemetry.py | verified | test_ir_reporting.py; acceptance/*/profiling.json | Synthetic development evidence; no scientific training claim |
| R23 | 3.5/9.1 | Quadratic energy and full/applied nonlinear residual | ir_cg.py; ir_telemetry.py | verified | test_ir_analytic_examples.py; test_ir_native_diagnostics.py | Synthetic development evidence; no scientific training claim |
| R24 | 9.2 | Isolated ten-step training-head recovery and disjoint fixed V monitor | ir_telemetry.py | verified | test_ir_native_diagnostics.py | Synthetic development evidence; no scientific training claim |
| R25 | 9.3 | Same-origin EG/IR AdamW displacement and common-clip counterfactual | ir_solver.py; ir_telemetry.py | verified | test_ir_diagnostics.py; test_ir_native_joint.py | Synthetic development evidence; no scientific training claim |
| R26 | 9.2/9.4 | Source TX/RX slice metrics, signed gains and null missing values | ir_telemetry.py | verified | test_ir_diagnostics.py; test_ir_reporting.py | Synthetic development evidence; no scientific training claim |
| R27 | 11 | CPU full development acceptance | scripts/check_ir_acceptance.py | verified | acceptance/cpu/result.json | Synthetic development evidence; no scientific training claim |
| R28 | 11 | Actual CUDA full development acceptance, unchanged declared tolerance | scripts/check_ir_acceptance.py | verified | acceptance/cuda/result.json | Synthetic development evidence; no scientific training claim |
| R29 | 10.1 | SIM/EG/IR/G0/OR/encoder-off and BR matrices, distinct seed roles | ir_experiments.py | verified | test_ir_experiment_manifest.py | Synthetic development evidence; no scientific training claim |
| R30 | 10.2 | Separate unfrozen source convergence preparation, historical budget unchanged | ir_experiments.py | verified | test_ir_experiment_manifest.py | Synthetic development evidence; no scientific training claim |
| R31 | Plan Tasks1/9 | Real CLI validation, no accidental launch, legacy config compatibility | scripts/train_response_games.py | verified | test_ir_legacy_config.py | Synthetic development evidence; no scientific training claim |
| R32 | Plan extra examples | Rotation matrix, nonmonotonic task risk, quadratic and asymmetric LU examples | ir_cg.py | verified | test_ir_analytic_examples.py | Synthetic development evidence; no scientific training claim |
| R33 | 10.3 | Measured bounded local profiling and history memory with explicit scope | scripts/profile_ir_acceptance.py | verified | acceptance/cpu/profiling.json; acceptance/cuda/profiling.json | Synthetic development evidence; no scientific training claim |
| D01 | 10.1B | Historical early/mid/late checkpoint probes | configs/; docs/experiment_protocol.md | deferred | not executed | No compliant checkpoint ancestry/data manifest provided or loaded; random initialization fixtures used. |
| D02 | 10.1D | Three-seed real joint training and ablation performance | configs/; docs/experiment_protocol.md | deferred | not executed | Future real-data training was not authorized or launched. |
| D03 | 10.2/10.3 | Complete convergence and full-cost efficiency confirmation | configs/; docs/experiment_protocol.md | deferred | not executed | Continuation LR/safety cap manifest is intentionally unfrozen; no source-convergence run. |
| D04 | 10.3 | Frozen target prediction and independent scoring | configs/; docs/experiment_protocol.md | deferred | not executed | Requires future frozen candidate/checkpoint; no target data or truth accessed. |

## 实施决定与边界

- 当前用户请求授权实施；附件的旧“仅规划”状态不是当前任务指令。未启动远端训练。
- 托管worktree工具因会话根目录不是Git而不可用，改用原生Git从精确基线创建隔离分支；历史证据未编辑。
- 同一原生CUDA模型的reference/reference也出现非确定性误差。所有新包比较臂统一确定性cuDNN策略，未放宽容差；这不是全局确定性保证。
- V监测固定前256条合法记录，切片只描述该面板；恢复头固定重放已登记Dropout掩码。
- Native端到端checkpoint入口测试跨E1/E2；活跃BR历史的真实序列化续训由独立受限联合fixture验证，不声称跑过E21之后完整真实数据epoch。
- IR严格实现设计中的有限步局部仿射近似；BR按设计是单独近似算法。最高剩余风险是实际数据上机制收益、稳定性和全成本尚未验证。
