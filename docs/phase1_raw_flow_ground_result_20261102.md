# 原始观测条件流首版 Phase1 地面测试

固定源训练 checkpoint `20261102-phase1-raw-conditioned-flow-source-s2026092701-r01`，四步推理，冻结 Tri-SIG-TIME SSL。测试 run `20261102-phase1-raw-conditioned-flow-ground-test-s2026092701-r01`，每景 168,000 条。七景预测固定后真值评分，独立程序复核 119 个 TX/RX/day 分组、总体与跨域指标。

| 场景 | 原始输入准确率 | 恢复后准确率 | 相对干净基座下降 |
|---|---:|---:|---:|
| clean | 86.5387% | 52.1256% | 39.77% |
| practical_high | 80.6155% | 52.2107% | 39.67% |
| practical_mid | 78.1077% | 50.5798% | 41.55% |
| practical_low_suburban | 75.7446% | 49.0351% | 43.34% |
| practical_high_urban | 77.0500% | 50.5673% | 41.57% |
| practical_mid_urban | 62.7958% | 43.3107% | 49.95% |
| practical_low_urban | 51.6048% | 37.5250% | 56.64% |

干净基座为 86.538690%，恢复后 clean 损失 34.413095 个百分点。六个受损场景均未满足相对下降 3% 或 3 个百分点。低城市 cross-RX 为 37.813492%，cross-RX+day 为 36.659524%。结果显著低于同一场景原始输入，首版不适合作为当前恢复器。

证据位于 N607 `runs/20261102-phase1-raw-conditioned-flow-ground-test-s2026092701-r01/flow/test_summary.json`，评分前检查和独立复算位于 `runs/20261102-phase1-raw-conditioned-flow-ground-test-s2026092701-r01-auto-audit/{pretruth_audit,independent_audit}.json`。两项审计状态 PASS，源物理编号与基座预测一致。

该地面数据及模拟信道测试此前反复用于研发，单种子结果不构成独立盲测或真实卫星性能。身份保持条件流是独立 checkpoint，其独立地面结果见 `phase1_identity_flow_ground_result_20261102.md`；不得将首版结论套用到该版本。
