# Phase1 已完成恢复网络与 DURA 的地面七景比较

统一范围：Phase1 Main split，固定 Tri-SIG-TIME SSL，模型种子 2026092701；每景 168000 条，完整七景预测固定后评分。已有测试数据曾用于研发观察，不能称作独立盲测或真实卫星链路结果。下表数值为识别准确率（%）。

| 方法 | clean | 高郊区 | 中郊区 | 低郊区 | 高城区 | 中城区 | 低城区 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DURA v1.0 | 86.615 | 84.343 | 83.016 | 81.322 | 83.049 | 75.706 | 68.801 |
| [直接注意力](../20261102-phase1-attention-waveform-ground-test-s2026092701-r01/test_summary.json) | 86.555 | 71.189 | 69.754 | 68.543 | 68.204 | 57.277 | 48.948 |
| [置信路由](../20261102-phase1-confidence-router-ground-test-s2026092701-r01/test_summary.json) | 86.439 | 84.268 | 82.598 | 80.581 | 82.167 | 72.158 | 63.263 |
| [全源蒸馏注意力](../20261102-phase1-distilled-attention-full-ground-test-s2026092701-r02/test_summary.json) | 86.533 | 68.757 | 68.367 | 67.585 | 68.020 | 63.706 | 58.957 |
| [源教师蒸馏注意力](../20261102-phase1-distilled-attention-ground-test-s2026092701-r01/test_summary.json) | 86.543 | 65.252 | 64.026 | 62.826 | 62.665 | 52.944 | 45.083 |
| [全源困难样本物理混合](../20261102-phase1-free-full-hard-ground-test-s2026092701-r01/test_summary.json) | 86.458 | 84.901 | 83.280 | 81.047 | 82.790 | 72.810 | 63.999 |
| [直接物理混合](../20261102-phase1-free-mixture-ground-test-s2026092701-r01/test_summary.json) | 86.449 | 85.249 | 83.671 | 81.473 | 83.325 | 74.191 | 65.695 |
| [分层先验](../20261102-phase1-hierarchical-prior-ground-test-s2026092701-r01/test_summary.json) | 86.445 | 84.585 | 82.834 | 80.526 | 82.360 | 71.806 | 62.770 |
| [独立先验 PnP](../20261102-phase1-independent-prior-pnp-ground-test-s2026092701-r02/test_summary.json) | 86.461 | 76.958 | 75.152 | 73.200 | 74.662 | 65.386 | 57.876 |
| [联合物理混合](../20261102-phase1-joint-mixture-ground-test-s2026092701-r01/test_summary.json) | 86.454 | 84.715 | 83.089 | 80.945 | 82.604 | 73.018 | 64.765 |
| [结构化先验](../20261102-phase1-structured-prior-ground-test-s2026092701-r01/test_summary.json) | 86.440 | 82.670 | 81.115 | 79.207 | 80.831 | 72.205 | 64.235 |
| [投票路由](../20261102-phase1-vote-router-ground-score-s2026092701-r01/test_summary.json) | 86.456 | 83.037 | 81.343 | 79.379 | 80.646 | 70.535 | 61.854 |

raw clean 为 86.538690%，以相差不超过 3 个百分点计及格线为 83.538690%；以相对下降不超过 3% 计及格线为 83.942530%。直接物理混合分别为 2/6 和 1/6；DURA 分别为 1/6 和 1/6。

直接物理混合与 DURA 的同记录配对证据见 [paired_dura.json](../20261102-phase1-free-mixture-ground-test-s2026092701-r01/paired_dura.json)。该方法在低城区比 DURA 低 3.106548 个百分点；不可把其他场景的提升当作已达成全部场景目标。

当前 iterative geometry、spectral prior、physical fusion、physical wide 和 physical local teacher 均未形成完整七景评分，本表不含它们的未完成预测。
