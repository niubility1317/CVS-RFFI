# D92 QIC Hard9+K1 truth-last分析

- 唯一裁决：`REJECT_ROUTE`
- 状态：`ANALYZED`
- 证据边界：prediction先封存，truth-last独立评分；K1仅作liveness，不进入性能聚合。

## 四态指标表

| 状态 | old accuracy | old floor | H_old_new | seen-new accuracy | 说明 |
|---|---:|---:|---:|---:|---|
| DA0_REG0 | N/A | N/A | N/A | N/A | 未提供该状态证据 |
| DA1_REG0 | 0.8691358024691358 | 0.6925925925925926 | N/A | N/A | 注册前 |
| DA0_REG1 | N/A | N/A | N/A | N/A | 未提供该状态证据 |
| DA1_REG1 | 0.4762345679012346 | 0.19444444444444445 | 0.388552259346257 | 0.3637037037037037 | 注册后 |

## 冻结门

| 门 | 结果 |
|---|---|
| complete_artifact_closure | PASS |
| performance_outer_closure | PASS |
| all_strict_pareto | FAIL |
| all_magnitude | FAIL |
| stability | FAIL |
| resource_integrity | PASS |
| resource_hard | FAIL |
| resource_target | FAIL |

## 结果边界

本文件只记录同排、同outer、同scene、同arm的机械证据，不构成科学解释或性能推广声明。
