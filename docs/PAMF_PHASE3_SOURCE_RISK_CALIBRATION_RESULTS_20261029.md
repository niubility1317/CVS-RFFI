# Phase3 源已知风险校准结果

**VERIFIED／ANALYZED_SOURCE_ONLY，科学结果为 NO_PROMOTION。** 固定MLP的未知TX宏拒识由76.6032%升至81.4603%，但整体与RX3/4/6/8仍低于95%。已知风险预算全部合格，没有技术故障。

run：`20261029-phase3-source-risk-calibration-s2026092701-r01`；code：`ba2b4ed63a03751b8934dd7385f16a05aa75bcf3`。

## 只校准操作点

复用完整六类PAMF、源几何和两冻结OE读出，不新训练、不F前向，CPU-only、新增GPU worker0、foreignstop0。真实拒识checkpoint smoke读取既有L特征6300行中的2条打分（2×7），不读取L标签，不冒充重新运行F。

known V27000的分数、标签及原head预测用于经验风险校准：总体及每源RX宏FRR≤4%，拒绝原正确样本造成的宏准确率损失≤1.5pp。各方法选择满足所有约束的最大单一全局阈值，保留分数ties；不按RX路由阈值。七方法均合格，下一阈值候选均违反约束。未知分数／标签未参与校准，阈值先冻结，再完成30150×7预测并独立评分。

“truth_read=false”在本轮只指未知私有truth；已知V标签明确用于允许的源校准。风险是在同一批known V上的经验值，没有独立泛化保证；novel3150与父run相同，是重复源开发，不能称新盲测。

## 结果与父方法比较

| 方法 | 父run未知拒识 | 本run未知拒识 | 已知FRR | 已知开放宏acc | 较原闭集损失/pp |
|---|---:|---:|---:|---:|---:|
| linear_oe | 68.4762% | 72.9206% | 1.3704% | 97.6741% | 0.7926 |
| mlp16_oe | 76.6032% | 81.4603% | 1.3667% | 97.6185% | 0.8481 |

原闭集宏acc仍为98.4667%。MLP多拒识153/3150条未知（+4.8571pp），同时多拒识100条已知，少72条旧类正确接受；没有增强表示或重训模型。linear未知增益4.4444pp。两者RX3的正确拒绝损失均达到1.5pp预算边界，不能根据未知结果继续提高预算或改阈值。

| 源RX | Linear未知宏拒识 | MLP未知宏拒识 |
|---|---:|---:|
| 1 | 97.7778% | 99.2063% |
| 3 | 53.6508% | 73.1746% |
| 4 | 84.2857% | 83.9683% |
| 6 | 91.2698% | 91.9048% |
| 8 | 37.6190% | 59.0476% |

仅RX1通过95%。MLP最弱RX8为59.0476%，RX3为73.1746%；其余RX4/6为83.9683%／91.9048%。三个验证TX（19-15、19-16、7-1）的MLP拒识为76.0000%、76.4762%、91.9048%，均不足95%。全部7方法／5RX／3TX及风险检查保存在机器摘要。

## 执行与边界

完整两日志5行、319字节，无异常；predict与score均COMPLETE，独立post_state无failure。pipeline 2.425367s（predict 2.094145s，score 0.330267s）。13项测试＋12子测试及一次独立P0/P1通过；本次不新增测试或重复父run的400条损失。

这是当前完整六类冻结模型在已见source RX、clean条件下的辅助新TX拒识；不是跨RX或星地验证，也没有适应、聚类或注册。正式RX24 U/Q未读。eligible空、selected=null，95%总目标未达，不扩seed、不改known预算挽救本run。

[评分与全部分组](E:/type10-7/local_artifacts/pamf_phase3_source_risk_calibration_20261029_r01/score/results.json)；[机器摘要及完整证据路径](E:/type10-7/local_artifacts/pamf_phase3_source_risk_calibration_20261029_r01/analysis_summary.json)；[父结果](E:/type10-7/docs/PAMF_PHASE3_SOURCE_OE_REJECTION_RESULTS_20261029.md)。
