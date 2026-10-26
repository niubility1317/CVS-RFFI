# 用户指定 F-only 的协方差源分析

状态：ANALYZED_SOURCE_ONLY。用户2026-10-26明确撤下DURA配合路线，只用clean及同包星地衍生IQ训练。对应协议和当前目标已更新；本独立分析不读取恢复IQ/权重，不用G性能列晋级，也不改变旧run的原判定。

四个现有模型都保留完整PAMF、单视图SIG_TIME、原L/T1跨RX、联合默认头和普通PL.95/weight1。它们实际训练已经只用clean+H，符合新输入范围；无需重训。对照复用已完成的同架构SIG1-PL，不重训、不重新预测旧已完成目标单元。

只读冻结sourceV27000的raw、lowSuburban、midUrban、lowUrban计数以及实际全U曝光。新范围的必要条件为相对PL的rawclean至多少27条、raw三景总正确数不减，并核实全56700U各30新增访问、1701000曝光和权限边界。逐行独立判断，G列不能影响结果。缺少实际final源结果的行只标等待；源通过不等于正式目标提升。

此范围在部分源结果已知后由用户明确变更，不能伪称原八视图run的预登记条件；旧G-inclusive判定保留。新covariance模型尚无任何正式target访问。后续直接F测试clean及六practical，保留原rawclean≥85.981746%、raw六景>72.096197%、严格同SIG1PL/Sup及原三seed等目标，不再使用DURA指标验收。

全部命令、输入、方法、输出、条件与声明边界见experiment.json。本项是CPU缓存计数分析，不训练或运行新模型、不读新IQ/隐藏U真值/target，不影响健康旧F训练或其他owner。

## 实际 F-only 结果

分析执行commit `ed0964bebe634259722bbda816af4acb7bd7bfa7`。全部四行实际源评估及U曝光已读回；只消费raw四视图计数，不用G列、模型/IQ或target。两项raw-only隔离/边界测试通过；实际完整训练日志另核验31080步/140条epoch、0致命标记。

| 方法 | raw clean 相对PL（pp） | raw三景平均相对PL（pp） | F-only必要条件 |
| --- | ---: | ---: | --- |
| CovSel-U | -0.022222 | +0.071605 | PASS |
| CovRandom-U | -0.018519 | +0.046914 | PASS |
| CovFull-U | -0.044444 | -0.038272 | FAIL |
| CovSel-LU | -0.022222 | +0.127160 | PASS |

选择性L/U比选择性U的三景多45条正确预测，即+0.055556pp；两者clean相同。选择性U比随机控制三景多20条，即+0.024691pp。只是单seed源结果，不能证明稳定泛化或选择规则的独立因果收益；原梯度压力不同的限制仍保留。

三个PASS行均进入同一固定正式rawclean+六practical矩阵，不按源分数再排一个“最优”；全相关项行停止此轮扩展。正式确认登记`20261026-phase1-PAMF-SIG1-Covariance-Fonly-confirm-m3-r01`，不调用DURA、不重训或重跑已完成对照。原raw绝对门槛、严格SIG1对照与三seed等目标保持，当前goal_complete=false。[完整源计数](source_results.json)。
