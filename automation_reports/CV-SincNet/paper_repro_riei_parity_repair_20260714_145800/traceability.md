# RIEI期刊Table III论文—代码一致性追踪

| ID | 要求 | 实现 | 状态 |
|---|---|---|---|
| RIEI-P01 | WiSig信道均衡；逐包RMS未由数据集专段明确规定 | RMS开关及no-RMS严格候选 | verified |
| RIEI-P02 | Eq.20–21交替梯度下降与中间FED更新 | SGD/Adam可控，交替顺序保持 | verified |
| RIEI-P03 | 期刊最终5个epoch均值/SD | Table III默认last5 | verified |
| RIEI-P04 | 两source到一target、14400/4800样本、完整12行 | 现有receiver holdout split保持 | verified |
| RIEI-P05 | ResNet1D-18、三层EC/RC、lambda均1.2 | 现有架构与损失保持 | verified |
| RIEI-P06 | 用同row受控实验定位优化器与loss reduction | 8候选完整1600epoch比较，P02 last5=`80.12±0.58%` | verified |
| RIEI-P07 | 胜出配置必须确认Table III全部12个receiver组合 | 固定SGD+mean+no-RMS+no-FN的12行last5确认launcher；本地语法/dry-run/pytest通过 | implemented |

论文未明确优化器名称和总epoch数，因此SGD仍与Adam做同row消融；正式分数固定200epoch last5，不用目标域峰值选配置。

当前反向审计：`verified=6`、`implemented=1`、`pending=0`、`deferred=0`、`rejected=0`、`blocked=0`。最高风险是P02尚未在Table III完整12行上确认。
