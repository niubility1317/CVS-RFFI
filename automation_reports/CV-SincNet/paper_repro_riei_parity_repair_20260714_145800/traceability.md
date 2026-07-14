# RIEI期刊Table III论文—代码一致性追踪

| ID | 要求 | 实现 | 状态 |
|---|---|---|---|
| RIEI-P01 | WiSig信道均衡；逐包RMS未由数据集专段明确规定 | RMS开关及no-RMS严格候选 | verified |
| RIEI-P02 | Eq.20–21交替梯度下降与中间FED更新 | SGD/Adam可控，交替顺序保持 | verified |
| RIEI-P03 | 期刊最终5个epoch均值/SD | Table III默认last5 | verified |
| RIEI-P04 | 两source到一target、14400/4800样本、完整12行 | 现有receiver holdout split保持 | verified |
| RIEI-P05 | ResNet1D-18、三层EC/RC、lambda均1.2 | 现有架构与损失保持 | verified |

论文未明确优化器名称和总epoch数，因此SGD仍与Adam做同row消融；正式分数固定200epoch last5，不用目标域峰值选配置。
