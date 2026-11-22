# LoRa Tri 独立 practical 观测试验结果

**VERIFIED / SCORED_COMPLETE**。三个种子均完成从同一 E140 起点出发的 2464 步和 315 份固定预测；同 30000 包、15 台接收机、各七景共 21 格。原候选集合权重、同种子的冻结教师和预测对照均已核对。

| 方法 | Clean | Practical 六景均值 |
| --- | ---: | ---: |
| 原 Tri | 71.6033% | 55.0511% |
| Tri 候选集合 | 71.6556% | 55.1180% |
| Tri 候选集合加独立 practical 学生观测 | 71.6478% | 55.0939% |

新增方法对候选集合对照净少 7 个 clean 正确和 130 个星地正确，星地逐种子净变化为 −137、+25、−18。相对原 Tri 星地多 231 个正确；相对普通 PL 多 185 个。所有种子的硬伪标签与候选集合位图逐数组等同原候选集合方法；每 seed 共使用 39190 次 practical U 和其中 19530 次第二份观测。由此可隔离损失接受规则和实际波形输入，加入 practical 学生观测没有在原候选集合收益上继续增加。此次设计同时改变了特征一致性和集合 logits 的输入，尚需单独消融才可进一步归因。

[正式目标评分](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/tri_paired_readout/scores.json) · [曝光和伪标签核对](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/tri_paired_analysis.json)
