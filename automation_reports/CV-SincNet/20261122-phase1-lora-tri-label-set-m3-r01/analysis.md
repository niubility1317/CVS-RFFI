# LoRa Tri 候选集合的正式配对结果

**VERIFIED / SCORED_COMPLETE**。三个种子各完成 2464 步，315 份盲预测固定后独立评分。21 个单元，30000 物理包，15 个未见 RX。原 Tri、普通 PL 和继续监督沿用同一 E140 初始化及原预算对照，逐 seed 校验同一冻结策略。标签集合和硬标签均独立离线核对。

| 方法 | clean | practical 六景均值 |
| --- | ---: | ---: |
| 继续监督 | 71.6533% | 55.0463% |
| 普通 PL | 71.3267% | 55.0596% |
| 原自适应 Tri | 71.6033% | 55.0511% |
| Tri 候选类别集合 | 71.6556% | 55.1180% |

候选集合相对原 Tri 的 clean 净纠正 +47（+0.0522 个百分点），三个种子分别 +49、−15、+13；practical +361（+0.0669 个百分点），分别 +228、+23、+110。相对 PL，clean +296、六景 +315。相对继续监督，clean +2、六景 +387。效果在星地三个种子均为正，clean 均值基本追平监督。

所有种子的硬标签使用量与正确数和原 Tri 一致，共 70930 次，正确 70575 次。额外 64749 次集合曝光与硬标签互斥，64399 次含真实类别（99.46%）。TX32、33、35、38 都获得了集合训练，TX38 占 23018 次，TX35 在种子 02 仅有 72 次，覆盖仍有不均衡。目标 clean 的 TX33 从 30.17% 升至 30.66%、TX35 从 45.66% 升至 46.27%，TX32 与 TX38 略降。集合覆盖增加提供了较稳定的星地小增益，仍需要提高难类之间的判别性。

[同包正式评分](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/tri_set_readout/scores.json) · [伪标签离线核对](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/tri_set_readout/u_offline_scores.json) · [候选集源域证据](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/tri_prototype_audit/set_results.json)
