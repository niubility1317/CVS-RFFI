# 完整 LoRa 适配与原 REPL 正式结果

**VERIFIED / SCORED_COMPLETE**。固定种子 2026092702，28 轮及 2,464 步均完成。正式 210 份预测冻结后独立完成 14 个评分单元和 U 正确率评分。

| 配方 | Clean | Practical 六景 | 城区低 | 接受 U | 已接受 U 正确率 |
|---|---:|---:|---:|---:|---:|
| 原 Tri-set 直通 | 70.3633% | 55.3956% | 41.2733% | — | — |
| 完整 LoRa + 原 REPL，普通 U | 68.8333% | 55.8156% | 41.7967% | 63,003 / 157,696 | 99.1461% |
| 完整 LoRa + 原 REPL，practical U | 68.6300% | 55.8872% | 41.9400% | 61,211 / 157,696 | 99.2420% |

两组初始源 L 为 80.1429%、源 V clean 为 73.5083%。末轮两组源 L 均为 76.3214%，源 V 分别 72.8583%、72.5667%。第二行实际使用 77,757 次同包 practical ordinary 输入。对比两行，practical U 相比普通 U 在星地六景净多 129 次正确，clean 净少 61 次。相比原 Tri-set 直通，practical U 星地净多 885 次，clean 净少 520 次。星地改善伴随 clean 下降。

伪标签正确率高而接受覆盖约四成。两条的 t2 组平均限幅率分别为 96.9562%、96.3880%，冲突率分别为 42.0049%、47.9708%。U 梯度与 strong practical 监督对星地有一定效果，当前保留绝对 clean 满足度仍是难点。后续冻结配对分别在同一源判别门控下接原 R69，并对适配识别器独立评估全条使用以及仅在恢复触发时使用的路径。

[正式评分](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/native_tri_compatible_readout/scores.json) · [离线 U 评分](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/native_tri_compatible_readout/u_offline_scores.json)
