# 单源接收机：物理分支软目标蒸馏

五种预登记的源 U 软目标蒸馏方式各训练三个种子；原完整 Tri SSL 的模型、固定时间重排及各自同种子 E20 起点保持一致。全部 15 个模型的七景预测完成并冻结后独立评分。

本设置及历史测试结果已曝光，本批是开发验证，不是独立盲测。主口径为 DURA-LU、cross_rx 与 cross_both 合并、每种子每景 144000 包；同包比较核验物理 ID。普通 PL 使用自然时间顺序及不同训练机制，差异不能归因于某个组件。

| 方法 | 场景 | 候选准确率 % | 相对 Tri pp（纠错 / 错失） | 相对 PL pp（纠错 / 错失） |
| --- | --- | ---: | ---: | ---: |
| soft-adapt025 | clean | 49.2606 | -0.1852（1038.3 / 1305.0） | -0.9641（7449.0 / 8837.3） |
| soft-adapt025 | practical_high | 55.4616 | +0.1292（994.3 / 808.3） | -0.4338（6621.7 / 7246.3） |
| soft-adapt025 | practical_mid | 54.2836 | +0.0792（880.3 / 766.3） | -0.7144（5628.7 / 6657.3） |
| soft-adapt025 | practical_low_suburban | 53.0507 | +0.1222（806.0 / 630.0） | -1.0921（4706.7 / 6279.3） |
| soft-adapt025 | practical_high_urban | 54.6799 | +0.1444（957.7 / 749.7） | -0.3465（6144.0 / 6643.0） |
| soft-adapt025 | practical_mid_urban | 50.9755 | +0.0579（624.7 / 541.3） | -0.5062（3949.7 / 4678.7） |
| soft-adapt025 | practical_low_urban | 47.6514 | +0.0903（487.0 / 357.0） | -0.4484（2711.7 / 3357.3） |
| soft-adapt1 | clean | 49.0088 | -0.4370（2498.3 / 3127.7） | -1.2160（7750.0 / 9501.0） |
| soft-adapt1 | practical_high | 54.9968 | -0.3356（1852.0 / 2335.3） | -0.8986（6471.0 / 7765.0） |
| soft-adapt1 | practical_mid | 53.8312 | -0.3731（1597.3 / 2134.7） | -1.1667（5472.0 / 7152.0） |
| soft-adapt1 | practical_low_suburban | 52.6093 | -0.3192（1427.3 / 1887.0） | -1.5336（4502.7 / 6711.0） |
| soft-adapt1 | practical_high_urban | 54.2053 | -0.3301（1716.0 / 2191.3） | -0.8211（5975.3 / 7157.7） |
| soft-adapt1 | practical_mid_urban | 50.6731 | -0.2444（1193.3 / 1545.3） | -0.8086（3876.7 / 5041.0） |
| soft-adapt1 | practical_low_urban | 47.5076 | -0.0535（953.0 / 1030.0） | -0.5921（2697.0 / 3549.7） |
| soft-uniform1 | clean | 40.4227 | -9.0231（4497.3 / 17490.7） | -9.8021（8421.0 / 22536.0） |
| soft-uniform1 | practical_high | 44.6674 | -10.6650（3987.3 / 19345.0） | -11.2280（7270.0 / 23438.3） |
| soft-uniform1 | practical_mid | 44.3229 | -9.8815（4333.0 / 18562.3） | -10.6750（7130.7 / 22502.7） |
| soft-uniform1 | practical_low_suburban | 43.8014 | -9.1271（4689.3 / 17832.3） | -10.3414（6910.7 / 21802.3） |
| soft-uniform1 | practical_high_urban | 44.2613 | -10.2741（4149.0 / 18943.7） | -10.7650（7206.0 / 22707.7） |
| soft-uniform1 | practical_mid_urban | 42.5634 | -8.3542（5036.0 / 17066.0） | -8.9183（6926.0 / 19768.3） |
| soft-uniform1 | practical_low_urban | 40.7292 | -6.8319（5584.3 / 15422.3） | -7.3706（6771.3 / 17385.0） |
| soft-identity1 | clean | 49.4796 | +0.0338（519.7 / 471.0） | -0.7451（7330.0 / 8403.0） |
| soft-identity1 | practical_high | 55.3414 | +0.0090（415.7 / 402.7） | -0.5539（6264.3 / 7062.0） |
| soft-identity1 | practical_mid | 54.1859 | -0.0185（352.0 / 378.7） | -0.8120（5316.0 / 6485.3） |
| soft-identity1 | practical_low_suburban | 52.9315 | +0.0030（331.7 / 327.3） | -1.2113（4439.3 / 6183.7） |
| soft-identity1 | practical_high_urban | 54.5236 | -0.0118（371.3 / 388.3） | -0.5028（5759.0 / 6483.0） |
| soft-identity1 | practical_mid_urban | 50.9051 | -0.0125（255.7 / 273.7） | -0.5766（3772.0 / 4602.3） |
| soft-identity1 | practical_low_urban | 47.5519 | -0.0093（189.3 / 202.7） | -0.5479（2556.0 / 3345.0） |
| soft-frequency1 | clean | 49.5542 | +0.1083（1765.7 / 1609.7） | -0.6706（8033.7 / 8999.3） |
| soft-frequency1 | practical_high | 55.4481 | +0.1157（1407.3 / 1240.7） | -0.4472（6676.0 / 7320.0） |
| soft-frequency1 | practical_mid | 54.2500 | +0.0456（1190.0 / 1124.3） | -0.7479（5639.3 / 6716.3） |
| soft-frequency1 | practical_low_suburban | 53.0185 | +0.0900（1077.7 / 948.0） | -1.1243（4693.0 / 6312.0） |
| soft-frequency1 | practical_high_urban | 54.6252 | +0.0898（1299.0 / 1169.7） | -0.4012（6141.0 / 6718.7） |
| soft-frequency1 | practical_mid_urban | 50.9562 | +0.0387（864.0 / 808.3） | -0.5255（3960.0 / 4716.7） |
| soft-frequency1 | practical_low_urban | 47.6167 | +0.0556（637.3 / 557.3） | -0.4831（2689.3 / 3385.0） |

## 蒸馏机制对照

| 候选 | 对照 | 场景 | 差异 pp（纠错 / 错失） |
| --- | --- | --- | ---: |
| soft-adapt1 | soft-uniform1 | clean | +8.5861（15703.3 / 3339.3） |
| soft-adapt1 | soft-uniform1 | practical_high | +10.3294（17930.3 / 3056.0） |
| soft-adapt1 | soft-uniform1 | practical_mid | +9.5083（17271.0 / 3579.0） |
| soft-adapt1 | soft-uniform1 | practical_low_suburban | +8.8079（16689.7 / 4006.3） |
| soft-adapt1 | soft-uniform1 | practical_high_urban | +9.9440（17618.3 / 3299.0） |
| soft-adapt1 | soft-uniform1 | practical_mid_urban | +8.1097（16129.0 / 4451.0） |
| soft-adapt1 | soft-uniform1 | practical_low_urban | +6.7785（14850.3 / 5089.3） |
| soft-adapt1 | soft-identity1 | clean | -0.4708（2596.7 / 3274.7） |
| soft-adapt1 | soft-identity1 | practical_high | -0.3447（1974.0 / 2470.3） |
| soft-adapt1 | soft-identity1 | practical_mid | -0.3546（1718.3 / 2229.0） |
| soft-adapt1 | soft-identity1 | practical_low_suburban | -0.3222（1525.7 / 1989.7） |
| soft-adapt1 | soft-identity1 | practical_high_urban | -0.3183（1839.7 / 2298.0） |
| soft-adapt1 | soft-identity1 | practical_mid_urban | -0.2319（1278.0 / 1612.0） |
| soft-adapt1 | soft-identity1 | practical_low_urban | -0.0442（1023.3 / 1087.0） |
| soft-frequency1 | soft-adapt1 | clean | +0.5454（2335.3 / 1550.0） |
| soft-frequency1 | soft-adapt1 | practical_high | +0.4514（1790.3 / 1140.3） |
| soft-frequency1 | soft-adapt1 | practical_mid | +0.4188（1641.3 / 1038.3） |
| soft-frequency1 | soft-adapt1 | practical_low_suburban | +0.4093（1494.7 / 905.3） |
| soft-frequency1 | soft-adapt1 | practical_high_urban | +0.4199（1691.0 / 1086.3） |
| soft-frequency1 | soft-adapt1 | practical_mid_urban | +0.2831（1163.0 / 755.3） |
| soft-frequency1 | soft-adapt1 | practical_low_urban | +0.1090（801.7 / 644.7） |

## 稳定性与分层

| 方法 | 相对 Tri 下降种子×场景 | 相对 PL 下降种子×场景 | 相对 PL 最差三种子分层 |
| --- | ---: | ---: | --- |
| soft-adapt025 | 3 / 21 | 18 / 21 | transmitter、practical_low_suburban、8-20、-11.4833 pp |
| soft-adapt1 | 14 / 21 | 20 / 21 | transmitter、clean、14-7、-13.5694 pp |
| soft-uniform1 | 19 / 21 | 19 / 21 | transmitter、practical_low_suburban、6-15、-28.0014 pp |
| soft-identity1 | 12 / 21 | 17 / 21 | transmitter、practical_low_suburban、8-20、-11.0514 pp |
| soft-frequency1 | 12 / 21 | 17 / 21 | transmitter、clean、14-7、-12.4625 pp |

完整逐种子、TX/RX/day、七场景纠错与错失见本地 `paired_softdistill_strata.csv`。identity 对照使用相同原始 IQ 且不衰减，均值匹配对照使用批内一致的平均衰减；不同方法的训练轨迹和接受数量不一定相同。测试目标逐景超过 PL 2 个百分点，须逐景判断，不能由均值替代。

五组在七景的三种子平均均低于普通 PL，未达到逐景领先 2 个百分点的目标。`soft-frequency1` 相对 Tri 的七景平均提高 0.0387 至 0.1157 个百分点，但 21 个种子与场景组合中有 12 个下降，不能认定稳定提高。`soft-adapt025` 的六个 practical 场景在三个种子中均提高，clean 在三个种子中均下降；这些结果均不能证明优于普通 PL。

`soft-uniform1` 的种子 03 出现明显失败：clean 为 20.392%，相对同种子 Tri 下降 27.612 个百分点；高郊区为 23.375%，下降 30.878 个百分点。源验证准确率在该续训段第 1 轮约 99.70%，第 11 轮降至 55.37%，第 20 轮为 50.07%，另两个种子末轮约为 99.70% 和 99.57%。日志中跳过非有限梯度和损失次数均为零；现有证据不能确定崩塌的具体原因，也不能把该种子排除。相对均匀衰减对照的平均优势受此失败显著影响，不足以单独证明逐包物理调节有效。

每个模型续训 30 轮执行 6660 次软目标调用，每次增加两个评估模式前向；每模型选取 846240 次源 U 样本暴露，其中接受约 76.0 万至 84.4 万次。这些是重复训练暴露次数，不是不同物理包数。源 U 平均参考缩放在非 identity 组约为 0.829，identity 为 1。续训墙钟时间约为 5231 至 5935 秒，GPU 共卡负载不同，不能作严格速度比较；完整统计见本地 `cost.json`。
