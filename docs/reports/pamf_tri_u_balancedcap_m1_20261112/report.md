# 单源接收机：宽松伪类别配额与物理分支敏感度排序

九种预登记的源 U 伪标签筛选方式各训练三个种子；原完整 Tri SSL 的模型、固定时间重排及各自同种子 E20 起点保持一致。全部 27 个模型的七景预测完成并冻结后独立评分。

本设置及历史测试结果已曝光，本批是开发验证，不是独立盲测。主口径为 DURA-LU、cross_rx 与 cross_both 合并、每种子每景 144000 包；同包比较核验物理 ID。普通 PL 使用自然时间顺序及不同训练机制，差异不能归因于某个组件。

| 方法 | 场景 | 候选准确率 % | 相对 Tri pp（纠错 / 错失） | 相对 PL pp（纠错 / 错失） |
| --- | --- | ---: | ---: | ---: |
| cap1125-random | clean | 49.1822 | -0.2637（568.0 / 947.7） | -1.0426（7093.7 / 8595.0） |
| cap1125-random | practical_high | 55.2032 | -0.1292（482.7 / 668.7） | -0.6921（6145.0 / 7141.7） |
| cap1125-random | practical_mid | 54.1199 | -0.0845（448.3 / 570.0） | -0.8780（5246.0 / 6510.3） |
| cap1125-random | practical_low_suburban | 52.9155 | -0.0130（451.3 / 470.0） | -1.2273（4384.3 / 6151.7） |
| cap1125-random | practical_high_urban | 54.3912 | -0.1442（429.3 / 637.0） | -0.6352（5646.7 / 6561.3） |
| cap1125-random | practical_mid_urban | 50.8625 | -0.0551（328.0 / 407.3） | -0.6192（3728.3 / 4620.0） |
| cap1125-random | practical_low_urban | 47.5593 | -0.0019（250.3 / 253.0） | -0.5405（2544.3 / 3322.7） |
| cap1125-confidence | clean | 49.9799 | +0.5340（2342.0 / 1573.0） | -0.2449（7483.0 / 7835.7） |
| cap1125-confidence | practical_high | 55.5500 | +0.2176（1499.0 / 1185.7） | -0.3454（6249.7 / 6747.0） |
| cap1125-confidence | practical_mid | 54.4477 | +0.2433（1409.3 / 1059.0） | -0.5502（5325.7 / 6118.0） |
| cap1125-confidence | practical_low_suburban | 53.2838 | +0.3553（1411.0 / 899.3） | -0.8590（4506.3 / 5743.3） |
| cap1125-confidence | practical_high_urban | 54.7150 | +0.1796（1359.7 / 1101.0） | -0.3113（5785.3 / 6233.7） |
| cap1125-confidence | practical_mid_urban | 51.0889 | +0.1713（1006.3 / 759.7） | -0.3928（3798.7 / 4364.3） |
| cap1125-confidence | practical_low_urban | 47.7451 | +0.1840（781.3 / 516.3） | -0.3546（2622.3 / 3133.0） |
| cap1125-physical | clean | 49.7620 | +0.3162（1766.3 / 1311.0） | -0.4627（7384.7 / 8051.0） |
| cap1125-physical | practical_high | 55.6514 | +0.3190（1399.0 / 939.7） | -0.2440（6345.0 / 6696.3） |
| cap1125-physical | practical_mid | 54.5340 | +0.3296（1324.7 / 850.0） | -0.4639（5397.7 / 6065.7） |
| cap1125-physical | practical_low_suburban | 53.3808 | +0.4523（1365.7 / 714.3） | -0.7620（4559.7 / 5657.0） |
| cap1125-physical | practical_high_urban | 54.7678 | +0.2324（1231.3 / 896.7） | -0.2586（5822.7 / 6195.0） |
| cap1125-physical | practical_mid_urban | 51.1637 | +0.2461（958.7 / 604.3） | -0.3181（3838.7 / 4296.7） |
| cap1125-physical | practical_low_urban | 47.7979 | +0.2368（770.3 / 429.3） | -0.3019（2633.7 / 3068.3） |
| cap125-random | clean | 49.4380 | -0.0079（554.3 / 565.7） | -0.7868（7292.3 / 8425.3） |
| cap125-random | practical_high | 55.2630 | -0.0694（400.7 / 500.7） | -0.6324（6170.7 / 7081.3） |
| cap125-random | practical_mid | 54.1778 | -0.0266（388.3 / 426.7） | -0.8201（5277.7 / 6458.7） |
| cap125-random | practical_low_suburban | 52.9329 | +0.0044（367.0 / 360.7） | -1.2100（4399.0 / 6141.3） |
| cap125-random | practical_high_urban | 54.4745 | -0.0609（373.3 / 461.0） | -0.5519（5686.7 / 6481.3） |
| cap125-random | practical_mid_urban | 50.9014 | -0.0162（285.0 / 308.3） | -0.5803（3749.3 / 4585.0） |
| cap125-random | practical_low_urban | 47.5562 | -0.0049（207.7 / 214.7） | -0.5435（2539.0 / 3321.7） |
| cap125-confidence | clean | 49.6662 | +0.2204（1341.7 / 1024.3） | -0.5586（7380.0 / 8184.3） |
| cap125-confidence | practical_high | 55.3819 | +0.0495（942.0 / 870.7） | -0.5134（6219.7 / 6959.0） |
| cap125-confidence | practical_mid | 54.2891 | +0.0847（902.0 / 780.0） | -0.7088（5316.0 / 6336.7） |
| cap125-confidence | practical_low_suburban | 53.1125 | +0.1840（914.3 / 649.3） | -1.0303（4473.0 / 5956.7） |
| cap125-confidence | practical_high_urban | 54.5819 | +0.0465（869.7 / 802.7） | -0.4444（5767.0 / 6407.0） |
| cap125-confidence | practical_mid_urban | 50.9900 | +0.0725（649.3 / 545.0） | -0.4917（3777.3 / 4485.3） |
| cap125-confidence | practical_low_urban | 47.6435 | +0.0824（500.7 / 382.0） | -0.4563（2576.7 / 3233.7） |
| cap125-physical | clean | 49.6454 | +0.1995（1120.3 / 833.0） | -0.5794（7344.3 / 8178.7） |
| cap125-physical | practical_high | 55.4847 | +0.1523（892.7 / 673.3） | -0.4106（6313.7 / 6905.0） |
| cap125-physical | practical_mid | 54.4109 | +0.2065（878.7 / 581.3） | -0.5870（5415.3 / 6260.7） |
| cap125-physical | practical_low_suburban | 53.1956 | +0.2671（882.0 / 497.3） | -0.9472（4531.0 / 5895.0） |
| cap125-physical | practical_high_urban | 54.6775 | +0.1421（826.7 / 622.0） | -0.3488（5830.7 / 6333.0） |
| cap125-physical | practical_mid_urban | 51.0764 | +0.1588（635.7 / 407.0） | -0.4053（3854.7 / 4438.3） |
| cap125-physical | practical_low_urban | 47.7062 | +0.1451（508.0 / 299.0） | -0.3935（2634.0 / 3200.7） |
| cap150-random | clean | 49.3252 | -0.1206（460.0 / 633.7） | -0.8995（7312.0 / 8607.3） |
| cap150-random | practical_high | 55.2104 | -0.1220（405.0 / 580.7） | -0.6850（6239.0 / 7225.3） |
| cap150-random | practical_mid | 54.0900 | -0.1144（349.7 / 514.3） | -0.9079（5303.7 / 6611.0） |
| cap150-random | practical_low_suburban | 52.8447 | -0.0838（299.0 / 419.7） | -1.2981（4399.3 / 6268.7） |
| cap150-random | practical_high_urban | 54.4067 | -0.1287（354.3 / 539.7） | -0.6197（5754.3 / 6646.7） |
| cap150-random | practical_mid_urban | 50.8407 | -0.0769（240.7 / 351.3） | -0.6410（3760.0 / 4683.0） |
| cap150-random | practical_low_urban | 47.5259 | -0.0352（179.0 / 229.7） | -0.5738（2550.7 / 3377.0） |
| cap150-confidence | clean | 49.5039 | +0.0581（617.0 / 533.3） | -0.7208（7389.7 / 8427.7） |
| cap150-confidence | practical_high | 55.3697 | +0.0373（506.7 / 453.0） | -0.5257（6276.0 / 7033.0） |
| cap150-confidence | practical_mid | 54.2333 | +0.0289（441.7 / 400.0） | -0.7646（5335.7 / 6436.7） |
| cap150-confidence | practical_low_suburban | 53.0007 | +0.0722（434.0 / 330.0） | -1.1421（4458.3 / 6103.0） |
| cap150-confidence | practical_high_urban | 54.5597 | +0.0243（458.7 / 423.7） | -0.4667（5785.0 / 6457.0） |
| cap150-confidence | practical_mid_urban | 50.9600 | +0.0424（348.0 / 287.0） | -0.5218（3794.0 / 4545.3） |
| cap150-confidence | practical_low_urban | 47.5914 | +0.0303（252.0 / 208.3） | -0.5083（2574.7 / 3306.7） |
| cap150-physical | clean | 49.4093 | -0.0366（620.3 / 673.0） | -0.8155（7297.0 / 8471.3） |
| cap150-physical | practical_high | 55.3289 | -0.0035（540.3 / 545.3） | -0.5664（6268.0 / 7083.7） |
| cap150-physical | practical_mid | 54.1921 | -0.0123（493.7 / 511.3） | -0.8058（5343.3 / 6503.7） |
| cap150-physical | practical_low_suburban | 52.9465 | +0.0181（456.3 / 430.3） | -1.1963（4453.0 / 6175.7） |
| cap150-physical | practical_high_urban | 54.5039 | -0.0315（495.7 / 541.0） | -0.5225（5766.7 / 6519.0） |
| cap150-physical | practical_mid_urban | 50.9155 | -0.0021（348.0 / 351.0） | -0.5662（3789.7 / 4605.0） |
| cap150-physical | practical_low_urban | 47.5674 | +0.0063（254.3 / 245.3） | -0.5324（2570.7 / 3337.3） |

## 均衡与排序机制对照

| 候选 | 对照 | 场景 | 差异 pp（纠错 / 错失） |
| --- | --- | --- | ---: |
| cap1125-physical | cap1125-random | clean | +0.5799（1832.7 / 997.7） |
| cap1125-physical | cap1125-random | practical_high | +0.4481（1398.3 / 753.0） |
| cap1125-physical | cap1125-random | practical_mid | +0.4141（1295.3 / 699.0） |
| cap1125-physical | cap1125-random | practical_low_suburban | +0.4653（1273.7 / 603.7） |
| cap1125-physical | cap1125-random | practical_high_urban | +0.3766（1242.7 / 700.3） |
| cap1125-physical | cap1125-random | practical_mid_urban | +0.3012（934.3 / 500.7） |
| cap1125-physical | cap1125-random | practical_low_urban | +0.2387（706.0 / 362.3） |
| cap1125-physical | cap1125-confidence | clean | -0.2178（808.0 / 1121.7） |
| cap1125-physical | cap1125-confidence | practical_high | +0.1014（774.3 / 628.3） |
| cap1125-physical | cap1125-confidence | practical_mid | +0.0863（687.3 / 563.0） |
| cap1125-physical | cap1125-confidence | practical_low_suburban | +0.0970（638.0 / 498.3） |
| cap1125-physical | cap1125-confidence | practical_high_urban | +0.0528（694.3 / 618.3） |
| cap1125-physical | cap1125-confidence | practical_mid_urban | +0.0748（519.0 / 411.3） |
| cap1125-physical | cap1125-confidence | practical_low_urban | +0.0528（391.7 / 315.7） |
| cap125-physical | cap125-random | clean | +0.2074（1071.3 / 772.7） |
| cap125-physical | cap125-random | practical_high | +0.2218（862.7 / 543.3） |
| cap125-physical | cap125-random | practical_mid | +0.2331（821.0 / 485.3） |
| cap125-physical | cap125-random | practical_low_suburban | +0.2627（797.3 / 419.0） |
| cap125-physical | cap125-random | practical_high_urban | +0.2030（805.3 / 513.0） |
| cap125-physical | cap125-random | practical_mid_urban | +0.1750（611.0 / 359.0） |
| cap125-physical | cap125-random | practical_low_urban | +0.1500（490.3 / 274.3） |
| cap125-physical | cap125-confidence | clean | -0.0208（600.7 / 630.7） |
| cap125-physical | cap125-confidence | practical_high | +0.1028（567.3 / 419.3） |
| cap125-physical | cap125-confidence | practical_mid | +0.1218（537.7 / 362.3） |
| cap125-physical | cap125-confidence | practical_low_suburban | +0.0831（463.0 / 343.3） |
| cap125-physical | cap125-confidence | practical_high_urban | +0.0956（528.7 / 391.0） |
| cap125-physical | cap125-confidence | practical_mid_urban | +0.0863（388.3 / 264.0） |
| cap125-physical | cap125-confidence | practical_low_urban | +0.0627（280.7 / 190.3） |
| cap150-physical | cap150-random | clean | +0.0840（706.3 / 585.3） |
| cap150-physical | cap150-random | practical_high | +0.1185（618.7 / 448.0） |
| cap150-physical | cap150-random | practical_mid | +0.1021（576.0 / 429.0） |
| cap150-physical | cap150-random | practical_low_suburban | +0.1019（508.3 / 361.7） |
| cap150-physical | cap150-random | practical_high_urban | +0.0972（577.0 / 437.0） |
| cap150-physical | cap150-random | practical_mid_urban | +0.0748（401.7 / 294.0） |
| cap150-physical | cap150-random | practical_low_urban | +0.0414（284.3 / 224.7） |
| cap150-physical | cap150-confidence | clean | -0.0947（557.7 / 694.0） |
| cap150-physical | cap150-confidence | practical_high | -0.0407（473.7 / 532.3） |
| cap150-physical | cap150-confidence | practical_mid | -0.0412（424.0 / 483.3） |
| cap150-physical | cap150-confidence | practical_low_suburban | -0.0542（362.7 / 440.7） |
| cap150-physical | cap150-confidence | practical_high_urban | -0.0558（441.7 / 522.0） |
| cap150-physical | cap150-confidence | practical_mid_urban | -0.0444（295.7 / 359.7） |
| cap150-physical | cap150-confidence | practical_low_urban | -0.0241（216.0 / 250.7） |

## 稳定性与分层

| 方法 | 相对 Tri 下降种子×场景 | 相对 PL 下降种子×场景 | 相对 PL 最差三种子分层 |
| --- | ---: | ---: | --- |
| cap1125-random | 18 / 21 | 18 / 21 | transmitter、practical_low_suburban、8-20、-11.0639 pp |
| cap1125-confidence | 0 / 21 | 15 / 21 | transmitter、practical_low_suburban、8-20、-9.7403 pp |
| cap1125-physical | 0 / 21 | 15 / 21 | transmitter、practical_low_suburban、8-20、-8.9931 pp |
| cap125-random | 10 / 21 | 17 / 21 | transmitter、practical_low_suburban、8-20、-10.9792 pp |
| cap125-confidence | 4 / 21 | 15 / 21 | transmitter、practical_low_suburban、8-20、-10.3069 pp |
| cap125-physical | 3 / 21 | 15 / 21 | transmitter、practical_low_suburban、8-20、-9.8778 pp |
| cap150-random | 21 / 21 | 17 / 21 | transmitter、practical_low_suburban、8-20、-11.3722 pp |
| cap150-confidence | 7 / 21 | 17 / 21 | transmitter、practical_low_suburban、8-20、-10.7764 pp |
| cap150-physical | 12 / 21 | 17 / 21 | transmitter、practical_low_suburban、8-20、-11.1236 pp |

完整逐种子、TX/RX/day、七场景纠错与错失见本地 `paired_balancedcap_strata.csv`。配额只在每批实际出现的已接受伪类别中计算，单类别批次不会均衡。不同方法轨迹分开后，各批接受数不保证一致；不同配额的选择数量不同。测试目标逐景超过 PL 2 个百分点，须逐景判断，不能由均值替代。

九种配置的七景三种子平均全部低于普通 PL，未达到逐景领先 2 个百分点。物理排序在配额 1.125 下相对 Tri 的七景平均提高 0.2324 至 0.4523 个百分点，但相对 PL 仍下降 0.2440 至 0.7620 个百分点；宽松到 1.5 时，物理排序相对 Tri 的三种子平均仅有两个场景略升。全部配置和同配额随机、置信度对照均保留在上表，不以单个较好场景代替完整比较。

配额 1.125、1.25、1.5 分别保留原阈值接受源 U 暴露的约 87.75% 至 87.83%、93.29% 至 93.33%、98.44% 至 98.47%。这些是训练中的重复暴露，不是不同物理包数。每次 U 调用增加两个教师前向；续训墙钟时间约 5029 至 6256 秒，受共享 GPU 负载影响，不能作为严格速度比较。完整源 U 伪类别计数和时间见本地 `cost.json`。已曝光的历史测试结果限制了本批作为独立泛化证据的解释。
