# 单源接收机：伪类别均衡与物理分支敏感度排序

六种预登记的源 U 伪标签筛选方式各训练三个种子；原完整 Tri SSL 的模型、固定时间重排及各自同种子 E20 起点保持一致。全部 18 个模型的七景预测完成并冻结后独立评分。

本设置及历史测试结果已曝光，本批是开发验证，不是独立盲测。主口径为 DURA-LU、cross_rx 与 cross_both 合并、每种子每景 144000 包；同包比较核验物理 ID。普通 PL 使用自然时间顺序及不同训练机制，差异不能归因于某个组件。

| 方法 | 场景 | 候选准确率 % | 相对 Tri pp（纠错 / 错失） | 相对 PL pp（纠错 / 错失） |
| --- | --- | ---: | ---: | ---: |
| balance-random | clean | 49.2539 | -0.1919（742.0 / 1018.3） | -0.9708（7073.3 / 8471.3） |
| balance-random | practical_high | 55.1963 | -0.1361（606.0 / 802.0） | -0.6991（6050.3 / 7057.0） |
| balance-random | practical_mid | 54.1144 | -0.0900（575.3 / 705.0） | -0.8836（5159.0 / 6431.3） |
| balance-random | practical_low_suburban | 52.8998 | -0.0287（562.7 / 604.0） | -1.2431（4286.3 / 6076.3） |
| balance-random | practical_high_urban | 54.3919 | -0.1435（546.0 / 752.7） | -0.6345（5571.0 / 6484.7） |
| balance-random | practical_mid_urban | 50.8479 | -0.0697（404.3 / 504.7） | -0.6338（3660.7 / 4573.3） |
| balance-random | practical_low_urban | 47.5620 | +0.0009（313.3 / 312.0） | -0.5377（2504.0 / 3278.3） |
| balance-confidence | clean | 50.2903 | +0.8444（3434.0 / 2218.0） | +0.0655（7603.7 / 7509.3） |
| balance-confidence | practical_high | 55.7324 | +0.4000（2245.0 / 1669.0） | -0.1630（6432.3 / 6667.0） |
| balance-confidence | practical_mid | 54.6442 | +0.4398（2096.7 / 1463.3） | -0.3537（5510.7 / 6020.0） |
| balance-confidence | practical_low_suburban | 53.4894 | +0.5609（2050.7 / 1243.0） | -0.6535（4669.7 / 5610.7） |
| balance-confidence | practical_high_urban | 54.9000 | +0.3646（2052.3 / 1527.3） | -0.1264（5966.3 / 6148.3） |
| balance-confidence | practical_mid_urban | 51.2285 | +0.3109（1482.3 / 1034.7） | -0.2532（3913.0 / 4277.7） |
| balance-confidence | practical_low_urban | 47.8398 | +0.2787（1111.0 / 709.7） | -0.2600（2687.0 / 3061.3） |
| balance-ref | clean | 50.2120 | +0.7662（2901.0 / 1797.7） | -0.0127（7433.3 / 7451.7） |
| balance-ref | practical_high | 55.9356 | +0.6032（2095.7 / 1227.0） | +0.0403（6491.7 / 6433.7） |
| balance-ref | practical_mid | 54.8507 | +0.6463（2004.3 / 1073.7） | -0.1472（5545.7 / 5757.7） |
| balance-ref | practical_low_suburban | 53.6794 | +0.7509（2007.0 / 925.7） | -0.4634（4698.0 / 5365.3） |
| balance-ref | practical_high_urban | 55.0641 | +0.5287（1898.3 / 1137.0） | +0.0377（5988.7 / 5934.3） |
| balance-ref | practical_mid_urban | 51.3687 | +0.4512（1429.0 / 779.3） | -0.1130（3959.3 / 4122.0） |
| balance-ref | practical_low_urban | 47.9475 | +0.3863（1127.0 / 570.7） | -0.1523（2725.3 / 2944.7） |
| balance-uniform | clean | 50.1014 | +0.6556（3260.3 / 2316.3） | -0.1234（7535.7 / 7713.3） |
| balance-uniform | practical_high | 55.6907 | +0.3583（2139.7 / 1623.7） | -0.2046（6483.7 / 6778.3） |
| balance-uniform | practical_mid | 54.5542 | +0.3498（1946.3 / 1442.7） | -0.4437（5510.0 / 6149.0） |
| balance-uniform | practical_low_suburban | 53.3845 | +0.4560（1902.3 / 1245.7） | -0.7583（4655.7 / 5747.7） |
| balance-uniform | practical_high_urban | 54.8326 | +0.2972（1940.0 / 1512.0） | -0.1938（5988.0 / 6267.0） |
| balance-uniform | practical_mid_urban | 51.1463 | +0.2287（1377.0 / 1047.7） | -0.3354（3900.3 / 4383.3） |
| balance-uniform | practical_low_urban | 47.7880 | +0.2269（1036.3 / 709.7） | -0.3118（2692.7 / 3141.7） |
| balance-reffreq | clean | 50.1215 | +0.6757（2620.0 / 1647.0） | -0.1032（7437.0 / 7585.7） |
| balance-reffreq | practical_high | 55.8602 | +0.5278（1945.7 / 1185.7） | -0.0352（6422.7 / 6473.3） |
| balance-reffreq | practical_mid | 54.7775 | +0.5731（1885.3 / 1060.0） | -0.2204（5506.0 / 5823.3） |
| balance-reffreq | practical_low_suburban | 53.6514 | +0.7229（1938.0 / 897.0） | -0.4914（4696.7 / 5404.3） |
| balance-reffreq | practical_high_urban | 54.9891 | +0.4537（1761.0 / 1107.7） | -0.0373（5922.0 / 5975.7） |
| balance-reffreq | practical_mid_urban | 51.3269 | +0.4093（1349.0 / 759.7） | -0.1549（3933.3 / 4156.3） |
| balance-reffreq | practical_low_urban | 47.9262 | +0.3650（1071.7 / 546.0） | -0.1736（2724.0 / 2974.0） |
| balance-ref-relaxed | clean | 49.4741 | +0.0282（498.7 / 458.0） | -0.7507（7313.7 / 8394.7） |
| balance-ref-relaxed | practical_high | 55.3194 | -0.0130（394.0 / 412.7） | -0.5759（6209.7 / 7039.0） |
| balance-ref-relaxed | practical_mid | 54.2012 | -0.0032（354.7 / 359.3） | -0.7968（5292.7 / 6440.0） |
| balance-ref-relaxed | practical_low_suburban | 52.9600 | +0.0315（349.0 / 303.7） | -1.1829（4415.7 / 6119.0） |
| balance-ref-relaxed | practical_high_urban | 54.5463 | +0.0109（371.3 / 355.7） | -0.4801（5741.7 / 6433.0） |
| balance-ref-relaxed | practical_mid_urban | 50.8988 | -0.0187（244.3 / 271.3） | -0.5829（3744.3 / 4583.7） |
| balance-ref-relaxed | practical_low_urban | 47.5752 | +0.0141（201.7 / 181.3） | -0.5245（2554.7 / 3310.0） |

## 均衡与排序机制对照

| 候选 | 对照 | 场景 | 差异 pp（纠错 / 错失） |
| --- | --- | --- | ---: |
| balance-ref | balance-random | clean | +0.9581（2687.7 / 1308.0） |
| balance-ref | balance-random | practical_high | +0.7394（1988.7 / 924.0） |
| balance-ref | balance-random | practical_mid | +0.7363（1870.0 / 809.7） |
| balance-ref | balance-random | practical_low_suburban | +0.7796（1807.7 / 685.0） |
| balance-ref | balance-random | practical_high_urban | +0.6722（1821.0 / 853.0） |
| balance-ref | balance-random | practical_mid_urban | +0.5208（1338.7 / 588.7） |
| balance-ref | balance-random | practical_low_urban | +0.3854（1011.3 / 456.3） |
| balance-ref | balance-confidence | clean | -0.0782（1147.3 / 1260.0） |
| balance-ref | balance-confidence | practical_high | +0.2032（1153.3 / 860.7） |
| balance-ref | balance-confidence | practical_mid | +0.2065（1056.3 / 759.0） |
| balance-ref | balance-confidence | practical_low_suburban | +0.1900（958.0 / 684.3） |
| balance-ref | balance-confidence | practical_high_urban | +0.1641（1047.7 / 811.3） |
| balance-ref | balance-confidence | practical_mid_urban | +0.1403（753.3 / 551.3） |
| balance-ref | balance-confidence | practical_low_urban | +0.1076（571.7 / 416.7） |
| balance-ref | balance-uniform | clean | +0.1106（1224.3 / 1065.0） |
| balance-ref | balance-uniform | practical_high | +0.2449（1096.3 / 743.7） |
| balance-ref | balance-uniform | practical_mid | +0.2965（1036.3 / 609.3） |
| balance-ref | balance-uniform | practical_low_suburban | +0.2949（957.7 / 533.0） |
| balance-ref | balance-uniform | practical_high_urban | +0.2315（1021.0 / 687.7） |
| balance-ref | balance-uniform | practical_mid_urban | +0.2225（755.7 / 435.3） |
| balance-ref | balance-uniform | practical_low_urban | +0.1595（561.0 / 331.3） |
| balance-reffreq | balance-ref | clean | -0.0905（661.7 / 792.0） |
| balance-reffreq | balance-ref | practical_high | -0.0755（521.7 / 630.3） |
| balance-reffreq | balance-ref | practical_mid | -0.0731（476.7 / 582.0） |
| balance-reffreq | balance-ref | practical_low_suburban | -0.0280（447.7 / 488.0） |
| balance-reffreq | balance-ref | practical_high_urban | -0.0750（493.7 / 601.7） |
| balance-reffreq | balance-ref | practical_mid_urban | -0.0419（335.7 / 396.0） |
| balance-reffreq | balance-ref | practical_low_urban | -0.0213（250.3 / 281.0） |
| balance-ref-relaxed | balance-ref | clean | -0.7380（1694.3 / 2757.0） |
| balance-ref-relaxed | balance-ref | practical_high | -0.6162（1142.3 / 2029.7） |
| balance-ref-relaxed | balance-ref | practical_mid | -0.6495（1001.3 / 1936.7） |
| balance-ref-relaxed | balance-ref | practical_low_suburban | -0.7194（875.3 / 1911.3） |
| balance-ref-relaxed | balance-ref | practical_high_urban | -0.5178（1094.0 / 1839.7） |
| balance-ref-relaxed | balance-ref | practical_mid_urban | -0.4699（726.0 / 1402.7） |
| balance-ref-relaxed | balance-ref | practical_low_urban | -0.3722（540.3 / 1076.3） |

## 稳定性与分层

| 方法 | 相对 Tri 下降种子×场景 | 相对 PL 下降种子×场景 | 相对 PL 最差三种子分层 |
| --- | ---: | ---: | --- |
| balance-random | 18 / 21 | 18 / 21 | transmitter、practical_low_suburban、8-20、-10.5778 pp |
| balance-confidence | 0 / 21 | 13 / 21 | transmitter、practical_low_suburban、8-20、-9.5333 pp |
| balance-ref | 0 / 21 | 12 / 21 | transmitter、practical_low_suburban、8-20、-8.5083 pp |
| balance-uniform | 0 / 21 | 15 / 21 | transmitter、practical_low_suburban、8-20、-9.2597 pp |
| balance-reffreq | 0 / 21 | 12 / 21 | transmitter、practical_low_suburban、8-20、-8.5083 pp |
| balance-ref-relaxed | 12 / 21 | 17 / 21 | transmitter、practical_low_suburban、8-20、-10.5931 pp |

完整逐种子、TX/RX/day、七场景纠错与错失见本地 `paired_balanced_strata.csv`。配额只在每批实际出现的已接受伪类别中计算，单类别批次不会均衡。不同方法轨迹分开后，各批接受数不保证一致；均值匹配对照使用批内一致的平均衰减。测试目标逐景超过 PL 2 个百分点，须逐景判断，不能由均值替代。

`balance-ref` 相对 Tri 七景三种子平均提高 0.3863 至 0.7662 个百分点，21 个种子与场景组合全部提高。相对普通 PL，仅高郊区和高城市的三种子平均分别提高 0.0403 和 0.0377 个百分点，其余五景下降；21 个种子与场景组合有 12 个下降，未达到逐景领先 2 个百分点。尤其 clean 的三个种子相对 PL 分别为 +2.2611、+0.8292 和 −3.1285 个百分点，不能用均值掩盖种子差异。

与同类配额的置信度排序相比，参考响应敏感度排序在六个 practical 场景的三个种子中全部提高，三种子平均幅度为 0.1076 至 0.2065 个百分点；clean 的三个种子均下降，平均下降 0.0782 个百分点。与均值匹配衰减相比，参考响应排序的六个 practical 场景和三个种子也均提高，clean 有一个种子略降。这提供了本设置下物理敏感度排序有作用的证据，但历史测试已暴露，尚不能宣称普遍优势或独立盲测成功。

严格均衡各组每模型保留约 135.5 万至 135.9 万次源 U 样本暴露，原阈值接受约 167.2 万至 167.8 万次；宽松配额几乎保留全部原接受样本。这些是重复训练暴露次数，不是不同物理包数。原接受伪类别在累计层面已相当接近，严格逐批配额仍丢弃约 19% 的接受暴露，因而不能把结果归因于纠正了整体真实类别比例。每次 U 调用增加两个教师前向，续训墙钟时间约 4549 至 5811 秒；GPU 共卡负载不同，不能作严格速度比较。详细伪类别计数与成本见本地 `cost.json`。
