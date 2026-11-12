# 单源接收机：物理分支敏感度排序的温度与分支

七种预登记的源 U 伪标签排序方式各训练三个种子；原完整 Tri SSL 的模型、固定时间重排及各自同种子 E20 起点保持一致。全部 21 个模型的七景预测完成并冻结后独立评分。

本设置及历史测试结果已曝光，本批是开发验证，不是独立盲测。主口径为 DURA-LU、cross_rx 与 cross_both 合并、每种子每景 144000 包；同包比较核验物理 ID。普通 PL 使用自然时间顺序及不同训练机制，差异不能归因于某个组件。

| 方法 | 场景 | 候选准确率 % | 相对 Tri pp（纠错 / 错失） | 相对 PL pp（纠错 / 错失） |
| --- | --- | ---: | ---: | ---: |
| ref-t2 | clean | 50.1426 | +0.6968（2502.7 / 1499.3） | -0.0822（7705.3 / 7823.7） |
| ref-t2 | practical_high | 55.9231 | +0.5907（1838.0 / 987.3） | +0.0278（6621.7 / 6581.7） |
| ref-t2 | practical_mid | 54.8234 | +0.6190（1771.3 / 880.0） | -0.1745（5650.3 / 5901.7） |
| ref-t2 | practical_low_suburban | 53.6648 | +0.7363（1810.0 / 749.7） | -0.4780（4807.7 / 5496.0） |
| ref-t2 | practical_high_urban | 55.0479 | +0.5125（1674.0 / 936.0） | +0.0215（6103.0 / 6072.0） |
| ref-t2 | practical_mid_urban | 51.3720 | +0.4544（1287.0 / 632.7） | -0.1097（4038.0 / 4196.0） |
| ref-t2 | practical_low_urban | 47.9308 | +0.3697（1005.0 / 472.7） | -0.1690（2775.0 / 3018.3） |
| ref-t4 | clean | 49.8463 | +0.4005（2160.0 / 1583.3） | -0.3785（7302.7 / 7847.7） |
| ref-t4 | practical_high | 55.5794 | +0.2470（1392.0 / 1036.3） | -0.3160（6345.7 / 6800.7） |
| ref-t4 | practical_mid | 54.5206 | +0.3162（1374.0 / 918.7） | -0.4773（5412.3 / 6099.7） |
| ref-t4 | practical_low_suburban | 53.3361 | +0.4076（1384.3 / 797.3） | -0.8067（4543.0 / 5704.7） |
| ref-t4 | practical_high_urban | 54.7458 | +0.2104（1282.7 / 979.7） | -0.2806（5837.0 / 6241.0） |
| ref-t4 | practical_mid_urban | 51.1606 | +0.2431（1021.0 / 671.0） | -0.3211（3886.3 / 4348.7） |
| ref-t4 | practical_low_urban | 47.7785 | +0.2174（804.7 / 491.7） | -0.3213（2663.0 / 3125.7） |
| reffreq-t2 | clean | 49.8581 | +0.4123（2195.7 / 1602.0） | -0.3667（7440.7 / 7968.7） |
| reffreq-t2 | practical_high | 55.7565 | +0.4241（1682.7 / 1072.0） | -0.1389（6494.7 / 6694.7） |
| reffreq-t2 | practical_mid | 54.6681 | +0.4637（1617.3 / 949.7） | -0.3299（5552.7 / 6027.7） |
| reffreq-t2 | practical_low_suburban | 53.5164 | +0.5880（1654.3 / 807.7） | -0.6264（4718.0 / 5620.0） |
| reffreq-t2 | practical_high_urban | 54.8785 | +0.3431（1517.7 / 1023.7） | -0.1479（5965.0 / 6178.0） |
| reffreq-t2 | practical_mid_urban | 51.2806 | +0.3630（1191.0 / 668.3） | -0.2012（3983.3 / 4273.0） |
| reffreq-t2 | practical_low_urban | 47.8553 | +0.2942（919.7 / 496.0） | -0.2444（2733.3 / 3085.3） |
| reffreq-t4 | clean | 49.8528 | +0.4069（2133.0 / 1547.0） | -0.3720（7449.3 / 7985.0） |
| reffreq-t4 | practical_high | 55.6771 | +0.3447（1531.7 / 1035.3） | -0.2183（6534.0 / 6848.3） |
| reffreq-t4 | practical_mid | 54.5685 | +0.3641（1460.3 / 936.0） | -0.4294（5571.7 / 6190.0） |
| reffreq-t4 | practical_low_suburban | 53.3896 | +0.4611（1460.0 / 796.0） | -0.7532（4702.0 / 5786.7） |
| reffreq-t4 | practical_high_urban | 54.8183 | +0.2829（1401.7 / 994.3） | -0.2081（5997.0 / 6296.7） |
| reffreq-t4 | practical_mid_urban | 51.2118 | +0.2942（1075.0 / 651.3） | -0.2699（3983.7 / 4372.3） |
| reffreq-t4 | practical_low_urban | 47.8058 | +0.2447（836.3 / 484.0） | -0.2940（2732.3 / 3155.7） |
| refpa-main-t1 | clean | 49.9933 | +0.5475（2439.3 / 1651.0） | -0.2315（7547.0 / 7880.3） |
| refpa-main-t1 | practical_high | 55.8363 | +0.5039（1885.7 / 1160.0） | -0.0590（6431.0 / 6516.0） |
| refpa-main-t1 | practical_mid | 54.7498 | +0.5454（1805.7 / 1020.3） | -0.2481（5503.7 / 5861.0） |
| refpa-main-t1 | practical_low_suburban | 53.6023 | +0.6738（1844.0 / 873.7） | -0.5405（4667.7 / 5446.0） |
| refpa-main-t1 | practical_high_urban | 54.9340 | +0.3986（1701.0 / 1127.0） | -0.0924（5924.0 / 6057.0） |
| refpa-main-t1 | practical_mid_urban | 51.3190 | +0.4014（1313.7 / 735.7） | -0.1627（3941.7 / 4176.0） |
| refpa-main-t1 | practical_low_urban | 47.9023 | +0.3412（1028.3 / 537.0） | -0.1975（2709.0 / 2993.3） |
| refpa-main-t2 | clean | 50.0248 | +0.5789（2177.7 / 1344.0） | -0.2000（7584.7 / 7872.7） |
| refpa-main-t2 | practical_high | 55.8255 | +0.4931（1623.3 / 913.3） | -0.0699（6495.7 / 6596.3） |
| refpa-main-t2 | practical_mid | 54.7162 | +0.5118（1556.3 / 819.3） | -0.2817（5535.0 / 5940.7） |
| refpa-main-t2 | practical_low_suburban | 53.5701 | +0.6417（1624.7 / 700.7） | -0.5727（4706.0 / 5530.7） |
| refpa-main-t2 | practical_high_urban | 54.9421 | +0.4067（1446.7 / 861.0） | -0.0843（5968.0 / 6089.3） |
| refpa-main-t2 | practical_mid_urban | 51.3118 | +0.3942（1157.3 / 589.7） | -0.1699（3966.3 / 4211.0） |
| refpa-main-t2 | practical_low_urban | 47.8910 | +0.3299（929.3 / 454.3） | -0.2088（2736.7 / 3037.3） |
| refpa-main-t4 | clean | 49.7139 | +0.2681（1688.0 / 1302.0） | -0.5109（7235.7 / 7971.3） |
| refpa-main-t4 | practical_high | 55.6243 | +0.2919（1307.3 / 887.0） | -0.2711（6282.0 / 6672.3） |
| refpa-main-t4 | practical_mid | 54.5498 | +0.3454（1267.7 / 770.3） | -0.4481（5360.3 / 6005.7） |
| refpa-main-t4 | practical_low_suburban | 53.3803 | +0.4519（1316.7 / 666.0） | -0.7625（4490.0 / 5588.0） |
| refpa-main-t4 | practical_high_urban | 54.7676 | +0.2322（1167.0 / 832.7） | -0.2588（5774.7 / 6147.3） |
| refpa-main-t4 | practical_mid_urban | 51.1873 | +0.2697（965.0 / 576.7） | -0.2944（3833.3 / 4257.3） |
| refpa-main-t4 | practical_low_urban | 47.8000 | +0.2389（765.7 / 421.7） | -0.2998（2628.7 / 3060.3） |

## 均衡与排序机制对照

| 候选 | 对照 | 场景 | 差异 pp（纠错 / 错失） |
| --- | --- | --- | ---: |
| ref-t2 | balance-ref | clean | -0.0694（1152.7 / 1252.7） |
| ref-t2 | balance-ref | practical_high | -0.0125（915.0 / 933.0） |
| ref-t2 | balance-ref | practical_mid | -0.0273（801.0 / 840.3） |
| ref-t2 | balance-ref | practical_low_suburban | -0.0146（723.3 / 744.3） |
| ref-t2 | balance-ref | practical_high_urban | -0.0162（861.3 / 884.7） |
| ref-t2 | balance-ref | practical_mid_urban | +0.0032（584.7 / 580.0） |
| ref-t2 | balance-ref | practical_low_urban | -0.0167（395.0 / 419.0） |
| ref-t4 | balance-ref | clean | -0.3657（1255.7 / 1782.3） |
| ref-t4 | balance-ref | practical_high | -0.3562（907.7 / 1420.7） |
| ref-t4 | balance-ref | practical_mid | -0.3301（793.0 / 1268.3） |
| ref-t4 | balance-ref | practical_low_suburban | -0.3433（675.7 / 1170.0） |
| ref-t4 | balance-ref | practical_high_urban | -0.3183（843.7 / 1302.0） |
| ref-t4 | balance-ref | practical_mid_urban | -0.2081（583.0 / 882.7） |
| ref-t4 | balance-ref | practical_low_urban | -0.1690（393.7 / 637.0） |
| reffreq-t2 | balance-reffreq | clean | -0.2634（884.0 / 1263.3） |
| reffreq-t2 | balance-reffreq | practical_high | -0.1037（772.7 / 922.0） |
| reffreq-t2 | balance-reffreq | practical_mid | -0.1095（674.7 / 832.3） |
| reffreq-t2 | balance-reffreq | practical_low_suburban | -0.1350（590.7 / 785.0） |
| reffreq-t2 | balance-reffreq | practical_high_urban | -0.1106（705.7 / 865.0） |
| reffreq-t2 | balance-reffreq | practical_mid_urban | -0.0463（502.3 / 569.0） |
| reffreq-t2 | balance-reffreq | practical_low_urban | -0.0708（336.3 / 438.3） |
| reffreq-t4 | balance-reffreq | clean | -0.2687（1050.3 / 1437.3） |
| reffreq-t4 | balance-reffreq | practical_high | -0.1831（909.3 / 1173.0） |
| reffreq-t4 | balance-reffreq | practical_mid | -0.2090（794.0 / 1095.0） |
| reffreq-t4 | balance-reffreq | practical_low_suburban | -0.2618（665.0 / 1042.0） |
| reffreq-t4 | balance-reffreq | practical_high_urban | -0.1708（863.0 / 1109.0） |
| reffreq-t4 | balance-reffreq | practical_mid_urban | -0.1150（589.7 / 755.3） |
| reffreq-t4 | balance-reffreq | practical_low_urban | -0.1204（389.0 / 562.3） |
| refpa-main-t1 | balance-ref | clean | -0.2188（1140.0 / 1455.0） |
| refpa-main-t1 | balance-ref | practical_high | -0.0993（825.0 / 968.0） |
| refpa-main-t1 | balance-ref | practical_mid | -0.1009（715.3 / 860.7） |
| refpa-main-t1 | balance-ref | practical_low_suburban | -0.0771（623.3 / 734.3） |
| refpa-main-t1 | balance-ref | practical_high_urban | -0.1301（764.0 / 951.3） |
| refpa-main-t1 | balance-ref | practical_mid_urban | -0.0498（509.3 / 581.0） |
| refpa-main-t1 | balance-ref | practical_low_urban | -0.0451（340.0 / 405.0） |
| refpa-main-t2 | balance-ref | clean | -0.1873（1340.7 / 1610.3） |
| refpa-main-t2 | balance-ref | practical_high | -0.1102（1010.7 / 1169.3） |
| refpa-main-t2 | balance-ref | practical_mid | -0.1345（886.3 / 1080.0） |
| refpa-main-t2 | balance-ref | practical_low_suburban | -0.1093（787.3 / 944.7） |
| refpa-main-t2 | balance-ref | practical_high_urban | -0.1220（925.0 / 1100.7） |
| refpa-main-t2 | balance-ref | practical_mid_urban | -0.0569（642.0 / 724.0） |
| refpa-main-t2 | balance-ref | practical_low_urban | -0.0565（448.3 / 529.7） |
| refpa-main-t4 | balance-ref | clean | -0.4981（1251.3 / 1968.7） |
| refpa-main-t4 | balance-ref | practical_high | -0.3113（921.0 / 1369.3） |
| refpa-main-t4 | balance-ref | practical_mid | -0.3009（813.0 / 1246.3） |
| refpa-main-t4 | balance-ref | practical_low_suburban | -0.2991（709.0 / 1139.7） |
| refpa-main-t4 | balance-ref | practical_high_urban | -0.2965（832.7 / 1259.7） |
| refpa-main-t4 | balance-ref | practical_mid_urban | -0.1815（595.0 / 856.3） |
| refpa-main-t4 | balance-ref | practical_low_urban | -0.1475（420.0 / 632.3） |

## 稳定性与分层

| 方法 | 相对 Tri 下降种子×场景 | 相对 PL 下降种子×场景 | 相对 PL 最差三种子分层 |
| --- | ---: | ---: | --- |
| ref-t2 | 0 / 21 | 11 / 21 | transmitter、practical_low_suburban、8-20、-8.9347 pp |
| ref-t4 | 2 / 21 | 13 / 21 | transmitter、practical_low_suburban、8-20、-9.7069 pp |
| reffreq-t2 | 0 / 21 | 13 / 21 | transmitter、practical_low_suburban、8-20、-9.2597 pp |
| reffreq-t4 | 1 / 21 | 15 / 21 | transmitter、practical_low_suburban、8-20、-10.1292 pp |
| refpa-main-t1 | 0 / 21 | 13 / 21 | transmitter、practical_low_suburban、8-20、-8.4778 pp |
| refpa-main-t2 | 0 / 21 | 13 / 21 | transmitter、practical_low_suburban、8-20、-8.9181 pp |
| refpa-main-t4 | 0 / 21 | 15 / 21 | transmitter、practical_low_suburban、8-20、-9.1181 pp |

完整逐种子、TX/RX/day、七场景纠错与错失见本地 `paired_ranktemp_strata.csv`。配额只在每批实际出现的已接受伪类别中计算，单类别批次不会均衡。不同方法轨迹分开后，各批接受数不保证一致；温度改变教师的排序分数，不改变伪标签阈值。PA 对照只缩放主投影，统计投影和 pa_delta 保持原样。测试目标逐景超过 PL 2 个百分点，须逐景判断，不能由均值替代。

七种配置均未达到逐景领先普通 PL 两个百分点。参考分支 T2 相对 PL 仅高郊区和高城市的三种子平均分别提高 0.0278 和 0.0215 个百分点，另外五景下降；其余六组在七景平均均低于 PL。参考分支 T2 相对 Tri 的七景三种子平均提高 0.3697 至 0.7363 个百分点，但温度增加并未产生稳定的额外收益；与前一轮参考分支 T1 的直接同包对照见上表。PA 配置只测试 PA 主投影，不能推断完整 PA 分支的作用。

每模型续训 30 轮保留约 135.6 万至 135.8 万次源 U 暴露，原阈值接受约 167.3 万至 167.7 万次。这些是重复训练暴露而不是不同物理包数。每次 U 调用增加两个教师前向，续训墙钟时间约 5448 至 6043 秒；共享 GPU 负载不同，不能作严格速度比较。源 U 伪类别计数与时间见本地 `cost.json`。该设置和历史测试已暴露，结论限于本批开发验证。
