# 单源接收机：物理分支教师集成

六种预登记的源 U 教师集成方式各训练三个种子；原完整 Tri SSL 的模型、固定时间重排及各自同种子 E20 起点保持一致。全部 18 个模型的七景预测完成并冻结后独立评分。

本设置及历史测试结果已曝光，本批是开发验证，不是独立盲测。主口径为 DURA-LU、cross_rx 与 cross_both 合并、每种子每景 144000 包；同包比较核验物理 ID。普通 PL 使用自然时间顺序及不同训练机制，差异不能归因于某个组件。

| 方法 | 场景 | 候选准确率 % | 相对 Tri pp（纠错 / 错失） | 相对 PL pp（纠错 / 错失） |
| --- | --- | ---: | ---: | ---: |
| teacher-full025 | clean | 49.3269 | -0.1190（1087.3 / 1258.7） | -0.8979（7114.7 / 8407.7） |
| teacher-full025 | practical_high | 55.2347 | -0.0977（885.7 / 1026.3） | -0.6606（6173.3 / 7124.7） |
| teacher-full025 | practical_mid | 54.0479 | -0.1565（784.3 / 1009.7） | -0.9500（5177.7 / 6545.7） |
| teacher-full025 | practical_low_suburban | 52.7947 | -0.1338（711.3 / 904.0） | -1.3481（4283.0 / 6224.3） |
| teacher-full025 | practical_high_urban | 54.4218 | -0.1137（834.3 / 998.0） | -0.6046（5694.3 / 6565.0） |
| teacher-full025 | practical_mid_urban | 50.8306 | -0.0870（589.7 / 715.0） | -0.6512（3693.0 / 4630.7） |
| teacher-full025 | practical_low_urban | 47.5310 | -0.0301（439.3 / 482.7） | -0.5687（2535.0 / 3354.0） |
| teacher-ref025 | clean | 49.2590 | -0.1868（1124.0 / 1393.0） | -0.9657（7071.7 / 8462.3） |
| teacher-ref025 | practical_high | 55.2030 | -0.1294（928.7 / 1115.0） | -0.6924（6121.0 / 7118.0） |
| teacher-ref025 | practical_mid | 54.0116 | -0.1928（834.3 / 1112.0） | -0.9863（5132.7 / 6553.0） |
| teacher-ref025 | practical_low_suburban | 52.7444 | -0.1840（752.0 / 1017.0） | -1.3984（4219.7 / 6233.3） |
| teacher-ref025 | practical_high_urban | 54.4111 | -0.1243（886.7 / 1065.7） | -0.6153（5660.0 / 6546.0） |
| teacher-ref025 | practical_mid_urban | 50.7993 | -0.1183（592.7 / 763.0） | -0.6824（3651.3 / 4634.0） |
| teacher-ref025 | practical_low_urban | 47.5039 | -0.0572（457.7 / 540.0） | -0.5958（2500.3 / 3358.3） |
| teacher-uniform025 | clean | 49.2081 | -0.2377（1113.7 / 1456.0） | -1.0167（7042.7 / 8506.7） |
| teacher-uniform025 | practical_high | 55.1958 | -0.1366（963.7 / 1160.3） | -0.6995（6179.7 / 7187.0） |
| teacher-uniform025 | practical_mid | 53.9912 | -0.2132（823.3 / 1130.3） | -1.0067（5155.3 / 6605.0） |
| teacher-uniform025 | practical_low_suburban | 52.7081 | -0.2204（717.0 / 1034.3） | -1.4347（4228.7 / 6294.7） |
| teacher-uniform025 | practical_high_urban | 54.3965 | -0.1389（894.7 / 1094.7） | -0.6299（5695.0 / 6602.0） |
| teacher-uniform025 | practical_mid_urban | 50.7921 | -0.1255（598.7 / 779.3） | -0.6896（3674.7 / 4667.7） |
| teacher-uniform025 | practical_low_urban | 47.4787 | -0.0824（421.7 / 540.3） | -0.6211（2505.3 / 3399.7） |
| teacher-ref1 | clean | 49.4051 | -0.0407（3759.0 / 3817.7） | -0.8197（7242.7 / 8423.0） |
| teacher-ref1 | practical_high | 55.3310 | -0.0014（2526.0 / 2528.0） | -0.5644（6155.0 / 6967.7） |
| teacher-ref1 | practical_mid | 54.0164 | -0.1880（2195.7 / 2466.3） | -0.9815（5134.3 / 6547.7） |
| teacher-ref1 | practical_low_suburban | 52.6484 | -0.2801（1962.7 / 2366.0） | -1.4944（4122.0 / 6274.0） |
| teacher-ref1 | practical_high_urban | 54.5347 | -0.0007（2338.0 / 2339.0） | -0.4917（5676.3 / 6384.3） |
| teacher-ref1 | practical_mid_urban | 50.8326 | -0.0850（1618.0 / 1740.3） | -0.6491（3698.7 / 4633.3） |
| teacher-ref1 | practical_low_urban | 47.5259 | -0.0352（1227.3 / 1278.0） | -0.5738（2576.3 / 3402.7） |
| teacher-reffreq025 | clean | 49.1241 | -0.3218（1073.7 / 1537.0） | -1.1007（6939.0 / 8524.0） |
| teacher-reffreq025 | practical_high | 55.1472 | -0.1852（875.3 / 1142.0） | -0.7481（6063.3 / 7140.7） |
| teacher-reffreq025 | practical_mid | 53.9428 | -0.2616（754.0 / 1130.7） | -1.0551（5048.3 / 6567.7） |
| teacher-reffreq025 | practical_low_suburban | 52.6593 | -0.2692（673.0 / 1060.7） | -1.4836（4118.0 / 6254.3） |
| teacher-reffreq025 | practical_high_urban | 54.3507 | -0.1847（816.3 / 1082.3） | -0.6757（5581.3 / 6554.3） |
| teacher-reffreq025 | practical_mid_urban | 50.7400 | -0.1775（544.0 / 799.7） | -0.7417（3596.0 / 4664.0） |
| teacher-reffreq025 | practical_low_urban | 47.4648 | -0.0963（408.0 / 546.7） | -0.6350（2453.3 / 3367.7） |
| teacher-refpa-main025 | clean | 49.2611 | -0.1847（1004.3 / 1270.3） | -0.9637（7082.7 / 8470.3） |
| teacher-refpa-main025 | practical_high | 55.2910 | -0.0414（850.3 / 910.0） | -0.6044（6225.3 / 7095.7） |
| teacher-refpa-main025 | practical_mid | 54.1157 | -0.0887（757.0 / 884.7） | -0.8822（5226.3 / 6496.7） |
| teacher-refpa-main025 | practical_low_suburban | 52.8519 | -0.0766（680.7 / 791.0） | -1.2910（4332.3 / 6191.3） |
| teacher-refpa-main025 | practical_high_urban | 54.5183 | -0.0171（828.0 / 852.7） | -0.5081（5755.0 / 6486.7） |
| teacher-refpa-main025 | practical_mid_urban | 50.8586 | -0.0590（542.7 / 627.7） | -0.6231（3719.3 / 4616.7） |
| teacher-refpa-main025 | practical_low_urban | 47.5405 | -0.0206（415.7 / 445.3） | -0.5593（2549.3 / 3354.7） |

## 蒸馏机制对照

| 候选 | 对照 | 场景 | 差异 pp（纠错 / 错失） |
| --- | --- | --- | ---: |
| teacher-ref025 | teacher-full025 | clean | -0.0678（538.3 / 636.0） |
| teacher-ref025 | teacher-full025 | practical_high | -0.0317（482.0 / 527.7） |
| teacher-ref025 | teacher-full025 | practical_mid | -0.0363（445.0 / 497.3） |
| teacher-ref025 | teacher-full025 | practical_low_suburban | -0.0502（384.7 / 457.0） |
| teacher-ref025 | teacher-full025 | practical_high_urban | -0.0106（468.0 / 483.3） |
| teacher-ref025 | teacher-full025 | practical_mid_urban | -0.0312（306.7 / 351.7） |
| teacher-ref025 | teacher-full025 | practical_low_urban | -0.0271（224.7 / 263.7） |
| teacher-ref025 | teacher-uniform025 | clean | +0.0509（749.7 / 676.3） |
| teacher-ref025 | teacher-uniform025 | practical_high | +0.0072（546.0 / 535.7） |
| teacher-ref025 | teacher-uniform025 | practical_mid | +0.0204（526.0 / 496.7） |
| teacher-ref025 | teacher-uniform025 | practical_low_suburban | +0.0363（497.3 / 445.0） |
| teacher-ref025 | teacher-uniform025 | practical_high_urban | +0.0146（515.0 / 494.0） |
| teacher-ref025 | teacher-uniform025 | practical_mid_urban | +0.0072（371.7 / 361.3） |
| teacher-ref025 | teacher-uniform025 | practical_low_urban | +0.0252（295.3 / 259.0） |
| teacher-ref1 | teacher-ref025 | clean | +0.1461（3097.7 / 2887.3） |
| teacher-ref1 | teacher-ref025 | practical_high | +0.1280（2082.7 / 1898.3） |
| teacher-ref1 | teacher-ref025 | practical_mid | +0.0049（1802.0 / 1795.0） |
| teacher-ref1 | teacher-ref025 | practical_low_suburban | -0.0961（1597.3 / 1735.7） |
| teacher-ref1 | teacher-ref025 | practical_high_urban | +0.1236（1926.0 / 1748.0） |
| teacher-ref1 | teacher-ref025 | practical_mid_urban | +0.0333（1321.3 / 1273.3） |
| teacher-ref1 | teacher-ref025 | practical_low_urban | +0.0220（972.0 / 940.3） |
| teacher-reffreq025 | teacher-ref025 | clean | -0.1350（550.7 / 745.0） |
| teacher-reffreq025 | teacher-ref025 | practical_high | -0.0558（528.0 / 608.3） |
| teacher-reffreq025 | teacher-ref025 | practical_mid | -0.0688（474.0 / 573.0） |
| teacher-reffreq025 | teacher-ref025 | practical_low_suburban | -0.0852（423.0 / 545.7） |
| teacher-reffreq025 | teacher-ref025 | practical_high_urban | -0.0604（486.3 / 573.3） |
| teacher-reffreq025 | teacher-ref025 | practical_mid_urban | -0.0593（338.0 / 423.3） |
| teacher-reffreq025 | teacher-ref025 | practical_low_urban | -0.0391（247.7 / 304.0） |
| teacher-refpa-main025 | teacher-ref025 | clean | +0.0021（608.0 / 605.0） |
| teacher-refpa-main025 | teacher-ref025 | practical_high | +0.0880（576.7 / 450.0） |
| teacher-refpa-main025 | teacher-ref025 | practical_mid | +0.1042（571.3 / 421.3） |
| teacher-refpa-main025 | teacher-ref025 | practical_low_suburban | +0.1074（543.3 / 388.7） |
| teacher-refpa-main025 | teacher-ref025 | practical_high_urban | +0.1072（563.0 / 408.7） |
| teacher-refpa-main025 | teacher-ref025 | practical_mid_urban | +0.0593（386.3 / 301.0） |
| teacher-refpa-main025 | teacher-ref025 | practical_low_urban | +0.0366（290.3 / 237.7） |

## 稳定性与分层

| 方法 | 相对 Tri 下降种子×场景 | 相对 PL 下降种子×场景 | 相对 PL 最差三种子分层 |
| --- | ---: | ---: | --- |
| teacher-full025 | 15 / 21 | 17 / 21 | transmitter、practical_low_suburban、8-20、-11.6667 pp |
| teacher-ref025 | 12 / 21 | 17 / 21 | transmitter、practical_low_suburban、8-20、-11.7375 pp |
| teacher-uniform025 | 17 / 21 | 17 / 21 | transmitter、practical_low_suburban、8-20、-11.5764 pp |
| teacher-ref1 | 13 / 21 | 17 / 21 | transmitter、practical_low_suburban、8-20、-11.3514 pp |
| teacher-reffreq025 | 20 / 21 | 18 / 21 | transmitter、practical_low_suburban、8-20、-11.4611 pp |
| teacher-refpa-main025 | 13 / 21 | 17 / 21 | transmitter、practical_low_suburban、8-20、-11.4389 pp |

完整逐种子、TX/RX/day、七场景纠错与错失见本地 `paired_teacherensemble_strata.csv`。完整教师对照不衰减，均值匹配对照使用批内一致的平均衰减；教师使用原始 IQ，学生使用原 Tri 的同包带噪训练视图。PA 对照只缩放 PA 主投影，统计投影和 pa_delta 保持原状。不同方法的训练轨迹和接受数量不一定相同。测试目标逐景超过 PL 2 个百分点，须逐景判断，不能由均值替代。

六种方法的七景三种子平均均低于 Tri，也均低于普通 PL；相对 PL 的差距为 0.4917 至 1.4944 个百分点。参考分支自适应教师相对完整教师的七景平均亦全部下降；与批均值匹配的参考教师相比则略有提高，但不能据此认定该方案有效。频域和 PA 主投影结果也不足以支持全面收益，更不能据此认定整个 PA 分支的作用。

每个模型续训 30 轮执行 6660 次蒸馏调用，每次增加两个评估模式教师前向；每模型选取 846240 次源 U 样本暴露，约 842475 至 844637 次被接受。这些是重复训练暴露次数，不是不同物理包数。非完整教师组的源 U 平均参考缩放约为 0.829，完整教师对照为 1。续训墙钟时间约 5012 至 5838 秒，GPU 共卡负载不同，不能作为严格速度比较；完整统计见本地 `cost.json`。
