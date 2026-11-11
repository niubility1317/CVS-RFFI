# 单源接收机：物理分支伪标签一致性筛选

六种预登记的源 U 伪标签筛选方式各训练三个种子；原完整 Tri SSL 的模型、固定时间重排及各自同种子 E20 起点保持一致。全部 18 个模型的七景预测完成并冻结后独立评分。

本设置及历史测试结果已曝光，本批是开发验证，不是独立盲测。主口径为 DURA-LU、cross_rx 与 cross_both 合并、每种子每景 144000 包；同包比较核验物理 ID。普通 PL 使用自然时间顺序及不同训练机制，差异不能归因于某个组件。

| 方法 | 场景 | 候选准确率 % | 相对 Tri pp（纠错 / 错失） | 相对 PL pp（纠错 / 错失） |
| --- | --- | ---: | ---: | ---: |
| adaptive-match | clean | 49.4074 | -0.0384（430.3 / 485.7） | -0.8174（7311.0 / 8488.0） |
| adaptive-match | practical_high | 55.2458 | -0.0866（341.3 / 466.0） | -0.6495（6210.7 / 7146.0） |
| adaptive-match | practical_mid | 54.1315 | -0.0729（323.3 / 428.3） | -0.8664（5286.3 / 6534.0） |
| adaptive-match | practical_low_suburban | 52.8873 | -0.0412（307.7 / 367.0） | -1.2556（4405.7 / 6213.7） |
| adaptive-match | practical_high_urban | 54.4586 | -0.0769（318.3 / 429.0） | -0.5678（5717.3 / 6535.0） |
| adaptive-match | practical_mid_urban | 50.8678 | -0.0498（224.7 / 296.3） | -0.6139（3755.3 / 4639.3） |
| adaptive-match | practical_low_urban | 47.5468 | -0.0144（178.3 / 199.0） | -0.5530（2557.3 / 3353.7） |
| uniform-match | clean | 49.4931 | +0.0472（477.0 / 409.0） | -0.7317（7376.0 / 8429.7） |
| uniform-match | practical_high | 55.2336 | -0.0988（270.0 / 412.3） | -0.6618（6147.0 / 7100.0） |
| uniform-match | practical_mid | 54.1238 | -0.0806（249.0 / 365.0） | -0.8741（5248.7 / 6507.3） |
| uniform-match | practical_low_suburban | 52.8762 | -0.0523（234.7 / 310.0） | -1.2667（4365.3 / 6189.3） |
| uniform-match | practical_high_urban | 54.4419 | -0.0935（256.7 / 391.3） | -0.5845（5667.0 / 6508.7） |
| uniform-match | practical_mid_urban | 50.8725 | -0.0451（190.7 / 255.7） | -0.6093（3732.0 / 4609.3） |
| uniform-match | practical_low_urban | 47.5366 | -0.0245（148.3 / 183.7） | -0.5632（2542.7 / 3353.7） |
| adaptive-conf | clean | 49.3822 | -0.0637（559.0 / 650.7） | -0.8426（7317.3 / 8530.7） |
| adaptive-conf | practical_high | 55.1345 | -0.1979（311.3 / 596.3） | -0.7609（6111.7 / 7207.3） |
| adaptive-conf | practical_mid | 54.0354 | -0.1690（292.7 / 536.0） | -0.9625（5237.3 / 6623.3） |
| adaptive-conf | practical_low_suburban | 52.8056 | -0.1229（296.7 / 473.7） | -1.3373（4360.3 / 6286.0） |
| adaptive-conf | practical_high_urban | 54.3421 | -0.1933（275.0 / 553.3） | -0.6843（5631.0 / 6616.3） |
| adaptive-conf | practical_mid_urban | 50.8007 | -0.1169（226.7 / 395.0） | -0.6810（3703.3 / 4684.0） |
| adaptive-conf | practical_low_urban | 47.4965 | -0.0646（174.7 / 267.7） | -0.6032（2521.3 / 3390.0） |
| uniform-conf | clean | 49.4104 | -0.0354（524.0 / 575.0） | -0.8144（7342.3 / 8515.0） |
| uniform-conf | practical_high | 55.2843 | -0.0481（422.3 / 491.7） | -0.6111（6230.7 / 7110.7） |
| uniform-conf | practical_mid | 54.1525 | -0.0519（386.7 / 461.3） | -0.8454（5315.3 / 6532.7） |
| uniform-conf | practical_low_suburban | 52.8961 | -0.0324（345.3 / 392.0） | -1.2468（4427.3 / 6222.7） |
| uniform-conf | practical_high_urban | 54.4734 | -0.0620（360.3 / 449.7） | -0.5530（5739.0 / 6535.3） |
| uniform-conf | practical_mid_urban | 50.8850 | -0.0326（272.0 / 319.0） | -0.5968（3774.0 / 4633.3） |
| uniform-conf | practical_low_urban | 47.5472 | -0.0139（215.3 / 235.3） | -0.5525（2567.0 / 3362.7） |
| adaptive-freq-conf | clean | 49.4197 | -0.0262（533.3 / 571.0） | -0.8051（7290.3 / 8449.7） |
| adaptive-freq-conf | practical_high | 55.2630 | -0.0694（386.0 / 486.0） | -0.6324（6168.3 / 7079.0） |
| adaptive-freq-conf | practical_mid | 54.1326 | -0.0718（337.7 / 441.0） | -0.8653（5246.7 / 6492.7） |
| adaptive-freq-conf | practical_low_suburban | 52.8917 | -0.0368（326.3 / 379.3） | -1.2512（4377.0 / 6178.7） |
| adaptive-freq-conf | practical_high_urban | 54.4699 | -0.0655（340.7 / 435.0） | -0.5565（5686.3 / 6487.7） |
| adaptive-freq-conf | practical_mid_urban | 50.8519 | -0.0657（224.7 / 319.3） | -0.6299（3707.0 / 4614.0） |
| adaptive-freq-conf | practical_low_urban | 47.5350 | -0.0262（188.0 / 225.7） | -0.5648（2539.3 / 3352.7） |
| random-conf | clean | 49.4843 | +0.0384（472.0 / 416.7） | -0.7405（7364.0 / 8430.3） |
| random-conf | practical_high | 55.2729 | -0.0595（310.7 / 396.3） | -0.6225（6165.3 / 7061.7） |
| random-conf | practical_mid | 54.1683 | -0.0361（266.7 / 318.7） | -0.8296（5262.3 / 6457.0） |
| random-conf | practical_low_suburban | 52.9442 | +0.0157（277.7 / 255.0） | -1.1986（4413.3 / 6139.3） |
| random-conf | practical_high_urban | 54.5007 | -0.0347（285.0 / 335.0） | -0.5257（5705.3 / 6462.3） |
| random-conf | practical_mid_urban | 50.8903 | -0.0273（183.7 / 223.0） | -0.5914（3739.0 / 4590.7） |
| random-conf | practical_low_urban | 47.5604 | -0.0007（154.3 / 155.3） | -0.5394（2547.7 / 3324.3） |

## 筛选机制对照

| 候选 | 对照 | 场景 | 差异 pp（纠错 / 错失） |
| --- | --- | --- | ---: |
| adaptive-match | uniform-match | clean | -0.0856（438.0 / 561.3） |
| adaptive-match | uniform-match | practical_high | +0.0123（416.3 / 398.7） |
| adaptive-match | uniform-match | practical_mid | +0.0076（391.0 / 380.0） |
| adaptive-match | uniform-match | practical_low_suburban | +0.0111（365.0 / 349.0） |
| adaptive-match | uniform-match | practical_high_urban | +0.0167（423.0 / 399.0） |
| adaptive-match | uniform-match | practical_mid_urban | -0.0046（279.0 / 285.7） |
| adaptive-match | uniform-match | practical_low_urban | +0.0102（200.3 / 185.7） |
| adaptive-conf | uniform-conf | clean | -0.0282（551.7 / 592.3） |
| adaptive-conf | uniform-conf | practical_high | -0.1498（344.3 / 560.0） |
| adaptive-conf | uniform-conf | practical_mid | -0.1171（339.3 / 508.0） |
| adaptive-conf | uniform-conf | practical_low_suburban | -0.0905（307.3 / 437.7） |
| adaptive-conf | uniform-conf | practical_high_urban | -0.1313（325.7 / 514.7） |
| adaptive-conf | uniform-conf | practical_mid_urban | -0.0843（252.0 / 373.3） |
| adaptive-conf | uniform-conf | practical_low_urban | -0.0507（185.0 / 258.0） |
| adaptive-conf | random-conf | clean | -0.1021（531.7 / 678.7） |
| adaptive-conf | random-conf | practical_high | -0.1384（379.3 / 578.7） |
| adaptive-conf | random-conf | practical_mid | -0.1329（344.0 / 535.3） |
| adaptive-conf | random-conf | practical_low_suburban | -0.1387（300.3 / 500.0） |
| adaptive-conf | random-conf | practical_high_urban | -0.1586（341.3 / 569.7） |
| adaptive-conf | random-conf | practical_mid_urban | -0.0896（255.0 / 384.0） |
| adaptive-conf | random-conf | practical_low_urban | -0.0639（180.3 / 272.3） |
| adaptive-freq-conf | adaptive-conf | clean | +0.0375（520.7 / 466.7） |
| adaptive-freq-conf | adaptive-conf | practical_high | +0.1285（542.3 / 357.3） |
| adaptive-freq-conf | adaptive-conf | practical_mid | +0.0972（495.7 / 355.7） |
| adaptive-freq-conf | adaptive-conf | practical_low_suburban | +0.0861（436.0 / 312.0） |
| adaptive-freq-conf | adaptive-conf | practical_high_urban | +0.1278（500.7 / 316.7） |
| adaptive-freq-conf | adaptive-conf | practical_mid_urban | +0.0512（337.0 / 263.3） |
| adaptive-freq-conf | adaptive-conf | practical_low_urban | +0.0384（242.7 / 187.3） |

## 稳定性与分层

| 方法 | 相对 Tri 下降种子×场景 | 相对 PL 下降种子×场景 | 相对 PL 最差三种子分层 |
| --- | ---: | ---: | --- |
| adaptive-match | 14 / 21 | 17 / 21 | transmitter、practical_low_suburban、8-20、-11.0514 pp |
| uniform-match | 19 / 21 | 17 / 21 | transmitter、practical_low_suburban、8-20、-10.9028 pp |
| adaptive-conf | 20 / 21 | 18 / 21 | transmitter、practical_low_suburban、8-20、-11.3042 pp |
| uniform-conf | 15 / 21 | 17 / 21 | transmitter、practical_low_suburban、8-20、-10.9653 pp |
| adaptive-freq-conf | 19 / 21 | 18 / 21 | transmitter、practical_low_suburban、8-20、-10.8778 pp |
| random-conf | 11 / 21 | 17 / 21 | transmitter、practical_low_suburban、8-20、-10.7542 pp |

完整逐种子、TX/RX/day、七场景纠错与错失见本地 `paired_consensus_strata.csv`。随机对照只在各自轨迹内与自适应门控匹配保留数量；不同方法的训练轨迹和总接收数量不一定相同。测试目标逐景超过 PL 2 个百分点，须逐景判断，不能由均值替代。

六组的七景三种子均值全部低于普通 PL，差距为 0.5257 至 1.3373 个百分点，未达到逐景领先 2 个百分点。相对 Tri，多数组合的场景均值也下降；仅少量组合和场景出现小幅正值，不构成全面提升。自适应置信筛选相对同数量随机筛选没有表现出稳定优势。

源 U 统计显示，单纯要求分支预测类别一致的 adaptive-match 和 uniform-match 在三个种子中均未额外拒绝任何原本接受的伪标签。加入置信阈值的组，每个模型在约 167.5 万至 167.8 万次原接受样本暴露中只额外拒绝几十至数百次；这是重复训练暴露次数，不是不同物理包数。因此结果不足以证明该物理筛选发挥了有效作用。该附加前向每次 U 训练调用执行一次，源 U 平均相干度约 0.99767、平均缩放约 0.829；续训 30 轮的实测墙钟时间约 5449 至 5966 秒，受共享 GPU 负载影响，不能作为严格速度对比。完整接受数和耗时见本地 `cost.json`。
