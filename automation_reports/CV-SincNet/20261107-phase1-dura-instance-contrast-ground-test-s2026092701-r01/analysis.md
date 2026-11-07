# DURA 源记录对比约束七景测试：完整七景结果

固定 Phase1 Main split、Tri-SIG-TIME SSL 原默认头、单模型 seed 2026092701。每景 168000 条，全部七景预测固定后独立 truth-last。训练不读取 F、TX 标签、V 或目标；此前已多次观察该地面测试，不是独立盲测或真实在轨结果。

raw clean 为 86.538690%。按不超过 3 个百分点或相对下降不超过 3%，分别 0/6、0/6 个受损场景达标；0/6 场景高于 DURA。低城区为 68.1292%。

| 场景 | raw % | 本版 % | DURA % | 相对 DURA（百分点） | 修复 | 损伤 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 86.5387 | 86.6155 | 86.6155 | +0.0000 | 2 | 2 |
| 高郊区 | 80.6155 | 83.4220 | 84.3435 | -0.9214 | 1186 | 2734 |
| 中郊区 | 78.1077 | 82.1387 | 83.0161 | -0.8774 | 1522 | 2996 |
| 低郊区 | 75.7446 | 80.4500 | 81.3220 | -0.8720 | 1943 | 3408 |
| 高城区 | 77.0500 | 82.2000 | 83.0494 | -0.8494 | 1516 | 2943 |
| 中城区 | 62.7958 | 74.9935 | 75.7060 | -0.7125 | 2798 | 3995 |
| 低城区 | 51.6048 | 68.1292 | 68.8012 | -0.6720 | 3640 | 4769 |

配对修复与损伤对应同一物理记录。逐 TX、RX、day 边际分组及 Macro-F1 见 [完整评分](test_summary.json)；[配对计数](paired_dura.json)、[预测完成记录](complete.json)和[源训练结果](source_summary.json)保留。结果不回流到本候选的选模、调参或选择性重跑。

## TX、RX、day 对照

以下分别按维度作边际分组；分组正确数差不是样本级修复数量，也不用于后续选模或调参。六个受损场景中：

- TX：36 个分组中，14 组高于 DURA、1 组相同、21 组低于 DURA。
- RX：42 个分组中，8 组高于 DURA、0 组相同、34 组低于 DURA。
- day：24 个分组中，0 组高于 DURA、0 组相同、24 组低于 DURA。

| 场景 | 维度 | 值 | 样本数 | 本版正确数 | DURA 正确数 | 差值（百分点） |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| clean | TX | 0 | 28000 | 27397 | 27398 | -0.0036 |
| clean | TX | 1 | 28000 | 19632 | 19632 | +0.0000 |
| clean | TX | 2 | 28000 | 22984 | 22985 | -0.0036 |
| clean | TX | 3 | 28000 | 20392 | 20391 | +0.0036 |
| clean | TX | 4 | 28000 | 27955 | 27955 | +0.0000 |
| clean | TX | 5 | 28000 | 27154 | 27153 | +0.0036 |
| clean | RX | 1-1 | 24000 | 19466 | 19466 | +0.0000 |
| clean | RX | 14-7 | 24000 | 19034 | 19033 | +0.0042 |
| clean | RX | 2-1 | 24000 | 22257 | 22259 | -0.0083 |
| clean | RX | 20-1 | 24000 | 20127 | 20126 | +0.0042 |
| clean | RX | 7-14 | 24000 | 22853 | 22853 | +0.0000 |
| clean | RX | 7-7 | 24000 | 20545 | 20545 | +0.0000 |
| clean | RX | 8-8 | 24000 | 21232 | 21232 | +0.0000 |
| clean | day | 2021_03_01 | 42000 | 34439 | 34440 | -0.0024 |
| clean | day | 2021_03_08 | 42000 | 37207 | 37206 | +0.0024 |
| clean | day | 2021_03_15 | 42000 | 36133 | 36133 | +0.0000 |
| clean | day | 2021_03_23 | 42000 | 37735 | 37735 | +0.0000 |
| practical_high | TX | 0 | 28000 | 26713 | 26906 | -0.6893 |
| practical_high | TX | 1 | 28000 | 14954 | 16333 | -4.9250 |
| practical_high | TX | 2 | 28000 | 24197 | 24123 | +0.2643 |
| practical_high | TX | 3 | 28000 | 18804 | 18862 | -0.2071 |
| practical_high | TX | 4 | 28000 | 27959 | 27959 | +0.0000 |
| practical_high | TX | 5 | 28000 | 27522 | 27514 | +0.0286 |
| practical_high | RX | 1-1 | 24000 | 18499 | 18335 | +0.6833 |
| practical_high | RX | 14-7 | 24000 | 20599 | 20930 | -1.3792 |
| practical_high | RX | 2-1 | 24000 | 20324 | 21003 | -2.8292 |
| practical_high | RX | 20-1 | 24000 | 18132 | 17986 | +0.6083 |
| practical_high | RX | 7-14 | 24000 | 22170 | 22523 | -1.4708 |
| practical_high | RX | 7-7 | 24000 | 19444 | 19498 | -0.2250 |
| practical_high | RX | 8-8 | 24000 | 20981 | 21422 | -1.8375 |
| practical_high | day | 2021_03_01 | 42000 | 33614 | 33969 | -0.8452 |
| practical_high | day | 2021_03_08 | 42000 | 35577 | 36055 | -1.1381 |
| practical_high | day | 2021_03_15 | 42000 | 34960 | 35355 | -0.9405 |
| practical_high | day | 2021_03_23 | 42000 | 35998 | 36318 | -0.7619 |
| practical_mid | TX | 0 | 28000 | 26526 | 26662 | -0.4857 |
| practical_mid | TX | 1 | 28000 | 14736 | 16224 | -5.3143 |
| practical_mid | TX | 2 | 28000 | 23951 | 23870 | +0.2893 |
| practical_mid | TX | 3 | 28000 | 18109 | 18044 | +0.2321 |
| practical_mid | TX | 4 | 28000 | 27928 | 27931 | -0.0107 |
| practical_mid | TX | 5 | 28000 | 26743 | 26736 | +0.0250 |
| practical_mid | RX | 1-1 | 24000 | 18295 | 18139 | +0.6500 |
| practical_mid | RX | 14-7 | 24000 | 20134 | 20501 | -1.5292 |
| practical_mid | RX | 2-1 | 24000 | 19809 | 20375 | -2.3583 |
| practical_mid | RX | 20-1 | 24000 | 17852 | 17809 | +0.1792 |
| practical_mid | RX | 7-14 | 24000 | 21729 | 21992 | -1.0958 |
| practical_mid | RX | 7-7 | 24000 | 19184 | 19250 | -0.2750 |
| practical_mid | RX | 8-8 | 24000 | 20990 | 21401 | -1.7125 |
| practical_mid | day | 2021_03_01 | 42000 | 33113 | 33508 | -0.9405 |
| practical_mid | day | 2021_03_08 | 42000 | 35201 | 35622 | -1.0024 |
| practical_mid | day | 2021_03_15 | 42000 | 34287 | 34694 | -0.9690 |
| practical_mid | day | 2021_03_23 | 42000 | 35392 | 35643 | -0.5976 |
| practical_low_suburban | TX | 0 | 28000 | 26075 | 26113 | -0.1357 |
| practical_low_suburban | TX | 1 | 28000 | 14429 | 16041 | -5.7571 |
| practical_low_suburban | TX | 2 | 28000 | 23655 | 23553 | +0.3643 |
| practical_low_suburban | TX | 3 | 28000 | 17239 | 17144 | +0.3393 |
| practical_low_suburban | TX | 4 | 28000 | 27825 | 27826 | -0.0036 |
| practical_low_suburban | TX | 5 | 28000 | 25933 | 25944 | -0.0393 |
| practical_low_suburban | RX | 1-1 | 24000 | 18014 | 17932 | +0.3417 |
| practical_low_suburban | RX | 14-7 | 24000 | 19557 | 19969 | -1.7167 |
| practical_low_suburban | RX | 2-1 | 24000 | 19235 | 19703 | -1.9500 |
| practical_low_suburban | RX | 20-1 | 24000 | 17519 | 17532 | -0.0542 |
| practical_low_suburban | RX | 7-14 | 24000 | 21100 | 21292 | -0.8000 |
| practical_low_suburban | RX | 7-7 | 24000 | 18903 | 18973 | -0.2917 |
| practical_low_suburban | RX | 8-8 | 24000 | 20828 | 21220 | -1.6333 |
| practical_low_suburban | day | 2021_03_01 | 42000 | 32516 | 32910 | -0.9381 |
| practical_low_suburban | day | 2021_03_08 | 42000 | 34502 | 34905 | -0.9595 |
| practical_low_suburban | day | 2021_03_15 | 42000 | 33591 | 34029 | -1.0429 |
| practical_low_suburban | day | 2021_03_23 | 42000 | 34547 | 34777 | -0.5476 |
| practical_high_urban | TX | 0 | 28000 | 26478 | 26592 | -0.4071 |
| practical_high_urban | TX | 1 | 28000 | 14700 | 15991 | -4.6107 |
| practical_high_urban | TX | 2 | 28000 | 23806 | 23778 | +0.1000 |
| practical_high_urban | TX | 3 | 28000 | 18594 | 18673 | -0.2821 |
| practical_high_urban | TX | 4 | 28000 | 27936 | 27935 | +0.0036 |
| practical_high_urban | TX | 5 | 28000 | 26582 | 26554 | +0.1000 |
| practical_high_urban | RX | 1-1 | 24000 | 18289 | 18145 | +0.6000 |
| practical_high_urban | RX | 14-7 | 24000 | 20197 | 20532 | -1.3958 |
| practical_high_urban | RX | 2-1 | 24000 | 20016 | 20578 | -2.3417 |
| practical_high_urban | RX | 20-1 | 24000 | 17861 | 17784 | +0.3208 |
| practical_high_urban | RX | 7-14 | 24000 | 21707 | 22030 | -1.3458 |
| practical_high_urban | RX | 7-7 | 24000 | 19137 | 19200 | -0.2625 |
| practical_high_urban | RX | 8-8 | 24000 | 20889 | 21254 | -1.5208 |
| practical_high_urban | day | 2021_03_01 | 42000 | 33145 | 33477 | -0.7905 |
| practical_high_urban | day | 2021_03_08 | 42000 | 35140 | 35619 | -1.1405 |
| practical_high_urban | day | 2021_03_15 | 42000 | 34405 | 34793 | -0.9238 |
| practical_high_urban | day | 2021_03_23 | 42000 | 35406 | 35634 | -0.5429 |
| practical_mid_urban | TX | 0 | 28000 | 24739 | 24470 | +0.9607 |
| practical_mid_urban | TX | 1 | 28000 | 12868 | 13818 | -3.3929 |
| practical_mid_urban | TX | 2 | 28000 | 21670 | 21788 | -0.4214 |
| practical_mid_urban | TX | 3 | 28000 | 16823 | 17182 | -1.2821 |
| practical_mid_urban | TX | 4 | 28000 | 27570 | 27565 | +0.0179 |
| practical_mid_urban | TX | 5 | 28000 | 22319 | 22363 | -0.1571 |
| practical_mid_urban | RX | 1-1 | 24000 | 17287 | 17254 | +0.1375 |
| practical_mid_urban | RX | 14-7 | 24000 | 17803 | 18051 | -1.0333 |
| practical_mid_urban | RX | 2-1 | 24000 | 18383 | 18662 | -1.1625 |
| practical_mid_urban | RX | 20-1 | 24000 | 16325 | 16445 | -0.5000 |
| practical_mid_urban | RX | 7-14 | 24000 | 18942 | 19155 | -0.8875 |
| practical_mid_urban | RX | 7-7 | 24000 | 17562 | 17648 | -0.3583 |
| practical_mid_urban | RX | 8-8 | 24000 | 19687 | 19971 | -1.1833 |
| practical_mid_urban | day | 2021_03_01 | 42000 | 30657 | 31016 | -0.8548 |
| practical_mid_urban | day | 2021_03_08 | 42000 | 32539 | 32895 | -0.8476 |
| practical_mid_urban | day | 2021_03_15 | 42000 | 31042 | 31317 | -0.6548 |
| practical_mid_urban | day | 2021_03_23 | 42000 | 31751 | 31958 | -0.4929 |
| practical_low_urban | TX | 0 | 28000 | 22752 | 22175 | +2.0607 |
| practical_low_urban | TX | 1 | 28000 | 11149 | 11903 | -2.6929 |
| practical_low_urban | TX | 2 | 28000 | 19597 | 19834 | -0.8464 |
| practical_low_urban | TX | 3 | 28000 | 15598 | 16259 | -2.3607 |
| practical_low_urban | TX | 4 | 28000 | 26713 | 26699 | +0.0500 |
| practical_low_urban | TX | 5 | 28000 | 18648 | 18716 | -0.2429 |
| practical_low_urban | RX | 1-1 | 24000 | 16047 | 16064 | -0.0708 |
| practical_low_urban | RX | 14-7 | 24000 | 15823 | 16123 | -1.2500 |
| practical_low_urban | RX | 2-1 | 24000 | 16734 | 16912 | -0.7417 |
| practical_low_urban | RX | 20-1 | 24000 | 14734 | 14953 | -0.9125 |
| practical_low_urban | RX | 7-14 | 24000 | 16757 | 16893 | -0.5667 |
| practical_low_urban | RX | 7-7 | 24000 | 16033 | 16090 | -0.2375 |
| practical_low_urban | RX | 8-8 | 24000 | 18329 | 18551 | -0.9250 |
| practical_low_urban | day | 2021_03_01 | 42000 | 28070 | 28414 | -0.8190 |
| practical_low_urban | day | 2021_03_08 | 42000 | 30025 | 30210 | -0.4405 |
| practical_low_urban | day | 2021_03_15 | 42000 | 27999 | 28325 | -0.7762 |
| practical_low_urban | day | 2021_03_23 | 42000 | 28363 | 28637 | -0.6524 |
