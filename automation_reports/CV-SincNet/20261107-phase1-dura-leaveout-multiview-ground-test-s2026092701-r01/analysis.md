# DURA 留一视图教师七景测试：完整七景结果

固定 Phase1 Main split、Tri-SIG-TIME SSL 原默认头、单模型 seed 2026092701。每景 168000 条，全部七景预测固定后独立 truth-last。训练不读取 F、TX 标签、V 或目标；此前已多次观察该地面测试，不是独立盲测或真实在轨结果。

raw clean 为 86.538690%。按不超过 3 个百分点或相对下降不超过 3%，分别 0/6、0/6 个受损场景达标；0/6 场景高于 DURA。低城区为 67.2149%。

| 场景 | raw % | 本版 % | DURA % | 相对 DURA（百分点） | 修复 | 损伤 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 86.5387 | 86.6155 | 86.6155 | +0.0000 | 1 | 1 |
| 高郊区 | 80.6155 | 83.2643 | 84.3435 | -1.0792 | 1493 | 3306 |
| 中郊区 | 78.1077 | 81.7036 | 83.0161 | -1.3125 | 1803 | 4008 |
| 低郊区 | 75.7446 | 79.8637 | 81.3220 | -1.4583 | 2179 | 4629 |
| 高城区 | 77.0500 | 81.8024 | 83.0494 | -1.2470 | 1655 | 3750 |
| 中城区 | 62.7958 | 74.2202 | 75.7060 | -1.4857 | 2584 | 5080 |
| 低城区 | 51.6048 | 67.2149 | 68.8012 | -1.5863 | 3146 | 5811 |

配对修复与损伤对应同一物理记录。逐 TX、RX、day 边际分组及 Macro-F1 见 [完整评分](test_summary.json)；[配对计数](paired_dura.json)、[预测完成记录](complete.json)和[源训练结果](source_summary.json)保留。结果不回流到本候选的选模、调参或选择性重跑。

## TX、RX、day 对照

以下分别按维度作边际分组；分组正确数差不是样本级修复数量，也不用于后续选模或调参。六个受损场景中：

- TX：36 个分组中，13 组高于 DURA、2 组相同、21 组低于 DURA。
- RX：42 个分组中，8 组高于 DURA、0 组相同、34 组低于 DURA。
- day：24 个分组中，0 组高于 DURA、0 组相同、24 组低于 DURA。

| 场景 | 维度 | 值 | 样本数 | 本版正确数 | DURA 正确数 | 差值（百分点） |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| clean | TX | 0 | 28000 | 27398 | 27398 | +0.0000 |
| clean | TX | 1 | 28000 | 19632 | 19632 | +0.0000 |
| clean | TX | 2 | 28000 | 22985 | 22985 | +0.0000 |
| clean | TX | 3 | 28000 | 20391 | 20391 | +0.0000 |
| clean | TX | 4 | 28000 | 27955 | 27955 | +0.0000 |
| clean | TX | 5 | 28000 | 27153 | 27153 | +0.0000 |
| clean | RX | 1-1 | 24000 | 19466 | 19466 | +0.0000 |
| clean | RX | 14-7 | 24000 | 19033 | 19033 | +0.0000 |
| clean | RX | 2-1 | 24000 | 22258 | 22259 | -0.0042 |
| clean | RX | 20-1 | 24000 | 20127 | 20126 | +0.0042 |
| clean | RX | 7-14 | 24000 | 22853 | 22853 | +0.0000 |
| clean | RX | 7-7 | 24000 | 20545 | 20545 | +0.0000 |
| clean | RX | 8-8 | 24000 | 21232 | 21232 | +0.0000 |
| clean | day | 2021_03_01 | 42000 | 34441 | 34440 | +0.0024 |
| clean | day | 2021_03_08 | 42000 | 37206 | 37206 | +0.0000 |
| clean | day | 2021_03_15 | 42000 | 36132 | 36133 | -0.0024 |
| clean | day | 2021_03_23 | 42000 | 37735 | 37735 | +0.0000 |
| practical_high | TX | 0 | 28000 | 26674 | 26906 | -0.8286 |
| practical_high | TX | 1 | 28000 | 14076 | 16333 | -8.0607 |
| practical_high | TX | 2 | 28000 | 24517 | 24123 | +1.4071 |
| practical_high | TX | 3 | 28000 | 19112 | 18862 | +0.8929 |
| practical_high | TX | 4 | 28000 | 27959 | 27959 | +0.0000 |
| practical_high | TX | 5 | 28000 | 27546 | 27514 | +0.1143 |
| practical_high | RX | 1-1 | 24000 | 18670 | 18335 | +1.3958 |
| practical_high | RX | 14-7 | 24000 | 20569 | 20930 | -1.5042 |
| practical_high | RX | 2-1 | 24000 | 20300 | 21003 | -2.9292 |
| practical_high | RX | 20-1 | 24000 | 17890 | 17986 | -0.4000 |
| practical_high | RX | 7-14 | 24000 | 22073 | 22523 | -1.8750 |
| practical_high | RX | 7-7 | 24000 | 19687 | 19498 | +0.7875 |
| practical_high | RX | 8-8 | 24000 | 20695 | 21422 | -3.0292 |
| practical_high | day | 2021_03_01 | 42000 | 33701 | 33969 | -0.6381 |
| practical_high | day | 2021_03_08 | 42000 | 35353 | 36055 | -1.6714 |
| practical_high | day | 2021_03_15 | 42000 | 34763 | 35355 | -1.4095 |
| practical_high | day | 2021_03_23 | 42000 | 36067 | 36318 | -0.5976 |
| practical_mid | TX | 0 | 28000 | 26415 | 26662 | -0.8821 |
| practical_mid | TX | 1 | 28000 | 13492 | 16224 | -9.7571 |
| practical_mid | TX | 2 | 28000 | 24097 | 23870 | +0.8107 |
| practical_mid | TX | 3 | 28000 | 18627 | 18044 | +2.0821 |
| practical_mid | TX | 4 | 28000 | 27932 | 27931 | +0.0036 |
| practical_mid | TX | 5 | 28000 | 26699 | 26736 | -0.1321 |
| practical_mid | RX | 1-1 | 24000 | 18419 | 18139 | +1.1667 |
| practical_mid | RX | 14-7 | 24000 | 19964 | 20501 | -2.2375 |
| practical_mid | RX | 2-1 | 24000 | 19794 | 20375 | -2.4208 |
| practical_mid | RX | 20-1 | 24000 | 17498 | 17809 | -1.2958 |
| practical_mid | RX | 7-14 | 24000 | 21539 | 21992 | -1.8875 |
| practical_mid | RX | 7-7 | 24000 | 19387 | 19250 | +0.5708 |
| practical_mid | RX | 8-8 | 24000 | 20661 | 21401 | -3.0833 |
| practical_mid | day | 2021_03_01 | 42000 | 33095 | 33508 | -0.9833 |
| practical_mid | day | 2021_03_08 | 42000 | 34869 | 35622 | -1.7929 |
| practical_mid | day | 2021_03_15 | 42000 | 34003 | 34694 | -1.6452 |
| practical_mid | day | 2021_03_23 | 42000 | 35295 | 35643 | -0.8286 |
| practical_low_suburban | TX | 0 | 28000 | 25928 | 26113 | -0.6607 |
| practical_low_suburban | TX | 1 | 28000 | 12920 | 16041 | -11.1464 |
| practical_low_suburban | TX | 2 | 28000 | 23665 | 23553 | +0.4000 |
| practical_low_suburban | TX | 3 | 28000 | 18052 | 17144 | +3.2429 |
| practical_low_suburban | TX | 4 | 28000 | 27820 | 27826 | -0.0214 |
| practical_low_suburban | TX | 5 | 28000 | 25786 | 25944 | -0.5643 |
| practical_low_suburban | RX | 1-1 | 24000 | 18063 | 17932 | +0.5458 |
| practical_low_suburban | RX | 14-7 | 24000 | 19258 | 19969 | -2.9625 |
| practical_low_suburban | RX | 2-1 | 24000 | 19283 | 19703 | -1.7500 |
| practical_low_suburban | RX | 20-1 | 24000 | 17115 | 17532 | -1.7375 |
| practical_low_suburban | RX | 7-14 | 24000 | 20870 | 21292 | -1.7583 |
| practical_low_suburban | RX | 7-7 | 24000 | 19073 | 18973 | +0.4167 |
| practical_low_suburban | RX | 8-8 | 24000 | 20509 | 21220 | -2.9625 |
| practical_low_suburban | day | 2021_03_01 | 42000 | 32326 | 32910 | -1.3905 |
| practical_low_suburban | day | 2021_03_08 | 42000 | 34196 | 34905 | -1.6881 |
| practical_low_suburban | day | 2021_03_15 | 42000 | 33230 | 34029 | -1.9024 |
| practical_low_suburban | day | 2021_03_23 | 42000 | 34419 | 34777 | -0.8524 |
| practical_high_urban | TX | 0 | 28000 | 26418 | 26592 | -0.6214 |
| practical_high_urban | TX | 1 | 28000 | 13530 | 15991 | -8.7893 |
| practical_high_urban | TX | 2 | 28000 | 24010 | 23778 | +0.8286 |
| practical_high_urban | TX | 3 | 28000 | 18999 | 18673 | +1.1643 |
| practical_high_urban | TX | 4 | 28000 | 27935 | 27935 | +0.0000 |
| practical_high_urban | TX | 5 | 28000 | 26536 | 26554 | -0.0643 |
| practical_high_urban | RX | 1-1 | 24000 | 18395 | 18145 | +1.0417 |
| practical_high_urban | RX | 14-7 | 24000 | 20118 | 20532 | -1.7250 |
| practical_high_urban | RX | 2-1 | 24000 | 19942 | 20578 | -2.6500 |
| practical_high_urban | RX | 20-1 | 24000 | 17583 | 17784 | -0.8375 |
| practical_high_urban | RX | 7-14 | 24000 | 21518 | 22030 | -2.1333 |
| practical_high_urban | RX | 7-7 | 24000 | 19321 | 19200 | +0.5042 |
| practical_high_urban | RX | 8-8 | 24000 | 20551 | 21254 | -2.9292 |
| practical_high_urban | day | 2021_03_01 | 42000 | 33154 | 33477 | -0.7690 |
| practical_high_urban | day | 2021_03_08 | 42000 | 34858 | 35619 | -1.8119 |
| practical_high_urban | day | 2021_03_15 | 42000 | 34131 | 34793 | -1.5762 |
| practical_high_urban | day | 2021_03_23 | 42000 | 35285 | 35634 | -0.8310 |
| practical_mid_urban | TX | 0 | 28000 | 24459 | 24470 | -0.0393 |
| practical_mid_urban | TX | 1 | 28000 | 10976 | 13818 | -10.1500 |
| practical_mid_urban | TX | 2 | 28000 | 21566 | 21788 | -0.7929 |
| practical_mid_urban | TX | 3 | 28000 | 18092 | 17182 | +3.2500 |
| practical_mid_urban | TX | 4 | 28000 | 27552 | 27565 | -0.0464 |
| practical_mid_urban | TX | 5 | 28000 | 22045 | 22363 | -1.1357 |
| practical_mid_urban | RX | 1-1 | 24000 | 17246 | 17254 | -0.0333 |
| practical_mid_urban | RX | 14-7 | 24000 | 17291 | 18051 | -3.1667 |
| practical_mid_urban | RX | 2-1 | 24000 | 18320 | 18662 | -1.4250 |
| practical_mid_urban | RX | 20-1 | 24000 | 16168 | 16445 | -1.1542 |
| practical_mid_urban | RX | 7-14 | 24000 | 18654 | 19155 | -2.0875 |
| practical_mid_urban | RX | 7-7 | 24000 | 17637 | 17648 | -0.0458 |
| practical_mid_urban | RX | 8-8 | 24000 | 19374 | 19971 | -2.4875 |
| practical_mid_urban | day | 2021_03_01 | 42000 | 30487 | 31016 | -1.2595 |
| practical_mid_urban | day | 2021_03_08 | 42000 | 32199 | 32895 | -1.6571 |
| practical_mid_urban | day | 2021_03_15 | 42000 | 30570 | 31317 | -1.7786 |
| practical_mid_urban | day | 2021_03_23 | 42000 | 31434 | 31958 | -1.2476 |
| practical_low_urban | TX | 0 | 28000 | 22290 | 22175 | +0.4107 |
| practical_low_urban | TX | 1 | 28000 | 8890 | 11903 | -10.7607 |
| practical_low_urban | TX | 2 | 28000 | 19274 | 19834 | -2.0000 |
| practical_low_urban | TX | 3 | 28000 | 17507 | 16259 | +4.4571 |
| practical_low_urban | TX | 4 | 28000 | 26635 | 26699 | -0.2286 |
| practical_low_urban | TX | 5 | 28000 | 18325 | 18716 | -1.3964 |
| practical_low_urban | RX | 1-1 | 24000 | 15933 | 16064 | -0.5458 |
| practical_low_urban | RX | 14-7 | 24000 | 15311 | 16123 | -3.3833 |
| practical_low_urban | RX | 2-1 | 24000 | 16614 | 16912 | -1.2417 |
| practical_low_urban | RX | 20-1 | 24000 | 14544 | 14953 | -1.7042 |
| practical_low_urban | RX | 7-14 | 24000 | 16387 | 16893 | -2.1083 |
| practical_low_urban | RX | 7-7 | 24000 | 16008 | 16090 | -0.3417 |
| practical_low_urban | RX | 8-8 | 24000 | 18124 | 18551 | -1.7792 |
| practical_low_urban | day | 2021_03_01 | 42000 | 27727 | 28414 | -1.6357 |
| practical_low_urban | day | 2021_03_08 | 42000 | 29591 | 30210 | -1.4738 |
| practical_low_urban | day | 2021_03_15 | 42000 | 27590 | 28325 | -1.7500 |
| practical_low_urban | day | 2021_03_23 | 42000 | 28013 | 28637 | -1.4857 |
