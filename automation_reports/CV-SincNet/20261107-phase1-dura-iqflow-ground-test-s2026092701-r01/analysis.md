# DURA 条件 IQ 流七景：完整七景结果

固定 Phase1 Main split、Tri-SIG-TIME SSL 原默认头、单模型 seed 2026092701。每景 168000 条，全部七景预测固定后独立 truth-last。训练不读取 F、TX 标签、V 或目标；此前已多次观察该地面测试，不是独立盲测或真实在轨结果。

raw clean 为 86.538690%。按不超过 3 个百分点或相对下降不超过 3%，分别 1/6、0/6 个受损场景达标；0/6 场景高于 DURA。低城区为 68.4554%。

| 场景 | raw % | 本版 % | DURA % | 相对 DURA（百分点） | 修复 | 损伤 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 86.5387 | 86.3696 | 86.6155 | -0.2458 | 1288 | 1701 |
| 高郊区 | 80.6155 | 83.8595 | 84.3435 | -0.4839 | 1472 | 2285 |
| 中郊区 | 78.1077 | 82.5143 | 83.0161 | -0.5018 | 1586 | 2429 |
| 低郊区 | 75.7446 | 80.7571 | 81.3220 | -0.5649 | 1760 | 2709 |
| 高城区 | 77.0500 | 82.5393 | 83.0494 | -0.5101 | 1605 | 2462 |
| 中城区 | 62.7958 | 75.2798 | 75.7060 | -0.4262 | 2599 | 3315 |
| 低城区 | 51.6048 | 68.4554 | 68.8012 | -0.3458 | 3104 | 3685 |

配对修复与损伤对应同一物理记录。逐 TX、RX、day 边际分组及 Macro-F1 见 [完整评分](test_summary.json)；[配对计数](paired_dura.json)、[预测完成记录](complete.json)和[源训练结果](source_summary.json)保留。结果不回流到本候选的选模、调参或选择性重跑。

## TX、RX、day 对照

以下分别按维度作边际分组；分组正确数差不是样本级修复数量，也不用于后续选模或调参。六个受损场景中：

- TX：36 个分组中，16 组高于 DURA、0 组相同、20 组低于 DURA。
- RX：42 个分组中，17 组高于 DURA、0 组相同、25 组低于 DURA。
- day：24 个分组中，0 组高于 DURA、0 组相同、24 组低于 DURA。

| 场景 | 维度 | 值 | 样本数 | 本版正确数 | DURA 正确数 | 差值（百分点） |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| clean | TX | 0 | 28000 | 27457 | 27398 | +0.2107 |
| clean | TX | 1 | 28000 | 19597 | 19632 | -0.1250 |
| clean | TX | 2 | 28000 | 23194 | 22985 | +0.7464 |
| clean | TX | 3 | 28000 | 19499 | 20391 | -3.1857 |
| clean | TX | 4 | 28000 | 27950 | 27955 | -0.0179 |
| clean | TX | 5 | 28000 | 27404 | 27153 | +0.8964 |
| clean | RX | 1-1 | 24000 | 19406 | 19466 | -0.2500 |
| clean | RX | 14-7 | 24000 | 19307 | 19033 | +1.1417 |
| clean | RX | 2-1 | 24000 | 22300 | 22259 | +0.1708 |
| clean | RX | 20-1 | 24000 | 20022 | 20126 | -0.4333 |
| clean | RX | 7-14 | 24000 | 22972 | 22853 | +0.4958 |
| clean | RX | 7-7 | 24000 | 19700 | 20545 | -3.5208 |
| clean | RX | 8-8 | 24000 | 21394 | 21232 | +0.6750 |
| clean | day | 2021_03_01 | 42000 | 34324 | 34440 | -0.2762 |
| clean | day | 2021_03_08 | 42000 | 37062 | 37206 | -0.3429 |
| clean | day | 2021_03_15 | 42000 | 36070 | 36133 | -0.1500 |
| clean | day | 2021_03_23 | 42000 | 37645 | 37735 | -0.2143 |
| practical_high | TX | 0 | 28000 | 27124 | 26906 | +0.7786 |
| practical_high | TX | 1 | 28000 | 15579 | 16333 | -2.6929 |
| practical_high | TX | 2 | 28000 | 24485 | 24123 | +1.2929 |
| practical_high | TX | 3 | 28000 | 18144 | 18862 | -2.5643 |
| practical_high | TX | 4 | 28000 | 27958 | 27959 | -0.0036 |
| practical_high | TX | 5 | 28000 | 27594 | 27514 | +0.2857 |
| practical_high | RX | 1-1 | 24000 | 17999 | 18335 | -1.4000 |
| practical_high | RX | 14-7 | 24000 | 21122 | 20930 | +0.8000 |
| practical_high | RX | 2-1 | 24000 | 20919 | 21003 | -0.3500 |
| practical_high | RX | 20-1 | 24000 | 18094 | 17986 | +0.4500 |
| practical_high | RX | 7-14 | 24000 | 22510 | 22523 | -0.0542 |
| practical_high | RX | 7-7 | 24000 | 19004 | 19498 | -2.0583 |
| practical_high | RX | 8-8 | 24000 | 21236 | 21422 | -0.7750 |
| practical_high | day | 2021_03_01 | 42000 | 33759 | 33969 | -0.5000 |
| practical_high | day | 2021_03_08 | 42000 | 35994 | 36055 | -0.1452 |
| practical_high | day | 2021_03_15 | 42000 | 35053 | 35355 | -0.7190 |
| practical_high | day | 2021_03_23 | 42000 | 36078 | 36318 | -0.5714 |
| practical_mid | TX | 0 | 28000 | 26976 | 26662 | +1.1214 |
| practical_mid | TX | 1 | 28000 | 15534 | 16224 | -2.4643 |
| practical_mid | TX | 2 | 28000 | 24147 | 23870 | +0.9893 |
| practical_mid | TX | 3 | 28000 | 17260 | 18044 | -2.8000 |
| practical_mid | TX | 4 | 28000 | 27928 | 27931 | -0.0107 |
| practical_mid | TX | 5 | 28000 | 26779 | 26736 | +0.1536 |
| practical_mid | RX | 1-1 | 24000 | 17765 | 18139 | -1.5583 |
| practical_mid | RX | 14-7 | 24000 | 20617 | 20501 | +0.4833 |
| practical_mid | RX | 2-1 | 24000 | 20317 | 20375 | -0.2417 |
| practical_mid | RX | 20-1 | 24000 | 17846 | 17809 | +0.1542 |
| practical_mid | RX | 7-14 | 24000 | 21994 | 21992 | +0.0083 |
| practical_mid | RX | 7-7 | 24000 | 18830 | 19250 | -1.7500 |
| practical_mid | RX | 8-8 | 24000 | 21255 | 21401 | -0.6083 |
| practical_mid | day | 2021_03_01 | 42000 | 33325 | 33508 | -0.4357 |
| practical_mid | day | 2021_03_08 | 42000 | 35501 | 35622 | -0.2881 |
| practical_mid | day | 2021_03_15 | 42000 | 34410 | 34694 | -0.6762 |
| practical_mid | day | 2021_03_23 | 42000 | 35388 | 35643 | -0.6071 |
| practical_low_suburban | TX | 0 | 28000 | 26570 | 26113 | +1.6321 |
| practical_low_suburban | TX | 1 | 28000 | 15354 | 16041 | -2.4536 |
| practical_low_suburban | TX | 2 | 28000 | 23737 | 23553 | +0.6571 |
| practical_low_suburban | TX | 3 | 28000 | 16275 | 17144 | -3.1036 |
| practical_low_suburban | TX | 4 | 28000 | 27818 | 27826 | -0.0286 |
| practical_low_suburban | TX | 5 | 28000 | 25918 | 25944 | -0.0929 |
| practical_low_suburban | RX | 1-1 | 24000 | 17482 | 17932 | -1.8750 |
| practical_low_suburban | RX | 14-7 | 24000 | 20037 | 19969 | +0.2833 |
| practical_low_suburban | RX | 2-1 | 24000 | 19735 | 19703 | +0.1333 |
| practical_low_suburban | RX | 20-1 | 24000 | 17495 | 17532 | -0.1542 |
| practical_low_suburban | RX | 7-14 | 24000 | 21253 | 21292 | -0.1625 |
| practical_low_suburban | RX | 7-7 | 24000 | 18520 | 18973 | -1.8875 |
| practical_low_suburban | RX | 8-8 | 24000 | 21150 | 21220 | -0.2917 |
| practical_low_suburban | day | 2021_03_01 | 42000 | 32575 | 32910 | -0.7976 |
| practical_low_suburban | day | 2021_03_08 | 42000 | 34838 | 34905 | -0.1595 |
| practical_low_suburban | day | 2021_03_15 | 42000 | 33675 | 34029 | -0.8429 |
| practical_low_suburban | day | 2021_03_23 | 42000 | 34584 | 34777 | -0.4595 |
| practical_high_urban | TX | 0 | 28000 | 26947 | 26592 | +1.2679 |
| practical_high_urban | TX | 1 | 28000 | 15402 | 15991 | -2.1036 |
| practical_high_urban | TX | 2 | 28000 | 24019 | 23778 | +0.8607 |
| practical_high_urban | TX | 3 | 28000 | 17781 | 18673 | -3.1857 |
| practical_high_urban | TX | 4 | 28000 | 27930 | 27935 | -0.0179 |
| practical_high_urban | TX | 5 | 28000 | 26587 | 26554 | +0.1179 |
| practical_high_urban | RX | 1-1 | 24000 | 17798 | 18145 | -1.4458 |
| practical_high_urban | RX | 14-7 | 24000 | 20695 | 20532 | +0.6792 |
| practical_high_urban | RX | 2-1 | 24000 | 20551 | 20578 | -0.1125 |
| practical_high_urban | RX | 20-1 | 24000 | 17825 | 17784 | +0.1708 |
| practical_high_urban | RX | 7-14 | 24000 | 22018 | 22030 | -0.0500 |
| practical_high_urban | RX | 7-7 | 24000 | 18683 | 19200 | -2.1542 |
| practical_high_urban | RX | 8-8 | 24000 | 21096 | 21254 | -0.6583 |
| practical_high_urban | day | 2021_03_01 | 42000 | 33277 | 33477 | -0.4762 |
| practical_high_urban | day | 2021_03_08 | 42000 | 35531 | 35619 | -0.2095 |
| practical_high_urban | day | 2021_03_15 | 42000 | 34472 | 34793 | -0.7643 |
| practical_high_urban | day | 2021_03_23 | 42000 | 35386 | 35634 | -0.5905 |
| practical_mid_urban | TX | 0 | 28000 | 25329 | 24470 | +3.0679 |
| practical_mid_urban | TX | 1 | 28000 | 13898 | 13818 | +0.2857 |
| practical_mid_urban | TX | 2 | 28000 | 21815 | 21788 | +0.0964 |
| practical_mid_urban | TX | 3 | 28000 | 15634 | 17182 | -5.5286 |
| practical_mid_urban | TX | 4 | 28000 | 27529 | 27565 | -0.1286 |
| practical_mid_urban | TX | 5 | 28000 | 22265 | 22363 | -0.3500 |
| practical_mid_urban | RX | 1-1 | 24000 | 16918 | 17254 | -1.4000 |
| practical_mid_urban | RX | 14-7 | 24000 | 18117 | 18051 | +0.2750 |
| practical_mid_urban | RX | 2-1 | 24000 | 18702 | 18662 | +0.1667 |
| practical_mid_urban | RX | 20-1 | 24000 | 16369 | 16445 | -0.3167 |
| practical_mid_urban | RX | 7-14 | 24000 | 19184 | 19155 | +0.1208 |
| practical_mid_urban | RX | 7-7 | 24000 | 17180 | 17648 | -1.9500 |
| practical_mid_urban | RX | 8-8 | 24000 | 20000 | 19971 | +0.1208 |
| practical_mid_urban | day | 2021_03_01 | 42000 | 30827 | 31016 | -0.4500 |
| practical_mid_urban | day | 2021_03_08 | 42000 | 32778 | 32895 | -0.2786 |
| practical_mid_urban | day | 2021_03_15 | 42000 | 31132 | 31317 | -0.4405 |
| practical_mid_urban | day | 2021_03_23 | 42000 | 31733 | 31958 | -0.5357 |
| practical_low_urban | TX | 0 | 28000 | 23325 | 22175 | +4.1071 |
| practical_low_urban | TX | 1 | 28000 | 12393 | 11903 | +1.7500 |
| practical_low_urban | TX | 2 | 28000 | 19702 | 19834 | -0.4714 |
| practical_low_urban | TX | 3 | 28000 | 14379 | 16259 | -6.7143 |
| practical_low_urban | TX | 4 | 28000 | 26623 | 26699 | -0.2714 |
| practical_low_urban | TX | 5 | 28000 | 18583 | 18716 | -0.4750 |
| practical_low_urban | RX | 1-1 | 24000 | 15713 | 16064 | -1.4625 |
| practical_low_urban | RX | 14-7 | 24000 | 16179 | 16123 | +0.2333 |
| practical_low_urban | RX | 2-1 | 24000 | 16960 | 16912 | +0.2000 |
| practical_low_urban | RX | 20-1 | 24000 | 14822 | 14953 | -0.5458 |
| practical_low_urban | RX | 7-14 | 24000 | 16923 | 16893 | +0.1250 |
| practical_low_urban | RX | 7-7 | 24000 | 15747 | 16090 | -1.4292 |
| practical_low_urban | RX | 8-8 | 24000 | 18661 | 18551 | +0.4583 |
| practical_low_urban | day | 2021_03_01 | 42000 | 28250 | 28414 | -0.3905 |
| practical_low_urban | day | 2021_03_08 | 42000 | 30092 | 30210 | -0.2810 |
| practical_low_urban | day | 2021_03_15 | 42000 | 28173 | 28325 | -0.3619 |
| practical_low_urban | day | 2021_03_23 | 42000 | 28490 | 28637 | -0.3500 |
