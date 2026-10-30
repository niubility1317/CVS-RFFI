# PAMF 预测身份条件密度结果（2026-10-31）

**结论：NO_PROMOTION，完整目标 ACTIVE。** 唯一主 `predicted_ratio` 比匹配 pooled-ratio 在两个跨 RX 条件都有局部收益，但 clean/star 的 C_K 与同 DURA 原 F0 2pp保护仍失败，星地未知拒识也未达90%。没有换主、目标回流或阈值扫描。

Run `20261031-phase3-class-density-s2026092701-r01`；actualcode `4af0668331682636ee212077b38cd912857a1b31`。全部三条件、四方法预测和独立评分 COMPLETE，pipeline `ANALYZED_DEVELOPMENT`。

## 固定改变与数据边界

原 F0 E50/11100 默认六类头、DURA v1.0（266951参数，E20/4940）及 P1 signal r03（headless232033＋MLP102688）均冻结。本轮只将 known90全类混合改为原F0**预测类**的15个RX/day模式、权重1/15；背景32模式与原密度11聚合不变。预测不按真实TX或query RX/day路由。不是另训头，也未消费此前mixture-UDA偏移。

源L6300/OE6134、U_s56700及原监督/2220步signal训练仅为祖先，本轮新消费0；原源训练有标签比例10%，源V27000是额外已知校准标签。**本轮实际用V校准两套新六阈值（12个标量）**，旧pooled两套六阈值精确复用；神经、密度及normalizer新拟合0不等于阈值校准0。known12规则为overall＋5RX＋6scene，另30交叉组仅诊断，无conformal保证。

源测试为既有星地→DURA的V27000（旧六TX各4500）/novel3150（19-15、19-16、7-1各1050）。源RX1/3/4/6/8对应1-19/18-2/19-2/2-19/3-19，日期2021-03-08/15/23。clean/star Q均为源外RX24-1(index26)、2021-03-15的同864物理记录（旧六/未知三类各96），两条件均先经过DURA；目标U、正式RX24-18未访问。源NV/Q历史曝光、单RX/seed及同物理配对保持，不能称独立盲测或新增物理样本。

四套τ在任一NV/Q前冻结，source/clean/star共享源星地τ；三条件全部预测完成后独立CPU scorer才连接源NV、cleanQ、starQ truth各一次。未读真实IQ/权重或原truth来做本报告。

## 全部12个运行点

A_K为旧类正确且接收的逐TX宏平均，C_K为接收率，R_U为未知拒识率，H为A_K/R_U调和均值。指标为%，损伤为同输入原F0闭集减A_K的百分点；AUROC/OSCR只描述固定margin排序。

| 方法 | 条件 | A_K | C_K | R_U | H | 原F0损伤pp | AUROC | OSCR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pooled_known | source | 89.948148 | 97.522222 | 46.888889 | 61.643672 | 1.033333 | 94.493082 | 87.505278 |
| pooled_known | clean_DURA | 63.541667 | 80.729167 | 88.541667 | 73.986872 | 16.493056 | 90.510465 | 72.142048 |
| pooled_known | star_DURA | 62.847222 | 86.631944 | 45.833333 | 53.008520 | 10.069444 | 77.519170 | 56.334394 |
| pooled_ratio | source | 89.770370 | 97.107407 | 60.888889 | 72.561330 | 1.211111 | 94.985255 | 87.944738 |
| pooled_ratio | clean_DURA | 68.923611 | 85.416667 | 99.652778 | 81.487441 | 11.111111 | 95.991874 | 77.463710 |
| pooled_ratio | star_DURA | 63.020833 | 84.895833 | 76.388889 | 69.063927 | 9.895833 | 84.728250 | 62.604890 |
| predicted_known | source | 89.914815 | 97.244444 | 49.206349 | 63.604697 | 1.066667 | 94.238086 | 87.432951 |
| predicted_known | clean_DURA | 68.750000 | 85.763889 | 88.541667 | 77.400662 | 11.284722 | 93.060378 | 74.845076 |
| predicted_known | star_DURA | 64.062500 | 87.500000 | 51.041667 | 56.815611 | 8.854167 | 80.401235 | 59.491946 |
| predicted_ratio | source | 89.744444 | 96.759259 | 61.936508 | 73.291437 | 1.237037 | 94.734250 | 87.937558 |
| predicted_ratio | clean_DURA | 70.833333 | 87.326389 | 100.000000 | 82.926829 | 9.201389 | 96.158251 | 77.770544 |
| predicted_ratio | star_DURA | 64.062500 | 85.590278 | 77.083333 | 69.972325 | 8.854167 | 85.027850 | 62.982253 |

两个新Q的pooled两列已只读对照dura-best：IDs、default_logits、closed、同输入baseline、signal特征、两raw/margin均逐值一致，pooled-ratio宏指标也完全一致。旧source P1成绩本来包含DURA；没有拿绕过G的旧clean或另训P0网络充当控制。

主source known12通过、known42未全过（5交叉组损伤超2pp），R_U61.936508%仅源开发诊断。主clean的R_U100%过95单项，但C_K87.326389%/损伤9.201389pp失败；主star的R_U77.083333%/C_K85.590278%/损伤8.854167pp均失败。原F0闭集source90.981481%、clean80.034722%、star72.916667%在全部方法中不变。

## 主方法相对pooled-ratio的同ID配对

计数格式为gain/loss/net；下面收益来自接收/拒绝集合变化，不能称原身份头纠错。

| 条件 | 正确接收 | 接受 | 未知拒对 | ΔA_K pp | ΔR_U pp | ΔH pp |
| --- | --- | --- | --- | ---: | ---: | ---: |
| source | 29/36/-7 | 38/132/-94 | 76/43/+33 | -0.025926 | 1.047619 | 0.730107 |
| clean_DURA | 11/0/+11 | 11/0/+11 | 1/0/+1 | 1.909722 | 0.347222 | 1.439388 |
| star_DURA | 8/2/+6 | 10/6/+4 | 4/2/+2 | 1.041667 | 0.694444 | 0.908398 |

主clean正确接收408/576、接受503/576，仍误拒53条原F0正确样本；star正确接收369/576、接受493/576，误拒51条原F0正确样本。14-7 clean虽接收95/96但仅28条身份正确（67条错身份被接收）；star接收96/96但仅32条正确（64条错误）。身份误分类与拒绝错误必须区分，C_K不能代替A_K。

主三未知TX拒对（19-15/19-16/7-1）：source为688/512/751（各1050），clean为96/96/96，star为75/71/76（各96）。完整TX/RX/day/42组及四组配对在独立results和私有分析中，不从局部较高控制替换主方法。

## 场景缺项与校准限制

| 主星地场景 | known/unknown | A_K% | C_K% | R_U% | 原F0损伤pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| practical_high | 33/6 | 72.857143 | 93.333333 | N/A | 3.333333 |
| practical_mid | 42/25 | 69.722222 | 87.222222 | 94.444444 | 12.777778 |
| practical_low_suburban | 203/117 | 66.240399 | 86.637225 | 88.691817 | 9.884793 |
| practical_high_urban | 38/16 | 75.330688 | 91.071429 | 86.666667 | 6.547619 |
| practical_mid_urban | 45/15 | 75.509259 | 94.444444 | 75.555556 | 5.555556 |
| practical_low_urban | 215/109 | 54.240154 | 79.663750 | 57.858423 | 9.289386 |

high未知三TX计数0/4/2，缺19-15，宏R_U/H为N/A，不能把6条全拒对写为场景通过。其余景覆盖完整但所有六景2pp保护均失败；总体通过也不能掩盖主景缺项。本轮没有事后删景、补样或按Q调τ。已有逐TX Wilson95%区间仅作历史开发池描述，不构成独立RX确认。

主新阈值的common-tail q=161/4958；已存下一候选q=138/4241因源RX3损伤2.018519pp违反2pp而不可选（C_K92.962963%仍过90%）。这是V-only既存限制，不是读取NV后挑阈值。源known12通过不等于42通过；主5个超限交叉组为RX3/high、RX3/mid_urban、RX3/low_urban、RX4/high_urban、RX8/low_urban。

## 成本与执行证据

新增真实G/F0/signal前向各1728（两个Q各864），synthetic实际checkpoint smoke各2且PASS；source新IQ/前向、L/OE消费、神经/密度拟合、目标适应/发现/注册均0。两套新六τ的源V校准实际发生，耗时包含predict阶段、未单独计时；忽略score中沿用的“no statistical fitting”笼统字符串。原F011100、signal特征11100＋2220/MLP2220及G训练成本仍继承。

部署活跃F0＋signal567714参数，含DURA为834665；原密度11数组364340B不变，全四套τ为192B（新12标量96B，旧12标量96B），单主仅需一套六τ。无source逐行bank部署。批处理predict29.048555s、CPUscore9.496420s、pipeline38.634129s，不是逐包B1时延。

4次独立观察捕获driver1021220、predictor1021309的CWD/argv/CUDA7及GPU7/UUID9c4eb614一次344MiB采样，后捕获CPUscorer1022185（CUDA空），终态live/owned_gpu空、foreign0。predict日志492→1168B；344MiB是离散样本，不是峰值。三完整运行stdout（21/1168/43B）均无warning/traceback/error/nonfinite；本地scorer测试另有旧np.trapz弃用告警2568次。

既有验证共23唯一项＋12子项：producer3＋4sub、scorer5＋8sub有持久stdout；core/launcher15项仅先前console PASS证据，无持久stdout，本轮未重复suite。一次P0/P1审查PASS；单archive578文件/data0、SHA比较/compile/runtime PASS。所有预测后truth-last，prepared来源与本次实际code区分保留。

结论仅支持“预测类条件密度＋各自源V再校准有局部开发收益”，不能把它全归因于条件算子、宣称跨RX达标或替代匿名注册。完整Phase3继续ACTIVE，A_N/H_OW为N/A，formal未访问。证据：[方案](PAMF_PHASE3_CLASS_DENSITY_PLAN_20261031.md)、[本run记录](../automation_reports/CV-SincNet/20261031-phase3-class-density-s2026092701-r01/report.md)、[独立评分](E:/type10-7/local_artifacts/pamf_phase3_class_density_20261031_r01/score/results.json)、[紧凑分析](E:/type10-7/local_artifacts/pamf_phase3_class_density_20261031_r01/analysis.json)。
