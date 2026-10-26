# 同包恢复残差插值的完整源验证

ANALYZED，独立核实VERIFIED。四个方法只有一个学生seed2026092701，全部own原E30+20→E50/11100，四份最终默认头及17760条新增训练记录通过。所有sourceV27000八视图已固定，544个分层单元直接由逐包预测重算；另有96个方法间配对单元，含复用的同系数G/G端点控制。没有新目标访问。

实际代码`e00d055490904c4d89a6953c84d5f0fd92ce9965`。真实E30/defaulthead/G精确端点、12批源剂量/CUDA RNG隔离、原L128/U256联合backward smoke、7项聚焦测试、local/remote compile及唯一一次P0/P1检查通过。archive传输及实际PID/CWD/argv/GPU/loggrowth核实；未热改健康任务。

## 原门槛与完整判断

相对同seed PLControl、相同物理源V与场景，单位百分点。heldRX1 G三景共16200次判断，≥+.3pp等价于净增至少49次；overallV rawclean27000条≥−.1pp等价于净增不少于−27。RX1只从额外分支留出，父模型/L/rawPL/G已见它，不是真正未见baseRX验证。

| 方法 | heldG三景差 | overallV rawclean差 | held纠正 / 新增错 / 净增 | 原源条件 |
| --- | ---: | ---: | ---: | --- |
| MixBase | +0.129630 | -0.018519 | 51 / 30 / 21 | FAIL |
| MixDose | +0.080247 | -0.014815 | 55 / 42 / 13 | FAIL |
| PairBase | +0.129630 | -0.003704 | 74 / 53 / 21 | FAIL |
| PairDose | +0.104938 | -0.022222 | 86 / 69 / 17 | FAIL |

**四条均未通过原源门槛，不扩种子，不启动本批新目标测试。** 保留所有负结果、原门槛与原联合目标；下一步依据输入/剂量配对结果确定方向。

## 实际训练执行

Mix为同物理包raw/G逐包Uniform[0,1)插值，MixMix用两个独立Mix，MixG为Mix加精确G端点。原L/rawPL不变，额外标签为冻结raw三E30共识95 hardargmax；附加分类weight梯度为0，特征融合norm/gain仍训练。Base系数0.059415326575376119；Dose源公式固定MixMix=0.13833314526398888、MixG=0.088849667264764703，初始全参数物理梯度代理10%，不是后续AdamW或性能保证。

每row U1134000、held226800、额外接受838780、额外学生FW2268000、私有α抽样2268000。全步原L768/AsKL0/ORD.1/rawPL/系数/升温/MG计数符合配置，六组接受共同前缀相同，Base/Dose两路线全部α统计前缀匹配，无非有限跳步、U真值、G/教师或另训头。

| 方法 | 系数 | Mix学生FW | 精确G学生FW | 使用α数 | α均值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MixBase | 0.059415327 | 2268000 | 0 | 2268000 | 0.499555 |
| MixDose | 0.138333145 | 2268000 | 0 | 2268000 | 0.499555 |
| PairBase | 0.059415327 | 1134000 | 1134000 | 1134000 | 0.499561 |
| PairDose | 0.088849667 | 1134000 | 1134000 | 1134000 | 0.499561 |

## 方法间held同包配对

candidate−control为直接同包差，区别于相对PL增益。Base输入对照匹配系数，Dose输入对照匹配初始梯度代理；同输入Base/Dose比较系数。旧GEndpointControl是已核实的GFeatureHard，不重训或作父权重。H/G仅为源三景，完整参与RX/TX/day分解见CSV。

| candidate − control | 源视图 | 纠正 / 新增错 / 净增 | 差（pp） |
| --- | --- | ---: | ---: |
| MixDose − MixBase | raw | 0 / 0 / 0 | +0.000000 |
| MixDose − MixBase | H_three_mean | 33 / 52 / -19 | -0.117284 |
| MixDose − MixBase | G_three_mean | 13 / 21 / -8 | -0.049383 |
| PairBase − MixBase | raw | 1 / 0 / 1 | +0.018519 |
| PairBase − MixBase | H_three_mean | 31 / 55 / -24 | -0.148148 |
| PairBase − MixBase | G_three_mean | 28 / 28 / 0 | +0.000000 |
| PairDose − MixDose | raw | 0 / 0 / 0 | +0.000000 |
| PairDose − MixDose | H_three_mean | 34 / 30 / 4 | +0.024691 |
| PairDose − MixDose | G_three_mean | 34 / 30 / 4 | +0.024691 |
| PairDose − PairBase | raw | 0 / 1 / -1 | -0.018519 |
| PairDose − PairBase | H_three_mean | 47 / 38 / 9 | +0.055556 |
| PairDose − PairBase | G_three_mean | 20 / 24 / -4 | -0.024691 |
| MixBase − GEndpointControl | raw | 0 / 0 / 0 | +0.000000 |
| MixBase − GEndpointControl | H_three_mean | 55 / 39 / 16 | +0.098765 |
| MixBase − GEndpointControl | G_three_mean | 52 / 54 / -2 | -0.012346 |
| MixDose − GEndpointControl | raw | 0 / 0 / 0 | +0.000000 |
| MixDose − GEndpointControl | H_three_mean | 34 / 37 / -3 | -0.018519 |
| MixDose − GEndpointControl | G_three_mean | 35 / 45 / -10 | -0.061728 |
| PairBase − GEndpointControl | raw | 1 / 0 / 1 | +0.018519 |
| PairBase − GEndpointControl | H_three_mean | 36 / 44 / -8 | -0.049383 |
| PairBase − GEndpointControl | G_three_mean | 29 / 31 / -2 | -0.012346 |
| PairDose − GEndpointControl | raw | 0 / 0 / 0 | +0.000000 |
| PairDose − GEndpointControl | H_three_mean | 40 / 39 / 1 | +0.006173 |
| PairDose − GEndpointControl | G_three_mean | 9 / 15 / -6 | -0.037037 |

## 输入几何与结论边界

现有56700U的无模型CPU复核中，原12批模型前向前raw/G几何与校准最大差1.788e−7。未筛选低RMS极端值不在附加监督有效集合；有效41939包/景无负相关对，混合RMS低于raw25%的均匀α解析期望仅约.000552%/.001034%/.003250%。这是波形解释，不能替代TX准确率或实际训练事件计数；分析未更改配方、门控或选择规则。

新增80F轮，共享三E30历史90轮及旧缓存/控制成本单列。最终仍要求同一方法三个学生种子的raw clean≥85.98174603174603%、raw六景>72.0961970899471%、同G-LU六景>78.07463624338624%；本源实验不能证明联合目标。

证据：[完整源单元](source_all_strata.csv)、[固定判断](source_candidate_decision.csv)、[方法配对](source_matched_route_pairs.csv)、[源迁移](source_transfer_breakdown.csv)、[逐包核算](source_packet_arithmetic.json)、[完整执行](finished_source_execution.csv)、[结果证据](source_result_evidence.json)、[源校准](source_gradient_calibration.json)、[IQ几何](source_IQ_geometry.json)。
