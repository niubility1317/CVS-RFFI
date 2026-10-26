# 恢复前后跨RX身份关系：完整源结果

ANALYZED，独立核实VERIFIED。四个方法只有一个学生seed2026092701，均own原ordinaryPL E30+20→E50/11100、累计13L37LU。四个最终默认头和17760条新增训练记录核实；全部sourceV27000八视图已固定，544个分层单元由逐包预测重算，另有72个方法间配对单元。没有新目标访问，不能称三个学生种子的性能。

实际代码`c0fc6a9a98e1a28669c2c41b59e26d2887311303`。8项聚焦测试、local/remote compile、唯一一次独立P0/P1检查、archive传输比较、真实E30/defaulthead/G与12批源校准、原L128/U256集成backward smoke通过。所属控制器与全部GPU子进程已自然退出；无技术失败或健康任务热改。

## 原门槛与判断

相对原sourceShift同seed PLControl、相同物理源V/场景，单位百分点。heldRX1 G3共16200次判断，≥+.3pp需净增至少49次；完整V rawclean27000条≥−.1pp需净增不少于−27。门槛按净正确数比较，保持原科学条件。RX1仅从附加分支留出，原父模型/L/rawPL/G已见它，不是全识别器未见RX证明。

| 方法 | heldG3差 | overallV rawclean差 | held纠正 / 新增错 / 净增 | 原源条件 |
| --- | ---: | ---: | ---: | --- |
| SeparateRG | +0.117284 | +0.000000 | 62 / 43 / 19 | FAIL |
| BridgeRG | +0.166667 | +0.000000 | 60 / 33 / 27 | FAIL |
| BridgeHG | +0.179012 | +0.000000 | 63 / 34 / 29 | FAIL |
| BridgeGG | +0.203704 | -0.003704 | 76 / 43 / 33 | FAIL |

**四条均未达到原源门槛，不扩种子，不启动这批方案的新目标测试。** 保留所有负结果和原门槛；后续依据同输入关系对照与H/G覆盖结果确定新机制，不直接继续增权。

## 实际机制与执行

SeparateRG/BridgeRG的额外输入都是raw/G，前者仅在同视图内、后者仅在跨视图间构造身份关系；两者输入、q/接受集合、归一化、正负pair数量、共同系数及预算匹配，只有关系结构不同。BridgeHG为同一受损/恢复对，BridgeGG为两个不同固定场景的恢复视图。所有正对为同伪类且不同RX，负对不同伪类；同包/同RX同类不作正对。

q来自原三E30 raw全同类且confidence≥.95，U真值隐藏。joint160采用两个视图的受理集合detached均值/std及L2归一，温度.2；无效/held记录不进入特征标准化。loss按2*完整U batch数归一，不按接受数放大。附加分支无分类weight梯度，featurefusion norm/gain仍训练；原完整T1 L/raw ordinaryPL与默认头主任务联合训练保持，无EMA/G/teacher/另训头更新。

12批72次无更新源全参数梯度一次确定共同λ=0.046777120618086768，444步升温。它是初始CENTER_UNITS代理，不能保证后续AdamW或目标效果。全步原L768/AsKL0/ORD.1、系数/ramp/视图路由及隐藏U核实；六组完整共有前缀的额外受理/held/有效anchor/正负pair计数一致，无非有限跳步。

每row U1134000、held226800、额外受理838780、额外学生FW2268000。下面anchor和pair为实际累计关系数量，不能当作独立物理包数。

| 方法 | raw FW | H FW | G FW | validanchor | positivepair | negativepair |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SeparateRG | 1134000 | 0 | 1134000 | 1677560 | 39544936 | 263214104 |
| BridgeRG | 1134000 | 0 | 1134000 | 1677560 | 39544936 | 263214104 |
| BridgeHG | 0 | 1134000 | 1134000 | 1677560 | 39544936 | 263214104 |
| BridgeGG | 0 | 0 | 2268000 | 1677560 | 39544936 | 263214104 |

## 完整训练曲线

全部17760条step记录按实际新增epoch汇总为80个完整epoch。附加loss和rawPL loss均为训练目标值，rawPL覆盖率为当前学生按置信度选择的曝光比例；三者不能替代伪标签真实正确率或目标准确率。下表只解释优化是否实际发生，不用于提前选模。

| 方法 | 新增E3关系loss | 新增E20关系loss | 新增E20 rawPL loss | 新增E20 rawPL覆盖率 |
| --- | ---: | ---: | ---: | ---: |
| SeparateRG | 0.361895 | 0.316049 | 0.006293 | 96.195767% |
| BridgeRG | 0.387657 | 0.339712 | 0.006273 | 96.148148% |
| BridgeHG | 0.864998 | 0.805288 | 0.006504 | 96.211640% |
| BridgeGG | 0.506671 | 0.462108 | 0.006334 | 96.093474% |

## held同包配对

下表为candidate−control的直接配对差，区别于各自相对PL的增益。SeparateRG/BridgeRG是同输入关系结构对照；HG/GG改变实际输入覆盖，不能单独归因于某一损失。完整参与RX及TX/day结果见CSV。

| candidate − control | 源视图 | 纠正 / 新增错 / 净增 | 差（pp） |
| --- | --- | ---: | ---: |
| BridgeRG − SeparateRG | raw | 0 / 0 / 0 | +0.000000 |
| BridgeRG − SeparateRG | H_three_mean | 37 / 39 / -2 | -0.012346 |
| BridgeRG − SeparateRG | G_three_mean | 18 / 10 / 8 | +0.049383 |
| BridgeHG − SeparateRG | raw | 0 / 1 / -1 | -0.018519 |
| BridgeHG − SeparateRG | H_three_mean | 72 / 60 / 12 | +0.074074 |
| BridgeHG − SeparateRG | G_three_mean | 19 / 9 / 10 | +0.061728 |
| BridgeGG − SeparateRG | raw | 0 / 0 / 0 | +0.000000 |
| BridgeGG − SeparateRG | H_three_mean | 48 / 47 / 1 | +0.006173 |
| BridgeGG − SeparateRG | G_three_mean | 29 / 15 / 14 | +0.086420 |
| BridgeHG − BridgeRG | raw | 0 / 1 / -1 | -0.018519 |
| BridgeHG − BridgeRG | H_three_mean | 77 / 63 / 14 | +0.086420 |
| BridgeHG − BridgeRG | G_three_mean | 20 / 18 / 2 | +0.012346 |
| BridgeGG − BridgeRG | raw | 0 / 0 / 0 | +0.000000 |
| BridgeGG − BridgeRG | H_three_mean | 53 / 50 / 3 | +0.018519 |
| BridgeGG − BridgeRG | G_three_mean | 22 / 16 / 6 | +0.037037 |
| BridgeGG − BridgeHG | raw | 1 / 0 / 1 | +0.018519 |
| BridgeGG − BridgeHG | H_three_mean | 62 / 73 / -11 | -0.067901 |
| BridgeGG − BridgeHG | G_three_mean | 29 / 25 / 4 | +0.024691 |

新增80F轮，共享三个E30历史90轮及旧q/H/G缓存成本单列。原同候选三个学生种子联合目标保持：raw clean≥85.98174603174603%、raw六景>72.0961970899471%、同G-LU六景>78.07463624338624%。本源实验不能证明联合目标。

证据：[完整源单元](source_all_strata.csv)、[固定判断](source_candidate_decision.csv)、[方法配对](source_matched_route_pairs.csv)、[源迁移](source_transfer_breakdown.csv)、[逐包核算](source_packet_arithmetic.json)、[完整执行](finished_source_execution.csv)、[完整训练曲线](source_training_curves.csv)、[曲线证据](source_training_curves_evidence.json)、[结果证据](source_result_evidence.json)、[源校准](source_gradient_calibration.json)。


## 事后源错误重叠：不直接相加两个弱分支

只读复用当前冻结预测与旧FeatureTask预测，所有V物理ID/y/场景及来源契约对齐，没有新模型前向、optimizer、U真值或目标读取。相同H/G输入的旧HGFeatureHard（CE）和当前BridgeHG（CL）在held G3中：共同纠正52/共同破坏24；CE独有纠正23/独有破坏33，CL独有纠正11/独有破坏10。在held H3中：共同纠正102/共同破坏75；CE独有纠正61/独有破坏42，CL独有纠正44/独有破坏46。

因此“CE和CL纠错有互补”不能推出相加有收益；G上的CE独有新增错误更多，H上的CE则提供额外净收益。两旧配方剂量和目标不同，不能将此描述统计写成G分支CE的因果证明。旧G/G CE与新双场景GG CL还存在输入曝光差异，完整18组见[源错误重叠](source_objective_overlap.csv)及[证据](source_objective_overlap.json)；不构造依赖真值的预测择优或宣称联合训练会达到这些计数。

下一可检验假设是非对称H/G联合目标：H承担硬伪标签分类，G提供跨RX身份关系参照。计划保持相同H/G两次学生前向，比较HCE+双向CL、HCE+仅H梯度CL（G特征detach）、GCE+双向CL、平均H/G CE+双向CL。CE和CL的固定相对系数各1，再通过12批无更新全参数源梯度一次确定四条共同总剂量。它不是原监督损失权重的照搬，也不保证原两类增益叠加。原L/rawPL、q、默认头、own原E30+20、原+.3/−.1源门槛和最终三学生种子联合目标保持；当前仅是下一方向，尚未发布该新矩阵。
