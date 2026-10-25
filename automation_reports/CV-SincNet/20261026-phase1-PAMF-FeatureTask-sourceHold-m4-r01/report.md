# 特征任务 CE 的完整源对照

ANALYZED，独立核验VERIFIED。四个方法只有一个学生seed2026092701；全部完成 own原E30+20→E50/11100、每条4440新增步，17760条实际训练记录及四份最终六类默认头checkpoint全部核实。所有源V27000八视图已固定，544个分层单元直接由逐包预测独立重算，36个方法间路由配对单元也由同包结果计算。没有新目标访问。

实际代码`1f34b3ac51a355fb5da73591ae43f9c28f73130d`。源H56700三景配对、12批L/U无更新校准、真实E30/defaulthead/G检查与集成HG联合backward smoke均通过；启动与四训练进程PID/CWD/argv/GPU/日志增长独立核实。7项聚焦测试、local/remote compile、唯一一次P0/P1检查和archive传输核对通过。健康任务按固定预算完成，未热改。

## 原门槛与完整结果

以下均相对既有同seed PLControl、相同源V物理ID与场景，单位百分点。heldRX1的G三景合计16200次判断，原门槛≥+.3pp等价于净增至少49次；整体V rawclean27000条门槛≥−.1pp等价于净增不少于−27。只有额外U分支排除RX1，原父/L/rawPL/G已见它，不是真正未见baseRX验证。

| 方法 | heldRX G三景差 | 整体V rawclean差 | held纠正 / 新增错 / 净增 | 原源条件 |
| --- | ---: | ---: | ---: | --- |
| GJointHard | +0.123457 | -0.018519 | 81 / 61 / 20 | FAIL |
| GFeatureHard | +0.141975 | -0.014815 | 95 / 72 / 23 | FAIL |
| HFeatureHard | +0.086420 | -0.007407 | 60 / 46 / 14 | FAIL |
| HGFeatureHard | +0.111111 | -0.014815 | 75 / 57 / 18 | FAIL |

**四条均未通过原源门槛，不扩三种子，不启动本批的新目标测试。** 保留全部负结果和原门槛。下一步依据匹配路由差异及既有主辅梯度冲突证据判断后续机制，不能只凭训练loss下降继续扩大系数，也不能把本批源收益写成目标收益。

## 原始信道与恢复视图的迁移

H三景为未恢复信道IQ，G三景为同包经固定G-LU恢复；这里只有源三景，不是正式目标六景。trainedRX三景共64800次判断，held三景16200次。

| 方法 | held H三景差 | held G三景差 | trained H三景差 | trained G三景差 |
| --- | ---: | ---: | ---: | ---: |
| GJointHard | +0.141975 | +0.123457 | +0.012346 | +0.470679 |
| GFeatureHard | +0.135802 | +0.141975 | +0.035494 | +0.442901 |
| HFeatureHard | +0.148148 | +0.086420 | +0.628086 | -0.007716 |
| HGFeatureHard | +0.283951 | +0.111111 | +0.375000 | +0.300926 |

下面直接比较方法之间的held同包预测；与上表相对PL的差值不同。G两组同额外G/G、硬标签和λG，隔离分类权重附加梯度；HG则改变第二种视图覆盖。

| candidate − control | 源视图 | 纠正 / 新增错 / 净增 | 配对差（pp） |
| --- | --- | ---: | ---: |
| GFeatureHard − GJointHard | raw | 0 / 0 / 0 | +0.000000 |
| GFeatureHard − GJointHard | H_three_mean | 44 / 45 / -1 | -0.006173 |
| GFeatureHard − GJointHard | G_three_mean | 26 / 23 / 3 | +0.018519 |
| HGFeatureHard − GFeatureHard | raw | 0 / 0 / 0 | +0.000000 |
| HGFeatureHard − GFeatureHard | H_three_mean | 87 / 63 / 24 | +0.148148 |
| HGFeatureHard − GFeatureHard | G_three_mean | 32 / 37 / -5 | -0.030864 |
| HGFeatureHard − HFeatureHard | raw | 0 / 0 / 0 | +0.000000 |
| HGFeatureHard − HFeatureHard | H_three_mean | 90 / 68 / 22 | +0.135802 |
| HGFeatureHard − HFeatureHard | G_three_mean | 41 / 37 / 4 | +0.024691 |

## 实际执行与边界

每条U曝光1134000、held曝光226800、额外接受838780、额外学生前向2268000；HG各H/G1134000，其他两次对应视图2268000。原L768包、AsKL0、ORD.1、普通rawPL、全U归一与444步升温记录符合配置，六组额外接受共同前缀完全一致，无非有限跳步、U真值、G/教师或另训头。原默认头由主任务正常联合更新；特征组只阻断附加分类阶段weight梯度，特征融合norm/gain仍更新。

源校准λG=0.059415326575376119、λH=0.11323142806482039按预登记完整参数CENTER_UNITS梯度规则固定。两G组12批CE完全相同，原norm分解误差约1.095e−7；屏蔽后主辅cos中位数−.249252→−.224113，H为−.145018，局部冲突没有消失。此为机制诊断，不是目标因果或收益证明。校准参数/buffers全部恢复；后续训练正常更新学生，最终源评估F/G参数与全部registered buffers保持不变。

新增四尾部80 F轮，共享三原E30历史90轮、旧q/G缓存与PLControl历史训练单列复用，不称新训练或零成本。最终目标仍要求同一方法三个学生种子的raw clean≥85.98174603174603%、raw六景>72.0961970899471%、同G-LU六景>78.07463624338624%；本源试验不能证明这些目标已实现。

证据：[全部源单元](source_all_strata.csv)、[固定判断](source_candidate_decision.csv)、[源迁移分解](source_transfer_breakdown.csv)、[方法间配对](source_matched_route_pairs.csv)、[逐包独立核算](source_packet_arithmetic.json)、[完整训练执行](finished_source_execution.csv)、[结果证据](source_result_evidence.json)、[源校准](source_gradient_calibration.json)。
