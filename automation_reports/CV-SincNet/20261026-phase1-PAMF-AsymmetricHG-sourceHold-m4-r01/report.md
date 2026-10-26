# 非对称H/G任务与关系：完整源结果

ANALYZED，独立核实VERIFIED。HJoint/HAnchor/GJoint/BothJoint只有一个学生seed2026092701，四条own原ordinaryPL E30+20至E50/11100，累计13L37LU。四份实际最终默认头、17760新增step、80完整训练epoch、544源预测单元及72方法配对全部核算，无技术失败、U真值或新目标访问。

实际代码`f064f495716de7434be99027ae383b2a49c88666`。7项聚焦检查/local+remote compile、唯一一次独立P0/P1、archive传输比较、真实E30/defaulthead/G与12批源校准、原L128/U256联合backward smoke通过。所属控制器与全部训练/评估子进程正常结束，健康任务没有热改。

## 固定源条件与结果

对照原sourceShift同seed PLControl、相同sourceV27000八视图。heldRX1 G3共16200次判断，原≥+.3pp等价于净增至少49次；overallV rawclean27000条≥−.1pp等价于净增不少于−27。RX1只从附加分支留出，原父模型/L/rawPL/G已经见过它，不是全网络未见RX测试。

| 方法 | heldG3差（pp） | overallrawclean差（pp） | held纠正 / 新增错 / 净增 | 原源条件 |
| --- | ---: | ---: | ---: | --- |
| HJoint | +0.160494 | -0.007407 | 62 / 36 / 26 | FAIL |
| HAnchor | +0.135802 | -0.011111 | 52 / 30 / 22 | FAIL |
| GJoint | +0.179012 | -0.025926 | 84 / 55 / 29 | FAIL |
| BothJoint | +0.166667 | -0.022222 | 64 / 37 / 27 | FAIL |

**四条均未通过原源门槛，不扩种子，不启动本批目标测试。** 非对称分工或CE/CL相加没有扩大到所需留出收益；保留原门槛和全部负结果。下一轮转向匹配的训练视图支持比较，不再只改这一组损失路由。

## 实际梯度与曝光

四路相同H/G两次学生前向、冻结三E30 raw共识95 q、附加RX1排除、原L/rawPL与默认头主任务联合训练保持。HJoint为H CE+双向CL；HAnchor为同值H CE+CL(H,detach(G上的F特征))；GJoint为G CE+双向CL；Both为平均H/G CE+双向CL。CE1/CL1相对系数在源校准前固定，附加分类weight梯度0，featurefusion norm/gain仍训练；无EMA/新头/G更新。

真实12批逐项核算确认HJoint/HAnchor数值一致、HAnchor的G特征CL梯度0、其他route H/G特征梯度非零；参数/buffers不变，原L768/AsKL0。按共同最大中位梯度公式一次冻结λ=0.042210454011369962，444步升温；仅初始CENTER_UNITS代理，不保证后续AdamW或性能。

每row U1134000、held226800、extraaccepted838780、H/G学生FW各1134000。下表CE计数是受理包曝光，Both两CE各.5；anchor/pair不是独立物理包。六组完整4440步共有前缀的extra/held/anchor/pairs完全一致，任务曝光符合实际路由，总附加loss=CE+CL，未见非有限跳步。

| 方法 | CE_H受理 | CE_G受理 | validanchor | positivepair | negativepair |
| --- | ---: | ---: | ---: | ---: | ---: |
| HJoint | 838780 | 0 | 1677560 | 39544936 | 263214104 |
| HAnchor | 838780 | 0 | 1677560 | 39544936 | 263214104 |
| GJoint | 0 | 838780 | 1677560 | 39544936 | 263214104 |
| BothJoint | 838780 | 838780 | 1677560 | 39544936 | 263214104 |

## 源迁移与直接配对

| 方法 | 额外分支RX组 | 源视图 | 相对PL差（pp） |
| --- | --- | --- | ---: |
| HJoint | held | H_three_mean | +0.166667 |
| HJoint | held | G_three_mean | +0.160494 |
| HJoint | trained | H_three_mean | +0.424383 |
| HJoint | trained | G_three_mean | +0.115741 |
| HAnchor | held | H_three_mean | +0.055556 |
| HAnchor | held | G_three_mean | +0.135802 |
| HAnchor | trained | H_three_mean | +0.390432 |
| HAnchor | trained | G_three_mean | +0.016975 |
| GJoint | held | H_three_mean | +0.216049 |
| GJoint | held | G_three_mean | +0.179012 |
| GJoint | trained | H_three_mean | +0.262346 |
| GJoint | trained | G_three_mean | +0.402778 |
| BothJoint | held | H_three_mean | +0.197531 |
| BothJoint | held | G_three_mean | +0.166667 |
| BothJoint | trained | H_three_mean | +0.341049 |
| BothJoint | trained | G_three_mean | +0.294753 |

| candidate − control | held源视图 | 纠正 / 新增错 / 净增 | 直接差（pp） |
| --- | --- | ---: | ---: |
| HAnchor − HJoint | H_three_mean | 36 / 54 / -18 | -0.111111 |
| HAnchor − HJoint | G_three_mean | 15 / 19 / -4 | -0.024691 |
| GJoint − HJoint | H_three_mean | 58 / 50 / 8 | +0.049383 |
| GJoint − HJoint | G_three_mean | 46 / 43 / 3 | +0.018519 |
| BothJoint − HJoint | H_three_mean | 60 / 55 / 5 | +0.030864 |
| BothJoint − HJoint | G_three_mean | 25 / 24 / 1 | +0.006173 |
| GJoint − HAnchor | H_three_mean | 79 / 53 / 26 | +0.160494 |
| GJoint − HAnchor | G_three_mean | 60 / 53 / 7 | +0.043210 |
| BothJoint − HAnchor | H_three_mean | 74 / 51 / 23 | +0.141975 |
| BothJoint − HAnchor | G_three_mean | 37 / 32 / 5 | +0.030864 |
| BothJoint − GJoint | H_three_mean | 42 / 45 / -3 | -0.018519 |
| BothJoint − GJoint | G_three_mean | 24 / 26 / -2 | -0.012346 |

HAnchor相对同输入、同值损失的HJoint没有扩大held H/G收益。G任务CE可同时增加纠正和新错，不能把两个独立弱分支的纠错并集当作联合训练收益。以上仅为一个seed源侧结果，不是目标性能或普遍因果结论。

## 完整训练曲线

| 方法 | 新增E3联合loss | 新增E20联合loss | 新增E20 rawPL loss | 新增E20 rawPL覆盖率 |
| --- | ---: | ---: | ---: | ---: |
| HJoint | 1.347201 | 1.245065 | 0.006609 | 96.229277% |
| HAnchor | 1.361728 | 1.271557 | 0.006650 | 96.266314% |
| GJoint | 1.068603 | 0.986484 | 0.006707 | 96.107584% |
| BothJoint | 1.210578 | 1.121608 | 0.006667 | 96.139330% |

全部17760记录汇总80个完整epoch，CE/CL组件见CSV。loss及覆盖率不是伪标签真实正确率。新增80F轮，原三E30历史90轮及q/H/G缓存成本共享单列。原同候选三个学生种子raw clean≥85.98174603174603%、raw六景>72.0961970899471%、同G-LU六景>78.07463624338624%的联合目标保持未达成，不更换默认方法。

见[固定判断](source_candidate_decision.csv)、[全部分层](source_all_strata.csv)、[方法配对](source_matched_route_pairs.csv)、[迁移分解](source_transfer_breakdown.csv)、[完整执行](finished_source_execution.csv)、[曲线](source_training_curves.csv)、[逐包核算](source_packet_arithmetic.json)、[最终证据](source_result_evidence.json)、[源校准](source_gradient_calibration.json)。
