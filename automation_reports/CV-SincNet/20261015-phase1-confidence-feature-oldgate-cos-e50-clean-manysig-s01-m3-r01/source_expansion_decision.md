# 置信度门控特征一致性：目标数值读回前的源扩展决定

FROZEN_SOURCE_ONLY，20261015-phase1-confidence-feature-oldgate-cos-e50-clean-manysig-s01-m3-r01固定源资格PASS；尚未读取该run新增目标数值摘要。完整source资格见source_expansion_decision.json，来源是原预登记规则，无阈值/权重扫描。

决定固定扩展2026092702/03，全SIG_TIME ZERO/CONF_FEATURE_MT两新模型各scratch50=33label+17pseudo/11100updates；旧门控+自身EMA+cos、默认头原联合训练、数据角色/物理ID/NoExplicitCFO不变。第34轮.02、第38轮起.1，teacherweak置信度>=studentstrong（固定gap0/ties保留），完整batchN分母，原UCE/entropy和门不改。每seed另复用核实历史OLD默认头测试，新4模型200轮/44400更新，6rowclean168000共1008000决策，逐row自动truth-last。

run 20261015-phase1-confidence-feature-oldgate-cos-e50-clean-manysig-s0203-m6-r01 / release cvs_confidence_feature_seeds_20261015_r01，矩阵code尚未实现或发布。先将本源决定commit+push并独立核实remoteOID，再读取首seed目标结果；读回后的好坏不改变已固定02/03、机制、权重、预算、父权重或停止规则。新增模型仅scratch，无目标选出的checkpoint。完整3seed仍是84%目标，n=1不冒充完成。
