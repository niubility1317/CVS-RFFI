# Phase1本轮续接交付与尚未完成的目标

目标为无显式CFO身份标量、原物理数据角色及L_s标签比例不变、纯DG下clean三primaryseed均值至少84%。用户明确本轮从头训练30轮。目标尚未完成。

正式已核实的改进：固定三成员native集成clean三primaryseed均值81.3254%，SD1.2267个百分点；对应单native30为79.4562%，配对提升1.8692pp，三个seed均正。三seed分别82.3774%、79.9780%、81.6208%；距84%仍2.6746pp。每ensemble531075可训练参数，为单模型3倍，累计训练成本也3倍。

[完整三seed、clean+六practical正式结果](../20261012-phase1-ensemble-final30-recovery-manysig-m3-r02/final_analysis.md)。3.528百万原始预测、378指标独立重算一致。原6新模型全部30轮后源汇总元数据复制参考而非运行契约导致预测前拒绝；旧权重/失败日志保留，只从真实子契约恢复的新run零训练，未伪造角色标记或降低检查。

本轮新增训练81模型×30轮=2430模型轮，共539460次更新；其中正式6新成员180模型轮/39960更新，其他源先导75模型2250模型轮/499500更新。不能把source探索成本隐藏成只做正式三seed的成本。原3正式member0及其他既有source父模型的历史训练额外披露在各row报告，零训练读出仍继承其训练成本。

|本轮路线|新增模型|新增更新|证据|
|---|---:|---:|---|
|质量条件FiLM|15|99900|[20261012-phase1-quality-film-source-manysig-s2026092701-r01](../20261012-phase1-quality-film-source-manysig-s2026092701-r01/final_analysis.md)|
|互补波形关系|10|66600|[20261012-phase1-relation-complement-source-manysig-s2026092701-r02](../20261012-phase1-relation-complement-source-manysig-s2026092701-r02/final_analysis.md)|
|三成员随机集成源先导|10|66600|[20261012-phase1-stochastic-ensemble-source-manysig-s2026092701-r01](../20261012-phase1-stochastic-ensemble-source-manysig-s2026092701-r01/final_analysis.md)|
|三成员全源最终30轮|6|39960|[20261012-phase1-ensemble-final30-manysig-m3-r01](../20261012-phase1-ensemble-final30-manysig-m3-r01/failure_analysis.md)|
|EMA及集成移交|10|66600|[20261012-phase1-bootstrap-ema-source-manysig-s2026092701-r01](../20261012-phase1-bootstrap-ema-source-manysig-s2026092701-r01/final_analysis.md)|
|独立全波形CNN|5|33300|[20261012-phase1-complement-cnn-source-manysig-s2026092701-r01](../20261012-phase1-complement-cnn-source-manysig-s2026092701-r01/final_analysis.md)|
|相位/统计池化消融|15|99900|[20261012-phase1-phase-pool-cnn-source-manysig-s2026092701-r01](../20261012-phase1-phase-pool-cnn-source-manysig-s2026092701-r01/final_analysis.md)|
|RX/随机分类头|10|66600|[20261012-phase1-rx-heads-source-manysig-s2026092701-r01](../20261012-phase1-rx-heads-source-manysig-s2026092701-r01/final_analysis.md)|
|RX分类器读出|0|0|[20261012-phase1-rx-head-readout-source-manysig-s2026092701-r01](../20261012-phase1-rx-head-readout-source-manysig-s2026092701-r01/final_analysis.md)|
|三成员证据读出|0|0|[20261012-phase1-ensemble-readout-source-manysig-s2026092701-r01](../20261012-phase1-ensemble-readout-source-manysig-s2026092701-r01/final_analysis.md)|

后续源端结论：EMA移交61.7185%，对照SELF EMA61.8074%，未通过；CNN固定混合54.7333%；相位/池化最好固定混合60.3148%；RX分组混合62.4000%相对随机61.4222%有源机制差异，但仅2/5 RX超过native3，未满足晋级；冻结RX读出最好62.4704%；三成员UNIT_MARGIN源61.9704%，未达到对原probmean61.7778%的+1pp阈值。所有未晋级路线均未进行目标测试，原判定不重写、不追逐微小涨分。

本轮全部上述run均完成或其直接失败已明确保留并完成零训练恢复；不存在只发布训练却未收尾的正式测试候选。原健康任务未被停止、重启或热修改，管理员账号未使用。代码/预登记/同row人可读报告提交推送；数据、weights、日志、原始预测和本机状态不进入Git。既有目标benchmark已暴露，当前结果不是新盲测或实际卫星部署。

下一项超出现有30轮预算的可审阅建议见[预算选项](budget_option.md)。它尚未获授权，未实施、登记为运行或发布；不把源候选失败当目标已完成，也不默认追加训练。
