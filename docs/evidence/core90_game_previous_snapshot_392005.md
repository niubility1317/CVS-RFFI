# CORE90完整产物分析

结论边界：仅源域证据，不作目标域泛化或晋级声明。缺失产物为pending，不按0计。

|行|seed|状态|主更新|头step|训练秒|源域评分|
|---|---|---|---|---|---|---|
|B0_seed392002|392002|pending_or_partial|608|0|pending|pending|
|B0_seed392003|392003|pending_or_partial|604|0|pending|pending|
|B1_seed392002|392002|pending_or_partial|610|0|pending|pending|
|B1_seed392003|392003|pending_or_partial|607|0|pending|pending|
|B2_seed392002|392002|pending_or_partial|596|0|pending|pending|
|B2_seed392003|392003|pending_or_partial|595|0|pending|pending|
|B4_seed392002|392002|pending_or_partial|323|646|pending|pending|
|B4_seed392003|392003|pending_or_partial|322|644|pending|pending|
|B5_seed392002|392002|pending_or_partial|394|0|pending|pending|
|B5_seed392003|392003|pending_or_partial|380|0|pending|pending|
|B6_seed392002|392002|pending_or_partial|277|0|pending|pending|

完整逐动作激活、失败、能力首次达标上界、四场景/RX指标和同seed差异见同名JSON。

三预算口径分别是固定训练轮次、实际训练时间、达到冻结源能力阈值的时间。相同seed、相同预算上限不等于实际计算成本匹配；训练总秒数已包含内部审计，不能重复累加。

