# 星地识别优先：E30 活动目标

2026-10-18 用户最新指令覆盖此前必须构造 Ag＋As、先独立证明额外跨 RX 泛化的安排。当前阶段不另设 low>60% 或 suburban距clean≤2pp门槛；后者原为探索建议。历史结果与原判据保留，不追溯翻案。当前主目标是提高固定六种practical识别，重点mid/lowurban；阶段clean相对同预算同seed同角色无As对照下降≤0.1个绝对百分点，最终通过As训练获得可重复clean正收益。阶段保持不等于最终完成；跨RX/day分层继续报告，单seed或<0.1pp上涨仅初步观察。

固定F＋CROSS_RX、默认cosine头随骨干联合训练、NoExplicitCFO及内部补偿、原L6300/U56700TX隐藏/V27000、全部源RX1/3/4/6/8和day1/2/3、split392005、原IQ256、原正式receivedIQ/信道。预算仍每模型E30，不主动扩展结构、epoch或U规则。目标不参与设计、选模、晋级或救跑；source检查聚焦标签/数值/明显损伤，不加逐包预测或全特征一致门槛。

当前最有价值参照：P43 clean保留式fullAs，三seed low+17.2849±3.4169pp、mid+21.9448±5.1968pp、urban均值+22.3079±5.5014pp，clean−0.3210±0.5189pp，未满足阶段0.1pp。各seedclean+.0935/−.1536/−.9030pp，不能称稳定双提升；完整逐seed/六practical/F1/RX/成本见既有As-replication-r02报告。原始普通拼接的收益仍保留，不把不同研究增益相加。

下一唯一训练假设：完整As外的featuredropout可能进一步破坏弱身份。新run 20261018-phase1-FcrossRX-satelliteDropout-e30-s2026092701-m2-r01 新AS_CE/AS_DET两行scratchown20+10，复用P43 B00/B01同seed01固定E30参照，6L/step和原U不变。AS_CE仅作去satelliteKL的匹配归因控制，旧AsPairCE失败保持关闭；AS_DET只在实际交替64As行绕过dropout输出但完整生成原mask，测试的是模型扰动而非跨视图mask相关性。另一对话正在连续E30轨迹试验，本轮不重复、不改健康任务。

详细矩阵、source数值判据、放弃条件、单次发布/自动七视图truth-last与成本在该run的report.md。独立source判据只决定后续保留；正式冻结后报告clean/六practical及paired变化、urban均值、macroF1、worstRX、修复/新错、全部seed/波动/成本/未完成。正式分数不回流研发。新模型尚未训练完成；不能预告提高或阶段达标。
