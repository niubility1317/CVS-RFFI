前三轮均未通过各自预登记的源判据，未扩 E50。P67 将 U 时间确认改为同一物理样本跨轮确认，As 行最终接受量由 8726 增至 456028／567000；但源 mid／low 均值对原 As 仅 +0.069444 pp，low −0.055556 pp，clean 对匹配 NoAs −0.118519 pp。接受覆盖增加不能替代伪标签精度或识别收益。[源判定][p67]、[训练计数][p67train]

P68 在既有 6L 预算内引入第二个独立完整信道视图。同父 INDEP−SAME 的源 mid／low 仅 +0.069444 pp、low −0.027778 pp；clean 对 NoAs −0.029630 pp，仍未达到 +0.5 pp 增量。两行共同使用 SAT18 AS_CE20 父，与原 P43 As20 的前期 KL 历史不同。[源判定][p68]

P70 共享 142080 个 nominal 观测，各用一次；质量分层对随机池的源 mid／low −0.027778 pp，clean 对 NoAs −0.125926 pp。保持 nominal 风险的分层组织未提供足够增量。[源判定][p70]

P71 唯一改变 As CE 内部风险：固定 γ=.25、v=[.15,.40,.25,.20]，π=.75p+.25v，总 CE 系数仍为 .25。复用 M2 的池与顺序，从原 P43 As20 独立训练；不改变 KL、网络、U 或前向预算，不称 nominal 无偏估计。目前训练中，尚无收益结论；历史均为单模型 seed，正式成绩不用于上述取舍。

[p67]: E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-temporalBank-e30-s2026092701-m2-r01/source_matrix_decision.json
[p67train]: E:/type10-7/local_artifacts/cvs_FcrossRX_temporal_bank_E30_20261018_r01/retrieved_r01/analysis_r01/training_summary.json
[p68]: E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-fullAsPair6-e30-s2026092701-m1-r01/source_matrix_decision.json
[p70]: E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-qualityStrat-e30-s2026092701-m2-r01/source_matrix_decision.json
