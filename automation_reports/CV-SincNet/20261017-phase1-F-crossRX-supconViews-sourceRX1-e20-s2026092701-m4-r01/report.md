# 原跨 RX 同 TX 对比项视图组织：完整源域 E20 结果

**ANALYZED／VERIFIED；FAIL_FIXED_VIEW_ROUTING_CLOSED。** 两个新 scratch E20 与两个匹配冻结控制的全部源域结果已独立复算。固定 primary 为 Q_mix，不根据结果切换为 Q_sat。正式目标数据未访问，完整联合目标仍 ACTIVE。

实际训练代码为 ed0ca91481f8c11a28cb8f4d4726a7c3d488720e；离线分析代码 f88e5cbf8870451a79bb679a091c6b203faa819d。原进程自然完成，未热改、重启或延长预算。[预登记](preregistration.md) 与 [配置](experiment.json) 保留。

## 数据与比较范围

单模型种子 2026092701，split392005。原可见 L 中 RX3/4/6/8 的 L5040 参与训练，U45360 真值隐藏，V21600 仅原源健康流程；原 RX1 L1260 只在 final20 后固定读出，每视图 1260。该契约不是完整 L6300，也不是 L_train5580/L_dev720。每行沿默认头得到 clean＋六 practical；全四行 35280 个源判决，其中 17640 个是复用的历史控制，17640 个来自本轮新行。没有正式 target 预测或 truth-last 测试结果。

Q0＝extra raw＋原 ordinary 对比输入；Q_ord＝extra 完整 IID As＋原 ordinary 对比输入；Q_sat＝同 As＋原对比项全 As 输入；Q_mix＝同 As＋每步出现位置 64 As/64 ordinary 输入。四行保持完整 raw .75/ordinary .25 基础监督、额外 .25 配对、总 pairweight1.25、8 次 L 前向、原 RDrop 两次同 IQ、原对比层/温度/系数和原 13L＋7U 规则。Q0/Q_ord 仅复用实际配置与完整物理角色一致的旧冻结结果；新行没有加载其权重。

本实验检验 A_s 训练组织，未引入新 IQ 泛化增强。Q0/Q_ord/Q_sat/Q_mix 不能冒充 B00/B10/B01/B11，因此不能计算或声称 A_g×A_s 因子交互。

## 冻结默认头识别结果

准确率，单位为 %。

| 行 | clean | high suburban | mid suburban | low suburban | high urban | mid urban | low urban | urban 均值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q0 | 80.158730 | 39.920635 | 40.238095 | 36.349206 | 38.968254 | 32.460317 | 27.777778 | 33.068783 |
| Q_ord | 84.444444 | 59.365079 | 55.714286 | 52.619048 | 54.365079 | 43.492063 | 33.888889 | 43.915344 |
| Q_sat | 82.936508 | 56.190476 | 51.587302 | 49.523810 | 52.777778 | 40.714286 | 32.619048 | 42.037037 |
| Q_mix | 83.809524 | 57.857143 | 53.253968 | 51.111111 | 52.619048 | 41.428571 | 33.730159 | 42.592593 |

宏平均 F1，单位为 %。

| 行 | clean | high suburban | mid suburban | low suburban | high urban | mid urban | low urban |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q0 | 79.742937 | 29.049466 | 30.224256 | 28.312996 | 29.583784 | 26.906529 | 24.569258 |
| Q_ord | 84.032268 | 57.619721 | 54.049281 | 50.935640 | 53.006631 | 42.830981 | 33.114787 |
| Q_sat | 82.239087 | 53.936453 | 49.089444 | 47.371959 | 51.122542 | 39.848863 | 31.800660 |
| Q_mix | 83.331596 | 56.075781 | 51.472446 | 49.415797 | 51.146733 | 40.780249 | 32.973340 |

## 固定主候选与匹配参照

| Q_mix 减参照 | clean | low urban | mid urban | urban 均值 | 判据结果 |
|---|---:|---:|---:|---:|---|
| Q0 | 3.650794 | 5.952381 | 8.968254 | 9.523810 | PASS |
| Q_ord | -0.634921 | -0.158730 | -2.063492 | -1.322751 | FAIL：clean, clean_F1, low, low_F1, mid, urban_mean, each_practical |
| Q_sat | 0.873016 | 1.111111 | 0.714286 | 0.555556 | FAIL：urban_mean |

差值单位为百分点。Q_mix 必须同时对 Q_ord/Q_sat 保持 clean/F1、mid、low F1，并使 low 与 urban 均值各 ≥+1pp；对 Q0 要求 low≥+5pp、urban 均值≥+3pp；每 practical 与 seenV 下降≤.5pp，基础目标≤.5，全部预算/权限正确。每项按预登记独立复算，未修改阈值。未通过项：Q_ord_clean, Q_ord_clean_F1, Q_ord_low, Q_ord_low_F1, Q_ord_mid, Q_ord_urban_mean, Q_ord_each_practical, Q_sat_urban_mean。

## suburban 绝对值与自身 clean 差距

| 行 | high suburban / clean差 | mid suburban / clean差 | low suburban / clean差 |
|---|---:|---:|---:|
| Q0 | 39.920635% / 40.238095pp | 40.238095% / 39.920635pp | 36.349206% / 43.809524pp |
| Q_ord | 59.365079% / 25.079365pp | 55.714286% / 28.730159pp | 52.619048% / 31.825397pp |
| Q_sat | 56.190476% / 26.746032pp | 51.587302% / 31.349206pp | 49.523810% / 33.412698pp |
| Q_mix | 57.857143% / 25.952381pp | 53.253968% / 30.555556pp | 51.111111% / 32.698413pp |

差距定义为该行 clean 准确率减该行 suburban 准确率。每景绝对差≤2pp 仍是探索目标；不得靠降低 clean 缩差。

## 配对身份错误

| 参照→Q_mix | urban 场景 | 全1260修复/新增 | 双方clean正确集合n | 集合内修复/新增 | 集合内净正确 |
|---|---|---:|---:|---:|---:|
| Q0 | practical_high_urban | 345/173 | 990 | 330/68 | 262 |
| Q0 | practical_mid_urban | 283/170 | 990 | 265/90 | 175 |
| Q0 | practical_low_urban | 207/132 | 990 | 190/74 | 116 |
| Q_ord | practical_high_urban | 18/40 | 1040 | 16/37 | -21 |
| Q_ord | practical_mid_urban | 19/45 | 1040 | 17/36 | -19 |
| Q_ord | practical_low_urban | 29/31 | 1040 | 23/24 | -1 |
| Q_sat | practical_high_urban | 29/31 | 1033 | 22/24 | -2 |
| Q_sat | practical_mid_urban | 24/15 | 1033 | 21/13 | 8 |
| Q_sat | practical_low_urban | 29/15 | 1033 | 22/12 | 10 |

双方 clean 正确集合是描述性固定子集，不改变晋级规则或过滤样本；全部 1260 个源样本仍参与评分。完整按 TX/day 的分层保存在本地 source_strata.csv。此处没有跨多个 held RX 的最差 RX 证明。

## 实际预算和完整轨迹

| 行 | 新训练/复用 | 更新 | 网络样本前向 | As生成/监督 | 训练秒 | 信道生成秒 |
|---|---|---:|---:|---:|---:|---:|
| Q0 | 历史 E20 复用 | 4440 | 5823318 | 0 | 1403.310508 | 0.000000 |
| Q_ord | 历史 E20 复用 | 4440 | 5823318 | 568320 | 2310.321066 | 871.221358 |
| Q_sat | 新 scratch E20 | 4440 | 5823318 | 568320 | 2291.135674 | 883.647509 |
| Q_mix | 新 scratch E20 | 4440 | 5823318 | 568320 | 2344.361997 | 874.430600 |

新增 40 modelE／8880 次更新／网络样本前向 11646638（含本轮 dispatcher 2 个合成 no-query smoke）；历史复用 40 modelE／8880 次更新／11646636 个行内样本前向，未重跑。控制来自原 3 行、60 modelE／13320 更新的 sceneCycle run，第三个 balanced 行未纳入本矩阵；原共享 smoke 不重复计入本轮。新旧训练成本分列，不把复用说成免费历史训练。

四行各自的前向实计相同：student train 4942928（L4546560＋原 U396368），student eval 475202；teacher train 0，teacher eval 原始计数396370，扣除继承的2个初始smoke计数后为396368；独立held读出8820，EMA成功更新4440。上述extra L1136640已包含在L/总量内，不重复相加。全量L/U观测序列逐项相同，原生参数仅run/candidate/output标识不同；实际teacher、extra、held和历史/新增成本分别保留。

每行实际 L 前向4546560，其中 extra1136640；As 行各生成568320次完整practical，Q0不生成As。所有有标签 As 均监督。每行原对比更新3330次／原层-phase项13320个，路由没有新增 NN 或 loss项。As map 捕获、描述提取、混合与暂存仍有计算和内存成本，不能称为零计算。完整逐步 loss 公式、CE/RDrop/SupCon尺度、场景决策及全量 L/U 轨迹均独立复算，有限性、默认头、初始化和权限检查通过。

- Q0：原纯L更新2886，U更新1554，U样本396368，EMA成功更新4440；U真值始终隐藏。路由As/ordinary出现次数0/426240，完整实际选择与计数已复核。
- Q_ord：原纯L更新2886，U更新1554，U样本396368，EMA成功更新4440；U真值始终隐藏。路由As/ordinary出现次数0/426240，完整实际选择与计数已复核。
- Q_sat：原纯L更新2886，U更新1554，U样本396368，EMA成功更新4440；U真值始终隐藏。路由As/ordinary出现次数426240/0，完整实际选择与计数已复核。
- Q_mix：原纯L更新2886，U更新1554，U样本396368，EMA成功更新4440；U真值始终隐藏。路由As/ordinary出现次数213120/213120，完整实际选择与计数已复核。

原生 final_L_CE 是 first ordinary smoothed CE，不能将其当作全部训练目标；完整 base objective、extra pair、原 SupCon 和权重分别保留。硬预测没有保存 logits，独立重算的是准确率/F1/配对错误，readout CE 仅作为运行报告值。所有80轮结构化轨迹及四行完整stdout均已检查；指标改善不能从辅助损失、anchors或源健康推出。

## 图表与原始证据

- [source_accuracy_all_views.png](E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_supcon_view_routing_E20_20261017/final/figures/source_accuracy_all_views.png)
- [paired_urban_identity_errors.png](E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_supcon_view_routing_E20_20261017/final/figures/paired_urban_identity_errors.png)
- [full_training_curves.png](E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_supcon_view_routing_E20_20261017/final/figures/full_training_curves.png)

完整 analysis_summary.json、training_curves.csv、source_strata.csv、全部源判决和轨迹保存在所属本地 final 根，权重保存在原 run，未提交 Git。收集仅取回本 run 元数据、源判决和日志，0 IQ/权重/新前向/更新。39 个离线反例检查通过；原候选唯一 P0/P1、真实匹配 no-query smoke 和发布证据保留，不重复审查或增设门槛。

## 结论与未完成项

固定 Q_mix 未通过预登记科学条件，关闭本次 existing-SupCon 视图组织路线。保留 Q_sat 和全部负结果，不改比例、层位、权重、场景概率、源 RX、seed 或预算救跑，也不按结果改选 Q_sat。

当前尚无验证有效的新 Ag；完整同契约 B00/B10/B01/B11 与交互、跨 seed 重现、正式 clean＋六 practical truth-last、clean 正收益、low urban>60%、suburban 接近自身 clean 均未完成。该源域终态不替代完整目标。
