# Phase1 可迁移特征提取器目标

以2026-10-14用户更新goal-objective.md为最新范围：持续复盘→合法源机制诊断→方向选择→最小实现→公平实验→结果分析。主任务仅Phase1，实际导出joint160的fθ增强与原头clean三预定义seed均值84%分别验收；不承诺必达，不追分调参。

默认ResidualPhysicalHead普通cos6×160/scale30随骨干按原损失联合训练，不通过复杂head/RX路由/类别专属参数优化。最新目标明确允许冻结骨干、丢弃旧W后仅L_s拟合同形式ordinarycosine作为特征诊断，预算单列、theta/buffers不变；此授权覆盖此前禁止一切source换头的目标表述。它不授权support调参或Phase2开发，不改变已发布PA单ORIGINAL_HEAD计划。

唯一后续基线：历史batch_neighbor伪标签门控＋EMA＋普通cos的LEGACY50，32label＋18pseudo，50epoch/11100updates，seeds2026092701/02/03，同原ManySigsource契约、NoExplicitCFO及内部补偿。MS/RC的33＋17与10轮AdamW诊断和本对话32＋18/20轮Adam诊断分别保留，不合并为同配方。

Phase1骨干每模型≤50，优先从零；L6300/U56700隐藏/V27000，训练池标注10%，split/sourceRX/day/TX/物理ID不变。V用途按预登记不更新模型，不增加星地训练。目标IQ/标签/统计/缓存/误差不进入结构、参数、父权重或停止选择；冻结后逐包预测与truth-last，已暴露基准不称独立盲测。

主复盘报告和必要台账集中在 [CVS_PHASE1_FEATURE_REVIEW_20261014.md](CVS_PHASE1_FEATURE_REVIEW_20261014.md)，不把Phase2完成作为Phase1前置条件。过去Phase2数据定位只作历史接口参考，不继续开发适配器/注册/capsule构建。

最新已核实：MS12、EPI9、RC15、CF9全部完整终态；各路线原头均值均未84%。源几何真实7/15行，不能据低秩认定塌缩。近期原头与冻结cos/proto、全部负结果和证据层级在主台账中保留。

当前健康run：20261014-phase1-pa-cross-delay-legacycos-clean-manysig-m9-r01，代码1f8814c046a8c68eba8a334e66a0aceb76e491e0。基线3同seed原默认决定复用并评分；6新scratch50、浅层PA加576参数（同容量卷积/因子化跨延迟三阶），分类头原对象，零单独head训练。真实baseline noquery smoke、focusedchecks、唯一独立P0/P1审查、archive传输/remotecompile PASS；6worker实际PID/CWD/argv/GPU/commit匹配，epoch4至5且loggrowth VERIFIED，failure0。此run保持固定预算和默认头单读出，训练后自动clean评分；目前无新性能结果。

下一步最多两个动作：只读真实Sinc checkpoint运动及回收既有表征诊断；继续当前PA闭环。源诊断无梯度/optimizer，不能仅非零gradient就宣称实际更新或迁移；后续候选先写源证据、接线、同容量控制、成本和失败判据。每次实现/config/主报告/必要聚合结果提交push，并独立比较remote OID；大原始产物保持原目录。

源证据已实际返回：9个原初始/终轮Sinc模型，23/24 cutoff与24/24 bandwidth参数未变，kernelrelativeL2≤2.4071e−8，非人工梯度替代结论；frozen_backbone=false。RC历史cos全15行/540CM已回收重计，RESTORE仅＋.0300ppvsBASE/2seed正、组合−.4518pp。下一优先机制改为基于源真实近静止证据的Sinc坐标单因素检查/修复，当前PA固定回收保持第二动作，不追加Phase2。
