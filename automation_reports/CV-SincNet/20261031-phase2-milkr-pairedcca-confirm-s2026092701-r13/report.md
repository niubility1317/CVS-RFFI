# MILKR 配对 CCA 六接收机确认

预登记。候选R14在首RX1-1通过事前四路均值门槛后，原rank16源投影/核候选/支持留出规则全部冻结，确认其余六RX。没有新候选排序、目标调参或重采样。源码、命令、环境/CWD、source ancestry、路径、owner、GPU/stoprule见experiment.json。

固定P01/MIX01、seed2026092701，RX14-7/2-1/20-1/7-14/7-7/8-8，K1/5/10/20×N2/5/10/20，各20个B/C工作点×四严格S/Q，480新增预测/评分行。每格全部注册类query补集。原版480行复用；r12先导R14/原版各80行保留，独立报告确认六RX和合并七RX，不用先导掩盖确认结果。

沿用真实Tri-SIG-TIME SSL及DURA v1.0冻结checkpoint。源投影参数从r12 source_cca.npz加载，匹配源数据/模型/增广契约及source-only元数据后原数组复制到新root；不重训、不重选rank。目标只当前K标签，Q0只raw编码，Q1只真实G后restored编码，两query路径共享同一support拟合状态。严格S0/S1不混用输入。source smoke真实checkpoint两行通过后即继续；源参数复用边界检查拒绝契约不一致或target-trained元数据，目标投影数组保持一致。R14原候选唯一P0/P1审查已PASS，不重复审查未变算法/runner。

均值验收分别核算确认六RX与合并七RX：两个S各Q0较匹配原版≥5pp、Q1>Q0、Q0旧新macro各≥原版−1pp；逐16KN/RX完整列出，每格+5pp作为更严格口径单独报告。首先导通过并未满足S0每格条件。全480预测固定后独立truth-last；仅在实际统计满足时宣称相应有限范围改善，不声称新盲测、多seed或全部物理池。

源smoke GPU4，六worker分配GPU0/2/4/6，每卡最多两个本任务新增worker，实际显存余量核实，其他owner保持。技术失败保留产物、不原地重启；结果失败不选择性重跑。F/G/默认分类头及源投影新增训练0。
