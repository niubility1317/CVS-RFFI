# 已完成 R17 精简候选的目标域对照

范围：R05 至 R10 已完成的全部16行，每行同168000×七目标场景、原F50/defaulthead。仅复用完整冻结汇总，0新IQ/checkpoint/模型前向/truth访问。不同run训练目标和可训练模块不同，以下是描述性跨run对照；不称单因素消融，不据此更改运行中R11/R12配方、初始化、预算、顺序或选择性重跑。

原R17完整参数4665073。七场景逐项下降≤1个百分点通过11/16行；已测试通过候选的最少完整参数为1592461，有两种不同权重版本，性能存在场景取舍。low urban 70%尚未由本表任何行达到；其他环境接近clean未自设阈值。所有负结果保留。

|版本|完整参数|较原R17减少|最大下降pp|clean|high|mid|low suburban|high urban|mid urban|low urban|1pp验收|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|[R05/step_norm_task](../20261020-phase1-restoration-r17-compact-target-comparison-s2026092701-m2-r05/report.md)|3118001|33.1629%|0.364286|84.586310|80.954762|79.319643|77.636310|79.989286|73.714881|67.356548|通过|
|[R05/activation_task](../20261020-phase1-restoration-r17-compact-target-comparison-s2026092701-m2-r05/report.md)|3775105|19.0773%|0.320238|84.586905|80.998810|79.272024|77.709524|80.057143|73.675595|67.298214|通过|
|[R06/shared_half](../20261020-phase1-restoration-r17-shared-factor-s2026092701-m2-r06/report.md)|2821345|39.5220%|0.957143|84.586310|80.815476|78.997619|77.223214|79.750595|72.910119|66.259524|通过|
|[R06/shared_quarter](../20261020-phase1-restoration-r17-shared-factor-s2026092701-m2-r06/report.md)|2582905|44.6331%|0.552381|84.586310|80.805952|79.130357|77.492262|79.801190|73.389286|66.873810|通过|
|[R07/shared_three_eighths](../20261020-phase1-restoration-r17-parallel-compact-s2026092701-m4-r07/report.md)|2702125|42.0775%|0.516071|84.586905|80.872024|79.144643|77.480357|79.904762|73.448214|67.154762|通过|
|[R07/shared_eighth](../20261020-phase1-restoration-r17-parallel-compact-s2026092701-m4-r07/report.md)|2463685|47.1887%|0.201190|84.586905|81.285714|79.890476|78.350000|80.234524|73.820238|67.015476|通过|
|[R07/shared_quarter_pointwise](../20261020-phase1-restoration-r17-parallel-compact-s2026092701-m4-r07/report.md)|2471609|47.0189%|0.338690|84.586310|80.980357|79.281548|77.693452|80.036310|73.627976|67.214881|通过|
|[R07/shared_eighth_pointwise](../20261020-phase1-restoration-r17-parallel-compact-s2026092701-m4-r07/report.md)|2352389|49.5744%|0.216071|84.586310|81.102976|79.751786|78.242262|80.146429|73.969643|67.361905|通过|
|[R08/context_half_prior_quarter](../20261020-phase1-restoration-r17-context-compact-s2026092701-m2-r08/report.md)|1822977|60.9229%|15.047024|69.539881|79.432738|77.857738|76.111310|77.217262|67.571429|58.908333|未通过|
|[R08/context_half_prior_eighth_pointwise](../20261020-phase1-restoration-r17-context-compact-s2026092701-m2-r08/report.md)|1592461|65.8642%|16.202381|68.384524|79.393452|77.239286|75.055357|76.647619|65.224405|55.435119|未通过|
|[R09/joint_teacher_quarter](../20261020-phase1-restoration-r17-joint-optimize-s2026092701-m4-r09/report.md)|1822977|60.9229%|0.782738|84.789286|80.686310|79.443452|78.061905|79.673214|73.268452|66.433929|通过|
|[R09/joint_clean_quarter](../20261020-phase1-restoration-r17-joint-optimize-s2026092701-m4-r09/report.md)|1822977|60.9229%|1.254167|85.267857|80.064881|78.713690|77.263690|79.115476|73.297619|67.082143|未通过|
|[R09/joint_teacher_eighth_pointwise](../20261020-phase1-restoration-r17-joint-optimize-s2026092701-m4-r09/report.md)|1592461|65.8642%|1.553571|84.434524|80.585714|79.296429|77.776786|79.446429|72.639286|65.663095|未通过|
|[R09/joint_clean_eighth_pointwise](../20261020-phase1-restoration-r17-joint-optimize-s2026092701-m4-r09/report.md)|1592461|65.8642%|0.109524|84.833333|81.450000|79.872619|78.294643|80.279762|73.789881|67.107143|通过|
|[R10/clean_only_quarter](../20261020-phase1-restoration-r17-clean-corrective-s2026092701-m2-r10/report.md)|1822977|60.9229%|2.004762|85.236310|79.373214|77.928571|76.441071|78.348810|72.808929|66.647619|未通过|
|[R10/clean_only_eighth_pointwise](../20261020-phase1-restoration-r17-clean-corrective-s2026092701-m2-r10/report.md)|1592461|65.8642%|0.797619|84.752976|80.552381|79.192857|77.615476|79.555952|73.713690|67.600595|通过|

逐场景支配仅表示：参数不少于对方，且七场景正确数全部不优于对方，至少一个维度严格较差。不同数据曝光/训练计算、延迟、稳定性、单样本配对显著性不包含在这一描述中，不能据此声称稳定总体优胜或全局最简。尚未被本表其他行逐场景支配的已验收版本：R05/activation_task, R07/shared_three_eighths, R07/shared_eighth, R07/shared_eighth_pointwise, R09/joint_clean_eighth_pointwise, R10/clean_only_eighth_pointwise。各行支配关系与整数正确数保存在JSON/CSV。

同1592461参数下，R09 clean版本相对原R17最大下降0.109524pp；R10直接clean版本low urban更高，但若干其他场景下降更多，最大下降0.797619pp。两版均保留，不能把low urban单项提升写成七场景同时改善。单seed/单F种子/已暴露benchmark，非新独立盲测，也非多seed总体保证。
