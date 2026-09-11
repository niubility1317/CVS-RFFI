# CORE90锚定几何：设计一致性与机制开启复查

审查日期：2026-09-11。受审生产代码：`939f5bf40891db0d5a44ffabe1e1ae8288214b4e`。

**结论：主线数学结构与默认参数基本按设计接入，但不能维持上次“全部实现验收通过”的结论。本轮发现2项P1、2项P2；生产代码未修复，未启动正式实验。**本报告以[设计原文](core90_anchored_geometry_design_source_20260911.md)与[实施计划](core90_anchored_geometry_implementation_plan_20260911.md)为准，重新检查可达路径、参数消费和实际梯度，不把已有测试通过当作完整符合性证明。

## 1.已复现的缺口

### F1/P1：OOF最终专家没有接入独立候选的校准/导出链

代码：[anchored_source.py:91](E:/type10-7/code/snapshots/core90_evidence_20260911_wt/code/cvsrffi/anchored_source.py:91)、[anchored_source.py:210](E:/type10-7/code/snapshots/core90_evidence_20260911_wt/code/cvsrffi/anchored_source.py:210)。对应R19/R31/R39。

`oof`阶段调用`run_expert_oof()`，已经拟合5个4RX模型和1个全source模型，但不接收其返回值、不生成最终专家的`system_state.pt`。`calibrate`的前置阶段白名单仅允许`fit/fuse`，也不接受OOF目录或`all_source/expert.pt`。裸`expert.pt`本身还是ExpertArtifact schema，不能按FrozenAnchoredSystem读取。

因此A2/A3/A4/C_angle/C_angle_keep无法通过现有CLI直接复用已经拟合的全source专家进入校准。沿当前可用入口只能另外执行`fit`或在CLI外手工装配。若5种专家×3seed都另外fit，会从90次基础拟合增至105次，条件性嵌套全部执行时从120次增至135次；这不是计划声明的预算闭环。

最小复现：向阶段校验提供匹配candidate/seed、但stage为`oof`的manifest，仍得到`input stage/candidate/head seed mismatch`。实际`all_source`目录还没有完整的stage manifest；该复现已给它更有利的完整manifest，仍被拒绝。

修复方向：OOF阶段直接把返回的final_expert装配为独立候选系统并保存，明确允许该合法OOF完成状态进入V校准；增加“6次拟合后直接calibrate/export且不再fit”的端到端断言。

### F2/P1：门控训练和部署的实际动作并不严格一致

代码：[anchored_crossfit.py:83](E:/type10-7/code/snapshots/core90_evidence_20260911_wt/code/cvsrffi/anchored_crossfit.py:83)、[anchored_crossfit.py:136](E:/type10-7/code/snapshots/core90_evidence_20260911_wt/code/cvsrffi/anchored_crossfit.py:136)，对照[anchored_pipeline.py:112](E:/type10-7/code/snapshots/core90_evidence_20260911_wt/code/cvsrffi/anchored_pipeline.py:112)。对应R25。

训练/嵌套评价先在FP32上执行`s0.log_softmax()`、`sG.log_softmax()`，之后`realize_actions()`再转FP64。部署先把原logits转FP64再log_softmax。两条路径虽然调用同名动作函数，输入概率已经不同；不可通过后来转double恢复丢失的精度。

已构造有限、正常幅度logits的α=0.5边界样本。一个可复现样本中：训练路径前两类概率为0.4352556052/0.4352556010，选0；部署路径为0.435255597771/0.435255597852，选1。若truth=0，训练utility是0，部署实际utility是−2。该样本位于保护区外，因此现有高间隔保护不能消除此问题。

10000个刻意靠近混合决策边界的合成样本中，3600个发生argmax不一致。**这是数值边界压力测试，不是实际数据上的36%错误率或发生率估计。**它足以推翻“utility标签与部署实际动作严格相同”的技术声明。

修复方向：训练、固定动作审计、嵌套评价和部署统一接受原始logits，由一个入口完成相同精度、相同顺序的概率变换/保护；增加逐动作输出、argmax和utility一致性测试。

### F3/P2：三seed汇总只有函数，没有可完成的CLI调用链

代码：[anchored_source.py:132](E:/type10-7/code/snapshots/core90_evidence_20260911_wt/code/cvsrffi/anchored_source.py:132)、[anchored_reporting.py:58](E:/type10-7/code/snapshots/core90_evidence_20260911_wt/code/cvsrffi/anchored_reporting.py:58)。对应R41/R42。

CLI每次只调用`assess_source_promotion({args.seed:nested_metrics})`。函数要求三个seed齐全，所以每次输出必然是`PENDING_ALL_HEAD_SEEDS`。即使三个独立seed目录都完成，也没有现成入口收集它们、核对配置/数据一致性并形成完整verdict；阶段report也只标记当前目录。

这不会误晋级，反而会一直pending，但与“已实现完整三seed判定链”的表述不符。配置中的三个head seed是允许值清单，不是自动运行/汇总三seed的开关。

修复方向：接入显式汇总入口，读取三个已完成、身份一致的嵌套产物，并保存全部RX/TX/day/seed及A6相对inner选定A5的差值。不需要为汇总重新训练。

### F4/P2：晋级函数可在缺少5个TX分组时返回通过

代码：[anchored_reporting.py:66](E:/type10-7/code/snapshots/core90_evidence_20260911_wt/code/cvsrffi/anchored_reporting.py:66)。对应R42。

完整性判定仅要求`len(tx)>0`、`len(rx)==5`，未核对实际TX/RX集合、唯一性和覆盖。已复现：每个seed仅提供TX0一行，其他5个TX全部缺失，仍返回`SOURCE_DEVELOPMENT_SUPPORTED`。当前内部`nested_reports()`通常生成全类行，所以这不是已证明的现有run误晋级；它是汇总接口对缺失/不完整报告的真实防错缺口。

修复方向：从冻结类别/RX定义核对完整且唯一的分组集合；缺组、重复或非有限值应返回不完整状态，不能进入最弱TX/RX判据。

## 2.机制与参数是否真的开启

|机制/参数|当前代码配置与消费|现有实测能证明什么|
|---|---|---|
|H0冻结、无标签margin|IdentityAnchor冻结并eval原模型，身份aux调用y=None；W/tau从实际CosFace读取|此前真实H0四场景72行三方对照通过、骨干逐tensor不变；本轮复核代码及原始证据|
|低秩G|r=4，ρ=log(2)/2；Q=QR(B)，a=ρtanh(b)；只训练B/b共644参数，cond≤2|真实两步拟合a变为非零；不是只有配置字段|
|CE及LEO配对|clean/LEO各0.5；A2为clean-only，A3/A4及角度对照为配对；每组2physical|实际history包含clean/LEO CE；6300/35batch属于已有技术测试，不是已跑完整训练|
|metric正则|λmetric=0.01，真实加入优化loss|第二步加权梯度范数4.80448×10⁻⁶；初始a=0时第一步正则梯度为0是预期行为|
|keep|A4/C_angle_keep开启；A2/A3/C_angle关闭；λkeep=1/τ²，实际τ=30，因此1/900；γmax=0.1τ=3|第二步加权梯度1.43174×10⁻¹⁰，确实接入但极弱；不能把非零等同有显著保护效果|
|普通角度对照|C_angle只训练W、不加bias；C_angle_keep额外加入keep；不训练M|配置分支和实际loss接线符合计划；正式同预算效果尚未测量|
|优化预算|Adam lr=0.001，weight_decay=0；默认80轮；20/40/80诊断；稳定窗口5轮、CE阈值0.002、M变化0.001|现有真实技术试验仅2步、1个head seed，状态budget_exhausted；不是80轮已完成|
|固定融合与utility|α∈{0,.25,.5,.75,1}；λH=2，utility=rescue−2harm；保护分位数0.9|函数与分支存在，但F2阻止“训练/部署严格同动作”的验收结论|
|A6门控|ridge=0.001，utility_margin=0；source固定动作正净收益后才进入嵌套拟合；11维类别对称特征|条件性入口存在，不能称已在正式source上激活或取得收益；F3尚未闭合完整seed汇总|
|最终校准|V最终输出温度范围0.05–20，目标coverage=0.9，内部尺度固定|校准模块及合成端到端有效；独立训练专家的OOF接入存在F1|
|P1|t/f/pa各160维；rank=4、shrinkage=0.1、观测方差floor=1e−5/ceiling=1；quality_bins=3、min_group_samples=20、consistency_quantile=.95、min_confidence=.5|独立E拟合/低秩边缘化/pattern路径可达，已有合成测试；没有真实完整P1实验效果证据|
|条件性后续模块|类别方向修正、state response、support后验/斜率、联合骨干、自适应预算均false|按计划应关闭，不能将其视为漏开；C_angle对照训练W不等于开启G方向修正|

真实第二步CE梯度范数为0.00294347，keep/CE约4.86×10⁻⁸。它说明保留约束在该短技术试验中几乎没有梯度贡献；是否因大部分H0正确样本已满足截断间隔、是否在更长拟合后开始保护，需要完整history和触发样本统计，不能现在断言有益或失效。

另一个遥测注意点：梯度取每轮`seed+epoch`打乱后的首batch，样本随epoch变化；M变化已记录，Q独立变化未单列。不能把跨epoch梯度变化全部解释成模型参数变化。此项未计入上述四个确定缺陷。

## 3.验证范围与状态修订

本轮实际执行：源码/配置/设计反向审查；[合成边界与接口复现脚本](audit_core90_anchored_recheck_20260911.py)；其[完整输出](audit_core90_anchored_recheck_20260911.json)。命令：`python -X utf8 analysis/audit_core90_anchored_recheck_20260911.py`，使用已验证ssr-gpu解释器，CPU、2线程。未读取真实target、未启动新训练、未访问或修改N607。

此前57项测试通过的历史事实仍成立，但没有覆盖本轮反例；本轮未把那57项重新执行结果冒充新增证明。没有修改生产模块，也没有把发现缺口后的状态继续写成全部通过。

[追踪表](core90_anchored_geometry_traceability.md)修订为：**36项verified、6项implemented但存在缺口（R19/R25/R31/R39/R41/R42）、6项deferred、0项rejected、0项blocked**。当前不满足严格设计/计划全量验收；最高风险是F2的实际动作utility不一致，其次是F1的最终专家交付/预算断链。
