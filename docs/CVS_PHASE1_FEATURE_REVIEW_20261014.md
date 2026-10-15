<!-- CURRENT_TRAINING_PRIORITY_20261014 -->
固定 SIG_TIME 和默认联合分类头，研发重点保持训练泛化与半监督机制。均值教师完整三seed已 ANALYZED/VERIFIED：FEATURE_MT82.4359%±1.5874pp，匹配SIG82.6806%±1.7906pp；配对−.2446±.2101个百分点，0/3正，净−1233/504000。三seed源U特征损失均下降约18%至21%，但未形成DG收益；保留完整负结果，不扫权重或换头补结论。完整报告 20261015-phase1-feature-mean-teacher-oldgate-cos-e50-clean-manysig-s0203-m6-r01/final_analysis.md。

RAW/TX_DAY_CENTERED骨干Fishr完整三seed已ANALYZED/VERIFIED：RAW82.6647%±1.7402pp、匹配SIG82.6504%±1.7118pp，配对+0.0143±0.1918个百分点，1/3正；COND82.5113%±1.7719pp，对SIG−0.1391±0.2340个百分点，1/3正，对RAW0/3正。源penalty均下降但未转成稳定DG收益；12模型/2016000默认头决策/216CM/936梯度矩阵独立复算，头拟合0。两扩展源决定先于新目标数字回收，无目标反馈改参；完整报告20261015-phase1-backbone-fishr-oldgate-cos-e50-clean-manysig-s0203-m8-r01/final_analysis.md。

同L强图CE/feature源诊断 20261015-phase1-source-feature-ce-gradient-manysig-m6-r01 已ANALYZED/VERIFIED：6模型、4950向量独立复算，头/模型/optimizer更新0。教师对学生错18/18对象同向，教师错学生对16/18反向，但每模型该组仅3至8个seen-L样本；整体符号混合，不支持普遍冲突或实际U因果解释。固定零置信差源诊断 20261015-phase1-source-teacher-confidence-manysig-m6-r01 已ANALYZED/VERIFIED：7200向量独立复算，6/6 V有益代理组保留69.04%至76.25% > 有害40.74%至48.44%，资格PASS；L全局方向仍混合（11/18相关对象cosine增加）。固定置信度门控feature-consistency完整三seed已ANALYZED/VERIFIED：CONF82.540675%±1.760937pp、匹配SIG82.776984%±1.607961pp；配对−0.236310±0.153150个百分点、0/3正、净−1191/504000。源保留feature均值下降17.91%至20.22%、全U下降17.20%至19.49%，但未转成默认头收益；9模型1512000决策/162CM/71928保存向量核实，原任务自然完成、无重跑或另训头。新方差形状900L源端检验已PASS（t1/t2同批CE cos0.646206/0.593616、5/5非零），固定02/03源扩展决定已push，尚未实现/发布；不自动新训练或调权。84%三seed目标仍未达到。

# Phase1 特征提取复盘与完整实验交付

当前主任务只做Phase1：实际joint160/id_feat_joint增强，默认ResidualPhysicalHead6×160/scale30按原损失与骨干正常联合训练，禁止单独头训练/重训/替换或RX诊断分类头。原頭clean三固定seed均值84%是研究目标；当前仍未达到。历史换头结果仅保留为档案，不能用于本轮主指标或新增fit。Phase2/newTX/未知拒识不在本轮开发范围。

## 近期完整实验与配方边界

|实验|配方/新骨干成本|匹配原默认头结果|判断|
|---|---|---|---|
|历史多尺度|33＋17；12scratch50/133200updates|OLD80.5268%，TIME81.6905%，PA79.0032%，JOINT81.2895%|TIME仅2/3正；PA/组合负结果保留，不归因RX捷径或梯度冲突|
|临时重建任务|32＋18；6scratch50/66600updates＋3base复用|LEGACY80.6454%，SAME80.6204%，CROSS80.7494%|原型历史增量3/3正，ordinarycos仅1/3正；不能宣称通用迁移|
|恢复/竞争|33＋17；12scratch50/133200updates＋3base复用|BASE80.5268%，ORD80.3075%，RESTORE80.2532%，COMP80.5663%，组合80.1298%|同容量RESTORE不胜ORD；COMP微小增量无稳定机制证据|
|跨频关系|32＋18；6scratch50/66600updates＋3base复用|LEGACY80.6454%，POWER76.8591%，COMPLEX79.3058%|COMPLEX胜POWER但低于BASE，完整负结果已交付|
|PA跨延迟|32＋18；6scratch50/66600updates＋3base复用|LEGACY80.6454%，CONV80.7738%，CROSS80.8133%|CROSS对同容量CONV仅＋.0395pp/2seed正；0独立head更新|
|当前TIME2×2|33＋17；12scratch50/133200updates＋3base复用|预登记LONG_SILU81.6417%，最高次要LONG_LINEAR81.9819%|完整3seed已结束，主指标差84%2.3583pp；不替换主指标|

33＋17与32＋18不合并。历史RC10轮AdamW ordinarycos：BASE78.5657%，ORD78.3613%，RESTORE78.5956%，COMP78.5095%，组合78.1139%；本对话EPI/CF20轮Adam诊断另列，预算不同不合并。历史source几何真实7/15行L6300，BASE有效rank7.648/7.831/8.185；低rank不等于塌缩，RX方差不能直接当捷径。全15行RC双读出已回收，540个保存CM独立重计；这不是重新预测或重训。其他owner历史原始日志证据层级、所有逐seed/paired及来源在机器台账中保留，没有重启/合并其他工作树。

唯一当前外部基线：OLD_GATE_COS50同33＋17三seed fixedfinal50；不悄悄改纯CE。L6300/U56700隐藏/V27000，训练池标注10%/全source7%，source split392005、RX/day/TX/物理ID保持；NoExplicitCFO及内部补偿保留，无新星地训练。原Torch初始化执行实测历史TIME与新矩阵主RNG不同，因此新训LONG_LINEAR，不按文件名复用。四组每seed真实initial_state全部200tensor相同。

## TIME完整三种子结果

run 20261014-phase1-time-factorial-oldgate-cos-e50-clean-manysig-m15-r01，实际代码 351e35cd0d298a6832c0865628bba28ab5a14cbe。source→fixedfinal50→默认头逐包预测→truth-last独立评分闭环；all15worker自然退出，失败0。唯一下一真实run P0/P1、archive单次SHA/remotecompile/noquery实际checkpoint smoke与PID/CWD/argv/GPU/loggrowth已完成，未加重复审批或hash链。

|条件|clean 均值 ± 样本 SD|Macro-F1|最差 RX 均值|最差 day 均值|
|---|---|---|---|---|
|OLD_GATE_COS50|80.5268% ± 2.3776 pp|80.0024%|67.9708%|76.0556%|
|COMPACT_LINEAR|81.8536% ± 0.8979 pp|81.3736%|74.5347%|79.0429%|
|LONG_LINEAR|81.9819% ± 0.8566 pp|81.5256%|74.7931%|79.4270%|
|COMPACT_SILU|81.5135% ± 2.2298 pp|81.0106%|71.3833%|78.3016%|
|LONG_SILU|81.6417% ± 1.7807 pp|81.1547%|72.4500%|78.0873%|

|固定对比|均值差（pp）|三个 seed 差（pp）|正向 seed|
|---|---|---|---|
|LONG_LINEAR − COMPACT_LINEAR|0.1284|0.5970 / 0.1976 / -0.4095|2/3|
|LONG_SILU − COMPACT_SILU|0.1282|0.1857 / 0.6024 / -0.4036|2/3|
|COMPACT_SILU − COMPACT_LINEAR|-0.3401|0.1982 / -1.8589 / 0.6405|2/3|
|LONG_SILU − LONG_LINEAR|-0.3403|-0.2131 / -1.4542 / 0.6464|1/3|
|COMPACT_LINEAR − OLD_GATE_COS50|1.3268|1.4637 / 2.7679 / -0.2512|2/3|
|LONG_LINEAR − OLD_GATE_COS50|1.4552|2.0607 / 2.9655 / -0.6607|2/3|
|COMPACT_SILU − OLD_GATE_COS50|0.9867|1.6619 / 0.9089 / 0.3893|3/3|
|LONG_SILU − OLD_GATE_COS50|1.1149|1.8476 / 1.5113 / -0.0143|2/3|

长覆盖增量两组均约＋.128pp，仅2/3正；SiLU相对紧凑/长线性−.3401/−.3403pp，正向2/3、1/3。交互−.0002pp，sampleSD.4081pp，未显示明确交互。新四组各237697identity参数，比OLD177025多60672；与OLD的均值差不是覆盖/非线性的纯归因，因为同时多了参数化/投影且初始化执行不同。新四组的配对才是本轮匹配对照。没有用目标分数挑选父权重、改参数、重跑或选最高seed。

独立完整解析133200新step、750epoch，11100成功optimizer/EMA更新/新模型，非有限跳过0，EMA.999/versioned/no startupaverage、33＋17一致。15×168000=2520000默认固定决定，270条overall/RX/TX/day CM、accuracy、Macro-F1由原NPZ和独立truth逐项重计；baseline504000原决定完全一致。全部三seed配对、纠错/新增错误、最差RX/day、完整曲线与资源统计见同runresults.json。新600model-epochs与历史baseline150epochs分别报告；epoch时长合计27274.47秒，非受控GPU速度对照。额外V24pass/648000包，U额外诊断前向0，独立headfit0。

## 真实feature与源机制

12新模型21个TIME tensor初始到final均非零更新，符合实际优化器/EMA路径。固定L720/模型（每TX/RX/day按ID前8个）真实initial/final z变化、原id_feat_joint与independent feature-only导出逐元素一致、临时复制W置零不改变z、finalθ/全部buffers未被诊断修改均VERIFIED；删除960个classW元素后导出236737θ参数，保留base_norm/pa_norm/gain/reference/inputpolicy。这里只证明学到了不同且可正确交接的真实特征，不证明可迁移性增益或新TX能力。

既有TIME3seed实际preGN等效核rank最多4，doublefold误差≤3.34e−15，外±4能量3.4%至4.8%；源branch撤除改变joint160/defaultmargin。GN数据相关，整块不是线性卷积；核rank≠feature rank，推理撤除≠独立训练消融。此源测量在新目标测试前固定并返回；未凭target下降归因为RX/梯度冲突。

新TIME末CE0.2264至0.2486，V98.39%至98.60%，后段V趋平台但CE有波动；50轮预算≠证明充分收敛。source33/34额外RX健康评估恢复Python/NumPy/Torch/loader RNG与模式，不进入梯度或选模。每新模型U963900包次，selected约1.65%至1.68%；OLD约1.57%至1.58%。各RX/day与预测类别计数、强student/弱EMA agreement完整；U真TX未打开，包次不是独立物理coverage，不能叫U准确率。

真实Sinc9模型（32＋18 LEGACY3、33＋17 OLD/TIME各3）source角色/祖先/预算核实：每个学生23/24cutoff与24/24band参数未变，kernelrelativeL2≤2.4071e−8。保存的全部183或192参数排序/moment形状匹配原AdamW收集函数，Sinc各24参数step11100，47/48非零firstmoments，排除漏入optimizer。最后保存LR/moments的adaptive项约2.26e−7至6.30e−7Hz，每48通道均小于半ULP；仅末端代数项，不冒充完整尝试步长历史。EMA cutoff/band最大变动5.625/21.875Hz，kernelrelativeL2约3.3578e−6。保持实际初始Sinc学生不变，CPU float32回放原.999/versioned/11100次，全部9真实EMA核逐元素复现；7模型48params完全一致，另2仅首cutoff差1ULP，核仍一致。回放是数值重建，真实保存态是训练证据；不能把EMA漂移算作学到滤波器，也不证明Sinc是泛化瓶颈。

## 三个可执行方向与最多两个下一动作

1. Sinc坐标单因素优先：源9模型的moment/ULP与实际核证据支持优化频率坐标。仅改真实Sinc参数坐标，RAW/native与centered-scaled保持同初始滤波器/初始joint160、48trainable参数、原time/frequency/PA/reference与默认头，旧门控/cos同50从零控制；同步检查cache、optimizer与EMA。失败判据必须在发布前固定，真实核/feature未改变或源训练失效不宣称有效；不拿target最高TIME作父权重。
2. TIME投影/基底归因：源实际等效核rank4与各支贡献证明路径活跃，但投影、容量仍未分开。可先用既有冻结weights做无训练projection/source撤除，若source支持再设同路径容量控制；不扫kernel/dilation、不叠加PA、不新增head。与本轮coverage×SiLU因素不同，预算≤50并预登记，推理撤除不冒充重训。
3. 旧门控覆盖审计：source真实约1.6%接收包次，可能限制U利用。先在合法有标签L/V代理只用既有defaultteacher/student诊断coverage/正确率，固定批次/视图/门控，不解封U、不拟合分类器、不反向V。只有source证据支持时才另外预登记一个SSL因素，与原旧门控＋cos控制同预算；不按target错误分组调门控。

当前最多两项：以完整源门控诊断为依据检验物理邻包成对随机采样；仅在预先固定的合法源条件通过后发布独立同路径scratch50训练矩阵。TIME/Sinc完整结果已回收，其他方向排队，不无限扫描直到84%。主目标仍active，未宣称达成。

完整当前 [同轮分析](../automation_reports/CV-SincNet/20261014-phase1-time-factorial-oldgate-cos-e50-clean-manysig-m15-r01/final_analysis.md) 与 [机器台账](research_data/CVS_PHASE1_FEATURE_REVIEW_20261014.json)。历史当前运行身份、代码/来源/证据层级在台账中；大IQ、checkpoint、原NPZ/日志留原目录。默认head及clean范围来自用户最新明确要求；历史换头/LEO结果不删改，当前不借clean声明部署。

## Sinc 完整三种子与真实特征

固定run 20261014-phase1-sinc-coordinate-oldgate-cos-e50-clean-manysig-m9-r01 全9row完成，代码 0f81373c25ba69a6cb9deec5c7ca652880fd62af。独立核验66600新step、450epoch、162CM/metric和1512000默认决策VERIFIED，全worker终止、无失败。CENTER_25K主候选80.7637%±2.1962pp，OLD80.5268%±2.3776pp，RAW80.5212%±2.2552pp；CENTER−OLD+0.2369pp、CENTER−RAW+0.2425pp，均3/3正差。RAW−OLD−0.0056pp负结果保留。主目标84%差3.2363pp，仍未达到。3模型不声称显著，也不改变主候选。

实际6模型optimizer全183参数与保存moments/step11100匹配；CENTER全部48坐标更新、真实核relativeL2=0.003487至0.004398，EMA接近学生；RAW核0/0/1.17e−8，EMA漂移约3.36e−6不能当作学到滤波器。固定L720的joint160feature-only导出与原前向逐元素一致，W置零不影响z，实际fθ176065保留全部norm/gain/reference。CENTER仅在临时final副本恢复初始Sinc使源z平均L2变化0.080334/0.043433/0.080260；不是独立重训因果DG消融，也不证明新TX。默认联合头正常训练，单独头拟合0。

新6骨干300model-epochs/66600updates，历史对照150model-epochs单列复用，额外V12次/324000包，U额外前向0。全部source曲线和U RX/day计数完整，963900U/模型且真值隐藏；接收覆盖约1.57%。已修正reader对历史epoch合计再次乘222的断言错误，无重跑实验。完整[同轮分析](../automation_reports/CV-SincNet/20261014-phase1-sinc-coordinate-oldgate-cos-e50-clean-manysig-m9-r01/final_analysis.md)、[结果](../automation_reports/CV-SincNet/20261014-phase1-sinc-coordinate-oldgate-cos-e50-clean-manysig-m9-r01/results.json)、[真实特征](../automation_reports/CV-SincNet/20261014-phase1-sinc-coordinate-oldgate-cos-e50-clean-manysig-m9-r01/source_features.json)。

## 源门控诊断与下一候选

源门控诊断run 20261014-phase1-source-legacy-gate-audit-manysig-m3-r01 完整3模型完成，实际代码 0f514fa1ba4302bc7e5e25d0ad41337881acce7f。L/V明确合法标签仅在mask之后作观察，无U/target/头拟合/优化器/模型状态更新；CPU104.59秒/199800包次/360条聚合计数。V完整旧门控覆盖3.4099%，本次接收923/913/926均无错；confidence+strong覆盖92.2815%，precision99.8221%，错误54/41/38。分RX/day/source类计数与比率重新核对。uniformB256/window2的V邻包几何期望3.6978%，实测3.7630%；U56700/cell630只用协议常数算期望1.7807%，不是U正确率。下一优先独立假设为物理邻包成对随机采样，保留原门控；尚未发布训练，不据目标分数选择。完整 [源同轮分析](../automation_reports/CV-SincNet/20261014-phase1-source-legacy-gate-audit-manysig-m3-r01/source_gate_analysis.md) 与 [计数结果](../automation_reports/CV-SincNet/20261014-phase1-source-legacy-gate-audit-manysig-m3-r01/results.json)。


## 物理邻包成对随机采样源反证完成

run 20261014-phase1-source-neighbor-pair-gate-manysig-m3-r01 全3模型source-only完成，实际执行7bde2923fae013495c3be1038504eb59d6e44602。旧门控全部保留，L/V每sampler所有包恰一次、pair内同允许RX/day/eq且sig/basegap1–2。360条RANDOM旧统计精确复现，720条总计数/比率/分组加和独立核验。V FULL覆盖3.4099%→89.9617%，PAIRS正确率99.9959%，接收24390/24069/24410，错误3/0/0；L覆盖13.3598%→92.9312%，错误0/0/0。全3通过固定coverage≥.5/precision≥.995/drop≤.0025源条件。0U/target/头拟合/参数更新、CPU215.21秒/399600包次，未访问U真值。支持独立采样训练假设，不证明U准确率或DG；不是丢弃temporal gate。下一动作只有匹配RANDOM/PAIRS的独立scratch50旧门控cos训练，与另一树单基time control路线不同，不叠加Sinc/TIME、不用target调参。完整[同轮分析](../automation_reports/CV-SincNet/20261014-phase1-source-neighbor-pair-gate-manysig-m3-r01/final_analysis.md)。


## 独立采样训练r02已启动

run 20261014-phase1-neighbor-pair-oldgate-cos-e50-clean-manysig-m9-r02，实际代码c5e67c46f2a04ec33ff3235d392037f38c000eff。r01仅合成预检NumPy dtype互操作失败，0真实训练/预测，失败保留；显式Python整数列表/torch.long修复在原环境PASS，唯一针对性复核PASS，r02新不可变root，配方/矩阵/阈值不变。首seed RANDOM/PAIR两个scratch50模型GPU2/4，独立启动及日志增长核验通过，均完成8轮sourceLCE约0.323。label33阶段U迭代/前向0，target训练0；后17轮才实施U随机vs物理pair对照。旧门控/EMA/loss/defaultcos头及全部网络不改，单独头拟合0，无Sinc/TIME叠加，与另一树单基time路线不同。50/11100终轮固定，首seed两LCE≤.5且PAIR17U接收率≥.25才放行后4models。未有新候选完整target结果，84%目标仍未达成。完整[预登记与启动证据](../automation_reports/CV-SincNet/20261014-phase1-neighbor-pair-oldgate-cos-e50-clean-manysig-m9-r02/report.md)。


## 邻包采样首种子固定源训练与真实fθ

20261014-phase1-neighbor-pair-oldgate-cos-e50-clean-manysig-m9-r02 首seed两新模型完整50/11100，sourceLCE0.289553/0.286946；22200step完整解析、EMA成功预算、原生optimizer183参数与moment形状核验。RANDOM U接收15226/963900（1.5796%），PAIR867095/963900（89.9569%），U真值隐藏且每17活跃epoch全池一次，label33不取U。固定源反证通过，其余4新模型自动运行，目标分数不参与放行。实际同seed初始全部states精确相同，L720 feature-only=原joint160、W独立、θ/buffers不变；终轮PAIR−RANDOM源z平均L2=2.446475、cos0.978069，fθ174804标量分量不同。此证明训练改变真实部署特征，不证明新TX或全部目标收益，单独头拟合0。完整[首seed源证据](../automation_reports/CV-SincNet/20261014-phase1-neighbor-pair-oldgate-cos-e50-clean-manysig-m9-r02/first_seed_source_evidence.json)；新候选完整3seed target结论仍待。


## 邻包采样完整三种子结果

20261014-phase1-neighbor-pair-oldgate-cos-e50-clean-manysig-m9-r02 全 9 行完成，实际代码 c5e67c46f2a04ec33ff3235d392037f38c000eff。独立核验 66600 新 step、450 epoch、1512000 个默认决策、162 条 CM/metric 和全部 6 模型实际源特征，VERIFIED，无缺项或健康任务变更。主候选 PAIR 80.7208% ± 2.1577 pp，旧门控＋cos 80.5268% ± 2.3776 pp，匹配 RANDOM 80.6224% ± 2.1857 pp；PAIR−OLD +0.1940 pp，PAIR−RANDOM +0.0984 pp，均 3/3 正差，RANDOM−OLD 仅 1/3 正差。84% 尚差 3.2792 pp，目标保持 active。PAIR 对 RANDOM 的最差 RX/day 均值略低，负结果完整保留，不把平均正差称普遍稳定提升。

17 轮 U 接收率 RANDOM 1.5727%→PAIR 89.3991%，真值隐藏，每活跃轮全池一次、label33 零 U 读取。源 RX3/8 的接收率与强弱一致性低于其他源 RX；排序改变几乎没有改变强弱一致性。6 模型实际 fθ176065 导出与原 joint160 完全一致、W 独立、分析零参数/buffer/头更新；同 seed 初始全状态一致，终轮源 z 的 PAIR−RANDOM L2 为 2.446475/1.793331/2.325156。只证明真实特征改变与接口正确，不证明新 TX，也不能把原头收益全归因于骨干。GPU 数值并非逐位确定，label33 前缀小差及 3 种子的证据边界如实披露。

成本为 6 新骨干 300 model-epochs/66600 updates，历史对照 150 epochs 独立复用；额外 V12/324000，额外 U 前向 0、单独分类头拟合 0。本轮没有叠加 Sinc/TIME，保留 rawHz 原 Sinc，学生实际核变化仍在浮点极小量级。r01 合成 dtype 失败保留且零真实训练；r02 无失败/重跑。模板中的 Sinc 描述已更正，实际冻结 design 和源放行规则一直为本轮邻包方案。完整 [同轮分析](../automation_reports/CV-SincNet/20261014-phase1-neighbor-pair-oldgate-cos-e50-clean-manysig-m9-r02/final_analysis.md)、[结果与完整核验](../automation_reports/CV-SincNet/20261014-phase1-neighbor-pair-oldgate-cos-e50-clean-manysig-m9-r02/results.json)、[真实特征](../automation_reports/CV-SincNet/20261014-phase1-neighbor-pair-oldgate-cos-e50-clean-manysig-m9-r02/source_features.json)。

下一步最多两项：先在合法 L/V 检查默认损失对浅层、中层及 joint160 的实际梯度与尺度，不更新模型、不打开 U 真值、不拟合头；只有源证据支持后才预登记一个独立特征机制与匹配旧门控＋cos 控制。该依据来自代码和源 RX3/8 证据，不按目标分数扩大 λ、修改门控或挑父权重；不无限扫描直到 84%。


## 源损失信号与首种子梯度诊断

完整6模型66600条source step记录已用代码d8e4514c42c1931685aa0b04176cf0960bf8c16a解析：PAIR加权U CE/L CE的每步比值均值为1.6065%/1.5691%/1.6617%，RANDOM为1.4163%/1.4200%/1.5078%；PAIR零接收step0，RANDOM460/475/475（各3774 pseudo step）。原生CE对已接收样本取mean，覆盖率不是loss乘数；entropy residual与逐轮/分布完整保留。标量比例不代表梯度比例，无IQ/权重tensor/真值读取或更新，完整[源信号](../automation_reports/CV-SincNet/20261014-phase1-neighbor-pair-oldgate-cos-e50-clean-manysig-m9-r02/source_signal.json)。

固定首seed三组actual final50/ownEMA的L_s6300源导数诊断20261014-phase1-source-feature-gradient-manysig-s01-m3-r01已实现、LOCAL_VERIFIED，尚未发布。默认头保留，fθ176065仅排除class W960，浅中层/time/freq/PA与joint160接线已本地PASS，autograd.grad不累积.grad、不更新模型；合法L梯度/观察标签在mask后进入，U/target无读取，V缓存materialize后丢弃、无前向或求导。新训练0/头拟合0/预计56700 source包次，eval无dropout的导数不冒充原训练梯度历史。一独立P0/P1入口问题已修复，唯一targeted复核PASS，完整[预登记](../automation_reports/CV-SincNet/20261014-phase1-source-feature-gradient-manysig-s01-m3-r01/report.md)。


## 首种子源特征导数已完成

20261014-phase1-source-feature-gradient-manysig-s01-m3-r01 全三组完成，实际代码 d4c1e9280854a3bfc988626f98224e737347260b；75 批/825 参数路径/4125 激活RX记录/56700 包次独立核对 VERIFIED。只有固定首模型 seed2026092701，三组不是三seed结果。默认头未拟合或更新、fθ参数/buffers逐元素不变、.grad未累积，V仅cache materialize后丢弃、无前向/求导，U/target读取0；新训练0，来源累计仍各50轮。所有11路径有源导数，U/L参数范数比中位数1.44%至2.51%，time_middle cosine0.0490/−0.1091/0.0680及负批次11/13/11，不能当原训练冲突或DG因果证据。PAIR joint160的RX3逐样本L导数约RX6的4.19倍，RX8约2.78倍；只是同路径源敏感度差异。

下一轮最多两动作：检查t1/t2源统计和实际身份保护调用，准备独立源L同TX跨RX MixStyle；有源支持后再固定单seed匹配控制、scratch≤50、默认头/旧门控cos和自动clean/truth-last。不叠加PAIRS/Sinc/TIME、不扩大λ追目标分。Fishr尚缺本网络分域逐样本梯度方差与明确fθ对象；不把norm/cosine当作Fishr测量。84%目标未变、未达成。本轮实际 [完整分析](../automation_reports/CV-SincNet/20261014-phase1-source-feature-gradient-manysig-s01-m3-r01/final_analysis.md)、[聚合及逐批证据](../automation_reports/CV-SincNet/20261014-phase1-source-feature-gradient-manysig-s01-m3-r01/results.json)。


## 浅中层源风格诊断完成

run：20261014-phase1-source-style-profile-manysig-s01-m1-r01；实际执行提交：d594ac63a95bf5c654d10dbc9e9a82608fe447b3。首种子 2026092701，固定旧门控＋cos final50/11100 来源；没有新训练、目标预测或分类头拟合。


同 TX/day 的 RX 统计差占中心平方和 t1 32.66%、t2 28.21%；固定身份保护干预实际进入 joint160（平均 L2 1.482290），默认头修正11、破坏7，无模型或头更新。这是源机制依据，不是训练/迁移收益。下一项只做匹配单seed MixStyle scratch50，不叠加结构，固定 t1/t2 p=.5 alpha=.1 strength=1；完整 [分析](../automation_reports/CV-SincNet/20261014-phase1-source-style-profile-manysig-s01-m1-r01/final_analysis.md)。用户新贴时序2×2推荐已完成，不重复。


## 身份保护 MixStyle 首种子训练已实现

run 20261014-phase1-identity-mixstyle-oldgate-cos-e50-clean-manysig-s01-m3-r01，2新骨干各scratch50/11100，33 label+17 pseudo，加1历史首seed旧门控cos对照。仅主L的t1/t2同TX/day跨RX统计混合，p=.5 alpha=.1 strength=1，无新参数/额外训练前向。默认头正常联合训练、无单独拟合；U/teacher/eval不混合。真实adapter梯度、数值更新、同seed初态、W与z边界、本地依赖和登记检查PASS；唯一P0/P1审查PASS，当前IMPLEMENTED_REVIEWED；尚未启动或有新目标结果。完整[预登记](../automation_reports/CV-SincNet/20261014-phase1-identity-mixstyle-oldgate-cos-e50-clean-manysig-s01-m3-r01/report.md)。


身份保护MixStyle 20261014-phase1-identity-mixstyle-oldgate-cos-e50-clean-manysig-s01-m3-r01 已发布，实际代码 b2a4146b7571d8713ddbd98cd096fb98b2b4c00c。remotecompile和real checkpoint no-query smoke PASS，GPU1/2两个owned进程/配置及日志增长VERIFIED，旧基线行已复用完成评分；新候选尚无target结果，固定50预算不变。完整[发布证据](../automation_reports/CV-SincNet/20261014-phase1-identity-mixstyle-oldgate-cos-e50-clean-manysig-s01-m3-r01/launch_evidence.json)。


## 身份保护 MixStyle 首种子完整结果

20261014-phase1-identity-mixstyle-oldgate-cos-e50-clean-manysig-s01-m3-r01 已全部完成，训练代码 b2a4146b7571d8713ddbd98cd096fb98b2b4c00c。全量22200新step/150epoch记录/504000默认决策/54CM及真实fθ重算VERIFIED，无缺项。旧门控cos80.6685%、matched80.8500%、MixStyle80.6095%；候选−matched−0.2405pp、−旧基线−0.0589pp，最差RX66.9833%较control70.9083%下降3.9250pp。负结果保留，一seed不定义SD，不冒充三seed84%。

实际t1/t2各11100主L调用，48.6239%/48.9502%包次改变；θ175127分量不同、固定L720 joint160 L2=6.010110、cos=.883807，导出=W独立且θ/buffers观察不变。新参数/额外训练forward/单独头拟合均0，保留NoExplicitCFO及旧门控EMA。源RX条件比例下降而绝对RX平方和上升，不能把比例当作删除RX/因果不变性证据。

源科学反证条件PASS；剩余预定义seed02/03的同参数同预算scratch验证决定已在读取target指标前提交 9e79dc413967aee78fa87e5d0f6c7408c44e4efc。不会按目标负差改p/alpha/层位/父权重或选择性重跑/停止；首seed不重训，下一动作只有登记并运行剩余4模型再合并真实三seed。完整[分析](../automation_reports/CV-SincNet/20261014-phase1-identity-mixstyle-oldgate-cos-e50-clean-manysig-s01-m3-r01/final_analysis.md)与[全量结果](../automation_reports/CV-SincNet/20261014-phase1-identity-mixstyle-oldgate-cos-e50-clean-manysig-s01-m3-r01/results.json)。


## MixStyle 固定剩余两种子已实现

20261014-phase1-identity-mixstyle-oldgate-cos-e50-clean-manysig-s0203-m6-r01 按source-only PASS及9e79dc413决定实施：seed02/03四个scratch50/11100，33+17，原p=.5/alpha=.1/strength=1、t1/t2、旧门控EMAcos和defaulthead正常联合训练均保持。2 个历史基线只按同seed固定final50复用；首seed不重训。只给已审core引入明确矩阵参数、复用源训练和默认预测pipeline，不修改算子。新增直接执行差异唯一P0/P1审查及真实native构建/paired初态/拒绝foreignseed/三seed统计/登记依赖PASS，当前IMPLEMENTED_REVIEWED，尚未发布。最终严格合并原3row+新6row、01/02/03均值/SD及配对差，保留首seed负结果，不按target调参。完整[预登记](../automation_reports/CV-SincNet/20261014-phase1-identity-mixstyle-oldgate-cos-e50-clean-manysig-s0203-m6-r01/report.md)。


MixStyle 剩余两种子现已发布并运行：实际代码 `2b8371027c11cab4b80c17a8c2d0797fac00524a`，四个从零训练模型在 GPU 1 至 GPU 4 运行，已完成 12/9/11/11 个 epoch；进程、CWD、命令、配置和日志增长 VERIFIED，无训练失败。两个历史基线行已复用；全部固定 50 轮，默认分类头正常联合训练，未单独拟合头。归档传输、远端编译和真实同种子旧 checkpoint 无查询 smoke 均 PASS。完整三种子指标尚未产生，首种子负结果保持；详见 [发布证据](../automation_reports/CV-SincNet/20261014-phase1-identity-mixstyle-oldgate-cos-e50-clean-manysig-s0203-m6-r01/launch_evidence.json)。


## 浅中层源梯度方差诊断

固定 source-only run 20261014-phase1-source-gradient-variance-manysig-s01-m1-r01 已实现，本地平衡采样/无偏方差/6协议负例/真实native无查询求导与CPU依赖检查 PASS。一个固定旧基线seed2026092701 final50，取 L_s180（每TX/day/RX2），对实际t1/t2 pointwise权重10368坐标求逐包默认CE导数，按5RX各36包统计 raw 与 TX/day均值去除后的方差。模型eval、参数/buffers不变、无新训练/头拟合/辅助classifier/目标读取；V cache物化后丢弃，无V前向。此为Fishr前的缺失源机制测量，未启用Fishr损失，不宣称有效。唯一新增直接P0/P1审查发现release执行提交读取路径错误，已改为既有release_commit.txt；尚未发布，当前LOCAL_VERIFIED。完整[预登记](../automation_reports/CV-SincNet/20261014-phase1-source-gradient-variance-manysig-s01-m1-r01/report.md)。

源梯度方差新增唯一直接审查现已 PASS，P1完成记录路径修复并针对性确认通过。用户新增明确要求：以SIG_TIME为基础综合MEAN_VAR/COV。将核实原配方，以同参数多SIG_TIME控制区分增容与统计信息；不读取或挑选目标最高权重继续训练，不重复加入SIG，不热改已发布任务。

源梯度方差已完成，执行c0f068f54，源180包/CPU1.8813秒，保存梯度独立重算VERIFIED、所有参数buffers不变。t1/t2归一化RX方差差异raw2.455803/1.938690，去TX/day共同均值后1.714985/1.373195；一模型seed描述统计，不证明Fishr有效或因果RX信息删除，无头拟合/更新或目标读取。当前按用户要求优先SIG_TIME＋MEAN_VAR＋COV组合，不将此诊断自动堆入。完整[源分析](../automation_reports/CV-SincNet/20261014-phase1-source-gradient-variance-manysig-s01-m1-r01/final_analysis.md)。


## 用户指定 SIG_TIME 统计组合

当前优先 run 20261014-phase1-sigtime-statistics-oldgate-cos-e50-clean-manysig-s01-m4-r01：单SIG_TIME、三SIG_TIME同参数控制、SIG_TIME＋MEAN_VAR＋COV独立局部投影组合，首seed2026092701三个新scratch50/11100、33+17，以及旧门控cos历史只读对照。后两个pool各55968参数，单pool18976；组合实际进入joint160，默认头正常联合训练，所有另训身份/RX/day/channel探针关闭。原单SIG_TIME初始函数逐元素等价，本地真实网络/梯度/数值更新/EMA/feature-only/W独立/发布依赖均PASS。独立完整审查定位2项P1（私有CPU初始化重置CUDA随机流、未接入的主L计数闭合），已定点修复，实际CPU/CUDA随机流保持与真实主L/U/eval/deepcopy合成检查PASS；正在完成原两项定点确认，不追加全面review。最终自动clean prediction与独立truthlast；单seed不替代完整3seed84%目标。完整[预登记](../automation_reports/CV-SincNet/20261014-phase1-sigtime-statistics-oldgate-cos-e50-clean-manysig-s01-m4-r01/report.md)。

SIG_TIME统计组合独立完整审查及仅两项P1修复定点确认均已 PASS，可以发布。旧MixStyle source入口收尾拟合了RX/day nuisance探针，之前泛称零拟合不准确；默认TX头没有单独重训/替换，源探针未用于目标预测，但不符合本轮禁止另训诊断头要求。历史产物保留并将在该run完整报告中补明；新SIG_TIME入口scoped禁用所有这类拟合，不再承袭该旧闭包行为。

SIG_TIME综合首seed矩阵已实际运行，代码eda57144d；真实CP noquery smoke/remotecompile/单archive比对PASS，GPU4/5/3三个owned模型首快照2/2/2 epoch，原生独立拟合probe禁用配置与日志增长VERIFIED，无failure。固定scratch50/11100、默认头联合训练。当前没有新target性能结果，84%仍未达；详见[发布证据](../automation_reports/CV-SincNet/20261014-phase1-sigtime-statistics-oldgate-cos-e50-clean-manysig-s01-m4-r01/launch_evidence.json)。


## MixStyle 三种子结果已完整回收（2026-10-14）

默认头均值：旧门控+cos 80.5268%，同路径对照 80.7337%，身份保护 MixStyle 80.7246%。候选相对同路径对照 −0.0091 ± 1.3001 个百分点，正差 1/3；最差 RX 均值降低 1.0056 个百分点。全 9 行 1512000 次默认预测与 162 组指标独立复算，6 个新模型 300 model-epochs / 66600 更新，完整结果见 [三种子报告](../automation_reports/CV-SincNet/20261014-phase1-identity-mixstyle-oldgate-cos-e50-clean-manysig-s0203-m6-r01/final_analysis.md)。历史 RX/day 探针拟合事实已补充到首 seed 及三种子报告；不能再表述为所有拟合均为零。当前 SIG_TIME 统计组合参数及源资格规则已在读取这些新目标均值前冻结并启动。


## SIG_TIME 组合剩余种子已准备（2026-10-14）

固定02/03 × 单 SIG_TIME、三路 SIG_TIME 与统计组合，六个新 scratch50，加两个历史 baseline。实现 a722e2593947ca804e2534ac75e4c081029d022c；真实模型上下文、初态、源资格拒绝及逐 seed 分组检查和唯一新增执行差异审查 PASS。当前仅登记准备，必须等待首 seed 既定源资格 PASS 和原四行闭环，不读目标分数决定发布。完整配置见 [条件式预登记](../automation_reports/CV-SincNet/20261014-phase1-sigtime-statistics-oldgate-cos-e50-clean-manysig-s0203-m8-r01/report.md)。首种子远端不可变运行保持。


## 用户报告路线一已实现（2026-10-14）

下一核心假设仅比较同6D路径的 μ/variance/offdiagCov 与 μ/variance/skewArea，27→28同lift、同初始化/容量和默认头，t2取点不变。三真实模型梯度/更新/export/W独立及算子检查 PASS，唯一直接P0/P1和定向修复确认PASS。当前只是已登记，尚无新训练成绩。参见 [落实记录](../analysis/moment_area_traceability_20261014.md) 与 [预登记](../automation_reports/CV-SincNet/20261014-phase1-moment-area-oldgate-cos-e50-clean-manysig-s01-m4-r01/report.md)。其他活动分支最新 de7f179f3 为 receiver tangent；本路线不同。


## SIG_TIME 统计组合三种子完成（2026-10-14）

组合82.4972±0.9973%，本轮单SIG_TIME82.6655±1.5986%，同参数X383.1855±1.2176%。组合−单分支−0.1683±0.6057个百分点（1/3正差），组合−X3−0.6883±0.2437个百分点（0/3）。12行2016000默认决策、216组CM及99900新step/600epoch已独立复算；9新模型实际无拟合探针，W独立导出与新参数更新通过。组合未达84%，无重训/换seed/按目标调参。原sequence50的83.0393%因随机流条件不同保留独立记录，不混合矩阵。详见 [完整三种子报告](../automation_reports/CV-SincNet/20261014-phase1-sigtime-statistics-oldgate-cos-e50-clean-manysig-s0203-m8-r01/final_analysis.md)。下一轮报告路线一在读取此完整均值前已冻结发布，保持原计划。


同投影 shape 首种子已闭环：SIG_TIME 83.7661%、COV_SHAPE 83.7351%、AREA_SHAPE 82.0667%。AREA−COV −1.6685 个百分点，源模块更新和导出通过不代表性能提高。独立 33300 steps／672000 predictions／72 metrics 复算 VERIFIED，三模型 native 拟合探针均关闭。20261014-phase1-moment-area-oldgate-cos-e50-clean-manysig-s0203-m8-r01 已在首种子目标分数读取前根据源 PASS 原样发布，六新模型运行。执行提交 8bd34db31787cd80c4ad53ee7327ff6a55daee72；详细正负结果见对应 final_analysis.md，保持 84% 未达。


报告路线三源诊断闭环：固定 first SIG_TIME 权重和 L720，8次CPU末端Dropout.1独立回放，63分层统计，0训练/头拟合/目标forward。feature-feature area噪声/clean RMS31.9940%，time area21.7255%，COV shape12.7249%，实际残差160为25.2901%；保存数组重算、eval pre/post一致、状态不变均 VERIFIED。固定分母和绝对RMS完整保留；不是完整训练流或性能因果效果。下一步单独固定SIG_TIME描述子比较t2与t2.pool，不叠加描述子替换/层级/U权重，依据此源证据与用户报告。详见 20261014-phase1-source-dropout-descriptor-manysig-s01-m1-r01/final_analysis.md。


路线三已实际发布 20261014-phase1-read-position-oldgate-cos-e50-clean-manysig-s01-m3-r01：两新scratch50/11100，SIG_TIME post 与 SIG_TIME_PRE pre，只改descriptor hook t2→t2.pool，主干Dropout.1/公式/18976pool参数/default联合头保持。执行 b4d55dba50c8601017604a2a0eb0d037f8d835e8；固定源诊断依据 20261014-phase1-source-dropout-descriptor-manysig-s01-m1-r01，不使用目标误差调参。两个resolved实际取点正确且native fitted probes关闭，默认clean/truthlast自动串联，当前source训练中，无性能结论。


取点02/03条件式扩展已实现登记 20261014-phase1-read-position-oldgate-cos-e50-clean-manysig-s0203-m6-r01，4新scratch50+2历史对照；首seed源资格未完成时不发布，不读取其目标分数控制扩展。实际4模型上下文/同初态/初始eval同/源资格负例/两种子配对及108/162指标汇总检查通过，复用该候选已有唯一核心审查。


同投影shape路线完整三seed闭环：SIG_TIME82.7931%±1.6294pp、COV_SHAPE82.6188%±1.9129pp、AREA_SHAPE82.0833%±1.6870pp。AREA−COV −0.5355pp±0.9818pp、AREA−SIG −0.7097pp±0.9055pp，均1/3正差；primary距84还差1.9167pp。9新/3历史、99900新增steps/600epoch记录/2016000默认pred/216组指标独立复算VERIFIED；9模型实际pool更新/精确导出/W独立/零拟合收尾全部通过，但无性能强化证据。详见 20261014-phase1-moment-area-oldgate-cos-e50-clean-manysig-s0203-m8-r01/final_analysis.md 与 results.json/results.png。取点实验和条件式02/03仍依原源规则不变。


取点首seed完整默认头闭环：POST83.7488%、PRE83.3595%（−.3893pp），PRE修正2343/引入2997/净−654；源LCE .248172/.238117、初始L720功能完全同、最终zL2 2.131700、导出/W独立/零额外拟合通过，22200steps/504000pred/54metrics复算VERIFIED。20261014-phase1-read-position-oldgate-cos-e50-clean-manysig-s0203-m6-r01在读取新首seedtarget前按SOURCE PASS决定发布，4新模型已核实CWD/PID/argv/GPU/实际t2或t2.pool/关闭fits/日志增长；执行038ec31397797e4631a969d0da8167de2ba861c3。单seed为负不改已冻结方法和种子，完整3seed仍pending，goal84=false。


报告路线二真实源几何已核实：20261014-phase1-source-hierarchy-geometry-manysig-s01-m1-r01，L720、3边界/global32、135统计、0训练/拟合/目标forward。连续组合direct64误差8.882e-16，native32为7.153e-7；省略cross-area的feature/time相对RMS37.0827%/87.0831%。下一步独立检验层级网络，单列global-time局部和同容量大窗统计控制，源差异不作性能结论。详见对应final_analysis.md/results.json。


取点完整三seed已回收：PRE82.5599%±1.5656pp、POST82.6909%±1.6836pp，配对−.1310±.2239pp、1/3正差，修正/引入错误净−660。全部9row/6新3历史、450epoch记录/66600新增steps/1512000默认pred/162指标复算VERIFIED，6新模型零额外拟合、真实pool更新/精确导出/W独立，原seed未重训。新机制未强化SIG_TIME，84未达；报告路线二已具备源几何证据，后续保持预定globaltime局部和同容量大窗统计控制。详见 20261014-phase1-read-position-oldgate-cos-e50-clean-manysig-s0203-m6-r01/final_analysis.md。


报告路线二已落实并实际发布：20261014-phase1-hierarchy-oldgate-cos-e50-clean-manysig-s01-m5-r01，4新scratch50/11100+1旧门控cos历史对照，默认头联合训练/额外fit关闭。SIG_TIME、LOCAL_GTIME、HIER_STAT、HIER_ORDER分辨clock、粗上下文统计容量与显式连续关系；两hierarchy同196→160/pool32416/fθ208481，3bridge/globaltime及actualjoint160接线。实现588209f4186e3056a310458237f4e478c6c15a64、执行7bdfdbf448b9c97ead5702f958bee293d646a09f，native4模型updates/export/W独立/teacher/连续代数gradient和一次P0/P1 PASS，真实四worker/GPU/日志增长VERIFIED，性能pending。20261014-phase1-hierarchy-oldgate-cos-e50-clean-manysig-s0203-m10-r01 条件式剩余8新模型已准备登记，仅首seed源PASS才发布，不读target决定扩展。


连续层级首seed已完整回收：HIER_ORDER82.7732%、HIER_STAT83.9994%、SIG_TIME83.5899%、LOCAL_GTIME83.5946%、OLD80.6685；候选−统计−1.2262pp、候选−SIG−.8167pp；统计控制精确低于84%，不更换primary。4新+1历史/44400steps/840000默认pred/90指标复算VERIFIED，新4模型真实pool更新/精确导出/W独立/零额外拟合，源HIER_ORDER−STAT zL2 2.944889。sourcePASS扩02/03决定 63f59df3933c6fcb778c76d9b1eaed3087f4213c 在target数值读取前提交发布，八新模型正常运行，完整3seedpending。结果见 20261014-phase1-hierarchy-oldgate-cos-e50-clean-manysig-s01-m5-r01/final_analysis.md。


可信门控内难度权重源代理已完成（20261014-phase1-source-trusted-weight-manysig-s01-m1-r01）：接受867/6300全部正确，uniform/weighted precision均100%，不能声称精度改善；t1/t2/t3平均λU/L由.244/.240/.228升至.357/.349/.320，两种U梯度cos .994至.996。0训练/头拟合/目标或隐藏U访问，15统计/75层批梯度保存复算VERIFIED。实现051de1a177819c0f8a9cfe5aba0fd09e7cefef6d、执行c84b17396bc0861b8856e867d17aab54206436c5；完整报告automation_reports/CV-SincNet/20261014-phase1-source-trusted-weight-manysig-s01-m1-r01/report.md。真实权重训练尚未发布，层级八模型保持原配方。


连续层级完整三seed已闭环：HIER_ORDER82.8734%±1.9546pp，vsSIG_TIME+.1784±.9861pp（2/3），vs同容量HIER_STAT+.3230±1.5336pp（2/3）；SIG_TIME82.6950%、LOCAL_GTIME82.7385%、HIER_STAT82.5504%、旧门控cos80.5268%。primary距84均值1.1266pp；最高seed84.8762不代替目标。15row/12新600epochs133200steps＋3历史150epochs，750epoch记录/2520000默认预测/270metrics/21错误表/12真实feature witness/3匹配初态完成VERIFIED，额外头拟合0。首seed未重训、source-only扩seed决定先于新target数字读取，不依据结果调参。完整报告automation_reports/CV-SincNet/20261014-phase1-hierarchy-oldgate-cos-e50-clean-manysig-s0203-m10-r01/report.md。下一步只按已完成源权重诊断与原报告路线四落实独立SIG_TIME权重训练，不加载目标高分层级权重或叠加机制。


报告路线四真实训练已实施并发布：20261014-phase1-trusted-weight-oldgate-cos-e50-clean-manysig-s01-m3-r01，实现23ea591bba17dd938d2b9a93f8b9d9dc5f02e97a、执行7708d4afcef141bb614e0c2171f2a9207c4469ff；固定历史oldgatecos＋scratch50 SIG_TIME uniform/weighted两新模型，结构/旧mask/EMA/cos/defaultjointhead相同，仅acceptedCE boundedstopgrad权重，entropy.01保持。局部真实模型update/RNG/state/33inactive/9negative PASS，一次独立直接P0/P1 PASS；真实CPnoquery smoke/归档compile/两实际workerPID-CWD-argv-GPU0/3-日志增长VERIFIED，目前2epoch。33/34/50全部222批/1998层批梯度的真实预算已接线，尚未到声明轮；未有新性能结果，不称有效或84达标。首seed自动默认头predict/truthlast，sourcePASS才原样扩剩余预定义seed，目标不回流。


可信权重首seed真实闭环：TRUSTED_WEIGHT83.5810%、SIG_TIME83.7345%，−.1536pp/0of1，修正585引入843净−258，旧gatecos80.6685%；3row/2新100epochs22200steps＋1历史50epochs，150源epoch/504000defaultpred/54metrics/3996真实骨干梯度记录独立复算VERIFIED，额外headfit0。33Uinactive；34/50同批weight放大t1/t2/t3梯度但cos约.9994至.9997，不能证明新增信息或性能。源距离比值4.3584→4.4402，但同TX跨RX距离也升高，旧基线ratio5.0771更高而target更差，因此不是泛化排序。sourcePASS（两LCE≤.5/导出/W独立/初态同/源zL2=1.736686/实际梯度budget）在target数字读取前冻结fcc252f3938a21b415f5354b2a2e9db7315934bd并原样发布4新02/03，执行a8ff278fc8dd1b54b69279503c4cda9282795261，真实GPU1/2/4/6进程与日志增长VERIFIED。完整3seed仍pending，84未达；首负结果与所有记录保留。报告automation_reports/CV-SincNet/20261014-phase1-trusted-weight-oldgate-cos-e50-clean-manysig-s01-m3-r01/report.md。


可信门控内权重完整三seed已闭环：weighted82.7030%±1.6081pp、uniformSIG_TIME82.7514%±1.6961pp、旧gatecos80.5268%±2.3776pp。增量−.0484±.1039pp（1/3正差），净正确−244/504000，84均值差1.2970pp未达。9row/6新300epochs66600steps＋3历史150epochs；450完整源epoch/1512000defaultpred/162metrics/11988真实33/34/50层批grad保存复算VERIFIED，额外headfit0，全部六新pool全元素更新/export差0/W独立/三初始source函数同，最终candidate-controlL720zL2 1.736686/1.700653/1.804470。实际grad强化约15.7至29.5%但cos .9994至.9997，不能说明新增泛化信息；源ratio三seed略升而同TX跨RX距离亦全升、target两负一正。报告四路线实际实施与矩阵分析完成，本权重不晋级有效机制，不继续扫描其参数；后续仅依据合法source证据设计W无关浅/中层特征目标。完整报告automation_reports/CV-SincNet/20261014-phase1-trusted-weight-oldgate-cos-e50-clean-manysig-s0203-m6-r01/report.md。


新报告31957f78路线：全L6300浅中层有界排序诊断已闭环，70L90cohort/每视图126000采样关系/1050实际辅助层批梯度+210CE基线，0update/headfit/query。t1/t2/joint未满足率87.8310%/81.3762%/10.7484%，活跃样本100%/100%/88.6508%，不能以关系比例替代样本覆盖。局部平均ORDER实际进入t1/t2且W无依赖；最终eval梯度不冒充真实早中期或target性能。实施1d15b241c7e43b78729c94230fe6ab011c6bacef，run 20261014-phase1-source-layer-order-manysig-s01-m1-r01，下一步独立浅中层有限身份监督三臂，区别另一分支joint160语义原型；停止权重、层级增容/日历扫描。


报告31957f78核心路线已发布 20261014-phase1-local-order-oldgate-cos-e50-clean-manysig-s01-m4-r01，执行f34078d1a0775ef6fa66365689481ad5a1932e4c：单路SIG_TIME ZERO/LOCAL_SUPCON/LOCAL_ORDER三新scratch50＋历史oldgatecos，直接t1/t2四有序格L身份关系，margin.1/tau.2/coeff.1固定；同2750extraL90forward/247500包每模型，原33+17/ownEMA/gate/cos/defaultjointhead不改。13negative/actualinitial/warmupgrad/modeRNG/vectorSupCon/nativejointupdate PASS，唯一直接P0/P1归档依赖fix后PASS。remote实CPsmoke/archive/compile及三actualGPU/PID/CWD/argv/loggrowthVERIFIED，目前3epochs，后续观察actual1/17/33/50梯度并自动defaultclean/truthlast；sourceonly条件才扩02/03，尚无新target分数或84结论。


浅中层有界排序首seed闭环：ORDER83.6238%、SIG83.5452%、SupCon83.4583%、旧gatecos80.6685%，ORDER−SIG+.0786pp/修正538引入406净+132；ORDER−SupCon+.1655pp。4row/3新150epochs33300steps＋1历史50epochs11100，200源epoch/8250auxsteps/672000defaultdecisions/72metrics/1980实际层cohort梯度保存复算VERIFIED，额外头拟合0。实际五pooltensor全元素更新/导出差0/W独立、初态函数同、ORDER−SIGL720zL2=1.592485。SupCon源未满足率更低/梯度更强而target首seed更低，不据源几何排序；单seed+.0786不宣称稳定增益或84。SOURCEPASS决定6d65f77a7659cb182bdb39852dc6c6b76001e81d先于新target数字读取并原样发布02/03六新，执行同commit，完整3seedpending。详见20261014-phase1-local-order-oldgate-cos-e50-clean-manysig-s01-m4-r01/final_analysis.md。


冻结 SIG_TIME 四视图源诊断完成：全 L6300/70cohort、各126000关系、840辅助＋210CE真实层批梯度保存复算 VERIFIED。固定时间增量4坐标严格为1、平方范数占比均值17.6511%；SIG_RAW/DYNAMIC未满足率29.0008%/26.0841%，错误次序19097/18859，样本覆盖99.8889%/99.6508%。直接签名梯度进入浅中层及实际SIG投影，W辅助无依赖，0训练/头拟合/query；几何变化不证明DG收益。运行 20261014-phase1-source-signature-order-manysig-s01-m1-r01，实现 6bf2fda405d58ba203f90fe69a6ced7f66231cd6，见同run final_analysis.md。


完整浅中层有界排序3seed闭环：ORDER82.7127%±1.4787pp，SIG82.6067%±1.7438pp，SupCon82.5843%±1.7028pp，旧gatecos80.5268%±2.3776pp。ORDER−SIG+.1060±.2932pp/2 of3，修正2168引入1634净+534/504000；−SupCon+.1284pp2/3。12row9新450epochs99900steps＋3历史150epochs33300，600完整源epoch/2016000defaultpred/216metrics/5940实际层cohortgrad保存复算VERIFIED，0头拟合/新增targetforward。9fθ实际更新/export0/W独立；初态同，最终ORDER−SIGsourceL2 1.592485/1.398321/1.142613。更低SupCon源关系率和更强梯度没有均值收益，不据几何调参。实现26bfdf044caa1be3c966370248c3bc3372009943，剩余执行6d65f77a7659cb182bdb39852dc6c6b76001e81d，报告20261014-phase1-local-order-oldgate-cos-e50-clean-manysig-s0203-m8-r01/final_analysis.md；84未达成。


下一独立签名内监督已实施：20261014-phase1-signature-order-oldgate-cos-e50-clean-manysig-s01-m4-r01，源依据20261014-phase1-source-signature-order-manysig-s01-m1-r01全L6300；三同网络ZERO/RAW112/DYNAMIC108，部署SIG112不变、只辅助去4常量，固定旧m.1/coeff.1/5+10/33+17/ownEMAgatecos/scratch50。8negative/actualnative同初态warmup/三参数更新/defaulthead joint/W辅助None/模式RNG与唯一P0/P1 PASS，代码f6dfdb587c505b5720469d4a4a101b8df8b321f4。目前LOCAL_VERIFIED，未发布/无新目标分数，不把实现当收益。


签名内监督首seed已发布 20261014-phase1-signature-order-oldgate-cos-e50-clean-manysig-s01-m4-r01，执行2658815a2fe5ef931f24558a6c6c61146433fd9c；真实SIG CP noquerysmoke、单归档/compile及三samePID/CWD/argv/GPU/loggrowth均VERIFIED，3/2/2epoch/GPU1/2/0，无failure，旧baseline默认预测复用已评分。固定50训练继续，RAW/DYNAMIC两候选不按target排行，source资格后才可原样扩02/03。暂无新目标数字或84结论。


2026-10-14 用户最新要求：固定现有单路SIG_TIME网络和默认分类头，探索重心转为训练机制，包括泛化训练与半监督学习。后续不新增/扩大骨干、分支、签名阶数或取特征位置，不重新以结构变体作为核心研发。当前SIG_RAW/DYNAMIC均只改辅助训练目标，部署SIG112及joint160/头结构不变，已发布健康任务保持原计划。下一步半监督工作首先统一SIG_TIME上的RANDOM/PAIR数据利用参照，区分覆盖、伪标签质量与实际梯度方向；不直接继续增大U权重、提早日程或叠加一致性。该整合为独立训练参照，不冒充新84%算法，不据目标结果选择模型/参数。


固定SIG_TIME半监督整合参照已实现 20261014-phase1-sigtime-pair-oldgate-cos-e50-clean-manysig-s01-m3-r01，代码c98bfcd54c870a03195f220644ab4f7a46ba8d8b：两新RANDOM/PAIR scratch50＋旧历史，单路网络/默认head/原oldgateownEMAcos33+17及λU.16/entropy.01不变，无新loss/额外forward。Epoch33/34/50各222真实批 t1/t2/SIG投影LCE/UCE/UCE+entropy梯度保存复算。6negative/隐藏TXtrap/56700once/randomRNGexact/两native同初态及正常W+fθ更新/export0/W独立/compile与唯一直接P0/P1 PASS；目前LOCAL_VERIFIED，未发布/无新target数字，已知PAIR整合不冒充新算法。


固定SIG_TIME半监督参照已真实发布 20261014-phase1-sigtime-pair-oldgate-cos-e50-clean-manysig-s01-m3-r01，执行86c22cb76e5eff4a7b106cca769d198879ab22b4：realSIGcheckpoint noquery90/单归档SHA/remotecompile与2samePID/CWD/argv/GPU4/5/loggrowth VERIFIED，3/3epoch且实际采样记录1至4轮均无Ufetch/迭代，原50/33+17不改。旧baseline默认预测复用已评分；暂无新target分数。另SIG有界身份机制首seed仍同PID健康42/41/42epoch，未修改或重启。


固定SIG_TIME签名内训练首seed闭环：RAW83.6363%、ZERO83.5815%（+.0548pp，修正588引入496净+92/168000）；DYNAMIC83.4476%（−.1339pp，修正573引入798净−225），DYNAMIC−RAW−.1887pp。4row/3新150model-epochs33300updates＋旧50/11100，200完整源epoch/672000defaultpred/72metrics/8250extraL90/1980实际层cohortgrad/7920向量保存复算VERIFIED，0头拟合。三fθ实际5pool张量全元素更新、export差0/W独立；初态函数同，源特征RAW/DYNAMIC−ZERO L2=1.310938/1.795317。更强gradient/去常量源几何不代表收益；仅n1、完整3seed84未达成。SOURCEPASS决定d56f9b79c9fea83629c525e362d473ff0fcafb1e先于新target数字回收，两候选原样02/03六新已发布同commit、samePID/CWD/argv/GPU/loggrowth VERIFIED（6/6/6/6/6/5），未择优、改参或重跑。报告20261014-phase1-signature-order-oldgate-cos-e50-clean-manysig-s01-m4-r01/final_analysis.md。


固定SIG_TIME半监督第34轮真实机制：RANDOM 908/56700接受（1.6014%）、32emptybatch；PAIR51378/56700（90.6138%）、0empty。t1 .16UCE/mainLCE均值12.7656%vs4.6682%，批间SD31.5486vs3.3222个百分点；t2/SIG投影同样均值与波动更低。epoch33两臂U梯度全0；不访问Utruth，不把覆盖/梯度统计当性能，不调lambda或日程。所有222批来自真实保存日志/grad记录；训练产物报告PT复算VERIFIED，独立全PT审计待矩阵完成。详见20261014-phase1-sigtime-pair-oldgate-cos-e50-clean-manysig-s01-m3-r01/mechanism_e34.md。


## 固定 SIG_TIME 半监督配对采样首种子（2026-10-14T14:08:41.344Z）

# 固定 SIG_TIME 的半监督配对采样：首种子结果

SIG_PAIR 的 clean 准确率为 83.8452%，同路径 SIG_TIME RANDOM 为 83.6738%，提高 0.1714 个百分点。仅预定义种子 2026092701，不能据此声称稳定增益或完成三种子均值 84% 的目标。其余两个种子均按原源域资格规则扩展，方案没有根据本轮目标分数调整。

运行 20261014-phase1-sigtime-pair-oldgate-cos-e50-clean-manysig-s01-m3-r01，实现 c98bfcd54c870a03195f220644ab4f7a46ba8d8b，执行 86c22cb76e5eff4a7b106cca769d198879ab22b4。源域扩展决定 f5b19b97bb259de87737dda7a2366036604d12b1 已提交推送，剩余训练已发布后，才回收本轮目标数值。源资格 PASS：最终 L CE RANDOM=0.247506658、PAIR=0.248212648；PAIR 接收 879408/963900（91.2344%）；实际 L720 joint160 特征差 L2=1.429918528。资格只验证机制接入、源健康、固定预算和特征变化，不是泛化有效性的证明。

|方法|clean 准确率|Macro-F1|最差 RX|最差 day|
|---|---:|---:|---:|---:|
|OLD_GATE_COS50|80.6685|80.1734|70.6167|75.4524|
|SIG_TIME|83.6738|83.2636|71.2583|79.1143|
|SIG_PAIR|83.8452|83.4227|71.5292|79.3500|

单位均为%。固定 ManySig clean 168000/row，旧 6 TX×7 目标 RX×4 days，split392005，种子 2026092701。全部 54 个整体/RX/day/TX 指标保存在 results.json。这是已暴露的研究基准，不称为新的独立盲测。L6300/U56700（TX 真值隐藏）/V27000、源 RX1/3/4/6/8 和 days1/2/3 保持。

|配对|差值（百分点）|修正错误|引入错误|净正确变化/168000|
|---|---:|---:|---:|---:|
|SIG_PAIR − SIG_TIME|0.1714|751|463|288|
|SIG_PAIR − OLD_GATE_COS50|3.1768|10508|5171|5337|
|SIG_TIME − OLD_GATE_COS50|3.0054|10341|5292|5049|

PAIR 对 RANDOM 净增加 288 个正确决策。相对历史旧门控+cos 的总改善包含 SIG_TIME 网络带来的差异，不能全部归给 PAIR。n=1 不计算种子样本标准差。

## 固定网络，改变半监督采样

两新模型使用相同 SIG_TIME112→joint160/id_feat_joint，fθ195041 参数；既有 cosine6×160、scale30 分类头按原流程与骨干联合训练，没有单独拟合、重训、替换分类头或 RX/day 诊断头。两个模型实际初始 body、W、完整参数和初始 eval 特征一致。

唯一机制差异是 U loader 顺序：RANDOM 复现原随机顺序与 Torch RNG；PAIR 按允许的 RX/day/eq/sig/物理索引组织不重复的相邻物理包对，不读取 U TX 真值或以 IQ 构造顺序。每个启用轮次仍遍历全部 56700 包一次，17 轮合计 963900，尾批 124。无新增损失、模型前向、优化器更新或额外结构。

两新模型均 scratch50、33 label+17 pseudo、11100 更新，保持原 batch_neighbor 门控、独立 EMA.999、cosine 日程、λU=.16 和 λentropy=.01。原生 U 损失对象直接返回，观察钩子不改值、不累积 .grad、不改变 RNG，也不新增梯度更新。训练中禁用原生另拟合诊断头。历史旧门控+cos 仅复用对应合法已冻结结果。

## 实际训练证据

RANDOM 共接收 15448/963900（1.6027%），PAIR 为 879408/963900（91.2344%）。接收率升高不等于伪标签更准确；没有解封 U TX 真值。U CE 按已接收样本求均值，不能将接收比例直接当损失倍数。

已独立重读完整 150 个源轮次、22200 个新更新、504000 个固定默认头决策及 54 组指标。两新模型累计 100 model-epochs，另列历史 50；新训练额外前向为 0。epoch33/34/50 各 222 批保存 main L CE、.16 U CE 和 .16 U CE+.01 entropy 在 t1PW、t2PW、实际 SIG 投影的真实导数，共 3996 个层批记录、11988 个保存向量。epoch33 U 导数均为 0，源 E34 完整分析另见 mechanism_e34.md。

E50 的加权 U CE / main L CE 梯度范数均值：

|方法|t1PW|t2PW|实际 SIG 投影|
|---|---:|---:|---:|
|SIG_TIME|10.4299%|10.7883%|10.0153%|
|SIG_PAIR|4.6716%|4.4129%|4.3877%|

E50 RANDOM 的非零 U CE 批为 202/222，PAIR 为 222/222；计入 entropy 后均为 222/222。这些是不同 L/U 包批的瞬时导数统计，不是同包反事实、累计 AdamW 贡献或种子方差。PAIR 覆盖提高而平均范数更低，因此不能用“梯度更强”解释效果。

两模型实际 fθ 均有数值更新，5 个 SIG pool 张量均更新；导出 joint160 与原模型最大差为 0，W 置零不改变特征，导出删除旧 W 并保留特征归一化/增益。PAIR−RANDOM 实际源特征差非零，但不证明新 TX 迁移。单独分类头拟合、诊断参数更新、分析中新目标前向均为 0。完整源曲线、门控覆盖和三期梯度图见 results.png。

## 三种子后续状态

剩余运行 20261014-phase1-sigtime-pair-oldgate-cos-e50-clean-manysig-s0203-m6-r01，实现 e1c5f12f68881f8757efae3d9d1bf8f2618f2982，执行 f5b19b97bb259de87737dda7a2366036604d12b1。四个新模型已完成真实 checkpoint 无 query smoke、源资格、进程/CWD/参数/GPU 和同 PID 日志增长核验；首种子不重训。单模型 50 轮，逐行固定后自动默认头预测和独立 truth-last 评分。所有候选和负结果都保留，不依据首种子测试分数调参、挑种子或重跑。

该试验是固定 SIG_TIME 下的半监督集成参照，不宣称新的核心算法。另一项签名身份泛化损失三种子试验仍按独立预登记完成。后续研发聚焦训练、泛化和半监督机制，不扩展网络结构。84% 目标未完成，无 Phase2、新 TX 或卫星能力声明。


## 固定 SIG_TIME 身份排序训练：三种子结果

RAW82.6952%、DYNAMIC82.6690%，对应 SIG_TIME82.6778%；配对 +0.0175/−0.0087 个百分点，2/3 与 1/3 种子为正，未形成稳定增益。全部 12 模型/600 源轮次/2016000 默认决策/216 指标及实际特征、梯度已独立复核。完整报告 20261014-phase1-signature-order-oldgate-cos-e50-clean-manysig-s0203-m8-r01/final_analysis.md，源扩展决定 d56f9b79c9fea83629c525e362d473ff0fcafb1e。固定网络的泛化训练目标确实改变 fθ，却未达到 84%。半监督剩余种子继续原配置运行。


## 固定 SIG_TIME 半监督参照：完整三种子

PAIR82.7667%±1.6363%，同路径RANDOM82.6645%±1.6928%；配对+0.1022±0.1220个百分点，2/3为正，净正确+515/504000。三种子U接收91.5781% vs1.6146%，实际平均U梯度更小；覆盖不等于伪标签准确率或可靠DG收益。9模型/450源轮次/1512000默认决策/162指标/11988层批真实梯度均已完整独立复核，0单训分类头、0额外训练前向，特征导出差0。实现 c98bfcd54c870a03195f220644ab4f7a46ba8d8b，首执行 86c22cb76e5eff4a7b106cca769d198879ab22b4，剩余执行/源资格 f5b19b97bb259de87737dda7a2366036604d12b1。完整报告 20261014-phase1-sigtime-pair-oldgate-cos-e50-clean-manysig-s0203-m6-r01/final_analysis.md。84%未达，无新TX/Phase2/卫星声明。

最新三个训练方向是骨干梯度统计、合法源伪标签误差与可靠身份监督、源优化扰动稳定性诊断；旧Sinc/投影/结构方向已被用户最新固定网络要求覆盖。下一轮最多两个优先动作：核实Fishr实际梯度对象和五源RX分组并落实最小泛化消融；用合法L诊断伪标签错误与fθ梯度后确定独立SSL机制。不依据目标分数调参、挑父权重或重跑。


## 固定 SIG_TIME 的真实骨干梯度方差训练

新运行 20261014-phase1-backbone-fishr-oldgate-cos-e50-clean-manysig-s01-m4-r01，实现 9c981766a0727cf072755ddd46cd5b639bae89ce，当前LOCAL_VERIFIED，尚未启动/没有性能结果。三新50轮ZERO/RAW/TXdaycentered，真实t1/t2参数逐包CE梯度、五RX方差匹配；无新增网络或头，原oldgateEMAcos33+17保持。9负例、真实二阶反传、训练hook、zero预热、默认W正常联合更新及唯一P0/P1修复后PASS。原始矩阵和梯度留远端，公平额外暴露650L180/model。完整预登记与精确命令见对应report.md/experiment.json。84%目标保持未达成。


骨干梯度方差矩阵 20261014-phase1-backbone-fishr-oldgate-cos-e50-clean-manysig-s01-m4-r01 已启动核实，执行 600afeeefe84e0d9f50f07c38ae0767bf1cabf07；三新GPU4/5/0，同PID/CWD/argv/loggrowth及首轮13L180/保存矩阵重算VERIFIED，无额外头。5轮预热故E1有效R为0，不把启动或局部梯度当性能证据。启动聚合见对应launch_verified.json与epoch1_source_evidence.json；完整固定50轮/自动truth-last仍运行。


## 固定 SIG_TIME 源域伪标签错误梯度：三模型完整诊断

20261014-phase1-source-sigtime-pseudo-gradient-manysig-m3-r01 ANALYZED，实现 d17f140c3528d8964ffedf464976532bc56e0760、执行 3bc96d53291ca3b2f1eeb0c19236e236a8d25866。三个预定SIG_TIME RANDOM final50/defaultstudent及actualownEMA；合法L6300/V27000，仅L梯度，U/query不访问，头/模型/优化器更新0。199800源包前向/786batch、75L梯度批、150scope批、450层scope记录/1800向量及L/V特征统计独立复算VERIFIED。

L的CONFIDENCE_STRONG接受6024/6061/6075、FULL_OLD接受848/851/855，均观察0错误；错误梯度全0、pseudo和matchedtrue梯度相等是标签相同的必然结果，未识别错误影响，不能推出U准确率或训练早期无害。V的CONFIDENCE_STRONG错误28/34/40；实际joint160错误/正确cos中位数为.700940/.682370、.713649/.675108、.699859/.680247，分位范围大量重叠，均值方向不一致；当前证据不支持增加简单特征一致性可靠度阈值。V无梯度/拟合，不扫描tau或新头。

完整源分析、成本、局限与图见 automation_reports/CV-SincNet/20261014-phase1-source-sigtime-pseudo-gradient-manysig-m3-r01/source_pseudo_gradient_analysis.md 与 results.json。下一步完成正在运行的真实骨干梯度方差训练，再检验独立SSL表征假设。84%未达成；本次没有新目标性能结果。


## 固定网络训练机制：实际梯度与独立半监督候选

BF 20261014-phase1-backbone-fishr-oldgate-cos-e50-clean-manysig-s01-m4-r01 RUNNING：epoch17三新各13个L180/2340源包，RAW/COND的实际t1/t2各13批effective梯度均非零，ZERO全0。RAW effective/mainLCE范数比均值t1/t2=.658707/.616297，COND=.512669/.492663，平均cos分别.073900/.097004和.006069/.034401。这是独立L128/auxL180图的实际导数观测，不代表累计AdamW贡献或DG收益。逐包矩阵存在，生产端保存复算VERIFIED；终轮完整独立审计仍待完成。

新独立SSL 20261014-phase1-feature-mean-teacher-oldgate-cos-e50-clean-manysig-s01-m3-r01 RUNNING，code 373262b51f037f03e3238471b3aefb88e45fba5b，execution 6b23a5b033ee063dea97d6783d5634e220a1d93d。同包原EMAweak/学生strong joint160的unit特征匹配；coefficient.1，epoch34至38五轮ramp，原hardUCE/entropy/ownEMA/RANDOM和默认头保持。ZERO同路径控制，旧基线复用；没有新模块、头、模型前向、TX/RX/day/confidence目标或阈值。辅助W梯度None，主W梯度不变，实际t1/t2/SIG及头正常更新与export/W独立局部验证PASS；唯一P0/P1 PASS。

两新scratch50/11100各，源预算100model-epochs22200更新＋历史50/11100，17x56700 U每模型。全部source合法，无query训练；真实semi机制在34轮前尚未启用，不把当前几轮当SSL效果。预登记source最后5轮U feature discrepancy比ZERO≤.95，配合源CE<=.5/梯度/特征边界才原样扩02/03，先冻结/提交源决定再看新目标摘要。不根据目标调权、选父或重跑；每row完成自动clean默认预测及独立truth-last评分。

一次archive/remotecompile/真实checkpoint no-query smoke及两worker PID/CWD/argv/GPU/loggrowth均VERIFIED，GPU7/2实际启动，所有owner每GPU≤3。报告、全矩阵与source规则在automation_reports/CV-SincNet/20261014-phase1-feature-mean-teacher-oldgate-cos-e50-clean-manysig-s01-m3-r01/report.md/experiment.json，阶段交付不宣称性能或84%达标。


## 实际U接线与固定种子扩展准备

FEATURE_MT两臂已到36轮，每臂观测U666批/170100包；epoch34候选t1/t2/SIGprojection各222批effective梯度非零，ZERO各0，逐批辅助W梯度None，producer保存复算VERIFIED；实际原生配置保持oldgate/ownEMA/33+17/.16/.01/smoothing.01。当前只证明接入，不宣称参数最终变化、完整预算、DG收益或84%。部分源证据见20261014-phase1-feature-mean-teacher-oldgate-cos-e50-clean-manysig-s01-m3-r01/actual_U_phase_source_evidence.json。

两个固定02/03执行配置已LOCAL_VERIFIED，均NOT_LAUNCHED：BF 20261015-phase1-backbone-fishr-oldgate-cos-e50-clean-manysig-s0203-m8-r01（code b69d6ffbbd229284c4cfde35124db10bf871fe07，6新/8row），FEATURE_MT 20261015-phase1-feature-mean-teacher-oldgate-cos-e50-clean-manysig-s0203-m6-r01（code 68b3143dbdfcde27c80997f970121a23e34e5f2e，4新/6row）。原机制字典/数值设置逐项一致，args仅种子/run/candidate/output差异，同seed初态一致，source资格10/11负例和实际新上下文检查PASS；没有新科学假设、组合或拟合头，复用已完成核心唯一P0/P1，不重复审未变代码。

首seed仍未完成source资格；准备不等于获准扩展或三seed完成。只在原固定SOURCEPASS后发布相应新矩阵，源决定先提交再读新numeric target摘要；首seed不重跑、双BF候选不按目标择优，预算和成本预先记录。原始健康archive与进程保持。


FEATURE_MT首seed三row已完成训练/默认预测/独立truth-last评分，原source资格PASS（last5 discrepancy ratio.821487695，终LCE ZERO .247665886/FEATURE .248785446，实际L720z差2.058244944，梯度/精确预算满足）。当前只取源判定，未回收新目标准确率；本次提交先冻结按原配方扩02/03的决定，再发布并回收目标摘要。源资格不是DG收益。


## 置信度特征一致性固定扩展（2026-10-15）

首源资格与02/03扩展决定已在 3b0b38c708029570cd1d6327cd4db96e2752a6ad 提交/推送核实，早于首种子新目标数值。当前适配器 fe2d8287ef1f9c167c5b68ee395e8bfe01b08360 本地/独立P0P1 PASS，固定四新scratch50模型，保持SIG_TIME及原默认联合头/旧门控EMAcos。首种子负差不改变既定扩展。见[最小预登记](../automation_reports/CV-SincNet/20261015-phase1-confidence-feature-oldgate-cos-e50-clean-manysig-s0203-m6-r01/report.md)。


## 置信度特征一致性首种子完整交付与固定扩展启动

首种子已ANALYZED/VERIFIED，CONF83.521429% 对SIG83.665476%，配对−0.144048pp（修正766/引入1008/净−242），n=1；源feature均值降低约17%至18%，未转成DG提升。默认头macroF1、最差RX/day同样未提升。完整报告[final_analysis.md](../automation_reports/CV-SincNet/20261015-phase1-confidence-feature-oldgate-cos-e50-clean-manysig-s01-m3-r01/final_analysis.md)，保留完整50轮/11100步、54CM和保存向量复算。源决定3b0b38c708029570cd1d6327cd4db96e2752a6ad在新目标数值前已推送，因此02/03按冻结不变配置执行；实际发布55ff29d676eb5f02afa8b3869ed4818122f09d9a、4个worker已完成5至6轮，详见[运行报告](../automation_reports/CV-SincNet/20261015-phase1-confidence-feature-oldgate-cos-e50-clean-manysig-s0203-m6-r01/report.md)。不新增头拟合、目标调权或结构分支。


## 源域方差约束追加解释（2026-10-15）

已完成9模型936保存梯度矩阵/1872方差视图的源域再分析，CPU25.93s，无新前向/梯度/模型或头更新，无IQ/预测/truth读取。原损失已尺度归一化，分母floor0；不能靠整体梯度缩小解释。固定终轮幅度项贡献85.92%至96.54%，候选幅度项12/12降低、加权角向项12/12升高；无幅度权重的形状cos改善RAW5/6、COND1/6。完整[源机制报告](../automation_reports/CV-SincNet/20261015-phase1-backbone-fishr-oldgate-cos-e50-clean-manysig-s0203-m8-r01/source_variance_decomposition.md)。下一DG方向需先证伪直接形状约束的真实身份梯度兼容性，当前无新训练候选发布。

CONF_FEATURE_MT02/03四个原worker仍live，最近完成23/27/27/26轮且loggrowth均正，failure0；按原50轮和自动clean链继续，当前没有完整三seed结果。


## 直接形状目标的最小源端证伪

新增20261015-phase1-source-backbone-variance-angle-manysig-s2026092701-r01，固定原BF SIG seed01终轮模型/前5个平衡L180，共900源样本。只比较同批默认CE与原RAW/纯角度目标的真实t1/t2导数，不训练/换头/访问query；两层汇总cos>=0且各5/5非零才源PASS，无自动训练发布。代码6907495c3c35147a08a5daa5c30f85fce9060ba8，本地11负例/实际可微公式和一次独立P0P1 PASS，当前未发布。见[最小预登记](../automation_reports/CV-SincNet/20261015-phase1-source-backbone-variance-angle-manysig-s2026092701-r01/report.md)。


## 方差形状目标900L真实源检验完成

20261015-phase1-source-backbone-variance-angle-manysig-s2026092701-r01 已ANALYZED/VERIFIED，执行30047816ba3beaa581a18cc0384ee947413fff6d，实际900L/10矩阵/30向量独立复算；t1/t2 ANGLE对同批CE全局cos0.646206/0.593616、5/5批非零，两层资格PASS，无模型/头/optimizer更新或query。旧RAW/新ANGLE的未加权导数幅度差异明显，不能仅据cos宣称几何机制更优或推断训练贡献。固定02/03同900L源扩展决定已写入原run source_extension_decision.json，当前未实现/发布，不自动开始训练。完整[源结果](../automation_reports/CV-SincNet/20261015-phase1-source-backbone-variance-angle-manysig-s2026092701-r01/final_analysis.md)。


## 置信度门控SSL完整三种子交付

20261015-phase1-confidence-feature-oldgate-cos-e50-clean-manysig-s0203-m6-r01已ANALYZED/VERIFIED，与保留首seed合并：9模型1512000默认头clean决策/162CM，CONF82.540675%±1.760937pp，匹配SIG82.776984%±1.607961pp；配对−0.236310±0.153150pp，0/3正，修正1837/引入3028/净−1191。源差异降低17%至20%未带来目标提升，完整450源epoch、23976层记录/71928向量独立核实；不调强正则/挑seed/换头补结论。完整[结果报告](../automation_reports/CV-SincNet/20261015-phase1-confidence-feature-oldgate-cos-e50-clean-manysig-s0203-m6-r01/final_analysis.md)。源资格独立文件的历史路径缺项已说明，实际firstsource/preflight证据完整；结果读取器的父子进程收尾间隙已独立核实，未重启原实验。下一项为已冻结的02/03方差形状源诊断，不是新训练。


## 方差方向源检验：三模型未通过，停止固定公式

20261015-phase1-source-backbone-variance-angle-manysig-s0203-m2-r01 已ANALYZED/VERIFIED，执行 e493b3e28c6ffb2824a52f7d6d71028c00015077。相同900个源物理样本、固定五个L180，保留原01结果，本次只新增02/03。ANGLE相对同批CE的t1/t2聚合余弦分别为01(.646206,.593616)、02(−.111682,−.024725)、03(−.009819,.087680)；只有1/3模型满足原两层非负判据，虽然六层均5/5批非零。独立逐行矩阵/向量复算通过，0训练/头拟合/optimizer更新，无U/V前向或query。按原规则停止该固定公式，不以调整标准/幅度或挑seed延续。完整[负结果](../automation_reports/CV-SincNet/20261015-phase1-source-backbone-variance-angle-manysig-s0203-m2-r01/final_analysis.md)。下一训练机制另行依据合法源证据设计，固定SIG_TIME/default联合头/旧门控cos/<=50，84%目标未完成。


## 固定 SIG_TIME：源接收机风险训练对照

20261015-phase1-receiver-dro-oldgate-cos-e50-clean-manysig-s01-m4-r01，实现 341d0dc0bdab44fce88e0af3d8fc29eb50b39912，LOCAL_VERIFIED/PREREGISTERED，尚未发布。首seed三新scratch50：原CE、源RX等权CE、GroupDRO式动态RX风险加权；旧门控/EMA/cos/默认联合头和WD不变，不增加训练前向或模块，不另训分类头。9负例、真实native接入、三实际网络同初态/骨干及默认W非零联合梯度与一次独立P0P1 PASS。源规则提前固定，PASS才原样扩三臂02/03，源决定先推送再回收新目标数值；每行自动clean默认头预测/独立truth-last。完整[预登记](../automation_reports/CV-SincNet/20261015-phase1-receiver-dro-oldgate-cos-e50-clean-manysig-s01-m4-r01/report.md)。没有新性能结果，84%仍未完成。


源接收机群风险矩阵已RUNNING/VERIFIED：执行 977d4fb609436bac23621ef8e32c2017495acef7，三新GPU1/2/0完成3轮，单归档/remotecompile/真实CP无query smoke和同PID-CWD-argv-loggrowth核实。每臂444原生源步骤的权重/CE消费独立重算通过，候选首轮真实t1/t2/SIG及默认W增量梯度13/13非零，ERM均0；无另训头。50轮完整训练/自动truth-last仍待完成，未回收新目标数值；源PASS决定先推送的要求保持。


## 接收机风险训练中期与固定扩展准备

原执行 977d4fb609436bac23621ef8e32c2017495acef7 的三进程保持，当前源流11322步群权/CE/原生消费独立重算通过；两个候选epoch17实际t1/t2/SIG及普通W的风险增量梯度各13/13非零，无另训头，无新目标准确率读回。完整source资格和50轮仍未完成。条件适配器 20261015-phase1-receiver-dro-oldgate-cos-e50-clean-manysig-s0203-m8-r01（code 8522fb75680c7b51c7d3d631f67c1b2b212eba8b）已LOCAL_VERIFIED/PREREGISTERED/NOT_LAUNCHED；六真实native上下文只seed/run/candidate/output差异，同seed初态相同，12负例/原U对象/默认头梯度和一次adapter-only P0P1 PASS。仅原sourcePASS且源决定先推送才发布；不是新机制或源资格通过。详见[条件预登记](../automation_reports/CV-SincNet/20261015-phase1-receiver-dro-oldgate-cos-e50-clean-manysig-s0203-m8-r01/report.md)。


接收机风险三臂已实测进入半监督，原batch_neighbor/ownEMA/cos、33+17、smoothing.01、UCE.16/entropy.01保持。源流快照U包处理SIG93820/BALANCED282608/DRO269296，U真值0、EMA逐步更新核实；监督risk流25086步独立回放，进入U后所用原生L loss仍一致。E33候选t1/t2/SIG增量各13/13，ERM0。当前仅源运行证据，原50轮/自动评分仍继续，未读本轮新目标准确率，未发布条件扩展。


2026-10-15接收机风险首seed源资格PASS：DRO/BALANCED最后5轮最差RX训练CE=0.817476，完整50/11100与实际骨干增量验证通过。新目标数值未读取，先冻结原样三臂02/03扩展并上传决定，再回收目标结果。源PASS不等于泛化提升。


2026-10-15接收机风险首seed完整默认头clean：SIG83.657738%、BALANCED83.587500%、DRO82.585119%，DRO−SIG=−1.072619个百分点。源资格改善未转成目标增益，n=1不作三seed结论。源决定`a283291279cb01b939325cf5ccba1d0a671b28c4`在新目标数值前已推送；固定02/03六新模型已核实运行。另有12既有模型/7992批/143856保存向量的只读SSL梯度分析：终轮F/L平均约0.9%至1.9%，无新增CE方向反转，不能凭近半负cos断言特征辅助过强。两份完整分析及产物路径已登记。


下一独立训练假设源预检：RSC启发的actual t2时序遮挡，固定SIG_TIME推理结构/默认头，无SAM/权重扰动/新头；run`20261015-phase1-source-temporal-self-challenge-manysig-s2026092701-r01`代码`50d265cb0e8e48cdc2580c9f7b13e342f26b50c8`。只读预定义SIG_TIME01父的L900，定向与同物理包同零数随机对照，3600前向包次/20层60向量/0更新。源规则预先固定，唯一P0/P1和本地检查PASS；当前未发布，不声称有效。源PASS只允许独立登记02/03源诊断，不能自动训练或据target调mask。


自挑战首模型源结果verified：RSC−匹配随机CE+.143074，900L/3600源forward/300选择包/3300时间零位置，骨干增量5/5真实非零。t1/t2导数余弦−.556406/−.620389，较大且反向；这是挑战接通，不是泛化改善，0更新/头fit/query。固定sourcePASS只准备02/03源复核，待原DRO矩阵SIG02/03父模型自然final50完成，不另选父或直接训练。


RSC剩余02/03源复核适配器准备完成：代码`1f5c259e9945913d7bb33fb88b1e270d861275dc`、13新负例和实际两个native seed上下文/唯一P0P1 PASS；sourcePASS决定`737b65502bc5d9f204a605108b80f430bfe59541`已上传。运行`20261015-phase1-source-temporal-self-challenge-manysig-s0203-m2-r01`尚未发布，等待原DRO矩阵SIG_TIME02/03父自然final50，固定同900物理包/原mask/0更新，新增7200sourceforward/40层120向量。不得使用中途或择优父，不重做01，不自动训练。


中层时序自挑战三源模型闭环：实际剩余执行`00c7213ac3ce8d2543e9443f5e6dea9ff5772a7b`，sourcePASS3/3；2700包×模型观察但实际ID并集/交集均900，10800源forward/60层180向量，0更新/头fit/query。t1/t2导数01/03反向、02同向，不能称稳定有益方向；TX/RX选择偏向跨seed仍有变化。完整报告`automation_reports/CV-SincNet/20261015-phase1-source-temporal-self-challenge-manysig-s0203-m2-r01/final_analysis.md`。下一独立RSC首seed训练假设仅据源/固定方法记录，尚未实现或发布；DRO全矩阵已自然完成，独立完整分析待汇总。


接收机风险完整3seed已闭环：SIG82.629167%±1.606482pp、等权82.743056%±1.622387pp、DRO81.655556%±1.892010pp。DRO−SIG−.973611±.384968pp/0of3，等权+.113889±.194466pp/2of3且最差RX/day均值略低。三seed源最差RX CE均下降（DRO/等权.8175/.8347/.8358）未转成目标收益。12row/9新450epochs99900steps＋3历史150epochs，2016000defaultheadclean决策/216指标/600源epoch/1872层5616向量独立复算VERIFIED，0headfit；84未达，固定DRO路线结束原样扩展。报告`automation_reports/CV-SincNet/20261015-phase1-receiver-dro-oldgate-cos-e50-clean-manysig-s0203-m8-r01/final_analysis.md`。RSC下一独立训练假设已在`84579f608289b6c103b503be816c16807ae4ddb7`记录后才读取本次剩余目标数值，仍按源和固定方法实现，不扫mask/层/父。


固定 SIG_TIME 的时序自挑战训练已 RUNNING/VERIFIED：`20261015-phase1-temporal-rsc-oldgate-cos-e50-clean-manysig-s01-m4-r02`，代码 `82cc7f1c24c2e5d0e20572b2654722fdc3e43a47`、执行 `e7a6215aa5d0a05e71c1c842e78cdfc845d57530`。三个新 scratch50/33＋17/11100 臂为 SIG_TIME、匹配随机遮挡、TEMPORAL_RSC；旧门控＋EMA＋cos/default 联合头保持，推理网络不变。每臂相同三 L 前向，22200 个额外前向单独计数，不声称历史计算量相同。单归档传输、远端编译、真实 checkpoint 无查询 smoke 和同 PID/CWD/argv/GPU/日志增长核实；三新进程均完成 2 轮。实际前444源步骤/臂的遮挡计数与原生 CE 消费独立复算通过；E1 每臂13批，两个变体 t1/t2/SIG/普通 W 的 CE 增量梯度各13/13非零，SIG控制为0；156层/468向量及39遮挡批复核通过。尚无源资格结论或新模型目标指标。失败 r01 在登记比较阶段退出，未启动训练，产物保留，r02 仅修复公开路径句柄读取和完整配置登记，未改数值配方。完整[登记与恢复说明](../automation_reports/CV-SincNet/20261015-phase1-temporal-rsc-oldgate-cos-e50-clean-manysig-s01-m4-r02/report.md)及[实际源运行证据](../automation_reports/CV-SincNet/20261015-phase1-temporal-rsc-oldgate-cos-e50-clean-manysig-s01-m4-r02/source_progress.json)。仍按发布前固定源规则决定扩种子，先推送决定再回收新目标数值；84% 目标未完成。


时序自挑战 02/03 条件适配器已 LOCAL_VERIFIED／PREREGISTERED／NOT_LAUNCHED：`20261015-phase1-temporal-rsc-oldgate-cos-e50-clean-manysig-s0203-m8-r01`，代码 `2f99d65f8a594ab1f56d84d1d7ee1011740da52c`。固定6新 scratch50＋2历史，不重训01，不增新机制；六个实际原生上下文仅 seed/run/candidate/output 不同、同 seed 三臂初态相等，实际前向/L损失/骨干与默认W梯度/原U对象/遮挡清理、12负例、完整公开配置还原与一次adapter-only P0/P1均通过。原首种子仍健康训练，完整source资格尚未完成；本适配器只在原规则PASS且源决定先提交推送后发布，不以target决定扩展，不重复审查未改机制。[条件预登记](../automation_reports/CV-SincNet/20261015-phase1-temporal-rsc-oldgate-cos-e50-clean-manysig-s0203-m8-r01/report.md)。


时序自挑战实际训练已核实21轮／原PID，E1/E17全部78个固定诊断批、312层936向量的只读梯度分析完成。RSC E17 t1/t2/SIG 的所选对原CE余弦均值 .84465/.85215/.74941，增量范数比 .61280/.59495/.90914；三个层13/13非零且无负余弦，SIG增量0，匹配随机也改变骨干。源探针大反向不能代替实际联合训练方向；这不是DG或导出特征提升证明，终轮feature witness仍待完成。原生／遮挡流13310匹配步及78mask批复核通过，无新目标分数、配方和源扩展规则保持。[中期机制记录](../automation_reports/CV-SincNet/20261015-phase1-temporal-rsc-oldgate-cos-e50-clean-manysig-s01-m4-r02/report.md)。


时序自挑战三原进程已40轮并实测进入U，原33＋17／EMA0.999／UCE0.16／entropy0.01／cos逐项核对，真U标签可用数始终0。完整36轮源曲线快照的后续可见U前缀为SIG221044／随机224116／RSC221812包，接纳率约1.64%／1.63%／1.62%，窗口不等且不是伪标签精度。E33实际梯度加入后117批／468层1404向量，RSC t1/t2/SIG余弦 .87216/.87034/.80670、增量范数比 .60237/.60692/.79518，保存logit CE与原生步对齐误差<=4.77e-7。源流25974匹配步和117mask批复核通过，源曲线图已渲染查看。终轮feature witness／源资格／新目标仍未完成，条件02/03不发布。[阶段证据](../automation_reports/CV-SincNet/20261015-phase1-temporal-rsc-oldgate-cos-e50-clean-manysig-s01-m4-r02/report.md)。


时序自挑战首seed终轮source资格已PASS，新模型目标数值尚未读取。三臂完整50／11100／原U963900／额外L前向22200；随机与RSC实际L720部署特征相对SIG距离3.324663／3.638345，195041实际f_theta中193912／193847元素改变，默认W正常联合更新960元素，W删除／清零不改z且feature-only导出逐元素一致。原17／33／50骨干增量条件与源CE健康通过，冻结原样全部三臂02/03扩展，先推送决定再读目标数值，不以sourcePASS称DG成功。源决定`automation_reports/CV-SincNet/20261015-phase1-temporal-rsc-oldgate-cos-e50-clean-manysig-s01-m4-r02/source_expansion_decision.json`。


时序自挑战首种子已完成完整训练／默认预测／独立truth-last与全量复核：SIG83.660714%、随机82.385119%、RSC81.823810%，RSC−SIG−1.836905pp、RSC−随机−.561310pp；n=1，不当作三seed结论。4row／200源epoch／33300新更新／672000默认决策／72指标及624层1872真实梯度向量复核VERIFIED，0headfit；实际fθ数值更新、W独立、feature-only导出差0通过，但sourcePASS未转成目标收益。源决定 `832485cace509543a7b4e921ac7f5483a2053749` 已先推送后取目标；02/03六新原样发布，执行同该提交，原PID/CWD/argv/GPU/loggrowth、4至5轮与6438源step／312层936向量已核实。完整[首种子结果](../automation_reports/CV-SincNet/20261015-phase1-temporal-rsc-oldgate-cos-e50-clean-manysig-s01-m4-r02/final_analysis.md)和[剩余运行](../automation_reports/CV-SincNet/20261015-phase1-temporal-rsc-oldgate-cos-e50-clean-manysig-s0203-m8-r01/report.md)。保留负结果，不按目标改mask／取消健康任务；84%仍未完成，网络及默认联合头保持，研发继续聚焦训练、泛化、半监督。


独立SSL下一源诊断已LOCAL_VERIFIED／PREREGISTERED：`20261015-phase1-source-soft-relation-manysig-s2026092701-r01`，代码`5ab6d390d5306a787c0efbf4a1532df861f0b877`。EMA类别概率定义软身份关系，直接约束原部署joint160的批内关系；ALL与跨源RX对照，无投影／新头／网络改动。固定原DRO首SIG final50及自身EMA、900L／1800前向／20层60向量／0优化或headfit；13负例和真实网络导数／W独立性／RNG及唯一P0/P1 PASS。XR源身份质量>=.90、真实三骨干层5/5非零和同strongCE全局余弦>=0才源PASS，且仅允许另登记02/03源复核，不自动训练；当前未发布、无新性能结果。完整[最小预登记](../automation_reports/CV-SincNet/20261015-phase1-source-soft-relation-manysig-s2026092701-r01/report.md)。RSC健康六新固定50轮继续，不据目标改变。


软关系源诊断已完整ANALYZED／VERIFIED，sourceFAIL。900唯一L／1800源前向／20层60向量、0更新或headfit/U/query；teacher软边身份质量98.697450%，XR t1/t2/SIG真实有效梯度范数比3.394885/3.936461/.511427，对同strongCE余弦−.262721/−.330844/.003583。只证明late训练L下梯度兼容性不满足，不能外推所有关系学习／U／DG；按预定规则不扩02/03、不训练或扫参数。实现`5ab6d390d5306a787c0efbf4a1532df861f0b877`、执行`e2242a9201afe4bf896305cc3d364f069f677a65`，完整[源结果](../automation_reports/CV-SincNet/20261015-phase1-source-soft-relation-manysig-s2026092701-r01/final_analysis.md)。RSC六新最新29至33轮，原PID/CWD/argv及日志增长仍核实、failure0，继续原50轮。


固定网络下一训练假设：原48个Sinc参数的Hz优化单位。真实9历史模型和当前3个SIG_TIME模型均几乎不更新；已有优化器矩非零且末轮步长低于半ULP。仅优化器组按Fs换算lr／eps／wd、保留原全局clip／cos／EMA和全部非Sinc参数（含默认W）选项，不改网络或参数化。组件本地7负例／真实native前向和同梯度检查PASS，STANDARD与原生更新/矩逐元素一致，unit首步仅Sinc不同并传到joint160；这只是合成机械证据，没有新真实训练或目标结果，完整训练接线与唯一审查尚待完成。[准备报告](../analysis/sinc_unit_optimizer_preparation_20261015.md)。依据在读取RSC剩余目标数值前固定，不扫倍率。


2026-10-15 更新：时序自挑战完整三种子结果已独立复核并全量保留。SIG_TIME 82.669048%，随机 82.002579%，RSC 81.588095%；RSC 相对 SIG 配对 −1.080952±0.766891 个百分点，0/3 为正。本固定机制不再扩展。完整矩阵、RX/day/F1、错误转移、实际梯度、部署特征和成本见 [完整结果](../automation_reports/CV-SincNet/20261015-phase1-temporal-rsc-oldgate-cos-e50-clean-manysig-s0203-m8-r01/final_analysis.md)。84% 目标仍未达成。下一项网络不变的 Sinc 优化器机制代码已固定于 00132f7d54aff45b6e52fc41f89d7705a043bc71，尚未发布或得到性能结论。


2026-10-15 训练机制更新：网络与默认头保持的Sinc优化器单位实验已在新R02发布，两个新模型均观察到E2，真实checkpoint无query smoke及PID/CWD/argv/GPU/日志增长已核实。E1实际13批对照有非零梯度但Sinc数值更新为0；单位臂48参数均真实更新。源资格和目标提升仍未知。R01因间接发布依赖在训练前失败，0更新且保留产物；修复只补依赖和输出ID。详见 [R02报告](../automation_reports/CV-SincNet/20261015-phase1-sinc-units-oldgate-cos-e50-clean-manysig-s01-m3-r02/report.md)。


2026-10-15 中期更新：Sinc单位两臂分别回收到20/21轮完整源前缀，E17真实13批重复观察到标准组非零梯度但0数值更新、单位组48参数持续更新。原U在E34启用；当前源资格与目标收益均未知。原样02/03条件扩展已完成最小适配、4原生上下文/15负例/隔离包/一次P0P1验证，代码03fd8a8c7b64663bc52edb83359d3e610cfeb361；仍未启动。详见 [首种子源证据](../automation_reports/CV-SincNet/20261015-phase1-sinc-units-oldgate-cos-e50-clean-manysig-s01-m3-r02/report.md)与 [条件扩展](../automation_reports/CV-SincNet/20261015-phase1-sinc-units-oldgate-cos-e50-clean-manysig-s0203-m6-r01/report.md)。


2026-10-15 原生U诊断：Sinc单位首种子两臂已观察到E35、原pseudo阶段，U曝光与门控实际执行。对E33/E34全部888批、2664层记录、7992输入向量作源端只读分解，熵梯度/L范数均值0.584%至0.803%，0个总L+U方向被熵反转；仅三参数张量/单seed/两个预定epoch，不作fθ全局或AdamW因果推断，不开启熵扫描。另确认现有PAIR是同RX邻近sig，而非跨RX同次发射配对，未据索引构造跨RX伪监督。详见 [本运行主报告](../automation_reports/CV-SincNet/20261015-phase1-sinc-units-oldgate-cos-e50-clean-manysig-s01-m3-r02/report.md)。


2026-10-15 Sinc单位首源资格PASS：两臂50/11100、各U963900完成；候选48参数全改变，student/EMA滤波核相对L2=0.307170421/0.306414366，真实L720 joint160差3.697698355；W独立feature-only边界最大差0。按原规则决定两臂原样扩02/03，源决定先提交推送，再读取本候选新目标数值；当前性能数值尚未回收，84目标不作已达成声明。[源决定](../automation_reports/CV-SincNet/20261015-phase1-sinc-units-oldgate-cos-e50-clean-manysig-s01-m3-r02/source_expansion_decision.json)。


2026-10-15 Sinc优化器首种子完整结果：SINC_UNITS83.354762% vs同路径SIG_TIME83.636905%，−0.282143个百分点；修正4258/引入4732，净−474。原Sinc48参数、EMA核与实际z确有更新，sourcePASS未转为本seed泛化收益，n1不替代3seed。全部54指标/504000决策、22200native步骤和3996真实L/U层梯度独立核实。[完整首结果](../automation_reports/CV-SincNet/20261015-phase1-sinc-units-oldgate-cos-e50-clean-manysig-s01-m3-r02/final_analysis.md)。源决定1a3be6d2568939ff56c38c8bc8dc9a879b7abe76推送后才读取首新目标分数；已按该预定决定原样启动02/03共4模型，真实PID/CWD/argv/GPU及日志增长核实，观察epoch15/16/14/16，无改参/重训01。[运行证据](../automation_reports/CV-SincNet/20261015-phase1-sinc-units-oldgate-cos-e50-clean-manysig-s0203-m6-r01/launch_verified.json)。84目标仍未达成。


## 下一训练机制的来源与重复路线排除

2026-10-15，在读取本轮 Sinc 剩余种子的目标数值之前，已核对另一分支的 `cvs_cross_rx_feedback` 与 `cvs_feedback_teacher` 实际公式：前者按源 L 导数给 U 微组分配权重，后者用源 L 反馈调整教师。因此本对话不再把这两类方法作为独立新路线发布；本次没有重新核验它们的完整性能或当前运行状态。一般原理可见 [Ren 等的样本重加权论文](https://proceedings.mlr.press/v80/ren18a.html)与 [Meta Pseudo Labels](https://openaccess.thecvf.com/content/CVPR2021/html/Pham_Meta_Pseudo_Labels_CVPR_2021_paper.html)，不能直接迁移其收益结论。

下一待检验假设为 dropout 两次随机前向的一致性。合法源依据仍是既有单模型 L720 的独立末端 dropout 回放：feature-feature area 噪声/clean RMS 31.9940%、实际残差160为25.2901%。该证据只覆盖隔离扰动，不能证明完整训练模式存在同幅度偏差，更不能证明它导致目标误差。[R-Drop 原论文](https://proceedings.neurips.cc/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html)以同输入的两个 dropout 子模型输出之间的双向 KL 作训练约束；这是一般方法依据，不是本任务的已验证效果。

先做小规模 L-only 冻结源检验：预定义原DRO首SIG seed2026092701 final50父，测实际随机前向差异和原骨干导数，0训练/头拟合/optimizer更新，无U/V/目标。此处仅固定研究来源与父选择，尚未实现、登记或发布新实验。若后续证据支持训练，最低对照必须同为两次 L 前向、平均两次CE；变体仅加对称KL，保持 SIG_TIME、默认联合头、旧门控/ownEMA/U损失/cos与每模型≤50轮，额外成本单列。不能用另一个前向或额外监督量解释候选增益，也不把KL降低当特征迁移证明。


2026-10-15 完整三种子交付：Sinc优化器单位 SINC_UNITS83.223214%±0.492234pp，匹配SIG82.696230%±1.624144pp；配对+0.526984±1.161397pp、2/3正，主要由seed02贡献，seed01负、seed03近零。所有9row/1512000默认决策/162指标、450源epoch、66600新更新、11988层L/U记录和实际fθ/W独立边界已核实，84%仍未完成。[完整结果](../automation_reports/CV-SincNet/20261015-phase1-sinc-units-oldgate-cos-e50-clean-manysig-s0203-m6-r01/final_analysis.md)。固定原网络、默认联合头，未换头或调倍率。

独立新R-Drop源检验`20261015-phase1-source-dropout-consistency-manysig-s2026092701-r01`实际完成，代码`eb21026149b1d1d02ddf4b213408b37f782bf2f5`、执行`842af6233e30dbfd9927670d1f0723905f592c9c`。首SIG父固定在Sinc剩余目标数值读取前；900唯一L、2700源前向、20层40向量、0模型/头/optimizer更新，原规则PASS：t1/t2/SIG对pairedCE聚合cos.200427/.189996/.488580、5/5非零。不同随机前向joint160 cosine均值.463282，但默认类别仅10/900不一致，不能把原始特征波动当DG失败原因。完整[源结果](../automation_reports/CV-SincNet/20261015-phase1-source-dropout-consistency-manysig-s2026092701-r01/final_analysis.md)。PASS只允许另登记02/03源复核，未自动训练，不重复另一分支的元重加权/反馈教师。


2026-10-15 R-Drop源复核完成：原样02/03各自SIG_TIME父与首模型使用相同900有序物理L、RNG和alpha1，0模型/头/optimizer更新。三模型原规则均PASS，t1/t2/SIG聚合KL对CE余弦分别(.200427,.189996,.488580)、(.852692,.819312,.508874)、(.452480,.513207,.390470)；KL/CE导数范数比1.28至1.95。新增5400前向/40层80向量、合并8100前向/60层120向量独立复算，实际唯一物理样本仍900。代码`b0903780256454eaebcbd1f7da66af359a39712e`、执行`9bf37c2edf97375167b8d891c698838a1061fecc`，[完整三模型源结果](../automation_reports/CV-SincNet/20261015-phase1-source-dropout-consistency-manysig-s0203-m2-r01/final_analysis.md)。只证明末轮L导数兼容，不证明训练/U/DG；下一步实现同双L前向平均CE对照vsalpha1 KL的正式scratch50研究，原SIG_TIME、默认联合头、旧门控/ownEMA/U损失/cos保持，尚未发布训练。84%仍未完成。


后续路线去重：只读核实另一对话已推进固定E20的R-Drop；其cleanL/LS0/排除RX1契约与本轮不同，不能合并，但本对话仍避免重复训练机制。三模型源PASS保留、不再发布自有R-Drop训练；已有VAT历史矩阵也复用不重跑。改为固定原SIG_TIME和默认头的V-REx源检验：五源RX的CE风险population variance，对比同前向均衡CE，不是GroupDRO权重或Fishr梯度方差。L900/900前向/0更新；本地14负例及实际骨干导数PASS，唯一P0/P1审查中，实际源资格未知。


2026-10-15 风险方差首源检验实际完成：`20261015-phase1-source-risk-variance-manysig-s2026092701-r01`，代码`a9b04efa1441fce2f0a2810e4e51c88b41532dde`、执行`3c340073b43ec07d0dbf11fe66b8737ae2d0dc49`。900唯一L/900前向/20参数层40向量，独立复算VERIFIED，0模型/头/optimizer更新或U/query。固定五RX风险population variance对t1/t2/SIG的CE方向余弦.553474/.586210/.585349，范数比.084406/.077715/.086460，均5/5非零，原规则PASS。均衡CE.096936944、方差.000494815，默认正确896/900；末轮L局部证据不证明早期/SSL/DG。下一步仅原样02/03零更新源复核，未发布正式训练。[完整源结果](../automation_reports/CV-SincNet/20261015-phase1-source-risk-variance-manysig-s2026092701-r01/final_analysis.md)。


2026-10-15 风险方差三模型源复核完成，原规则3/3PASS。02/03 t1/t2/SIG的均衡CE导数余弦分别(.906626,.881356,.753361)、(.830733,.866633,.713499)，首模型(.553474,.586210,.585349)保留不重测；范数比全体3.69%至8.65%。新增1800前向/40层80向量，合并2700前向/60层120向量独立复算，同900唯一物理L，0模型/头/optimizer更新或U/query。代码`c514c20a11c88f35b0154eb2a15ede7a4bc992fe`、执行`e06ce7bc4957efafe4b92f4d5d7dddf180c0ae0d`，[三模型源结果](../automation_reports/CV-SincNet/20261015-phase1-source-risk-variance-manysig-s0203-m2-r01/final_analysis.md)。只支持另行实现单seed公平训练，不证明早期或DG；下一步固定SIG_TIME/默认联合头、旧门控ownEMA/U/cos，均衡RX CE对照vs系数1方差，从零50轮。


2026-10-15 新训练路线已完成最小实现及预登记：`20261015-phase1-receiver-vrex-oldgate-cos-e50-clean-manysig-s01-m3-r01`，代码`dcf5d36f1af3a1f29796b282c28fb3c6158ff34b`（核心`6553ed11ac2513a5d4a3622481138139c3195ddf`）。三冻结模型源PASS基础上，两新模型scratch50/33L+17U，同接收机均衡CE对照vs系数1风险方差，原SIG_TIME/默认联合头/旧门控ownEMA/U/cos保持，不重复另一对话R-Drop。16负例、实际native hook、两初始state及各1次合成joint更新/zeroW不改zPASS；唯一P0/P1的build接口/FP32复算容差已定点修复。新100轮/22200更新、无新增训练模型前向或参数，自动cleantruth-last；当前尚未发布、无新性能。[最小报告](../automation_reports/CV-SincNet/20261015-phase1-receiver-vrex-oldgate-cos-e50-clean-manysig-s01-m3-r01/report.md)。

2026-10-15 发布恢复：r01 已通过远端编译/真实 checkpoint no-query smoke，但公开 PROJECT 路径未解析，调度在启动任何 worker 前退出。原产物保留，0训练/预测；旧模板错误继承的 DRO completion/result 已更正。恢复代码 `1bcaa4919b835601c01cc4095ab8ed869c2e4b9b` 仅对登记 config 解析 PROJECT，科学差异仍严格比较，定向复现及独立定点审查 PASS。独立 r02 `20261015-phase1-receiver-vrex-oldgate-cos-e50-clean-manysig-s01-m3-r02` 保持原矩阵、预算、源资格和科学配方，当前待发布。

2026-10-15 r02 实际启动 VERIFIED：执行 `b7987055f7ad9e95f5fe65f2d368ce528dbac017`，两个新模型在 GPU7/1 均完成首轮，日志增长且无失败；真实 checkpoint no-query smoke PASS、全owner每卡≤3，历史参考已评分关闭。首轮各约49.8秒，后续预算不变；源训练资格/新目标性能待完成，84% 未达成。

2026-10-15 V-REx在训23/27轮（本次最新核实），前4轮1776步L风险公式/native消费及首轮104层312向量已独立复算，方差项确实进入t1/t2/SIG。早期源损失/最差RX/方差变化不作为晋级或DG证据。控制39个body梯度重复GPU反传存在约2.6e-7相对L2舍入差，原bitwise复算会误拒绝；仅离线修复保留原allclose，已通过真实向量/260负例及定点P0/P1，并独立部署提交`0416c8c4f7865b7bb879f9e1dba69a95dde9878d`。健康训练不热改，冻结后单独analysis_recovery_r01收尾；84%仍未完成。

2026-10-15 V-REx首种子矩阵已冻结/逐行评分；修复分析完整22200步/416层1248向量通过。源方差比0.812076、最差源RX训练CE比0.955575、同初始L720特征差2.696216，原规则PASS。新目标数值读取前冻结原样02/03两臂scratch50决定（4模型/44400新更新），未以源风险当DG，未启动扩展。目标汇总待读，84%仍未完成。

2026-10-15 V-REx首seed完整结果：83.2851%，匹配均衡对照83.6405%，差−0.3554个百分点；修正1127/引入1724，净−597。源风险降低/实际f_theta改变未转换为该seedDG增益。三row504000决策/54指标及完整源记录复算通过。读分前4056e9fb6已冻结原样02/03，因此继续既定4模型，不以此负分取消或调参；84%三seed均值尚未完成。[首种子完整分析](../automation_reports/CV-SincNet/20261015-phase1-receiver-vrex-oldgate-cos-e50-clean-manysig-s01-m3-r02/final_analysis.md)。

2026-10-15 原样02/03适配已本地验证/登记：20261015-phase1-receiver-vrex-oldgate-cos-e50-clean-manysig-s0203-m6-r01，代码8ce0a02e4b0c56d1aaeda0171b7c08813d4a7811。4新模型/6rows，原生上下文只变seed/runID/candidate/output，同seed两臂相同、跨seed不同；原核心16＋适配11负例、PROJECT展开及定向P0/P1 PASS。仍执行读分前4056e9fb6决定，尚待发布，不因首seed负差改变矩阵。

### V-REx 剩余种子已发布（2026-10-15）

实现和预登记已上传，执行提交 `f7aded2fc5c607064972991d07ab6fca227cd626`。4 个新模型已通过真实 checkpoint 无查询检查并分别在空闲显卡启动，已独立核实实际种子、运行目录、命令、首轮增长；2 个历史对照复用。固定 50 epoch、33+17 阶段、默认头联合训练及旧门控均不改变。源域扩种子决定已先于目标结果冻结，首种子负结果不触发调参或选择性撤销。完整三种子结果尚未产生。

### V-REx 完整三种子结果（2026-10-15）

4 个剩余模型均完成 50 epoch、默认头预测与独立 truth-last 评分。合并首种子，共 9 行、1,512,000 条 clean 固定预测和 162 条指标记录，无缺项。旧门控 + cos 的三种子均值为 80.5268% ± 2.3776 个百分点；接收机均衡 SIG_TIME 对照为 82.7532% ± 1.6896 个百分点；V-REx 为 82.4778% ± 1.8898 个百分点。V-REx − 同配方对照为 −0.2754 ± 0.2603 个百分点，正收益 1/3；距 84% 均值目标仍差 1.5222 个百分点。

三个种子的源域方差比值分别为 0.8121、0.7971、0.8029，最差源 RX 训练 CE 也均下降。机制确实进入浅层、中层和 SIG_TIME 投影，六个新模型的实际 f_theta 更新和特征导出已核实；这没有转化为稳定目标收益。本轮作为完整负结果保留，不晋级该候选，不根据目标结果调系数或选择性重跑。完整曲线、逐 RX/day/TX 指标、错误转移、成本和默认头边界见 `automation_reports/CV-SincNet/20261015-phase1-receiver-vrex-oldgate-cos-e50-clean-manysig-s0203-m6-r01/final_analysis.md` 与 `results.json`。

### 下一项独立源检验：骨干梯度共识

本轮只检验 AND-mask 启发的跨源 RX 梯度共识，保持 SIG_TIME、默认头和全部输入协议。源端已有梯度方差分解主要体现径向尺度变化，尚不能说明哪些坐标在各 RX 共同指向身份损失下降；因此直接检查完整 195,041 个 f_theta 参数，而不是再叠加风险方差损失。现有 U 特征一致性及 SAM 路线保留原记录，本轮不重复，不叠加其他核心机制；另一对话的 RDrop 和随机差分交换也不复用为本候选。

最小源诊断：原预定义 SIG_TIME final50 控制权重，固定同 900L／五个 180 包 cohort，每包单次 .015 strong IQ + 原 dropout 前向；原 CE、跨 RX 符号过滤与逐参数同 eligible/keep count 随机过滤比较。只排除具体 class_W，保留归一化、增益、参考响应。0 优化更新、0 头拟合、0 U 或目标前向。先测01；只有固定源规则 PASS 才自动测02/03，后两行不按各自结果选择性取消。即使源 PASS，也只允许另登记训练研究，不等于 DG 或 84% 达标。

当前代码局部检查通过：13 个协议负例、实际随机初始化网络的完整特征范围、六次导数、默认 W 独立、逐参数随机数量匹配、RNG 和参数／buffer 不变。一次限定 P0/P1 独立审查 PASS。正式源实验尚未发布，没有新的性能或真实源结果。来源：[AND-mask 原论文](https://arxiv.org/abs/2009.00329)。

跨 RX 梯度方向共识源诊断已完成（20261015-phase1-source-rx-consensus-manysig-m3-r01）：900 L、30 次导数、独立复算 VERIFIED。固定源规则 FAIL：聚合最差 RX 余弦 −0.02687565，逐参数同数量随机过滤 +0.01730920；保留 77.5100% 梯度范数。02/03 按预登记跳过，不进入完整训练、不调阈值。默认头和模型状态未更新，未访问目标数据。详见该行 final_analysis.md／results.json。

下一训练机制已实现：RX_CAGRAD 直接使用完整 f_theta 的五接收机梯度 Gram，在原 CE 方向周围半径 0.5 倍梯度范数的范围内协调最差局部下降；同范数原方向对照以区分方向和步长作用。默认 W 原 CE 梯度及原 U 原样保持，SIG_TIME、全局裁剪、AdamW、EMA 和 33L+17U 不变。首种子两新模型各从零50轮，旧门控+cos50历史对照复用；尚未发布，无新性能结论。18负例、完整195041梯度公式/同范数/原生hook/实际joint160更新均PASS，同范数判据100组直接检查PASS，一次限定P0/P1审查PASS。此为 CAGrad 原理向源RX的改编，原样AdamW下不宣称论文收敛或未见域保证。

RX_CAGRAD首种子已真实发布（执行 22de3258841f6068688885ba99d8ceeb866be17c），两个新模型在 GPU4/5 运行，最新原生epoch {"OLD_GATE_COS50-s2026092701":50,"NORM_CONTROL-s2026092701":4,"RX_CAGRAD-s2026092701":3}，默认头/原33+17与旧门控不变。真实checkpoint烟测、PID/CWD/argv/GPU/日志增长已核实；首轮104层416向量独立复算VERIFIED，默认W校正严格为0。源资格和新目标结果未知；不把843步源前缀和0求解回退当DG或84%证据。

CAGrad条件扩种子适配已完成局部验证：四个02/03实际native上下文只改变seed与运行标识，同seed初始权重相同、跨seed不同；L/U和默认W路径、11适配负例及portable登记展开PASS，核心检查不重复。未读取首新目标汇总，仍未登记或发布扩展，源决定待50轮完整资格；首种子健康训练保持。
