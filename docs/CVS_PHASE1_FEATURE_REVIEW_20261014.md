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

当前最多两项：实现源证据驱动的Sinc坐标候选及其公平控制；补合法源代理门控诊断。当前TIME实现/配置/完整正负结果集中交付，其他方向排队，不无限扫描直到84%。主目标仍active，未宣称达成。

完整当前 [同轮分析](../automation_reports/CV-SincNet/20261014-phase1-time-factorial-oldgate-cos-e50-clean-manysig-m15-r01/final_analysis.md) 与 [机器台账](research_data/CVS_PHASE1_FEATURE_REVIEW_20261014.json)。历史当前运行身份、代码/来源/证据层级在台账中；大IQ、checkpoint、原NPZ/日志留原目录。默认head及clean范围来自用户最新明确要求；历史换头/LEO结果不删改，当前不借clean声明部署。

## Sinc 坐标探索：源首种子已启动

新 run `20261014-phase1-sinc-coordinate-oldgate-cos-e50-clean-manysig-m9-r01` 已固定 RAW_HZ、CENTER_25K 及历史 OLD_GATE_COS50 三组 × 三种子。6 个新模型各从零 50 轮、33＋17、旧门控＋EMA＋cos；原时序/PA/频域/参考结构和默认联合训练头不改。两坐标方案同初始滤波器、joint160、logits、其他权重和构建随机状态，48 个 Sinc 参数与 optimizer/EMA 路径检查通过。每模型增加 48 个固定 origin buffer 元素，不能将完整坐标方案中的权重衰减/EMA 算术变化冒充纯学习率效果。

首种子仅按两组 source final labeledCE≤.5 和 CENTER 实际核相对 L2≥1e−4 放行其余种子；每个合法训练完成 row 自动默认头 clean prediction 和独立 truth-last，目标分数不进入放行。合成整网、缓存/strict load、W 独立导出及源健康 RNG 检查通过，唯一 P0/P1 启动审查 PASS。实际执行 0f81373c25ba69a6cb9deec5c7ca652880fd62af，归档/compile/真实无查询 smoke 与 PID/CWD/argv/GPU/log-growth 已 VERIFIED；首2轮记录显示 RAW 核相对 L2 约1.17e−8、CENTER 约0.0015867。此为中间源机制记录，尚无新模型终轮性能或完整三种子结果，不能称性能改善。完整预算、来源、阈值、代码提交身份和执行路径见 [Sinc 预登记](../automation_reports/CV-SincNet/20261014-phase1-sinc-coordinate-oldgate-cos-e50-clean-manysig-m9-r01/report.md)。
