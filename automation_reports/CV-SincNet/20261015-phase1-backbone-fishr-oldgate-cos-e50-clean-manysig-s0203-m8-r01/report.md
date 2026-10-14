# 真实骨干梯度方差：固定02/03条件扩展

状态 LOCAL_VERIFIED / NOT_LAUNCHED，代码 b69d6ffbbd229284c4cfde35124db10bf871fe07。这不是新机制；首种子仍在健康训练，只有原预登记两候选均SOURCEPASS后才发布。没有source资格通过、新目标结果或完整三种子性能的声明。

原始run 20261014-phase1-backbone-fishr-oldgate-cos-e50-clean-manysig-s01-m4-r01 / execution600afeeefe84e0d9f50f07c38ae0767bf1cabf07 的SIGZERO、RAW、COND、旧门控+cos各保留。新run 20261015-phase1-backbone-fishr-oldgate-cos-e50-clean-manysig-s0203-m8-r01，固定模型seed2026092702/03，各4row，共8行，其中6新scratch50/11100、2历史baseline复用。首seed不重训，RAW和COND一并原样扩，不按目标排名、换父或改参数。

科学配方完全沿用：固定SIG112→joint160/fθ195041，默认cosine6×160 scale30正常联合训练；33label+17pseudo、原oldgate/ownEMA/cos、λUCE.16/entropy.01/L smoothing.01、NoExplicitCFO及原内部补偿。L6300/U56700隐藏TX/V27000、split392005/sourceRX1/3/4/6/8/days1/2/3/6TX/physicalIDs不变，V无梯度或拟合，不新增星地训练。

真实L180 cohort每TX/day/RX2包，各RX36包；每轮13次、650/model，原cohortseed2026210712和35组轮转保持。实际t1PW3456/t2PW6912逐包CE参数导数、RX方差n−1=35、RAW或TX/day去均值、可微归一化分母floor1e-30、两层均值、系数.01、5预热+10ramp不变。第三SIG投影仅诊断，不变成资格门槛。aux图中当前W常量，默认头原L/U联合更新保持，无新头或参数。

源码仅增加薄的固定seed执行/汇总封装，并让已有完整审计接受design/build参数，默认调用含义不变；已发布immutable archive与健康进程不修改。复用原核心一次独立P0/P1审查，不再审查未变训练核心。

## 检查、源资格和成本

六个新模型上下文实际构建与合成无query前向PASS；每seed三臂初态参数逐值相同，原生args差异仅seed/run_id/candidate_id/output_dir，原方法字典逐项一致；10个source资格拒绝条件PASS。两个历史NaN元数据字段使用NaN-aware比较，不改训练值。没有读真实数据、目标或checkpoint，没有优化器/头更新。编译及发布脚本编译PASS，真实checkpoint no-query smoke随实际发布自动执行。

发布前检查原source_qualification.json、source_features.json和first固定row完整计量状态，不读scores或预测值。必须匹配原源规则：三新终轮LCE≤.5、两候选实际L720特征差≥1e-6、t1/t2在17/33/50每期有非零effective梯度、精确650extra/52gradientcohorts/11100updates及默认特征边界，且两候选均PASS。原资格回收与决定提交先于新numeric target摘要；不新增seal、receipt或审批，不改变规则。

本run计划6新300model-epochs/66600更新、3900extraL180/702000额外源包、936层cohort/624逐包梯度矩阵/3744保存向量；历史2×50/11100成本另列。完整01/02/03计划9新450epochs/99900更新＋3历史150epochs/33300更新，12模型2016000默认决策/216组指标，不把计划数写成已完成数。所有原始PT/权重/日志/预测留远端。

## 执行与自动收尾

普通N607既有CVS-RFFI/fullFP32/每worker2CPUthreads；唯一owner codex/root/backbone-fishr-seeds-20261015。CWD PROJECT/releases/cvs_backbone_fishr_seeds_20261015_r01，run PROJECT/runs/20261015-phase1-backbone-fishr-oldgate-cos-e50-clean-manysig-s0203-m8-r01，各row source/predictions/scores与logs独占。远端入口 python -u -m experiments.cvs_backbone_fishr_seeds.dispatch；本地发布 experiments.cvs_backbone_fishr_seeds.publish --output local_artifacts/backbone_fishr_seeds_delivery_r01。全命令、矩阵、路径在experiment.json。

只有首种子source条件满足才执行普通账户标准preflight、一次release归档和本地/远端SHA比较、远端compile/真实checkpoint no-query smoke，PASS自动继续。共享锁按全部owner每GPU≤3实验进程、启动前空闲≥12000MiB，实际GPU由launch记录；无其他进程修改。

各row final50冻结后自动clean168000原默认头预测→prediction固定→独立truth-last评分→同row报告与登记/Git，首seed产物直接复用，合并固定三seed配对均值/样本SD/正差数、MacroF1、最差RX/day、纠错/引入错误、完整源曲线和实际梯度/feature边界。84%只认预定义三种子原头口径，不以最高secondary或其他读出替代。

停止规则沿用所属run系统技术失败：数据/query/来源/defaulthead权限、非有限/跳过更新、错误预算/路径冲突/无法执行或合法预测/scorer故障；先核对run所属PID/CWD/argv，保留partial，无自动重试，不按分数停止、重启或选择性重跑，不触碰健康任务。当前未发布，本轮只提交可复用执行配置和局部证据。


技术预防修复 729c01e878a0df790c1056d00d1b547f709a57be：同一汇总依赖改用发布包已有且AST相同的signature_stats_seeds helper，preflight导入dispatcher确认真实链可用。科学配置未变，仍未发布，原source条件待。


## 首 seed 源决定已回收，目标数值尚未读回

首 run 20261014-phase1-backbone-fishr-oldgate-cos-e50-clean-manysig-s01-m4-r01 的四行已闭环。原 source 资格 PASS：终 LCE ZERO=.2472100322、RAW=.2489144201、TX_DAY_CENTERED=.2482162956；候选−ZERO 合法 L720 zL2 分别1.6659400463/1.4148471355；两个实际 t1/t2 对象在17/33/50有效梯度均非零；每新11100更新/650额外L180/117000包/52诊断cohort完整。当前只读源资格与闭环计量，未回收新目标准确率。按原计划两候选均原样扩固定02/03，不重排、改权重或重跑；源资格不等于DG收益。技术依赖修复代码 729c01e878a0df790c1056d00d1b547f709a57be，原算法/矩阵不变。详情first_source_decision.json。
