# SIG_TIME可信加权：剩余两个种子条件式预登记

首种子 20261014-phase1-trusted-weight-oldgate-cos-e50-clean-manysig-s01-m3-r01 健康训练中，源资格尚未完成。本登记只准备2026092702/03；发布入口必须读取固定实际执行7708d4afcef141bb614e0c2171f2a9207c4469ff的首run source_qualification/source_features/completion/experiment，原源阈值、默认头特征边界和实际梯度预算全部PASS且3row收尾完整才继续。入口不读取target analysis/metrics决定扩展；source FAIL保留首结果，不改weight/scale/seed或重跑。首seed不重训。

固定每seed历史OLD_GATE_COS50＋新SIG_TIME uniform＋新TRUSTED_WEIGHT，共6row/4新。与首seed结构、门控、数据、损失、50轮及33＋17安排完全一致，只改预定义modelseed/run/output owner。raw stopgrad(.75+.5×clip((1−p_student,strong(teacher伪类别))/.1,0,1))，raw[.75,1.25]/effective理论[.6,1.6667]，只在原已接受集合中按权重和归一化CE；lambdaU.16/全Uentropy.01保持，ownEMA.999/cos不变。SIG_TIME t2完整postDropout特征保持，不叠加层级/PRE/Sinc/新卷积/SiLU。默认cos6×160scale30随骨干原流程联合，无冻结、重训或另训头/身份/RX/day/channel诊断分类器。

复用合法roles/split392005/VALIDATED_ONCE、L6300/U56700隐藏TX/V27000health-only、sourceRX1/3/4/6/8/day1/2/3/旧6TX及physicalIDs，NoExplicitCFO及内部补偿不变；无新信道增强/源或目标权重继承、target调参。每新模型真实33/34/50轮全部222个原训练批记录t1/t2/t3的L/UniformU/WeightedU/Entropy参数梯度，666batch/1998层批，0额外IQforward/RNG抽样，正常backward/optimizer/EMA/默认头联合训练随后执行；保存三轮私有vectors后复算。33轮U inactive为0。源V仅健康，终轮固定。

源条件严格复用首登记：两新finalLCE≤.5、pool真实更新/导出差0/W独立、初始全部参数与源z相同、最终candidate−controlL720zL2≥1e−6、实际1998梯度记录及预算/默认头/无Utruth核实。源PASS不是性能PASS；当前首source尚pending，不把REGISTERED当RUNNING。完整3seed目标只取预登记TRUSTED_WEIGHT均值≥84%，剩余2seed或最高单seed不得替代。

成本：4×50=200新增model-epochs/44400updates＋2历史100epochs/22200updates单列复用；加首seed6新300epochs/66600updates＋3历史150epochs/33300updates，0额外头fit。每row clean168000固定预测→独立truthlast，剩余1008000决定/108metrics；合并9row1512000决定/162metrics、6新真实11988层批梯度记录。完整逐seed/paired均值sampleSD/正差seed/CM/MacroF1/最差RXday/错误修正引入/源曲线及负结果保存。已暴露clean旧TX基准，不称新TX、Phase2、盲测或卫星部署。

实现 1859aa924e08210e7d23193d908f8e02c4fb2024，核心候选 23ea591bba17dd938d2b9a93f8b9d9dc5f02e97a 及其一次独立直接P0/P1 PASS复用，无重复审查。已完成4实际native上下文、同seed初态全部参数/z完全相同、权重配置反例、10源资格反例及剩余observer配置绑定/scope恢复、12py compile PASS；本地不读真实IQ或目标。封装只绑定原observer设计命名空间，不改核心公式/训练实现，不热改当前远端release。局部检查不冒充真实训练。

入口python -u -m experiments.cvs_trusted_weight_seeds.dispatch；普通N607既有CVS-RFFI fullFP32/2threads；CWD /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_trusted_weight_seeds_20261014_r01；immutable output /home/szu2070436088/2510044040/CV-SincNet/runs/20261014-phase1-trusted-weight-oldgate-cos-e50-clean-manysig-s0203-m6-r01；owner codex/root/trusted-weight-seeds-20261014。现有capacity锁计入所有owner，GPU0至7自动分配、每GPU≤3总实验进程/free≥12000MiB；保护健康任务。sourcePASS后才一个releasearchive单次SHA比较/remotecompile/一个真实同seedhistorical CP2zeroIQ noquery smoke PASS直接继续，核对实际PID/CWD/argv/GPU/loggrowth。

Stop：复用角色/provenance/初始化/非有限/shape/默认头/门控/损失对齐/预算/保存梯度复算/export/预测truthlast系统技术失败规则，停止新增dispatch、保留健康row和产物，无自动retry/hotedit/目标早停。预期source checkpoints/full50epoch/fullstep/trustedweightsteps/三轮rawgrad+records/source witness/freeze、fixedpred/scores/rowcompletion、sourcefeatures/analysis/combined3seed/launch/queue/completion及独立固定产物复算。Git仅实现/config/聚合结果与报告；原IQ/weights/gradarrays/predictions/大日志/私密状态不入Git。当前REGISTERED，未发布。


首固定执行7708d4afcef141bb614e0c2171f2a9207c4469ff的源资格已只读核实PASS，读取target summary/metrics数值前冻结seed_expansion_decision.json。两新finalLCE .247590639/.247829630，candidate−control源L720zL2 1.736685634；每模型真实11100updates/3774Usteps/963900U包及33/34/50共1998层批记录保存复算VERIFIED，额外headfit0，actualexport/W独立及matchedinitial通过。sourcePASS不等于性能PASS，首target数字尚未读取。发布的source qualification检查放入远端preflight，复用既有正确N607 Linux文件读取路径；当前仍未发布剩余训练。
