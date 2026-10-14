# 固定 SIG_TIME：教师学生特征一致性训练预登记

状态 RUNNING，代码 373262b51f037f03e3238471b3aefb88e45fba5b，分支 codex/phase1-feature-challenge-20261013。一个独立SSL假设，不叠加运行中的BF/PAIR/签名排序。一次P0/P1审查与实际网络检查PASS；没有新目标结果。

## 依据、方法与固定矩阵

三模型源诊断20261014-phase1-source-sigtime-pseudo-gradient-manysig-m3-r01已完成：L接受集合无错使错误梯度不可识别；V错误组特征cos没有稳定更低，不支持可靠度阈值。正确组weak/strong joint160 cosine约.66至.67，类别预测一致仍可伴随特征位移；这只支持可检验假设，未证明一致性训练有用。

一般一致性思想参考[Mean Teacher原论文](https://arxiv.org/abs/1703.01780)与[作者实现](https://github.com/CuriousAI/mean-teacher)。原方法匹配输出概率，本轮直接约束现有joint160，是项目特征级变体，无新projector/predictor，不称逐项复现或新理论保证。参数/父权重/候选不依据目标分数。

Run 20261014-phase1-feature-mean-teacher-oldgate-cos-e50-clean-manysig-s01-m3-r01，第一步seed2026092701：历史OLD_GATE_COS50同33+17/final50/11100复用；SIG_TIME ZERO与FEATURE_MT两新臂scratch各50轮。固定SIG112→实际joint160/id_feat_joint/fθ195041，默认cosine6×160 scale30原联合训练，无新模块/参数/头，无单独头训练、替换或RX诊断头。NoExplicitCFO及内部补偿保持。

原batch_neighbor门控、ownEMA.999、λUCE=.16/λentropy=.01、L smoothing=.01、cos日程、L128/U256、33label+17pseudo/11100更新全部不变。合法源L6300/U56700隐藏TX/V27000、split392005/RX1/3/4/6/8/days1/2/3/6TX/physicalIDs不变；V仅健康验证，无梯度/EMA/拟合，不新增星地训练或数据重验hash链。

原有真实U EMAweak与学生strong(.015Gaussian+原dropout)已经前向；新增默认None观测接口只传两者现有joint160，训练从零教师为自身EMA。新增loss=全批mean half squared L2 distance between unit student and stopgrad unit teacher features，normalize eps1e-8，尾批124保留。同一物理包自配对，不使用TX/RX/day/pseudo/confidence，不跨样本配对或设新tau。

FEATURE_MT权重固定.1×clip((epoch−33)/5,0,1)：34开始，38到.1；此前不取U。ZERO同计算权重0。原RANDOM U顺序/RNG、原UCE对象及其所有门控保持，aux经既有_source_feature_auxiliary_hook加入总loss，不增加模型前向。辅助图不依赖类别W、W梯度None；原L/U默认头梯度和更新保持。实际t1/t2/SIGprojection合成检查非零更新、z导出一致/W置零不改z；最终真实fθ更新仍待证据，不把检查当收益。已发布健康BF archive不变。

## 预算、源资格与验证

两新100model-epochs/22200更新，旧50/11100另列。每新模型17×56700=963900真实U包，原生两个U视图不增加。额外模型前向0，但诊断反传/存储有成本：每模型原L/U与新feature分别33/34/50各222批=666批/1998层记录/5994向量，两模型总7992层记录/23976向量。原PT/权重/日志/预测留远端，Git只交聚合。

预登记source资格：两臂终LCE≤.5、精确预算/默认特征边界；候选−ZERO合法L720 zL2≥1e-6；候选34/50每期实际t1/t2/SIGprojection各至少一批effective梯度非零；最后5轮（46..50）按U包数加权原始feature discrepancy，FEATURE_MT/ZERO≤.95。全部满足才原样扩固定02/03，source决定回收并提交先于新numeric target摘要。不合格保留结果，不扫参/重跑；source健康和优化资格不是DG证明或U正确率。

## 执行和自动收尾

普通N607既有CVS-RFFI/fullFP32/每worker2CPUthreads；CWD PROJECT/releases/cvs_feature_mean_teacher50_20261014_r01，唯一owner codex/root/feature-mean-teacher50-20261014；不可变run PROJECT/runs/20261014-phase1-feature-mean-teacher-oldgate-cos-e50-clean-manysig-s01-m3-r01，各row source/predictions/scores及logs独占。入口 python -u -m experiments.cvs_feature_mean_teacher50.dispatch；本地发布 experiments.cvs_feature_mean_teacher50.publish --output local_artifacts/feature_mean_teacher50_delivery_r01。全部命令、配置和路径在experiment.json。

全部owner共享容量锁每GPU≤3实验进程、分配前空闲≥12000MiB，实际GPU由launch.json记录；保护健康他人任务。2026-10-14 23:48CST标准只读N607预检PASS。一次归档SHA比较、远端编译、真实预定源checkpoint的合成IQ no-query smoke PASS后自动继续；smoke权重不进入scratch训练。不增加seal/签名/额外设计审批/重复审查。

逐row final50后自动clean168000默认头预测→prediction固定→独立truth-last评分→报告/登记。3row504000决定/54组指标；逐seed配对、MacroF1、最差RX/day、纠错/新增错误、完整源曲线和梯度/feature边界。单种子不完成三种子84%；无Phase2、新TX、卫星部署或新独立盲测声明。

停止规则仅所属run数据/query/来源/默认头协议、无法执行或合法预测、非有限/跳过更新、预算/路径碰撞/scorer等系统技术失败：先核实所属PID/CWD/argv/产物，保留partial、无自动重试，不按分数停机，不改动其他owner。50轮预算不冒充收敛，完整源趋势在结果返回后分析。

本地8协议负例、同初态/RANDOM顺序RNG、真实feature/head更新、原UCE对象保持、auxW梯度None/mainW梯度不变、warmup主梯度一致、teacher停梯度、hook恢复、feature导出/W独立与发布脚本编译PASS。唯一独立P0/P1未发现直接问题。

## 实际启动证据

执行提交 6b23a5b033ee063dea97d6783d5634e220a1d93d。一次归档比较/remotecompile/真实checkpoint no-query feature smoke PASS；两新worker实际在GPU7/2启动，各PID/CWD/argv匹配、日志增长与首轮完成已VERIFIED。全部owner每GPU≤3；前33轮没有U特征项，真正半监督阶段及最终性能仍待证据，当前不宣称有效。


## 真实U阶段证据

actual_U_phase_source_evidence.json已从运行中源产物回收：两臂完成epoch34全部222梯度批；FEATURE_MT的t1/t2/SIGprojection每项222批effective梯度均非零，ZERO每项均0。逐批实际feature项无classW梯度检查通过，生产端保存PT复算VERIFIED。读取的实际原生配置仍为ownEMA=true、batch_neighbor、λUCE=.16/λentropy=.01/L smoothing=.01、50=33+17。当前已刷新至36轮，每臂观测666个U批/170100包，属于部分计数，未完成963900总预算。终轮数值参数变化、完整独立审计、源资格与目标性能尚待。
