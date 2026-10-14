# 固定SIG_TIME半监督参照：剩余固定种子条件扩展

首run 20261014-phase1-sigtime-pair-oldgate-cos-e50-clean-manysig-s01-m3-r01（执行 86c22cb76e5eff4a7b106cca769d198879ab22b4）健康运行时准备包装，当前未发布。只有首3row完整默认头闭环且SOURCE资格PASS才原样扩02/03；先读source证据，不读取新target数字决定扩种。固定SIG_TIME195041/joint160、默认cos头原流程联合、oldgate/ownEMA/cos33+17、λU.16/λent.01、每模型scratch50/11100更新不变；只沿用RANDOMvsPAIR的U顺序差异，无新loss/额外forward或头拟合，首seed不重训。

6row：每seed旧gatecos历史＋SIG_TIME RANDOM/SIG_PAIR PAIRS，四新200model-epochs44400updates＋两历史100epochs22200updates。合首seed共9row，六新300epochs66600updates＋三历史150epochs33300，450完整源epoch/1512000固定defaultpred/162组指标/11988实际层批梯度。每new同963900U曝光/17次全遍历/源角色和physicalIDs不变。33/34/50全部222真实批L/U梯度保存复算。

源规则原样：两firstfinalLCE<=.5，PAIR整17轮Uaccepted/963900>=.25，实际featureexport/W独立/PAIR−RANDOML720zL2>=1e-6，精确steps/sampling/666gradbatches/headfit0。qualification强制核实first执行commit/原矩阵/源产物，不按目标排行。自动逐rowfinal50冻结→defaultclean168000predict→独立truthlast评分→完整3seed统计；不声称新算法/Phase2/卫星/新TX能力。

复用唯一已审核心，无新科学假设或重复核心审查。8SOURCE负例、四actualnative上下文及same-seed同初态、原SSL配置相同、observer/model design绑定、关键archive依赖/remotecompile PASS。复现入口 python -u -m experiments.cvs_sigtime_pair_seeds.dispatch，普通N607 CVS-RFFI既有环境/CWD/路径/每row命令/seed/预算/stoprule均在experiment.json。每GPU含所有owner最多3进程/free>=12000MiB，不动健康任务；不可变输出，技术失败保留、不盲重试。原始IQ/CP/pred/log/gradPT远端保留，Git只实现/必要聚合。首SOURCE仍pending，不宣称已扩种或得到目标准确率。

包装代码 e1c5f12f68881f8757efae3d9d1bf8f2618f2982；8SOURCE负例/4native/observer与modeldesign绑定/6row4new1008000pred监控计数/源码compile PASS。真实首SOURCE资格仍pending，未启动扩种。

首seed3row闭环后仅读SOURCE资格PASS，尚未回收新target数字：RANDOM/PAIR LCE0.247506658/0.248212648，PAIR整17轮accepted879408/963900=91.2344%，RANDOM15448/963900；PAIR−RANDOM实际L720zL2=1.429918528，精确11100steps/17samplerepochs/666gradientbatches/0headfit/export/W独立均PASS。按原固定02/03四新扩展，不改权重或按目标排行。
