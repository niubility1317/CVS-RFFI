# SIG_TIME 可信门控内有界权重：首种子预登记

依据用户附件报告路线四及固定源L6300权重诊断（20261014-phase1-source-trusted-weight-manysig-s01-m1-r01），实现学生strong对teacher伪类别难度的有界stopgrad权重。源代理只表明t1/t2/t3贡献变化，不能证明性能收益。保留SIG_TIME实际joint160结构，不叠加层级、PRE取点、Sinc/卷积/SiLU改动，不选目标高分checkpoint继续训练。

固定3row：历史OLD_GATE_COS50同seed默认头复用；新SIG_TIME均匀原loss；新TRUSTED_WEIGHT。两新模型相同初始参数/实际特征、pool18976/fθ195041/defaultclassW960，分别scratch50/11100更新，33label＋17pseudo，ownEMA.999、old batch_neighbor窗口2/confidence/strong一致门控、cos、lambdaU.16/entropy.01保持。默认cos6×160scale30随骨干正常联合，不冻结或单独训练/替换头，不另训身份/RX/day/channel诊断分类器。仅修改已接受U的CE权重：h=1−p_studentstrong(teacher伪类别)，raw w=stopgrad(.75+.5×clip(h/.1,0,1))；CE_U=sum(a*w*CE)/sum(a*w)，空accept0；raw界[.75,1.25]，归一化effective理论界[.6,1.6667]，不混称。scale/bounds观测及发布前已固定，无扫描。Uniform control返回native原CE tensor，门控算法不变；两个模型训练后伪标签/接受集合可随权重更新不同，不能宣称跨模型mask相同。熵项覆盖原全U且系数.01不变。

旧合法roles/split392005/VALIDATED_ONCE、L6300/U56700隐藏TX/V27000health-only、sourceRX1/3/4/6/8/day1/2/3/旧6TX/physicalIDs不变。训练池标注率10%，原native train_ratio.07分母含V保持。NoExplicitCFO显式坐标关闭、内部逐包补偿保留，无星地或新训练增强。V无梯度，仅健康；终轮固定，无V微小涨分/target调参选模。新模型无父权重/EMA继承，historical CP仅baseline比较和无query smoke。

真实训练第33/34/50轮全部222批，在原门控后、总损失组装前记录实际t1/t2/t3.pw.weight的L CE、lambdaU uniformCE、lambdaU weightedCE、lambdaEnt熵梯度；3×222=666批，1998层批记录/模型，额外2664次参数autograd.grad retain_graph，无新IQforward/RNG抽样，不累积.grad、不改变正常backward/optimizer/EMA/默认头学习。33轮U inactive显式0；真实L128/U256各自原采样，不用先前L代理冒充U。保存每个声明轮的私有梯度向量，重读复算norm/同层U/L/cos和记录数；.pt约70MB/声明轮/模型，原数组不入Git。完整11100step权重/门控接受与hardness记录，实际3774Usteps/963900U包；U真值不读取，不报告其精度。不同层对象元素3456/6912/9216，跨层norm不解释为重要性。

源扩seed规则预先固定：两新终轮LCE≤.5，新增pool实际更新、joint160导出差0/W移除独立、初始全部参数及source函数一致、最终candidate−control源L720zL2≥1e−6、两个模型实际33/34/50保存复算1998记录及预算/default头/无Utruth均核实。仅source PASS可原样扩预定义2026092702/03，target数字不控制扩展；source FAIL保留三row结果，不改weight/scale/seed或重采样。首次source资格在读取target summary数字前冻结；逐row原计划测试不等待其他训练。

预算2×50=100新增model-epochs/22200updates，1历史50epochs/11100updates单列复用；0额外头拟合。每row clean168000旧6TX/7目标RX/4day，3row504000默认决定/54组指标；固定预测后独立truthlast，全部正负指标/CM/MacroF1/最差RX/day/配对修正与引入错误及完整源曲线保留。已暴露基准，不称独立盲测、新TX、Phase2或卫星部署；首seed不代表84%三seed均值目标完成。

入口python -u -m experiments.cvs_trusted_weight50.dispatch；普通N607既有CVS-RFFI fullFP32/2threads；CWD /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_trusted_weight50_20261014_r01；immutable output /home/szu2070436088/2510044040/CV-SincNet/runs/20261014-phase1-trusted-weight-oldgate-cos-e50-clean-manysig-s01-m3-r01；owner codex/root/trusted-weight50-20261014。GPU由现有capacity锁在0至7自动分配，计入所有owner，每GPU≤3总实验进程且free≥12000MiB，禁止停止/热改/重启其他任务。发布一个archive本地/remote SHA比较/remotecompile/一个真实historical CP2zeroIQ noquery smoke PASS立即继续，独立核对PID/CWD/argv/GPU/loggrowth。

Stop：role/provenance/初始化/非有限/shape/默认头/实际oldgate/原loss对齐/预算/保存梯度复算/export/predictions或truthlast系统技术失败停止新增dispatch、保留健康row和输出，无自动retry或目标早停。预期source checkpoint/initial/defaultguard/fullstep/full50epochs/trustedweightsteps/三轮gradpt与records/source witness/sourcefreeze、逐rowfixedpred/scores/completion、矩阵source qualification/sourcefeatures/analysis/launch/queue/completion。Git仅实现/config/聚合结果与报告，不提交IQ/checkpoint/rawgrad/prediction/大日志或私密本机状态。

局部真实两native模型loss/梯度/正常joint默认头更新、pool5tensor更新、observer无参数/.grad/RNG干扰、初始全部参数及真实z相同、9负配置、33轮inactive梯度及scope恢复PASS；12新py＋native compile PASS。本地使用2zero/ones合成IQ，不读真实源/目标；局部检查不冒充实际训练。一次独立仅直接P0/P1审查PASS，无直接缺陷；实现/config 23ea591bba17dd938d2b9a93f8b9d9dc5f02e97a，尚未发布或训练。
