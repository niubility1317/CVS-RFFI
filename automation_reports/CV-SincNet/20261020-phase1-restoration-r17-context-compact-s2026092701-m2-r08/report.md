# 同时压缩粗恢复与信道估计的 R17 精简 R08

状态：RUNNING。用户要求最大程度精简R17及多GPU批量快速迭代。本轮在R06/R07六行仍训练或预测、未读其新目标准确率时预先固定。动机为纯结构盘点：原coarse1305958/action851589合计2157547参数，占最小R07完整2352389参数约91.7%，仅压prior已接近这条路线的结构下限。CPU半秩分解context长卷积后context为1397619；此计数不证明身份性能。

|固定行|完整参数|比原4665073减少|冻结context|可训练prior|GPU|
|---|---:|---:|---:|---:|---:|
|context_half_prior_quarter|1822977|60.9229%|1397619|425358|4|
|context_half_prior_eighth_pointwise|1592461|65.8642%|1397619|194842|5|

只从原source-selected R17/cd977448/E20/4940初始化与蒸馏，无R04/R05/R06/R07学生。原source-only粗恢复和action估计的84个长Conv1d独立CPU半秩平衡SVD，保留通道数、bias/GN/1x1、谱变换/物理算子/nativevariance范围；分解后context完全冻结。prior为原三步长权重均值后quarter，或eighth并共享原1x1权重；保留各步bias/GN。0初始化IQ/TX/F/V/target/优化更新，CPU分解时间另报。context是原source权重的确定性近似，不再宣称context逐位相同；训练来源与物理数据契约不变。

关键差异：原R17教师必须用自身原coarse/action/nativevariance，学生用自身压缩context。teacher_and_student_context先计算原teacher上下文与原三步目标，再计算student上下文；禁止用student近似物理估计生成teacher目标。源训练、真实smoke均调用同一helper，sentinel负向测试区分两个上下文。仍只优化prior，coarse/action/teacher/F50/defaulthead参数0更新。原3步×每步2CG不变，恢复推理仅当前IQ，0TX/RX/场景/clean/教师/F运行时输入或状态更新。

固定训练同R04/R06/R07：R07原63000=L6300/U56700缓存epochs1..20、15%clean/85%residual，batch256/model2701/data2026102001；AdamW2e-4/warm247/cos2e-5/WD1e-4/clip1/FP32/noTF32，E20/4940。损失10×三步IQNMSE_to原R17 + 冻结原F50/defaulthead T1 KL + .5×unitF160距离平方，task系数1；clean仅detached能量分母floor1e-4/日志。每行L126000/U1134000，原R17/Fteacher/Fstudent各1260000输入；相较此前共享同一context的版本，本轮teacher context和student context各独立前向1260000次，不能忽略额外教师计算。prior继承20+新20=40，coarse20/action20共80继承串行pass，context分解0优化；原F50预训练50/11100、F输入反传及CPU初始化成本分列。使用源监督F知识，不称F-free无监督。

两个候选/秩/预算在新结果前固定，不根据source或target分数筛选/早停/重排/改参数/重跑。每行E20冻结后自动source七视图诊断/profile/独立truth-last，再同168000物理样本×clean与六practical完整target预测，七视图IQ/logit/feature/IDs全部固定后独立CPU才读truth/clean。相对原R17逐场景准确率下降<=1个百分点，以最多损失1680条正确数判断；0pp另诊断。报告accuracy/MacroF1、TX/RX/day/confusion、raw repair/damage/net、full256/prefix232/tail24波形/FFT与配对clean特征差距、完整成本。low urban≥70%及其他环境接近clean的长期目标另报，未自设“接近”阈值；不称新盲测/全局最简/Phase2/在轨保证。

35项合成/协议检查PASS，覆盖两结构完整参数、context冻结、共享alias、原teacher保持、独立context、strictreload、源预算/权限、七视图truth-last及失败隔离。唯一独立P0/P1审查后固定code/config commit。launcher第一步真实原R17/F50/2Ls CPU noquery smoke：teacher阶段输出必须逐位等于原R17自身forward；student prior/F输入梯度有限非零、context/teacher/F/head冻结，checkpoint严格重载逐位一致；0TX/V/target/optimizer更新。PASS立即排队，无批准token。

GPU4/5排队：等待R06根complete/failure终态且其实际ownedGPU句柄退出，仅读PID/命令和终态存在性，不读评分；不修改、停止或重启当前健康六行。之后每卡一个own worker、sourcefree≥18GB/allowner<2，R07健康2/3/6/7继续，保持最多六条own GPU。CPU2线程，所有R17候选共用R06score.lock、最多一个own scorer。一次普通N607预检/新不可覆盖root与release/单archive SHA比较/远端compile/实际PID-CWD-cmd-GPU-loggrowth。契约/nonfinite/OOM/技术错误停止失败行并保留，健康行继续，无自动重试/回退；性能失败不作重启理由。source profile排除GPU checkpointpayload/optimizer，报告resident和增量peak，仍有coowner/顺序限制。命令、环境/CWD、路径与产物见experiment.json。

发布前证据：35项合成/协议检查和唯一独立P0/P1审查PASS，独立复跑35项，未访问真实IQ/权重/远端；检查教师与学生独立context、压缩context冻结/复载、R06实际句柄等待、共享score.lock及自动目标链路。普通N607预检VERIFIED，R06实际GPU句柄仍2条，新root/release不存在；只提交CPU smoke和等待队列，GPU释放后再启动，保留六条健康任务。

实际发布 VERIFIED：方法commit `2e23af45f4a5fc26c3d44c4efa4305ac36ef45db` 已推送且远端OID独立比较一致；单归档SHA比较/远端compile/help通过。实际原R17/F50/2Ls noquery smoke PASS，teacher阶段逐位等于其自身原context前向，student compressed-context/prior梯度和F输入梯度有限非零，context/teacher/F/head0更新、严格复载逐位输出一致。先等待R06目标测试和评分结束、原实际controller及GPU句柄退出，再在GPU4/5各启动一个新worker；实际PID/CWD/完整命令/commit/GPU/loggrowth与首轮完成均核实，R07健康四卡继续。未读取R06结果改变R08设计/训练/矩阵。

完整汇总工具9项合成检查PASS，包括context_parameters1397619、原teacher context与student context各1260000前向计费、prior-only预算、两固定row/全七视图/分组混淆矩阵、独立assessment与整数1pp重算一致、payload-free推理profile。该工具未改运行中的方法commit；当前无R08完整训练或目标性能结果。
