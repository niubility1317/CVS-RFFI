# R17 精简版源 clean 直接纠正 R10

状态：RUNNING。用户要求压缩同时优化，本轮固定两种既定R08/R09结构，检验去掉模仿原R17恢复波形后，直接源clean波形/身份约束能否提高识别。原R17仅作CPU权重初始化；原F50/defaulthead冻结。目标逐场景原R17−1pp为下降上限，不能把容差通过等同提分。low urban≥70%、其他环境距clean和配对clean恢复差距独立报告，不自设“接近”的阈值。

|固定行|完整/可训练参数|比原4665073减少|GPU|波形教师权重|F参照|
|---|---:|---:|---:|---:|---|
|clean_only_quarter|1822977|60.9229%|4|0|source paired clean|
|clean_only_eighth_pointwise|1592461|65.8642%|5|0|source paired clean|

只原source-selected R17/cd977448/E20/4940初始化，无已测学生继承。84个context长卷积CPU半秩、prior quarter共享或eighth并共享原1×1，保留各步bias/GN；原结构/初始权重构造同R08/R09，三步×2CG不变。全部coarse/action/prior可训练；直接调用coarse与action estimator保留梯度，不走旧coarse no_grad路径。原R17初始化后保留CPU，训练开始即删除，optimizer只含compact参数。

固定目标：最终恢复与对应source-clean IQ的NMSE（clean mean energy detached、下限1e-4）+ KL_T1(Fclean||Frestored) + 0.5×单位160维特征距离平方。IQ、KL系数各1，feature系数0.5，无原R17输出/中间阶段模仿项。与对应R09 clean arm相比只删除10倍R17三阶段波形模仿，原架构/初始化/数据/顺序/seed/优化器/预算保持；同时移除无用teacher前向，因此不称相同训练算量。不预先断言去掉teacher一定提分。

只原source训练缓存63000=L6300/U56700、20个既定received视图epoch、15%clean/85%residual；配对clean来自合法source L/U，0新TX标签、0V/target训练、0头拟合。使用配对波形及监督F来源知识，不称严格无监督或F-free。推理只当前received IQ，0clean/Fteacher/TX/RX/场景/cache/测试状态更新；识别仍接原F50及其联合训练默认头。

E20/4940，B256/model2026092701/data2026102001，AdamW2e-4/warm247/cos2e-5/WD1e-4/clip1/FP32/noTF32。每行L126000/U1134000/source1260000，原R17 teacher/context训练前向0，studentcontext1260000、Fclean1260000、Fstudent1260000并对student IQ反传。继承coarse20/action20/R17prior20加一次joint20共80串行训练pass，各组件累计40、component-epoch sum120；F50预训练50/11100及smoke/profile开销另计。不能把20轮当整体成本，不能把120当串行轮数。

固定最终E20，无早停、V选模或源性能资格门槛；每行自动source七视图诊断/profile/独立truth-last，再正式168000×clean和六practical全部预测，所有七视图IQ/logit/feature/physical IDs固定后独立CPU才读truth/clean。报告原始正确数1pp整数验收、真实涨分、MacroF1、TX/RX/day、raw修复/损伤/净修复、full256/prefix232/tail24/FFT/clean特征距离、完整参数/文件/延迟/显存与20轮日志。全部固定行和负结果保留。

38项合成/协议检查PASS，唯一独立P0/P1审查PASS，独立复跑38项。真实原R17初始化/F50/2Ls noquery smoke为launcher首步，检查coarse/action/各prior与F输入有限非零梯度、原初始化父/F/head不更新、严格复载逐位输出，0TX/V/target/optimizer更新，PASS即继续。

GPU4/5先等R08根终态且实际ownedGPU进程全部退出，不读评分；随后各一own worker，R09健康2/3/6/7继续，共六路。所有R17 scorer共享R06/score.lock，CPU2线程/最多一个own scorer。新不可覆盖run/output与release；一次普通N607预检、一个archive SHA比较/远端compile/help与实际PID/CWD/完整命令/commit/GPU/loggrowth。技术错误停止失败行、保留产物，健康行继续，无自动重试/回退；性能负结果不重启。profile排除GPU payload/optimizer，source-only16/batch1/2warmup，不把共存负载/测量顺序差称部署加速。

新方案由用户要求与合法源训练目标的匹配消融决定，在任何R08/R09目标分数前固定，已暴露基准不称新盲测。不以目标结果调秩、系数、预算、父权重、重排或选择性重跑。具体环境/CWD/命令/路径/停止规则和预期产物见experiment.json。

发布前证据：38项聚焦合成/协议检查和唯一独立P0/P1审查PASS，审查0真实IQ/checkpoint/远端访问；已核对正式训练0R17前向、完整联合梯度、F冻结、严格复载、实际R08句柄等待和自动全目标truth-last。普通N607预检VERIFIED，R08仍2个实际GPU进程，新root/release不存在；source/R17/F50句柄复用，只提交CPU smoke与等待，不抢占健康任务。

实际发布 VERIFIED：方法commit `d113672a6e51e75eb3ea0f7ecce0500e67f92966` 已推送且远端OID独立比较一致；单归档SHA比较/远端compile/help通过。实际原R17初始化/F50/2Ls noquery smoke PASS，两行student coarse/action/各prior及F输入梯度有限非零，原CPU初始化父/F/head0更新，严格复载输出逐位一致。R08完整测试/评分及原实际GPU句柄退出后，在GPU4/5各启动一个新worker。实际PID/CWD/完整命令/commit/GPU/loggrowth与E1均核实，R09健康2/3/6/7继续；原R17正式训练前向删除，未用新目标结果改变设计/矩阵。

完整汇总工具26项聚焦检查PASS，检查全联合组件预算、0R17 teacher/context计费/source曝光、直接clean损失、同全七视图/分组/原始整数1pp、实际涨分/70%与容差分开、payload-free profile。汇总工具未改运行中方法commit；当前尚无R10目标准确率。
