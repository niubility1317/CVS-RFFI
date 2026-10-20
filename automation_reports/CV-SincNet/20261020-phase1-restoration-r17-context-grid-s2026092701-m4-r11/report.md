# R17 压缩与源 clean 联合优化上下文矩阵 R11

状态：RUNNING。同一源 clean 纠正目标下，固定两种 prior，补齐 full/quarter context；R10 已登记 half context 作为匹配对照。性能对照保留完整 context，更小两组检验继续压缩。验收以同一正式目标七场景为准：各场景相对原 R17 下降不超过 1 个百分点；另报实际提分、low urban≥70%、其他场景距 clean 和配对 clean 恢复差距。容差通过不等于提分。

|固定行|完整/可训练参数|context参数|context保留比例|比原4665073减少|GPU|
|---|---:|---:|---:|---:|---:|
|full_context_quarter|2582905|2157547|1.0|44.6331%|2|
|full_context_eighth_pointwise|2352389|2157547|1.0|49.5744%|3|
|quarter_context_quarter|1212813|787455|0.25|74.0023%|6|
|quarter_context_eighth_pointwise|982297|787455|0.25|78.9436%|7|

只原 source-selected R17/cd977448/E20/4940 CPU 初始化，无已测学生继承。完整 context 逐位复制原权重；quarter context 的84个长卷积做无IQ CPU平衡SVD；prior固定quarter共享或eighth并共享原1×1，保留各步bias/GN。三步×2CG不变，coarse/action/prior全部联合优化。原R17对象在建立optimizer前删除，正式训练0R17 teacher/context前向。

固定损失与R10完全相同：最终恢复与paired source-clean的IQ NMSE（clean能量detach、下限1e-4）+T1 KL(Fclean||Frestored)+0.5×单位F160距离平方，系数1/1/0.5，无R17波形模仿。原F50/默认联合训练头冻结，0头拟合。源数据为原63000=L6300/U56700及原20个received缓存、15%clean/85%residual；0新TX、V或target训练标签。使用源配对波形及既有监督F知识，不称严格无监督或F-free。推理只当前received IQ，0clean/Fteacher/TX/RX/场景/cache/状态更新。

E20/4940、B256、model2026092701/data2026102001、AdamW2e-4/warm247/cos2e-5/WD1e-4/clip1/FP32/noTF32。每行L126000/U1134000/source1260000、studentcontext1260000、Fclean/student各1260000、R17训练前向0。继承coarse20/action20/R17prior20加一次joint20为80串行训练pass；各组件累计40，component-epoch sum120。F50预训练50/11100及smoke/profile成本另计。

固定最终E20，无V选模、早停或source性能门槛。每行自动source1800×7诊断/profile后执行正式168000×7预测；同row七视图IQ/logit/features/physical IDs全部固定后，独立CPU才读truth/clean。报告准确率/MacroF1/TX/RX/day/净纠错、完整参数和文件、full256/prefix232/tail24/FFT/配对clean特征差、全部20轮日志和batch1 profile（排除GPU payload/optimizer）。精确1pp按100×(原正确数−新正确数)≤N验收，不能把相对百分比当百分点。所有固定行及负结果保留。

full/half/quarter context成本在R08目标暴露前已做0IQ静态审计，本轮沿用固定两种prior和R10目标，补齐上下文层级。已看到R08/R09的benchmark结果，如实披露，不称新独立盲测；不据目标改秩、损失、预算、父权重、排序或选择性重跑。

42项合成/协议检查和唯一独立P0/P1审查PASS（独立复跑42项）。真实原R17/F50/2Ls noquery smoke是launcher首步，检查全部组件和F输入有限非零梯度、原父/F/head0更新、严格复载逐位输出，0V/target/TX/optimizer更新，PASS立即继续。

R09根终态且其实际GPU句柄退出后接续GPU2/3/6/7，每卡一own worker；R10健康4/5保持，共最多6路。所有R17 scorer共享R06 score.lock，最多1个own CPU scorer。新不可覆盖run/release/output；一次普通N607预检、单archive SHA比较、远端compile/help、实际PID/CWD/cmd/commit/GPU/loggrowth核实。技术错误停止受影响行并保留，健康行继续；任一失败则完成健康行后根状态FAILED，不自动重试或回退。精确命令、环境、路径和预期产物见experiment.json。

发布前预检 VERIFIED：普通N607账户、R09旧实际GPU句柄0、新run/release不存在，GPU2/3/6/7各24124MiB空闲。R10在GPU4/5继续；复用合法source/R17/F50与formal句柄，无新增数据重验。

实际发布 VERIFIED：方法commit `5f8c563e96766fee9f4601fe64bf0da4395892ae` 已推送并独立核实远端OID；单归档SHA比较、远端compile/help通过。真实原R17/F50/2Ls noquery smoke PASS，四行coarse/action/各prior与F输入有限非零梯度，原初始化父/F/head0更新、严格复载逐位输出。R09实际句柄退出后，在GPU2/3/6/7各一新worker；实际PID/CWD/完整命令/commit/GPU/日志增长及E1均核实。R10保持原固定方案完成，不以其新结果改R11设计、预算或顺序。24项完整汇总检查PASS，检查动态context量/比例、全部预算和七视图/整数1pp/真实涨分/成本；汇总工具不修改运行中方法commit。当前尚无R11目标结果。
