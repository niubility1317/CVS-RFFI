# R30：原35.05万父版上的 F-free 四阶 IQ 统计保持

状态：VERIFIED_QUEUED_AFTER_REAL_SMOKE。已发布并通过真实检查，两版等待R28各自实际worker退出，无新版正式分数。

目的为联合压缩和性能优化。R29固定四模型、256个源物理位置、clean/received01/20、0F/TX/V/target诊断发现，R22两版较R18的pairedIQ误差约低1.1%，11维四阶统计误差却高4.94%/6.02%。这些只是源保真取舍，不证明正式TX1退化的唯一原因，也不是识别代用指标。本轮固定两种统计参考，检验其实际训练和正式表现。

|固定版本|完整参数|新增moment MSE权重|统计参考|GPU|
|---|---:|---:|---|---:|
|moment_clean|278180|1.0|合法模拟配对cleanIQ|2|
|moment_teacher|278180|1.0|原350521在同received上的最终IQ|3|

唯一初始化/IQ教师为用户原R46 prune_refine350521 source-final E20/4940，方法bbb8daf817de050b958d217b70d4c33f4aefe341；独立全链data-free SVD625。原7/63/30与完整R44/R38/R17/R07/R04源继承契约保持，不读取R18/R22/R23学生权重作为初始化或教师。完整coarse106658/action80045/prior91477，较350521减20.6381%；所有通道、GN4、skip、三stage/六CG保留且全部联合训练。其他主对话研发通道组裁剪，本轮用保通道低秩和参数自由IQ统计约束，路线不同。

基础目标与R22恒等版相同：pairedclean IQ NMSE+.05三尺度powerSTFT+三stage教师IQ权重线性1至.1+.02分组复FFT+.25G(clean)恒等IQ。新增项是最终restoredIQ与指定参考的11维描述向量每例MSE。两行均计算restored/clean/原父三个描述，记录两个误差及所选项/加权项；仅所选项进入梯度，所有参考detached。教师received输出已由基础目标计算，0新增完整G前向，0F权重/输出、TX标签、V或target用于训练。

描述量逐窗口去复均值：|C40|/P²、C42/P²、M42/P²及lag1/2/4/8的cum(a,a,a*,b*)归一化实/虚部，共11维；算术逐值匹配R29。每个power下限1e-4，不除接近0的累积量。系数1.0由已有R29源误差量级固定：pairedmoment约.0012、pairedIQ约.0195，约几个百分点；并非梯度等强或两参考误差等量。有限窗口描述不是无偏总体统计、TX真实性证明或理论收益保证。训练聚合loss和clip梯度检查有限，避免每个GPU描述量都回传主机同步。

每版新增固定20轮、19700更新、B64/micro64、received01..20；L126000/U1134000，teacher1260000输入、received学生1260000、额外clean学生1260000，学生总2520000；描述量3×1260000=3780000次。ResetAdamW2e-4、warm985、cos至2e-5、WD1e-4、clip1、FP32/TF32off。所有组件累计100轮、和300，祖先串行120+新增20=140；固定finalE20/19700，无早停或新源选模。

每版自动source1800×7、同卡完整G/G+F B1/B32成本、独立源评分、formal168000×clean/三郊区/三urban。无源收益门槛，七景全部预测固定后独立CPU truth-last；原F50 seed2026092701/defaulthead，仅冻结评测和真实2Ls smoke使用，不另训头。逐景accuracy/MacroF1、逐TX/RX/day/混淆、repair/damage、clean损伤、波形/F几何、完整20轮曲线与参数/MAC/文件/延迟/峰值均保留。原父逐景0.8pp/1344条、原R17历史1pp是容忍上限；70%/5pp/clean保持和强身份真实性另列。历史R22恒等版只作结果参考，不重复旧目标或称同批随机重复。

GPU2/3最多两条own、每卡一条，逐卡等待R28相应worker终态及实际owned PID/module退出；其workers尚空时继续等待。之后共享容量锁free≥18GB/allowner<2，健康任务保持。CPU评分两线程、R06score.lock、最多一个own scorer；普通账户、新不可覆盖run/release/root。一个方法commit、一次preflight、一次归档两端SHA和compile；launcher首步原父/F50/2Ls smoke，PASS立即排队，不额外审批。仅预登记系统技术失败保留影响行，健康行继续，无热改、自动重启或回退。精确命令、环境/CWD、数据/六类seed、输出与预期artifact见experiment.json。

45项聚焦检查覆盖R29算术匹配/相位幅度关系/能量floor梯度、目标选择/参考detached、完整联合梯度/同前向与初始化、来源与F/target权限负例、strictload/共享别名、前驱未启动等待、完整七景及truth-last和旧输出保护。唯一独立P0/P1审查PASS，无P0/P1，不额外重复审查；通用设计审批与全库测试为REJECTED_EXTRA_GATE/NONBLOCKING。完整联合优化目标仍ACTIVE。


## 实际发布与独立交付

方法commit 0b2c778d566ef83f159f236328cc704897a24507已推送并独立匹配远端OID。一次普通账户N607预检复核原350521 source-final20/4940、原源READY/EXACT_MATCH和正式七景最小句柄；无目标IQ或truth读取。一次归档两端SHA一致，远端compile通过，实际release/CWD为r30，既有输出保持。

真实原父/F50/2Ls no-query smoke两版PASS：完整278180参数，全部coarse/action/三prior梯度有限非零，原父与F状态不变；参考均detached、无F损失知识，新增统计约束0额外G调用。完整权重严格重载输出逐位一致；smoke记录两个描述误差、selected与weighted项，不是实际训练或识别收益。

实际CPU控制器PID/CWD/完整命令/commit与日志376至664字节增长独立核实，无failure/Traceback。两版GPUworker均0，R28前驱workers尚空时继续等待；不因GPU当前空闲绕过固定队列，不停止或热改健康任务。R23小版七景预测已固定、评分仍在进行，R26控制训练2/20，原矩阵保持。

20项结果工具检查PASS，使用合成完整损失和七景统计fixture，不是实际识别成绩。工具逐条核对20轮完整base/两参考/selected/weighted项、固定1.0与参考、3780000次描述/2520000学生/1260000教师输入、0额外G、原曝光和LR，并核对全部七预测/独立评分及实际owned句柄退出。汇总保留逐TX/RX/day/混淆/净纠错/clean损伤、波形/F几何、全训练曲线和完整成本，以及两版配对和历史R22恒等参考。

单一本地完成owner只读等待原controller/worker退出后收集完整source7/formal7与独立truth-last，取回两份独立完整权重并CPU严格加载，核对组件/MAC/有限输出，自动报告、提交推送并独立核对远端OID。此owner不启动、修改、停止或重试远端训练；不读取新增实际IQ或另做评分。已有R23/R24/R26/R28完成owner保持，完整联合优化目标ACTIVE。
