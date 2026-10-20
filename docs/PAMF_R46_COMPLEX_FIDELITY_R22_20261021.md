# R22：27.82万恢复器复频谱保真与clean恒等约束

状态：LOCAL_VERIFIED，尚未发布，无新版正式性能结论。

从用户指定原R46 prune_refine350521 source-final E20/4940独立初始化及作IQ教师，方法bbb8daf817de050b958d217b70d4c33f4aefe341。两版均为R18阶段对照的完整SVD625架构：278180参数，其中coarse106658、action80045、prior91477；保留全部通道/GN4/skip、共享prior长卷积权重及各stage独立bias/norm，三stage/六次CG全部执行。原R18学生不作为训练父或教师。

R21源诊断固定了128Ls+128Us不透明均匀位置、clean及原received01/20，三份冻结原模型均只读源训练IQ。阶段版received20弱功率Q1复FFT相对误差为强功率Q4的41.19倍、tail24为prefix232的4.15倍；这些是各自归一化误差之比，不证明TX身份或可恢复性。源quartile复误差均值约.1741/.2091、pairedIQ约.01991/.01950，选用.02权重约占对应IQ项17%至21%的未加floor参考量级；能量floor只会降低该额外项。此依据不是目标调参。

|固定版本|完整参数|复FFT权重|clean恒等权重|额外clean学生前向|GPU|
|---|---:|---:|---:|---:|---:|
|complex625|278180|.02|0|0|2|
|complex625_identity|278180|.02|.25|1260000|3|

基础目标与R18阶段版相同：pairedclean IQNMSE+.05三尺度功率STFT64/128/256+三stage教师IQNMSE，教师权重20轮线性1至.1。两版额外匹配FFT256的复数残差：每例按detached clean功率排序，四等份各64频点，各份误差除以max(本份clean平均功率,.01全频平均功率,1e-4)，再平均四份。该项包含相位信息，绝对能量下限抑制弱频点放大。第二版另外前向G(detached clean)，以原IQ能量clamp1e-4归一化计算.25恒等误差；第一版不执行这条额外前向。无条件teacherprior分支、F损失或标签分支。

每版固定20轮/19700更新、B64micro64、每轮985，6300Ls/56700Us每轮全曝光，总L126000/U1134000；原R07缓存received01至20及原data/model seeds复用。ResetAdamW2e-4、warm985、cos至2e-5、WD1e-4、clip1、FP32/TF32off；全部coarse/action/prior联合更新。第二版额外1260000次clean学生前向单列，不冒充相同前向成本。各组件累计100轮、组件和300、祖先串行120+新增20=140；F50训练另计。恢复器训练读取TX标签/F权重输出/V/target为0，合法模拟pairedclean及IQ教师不称无clean盲学习。

历史R18factor625_stage E20/19700作为匹配配方对照，复用已完成七景结果，不重复旧目标前向；不是新同批运行的对照，不作壁钟或统计因果归因。新直接float32 buffer在真实2Ls smoke中与原传输逐位比较。原7/63/30物理角色和完整祖先来源核对复用READY/EXACT_MATCH，不新增数据重验。

每版最终E20/19700冻结后自动source1800×7、同卡完整成本、独立源评分，再formal168000×7及独立CPU truth-last。全部七预测固定后才连接truth；源收益不是正式测试门槛，负结果也全部测试。使用原F50 seed2026092701及默认头，不拟合新头。逐景accuracy/MacroF1、TX/RX/day、净纠错、clean损伤、波形/FFT/F几何、完整训练曲线及成本全保留。父逐景0.8pp/1344条退化上限、原R17历史1pp、lowurban70%及其他五景距rawclean5pp分别报告，容差不等于性能优化目标。

同卡64sourceV测完整G及G+原F，B1/B32三轮正反顺序，报告median/p90/p95、每轮中位、吞吐、共享suite驻留/peak/incremental、完整参数/文件字节/ConvLinearMAC。输入预加载GPU，不含传输/前处理/loader；MAC不含FFT/CG/norm/激活/访存/F。R20图执行速度不自动归于新loss；实际独立G部署无需teacher/F/clean或训练loss分支。

N607普通账户、GPU2/3最多2own且每卡1；原R19实际owned句柄退出后，由共享容量锁下free≥18GB/allowner<2取得GPU，忙时等待，不修改其他owner健康任务。CPU评分2线程/R06 score.lock/max1ownscorer。新不可覆盖run/release/output，一次preflight、一次归档两端SHA、一次compile；launcher首步原父/F50/2Ls真实noquery smoke PASS即继续。31项聚焦检查及唯一独立P0/P1审查PASS，无P0/P1问题；真实原父/F50/2Ls noquery smoke由launcher首步执行。技术失败保留影响行、健康行继续，无自动重试/热改/回退/误杀。精确方法commit、环境/CWD、命令、路径与矩阵见experiment.json。

R18/R19/R20正式成绩已暴露，R22仅依据R21源IQ证据固定，不称新盲测或TX逐细节证明。主对话保留锚定/通道剪枝路线，本组保留通道维度并检验解析复频谱保真与clean恒等约束。
