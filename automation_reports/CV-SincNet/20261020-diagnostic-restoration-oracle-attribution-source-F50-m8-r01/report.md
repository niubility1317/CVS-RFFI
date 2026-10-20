# 原 R03 恢复误差的源域位置归因

状态：ANALYZED / VERIFIED（source-only diagnostic）。仅source机制诊断，不训练模型、不读取目标IQ/真值、不提供可部署oracle方法，不能据此完成low urban≥70%、其余环境接近原clean的用户目标。当前四个GPU训练行保持固定方案。

固定原1800sourceV中RX1/3/4/6/8各按原数组顺序用linspace取36位置，共180物理样本，不读TX标签选样；不是把V再分成校准/选模角色。全部clean+六practical、原F50 seeds01/02/03/defaultheads独立argmax。R03基线波形取已完整结束spatial/pooled_features的SOURCE parent_R03.npy，未用新field输出或target波形；原R03 b01b534f实际权重只做launcher source smoke，三原F4df7b94d全冻。source/data/checkpoint契约已核实且不改变。0训练、0head拟合、0模型/normalization/持久状态更新。

|固定读出|源clean依赖|诊断含义|
|---|---|---|
|identity|无|原received|
|parent_R03|无|原R03恢复|
|clean_tail24|替换末24点|名义边界误差；不证明严格不可观测维度|
|clean_prefix232|替换前232点|当前主体波形残差影响|
|clean_low64|替换sourceclean最低功率64个复FFT bins|低功率复误差影响，不假设其必为TX指纹|
|clean_high64|替换最高功率64个复FFT bins|同频点预算对照|
|clean_gain|用sourceclean求全256点最优复标量后乘R03|全局幅相残差影响，无oracle时延对齐|
|clean_reference|整个配对clean|同样本原F clean参考，不是数学上界|

复杂FFT使用FP64 NumPy/orthonormal，按sourceclean功率稳定排序，每类精确替换64 bins后转FP32；时间patch其余位置不改，增益不作为runtime路径。tail24为48个实坐标、prefix232为464个、64复频点为128个、gain含2个依赖clean的实系数，oracle信息量不同，不把它们冒充同预算算法。原始输入不变，所有改变都在源counterfactual副本上。

7视图×8读出×3原头的IQ/ID/预测全部固定后，独立无CUDA CPU子进程才打开V TX标签。报告逐F/TX/RX/day accuracy/MacroF1、confusion、相对received/R03修复/损坏/净纠错和同物理clean full256/prefix232/tail24 NMSE/logpower，实际180样本TX分布在truth-last后给出。不把180×7×8×3个决策当独立样本，不把三F读出当三次恢复器训练。不存在target入口、目标指标、source候选排序或oracle部署性能宣称。

24项聚焦检查PASS，覆盖systematic选样/源人口负例、时间与频谱patch和gain、输入不变/有限值、完整预测包及禁止提前标签/target/runtime宣称、独立scorer前置拒绝、双CPU锁持有期间运行child再释放。唯一一次独立P0/P1审查PASS，审查者独立复跑24项通过，未发现确定P0/P1。launcher首步实际原R03/三原F/四Ls no-query smoke，原头输出和counterfactual有限、0TX/V/target/optimizer；PASS立即全部源预测。

N607普通账号，CPU2线程、0GPU worker。资源层按固定顺序同时持有现有scratch_guidance/score.lock和spectral_precision/score.lock，从真实smoke持续到全部source预测和独立child评分；与两个健康run CPU scorer互斥，旧任务从不同时取两锁，无环。只使用其既有资源锁，不改代码/配置/预算/过程，也不停止任何健康任务。新run/output不可覆盖，一次preflight、一实际代码配置commit、一次归档本地/远端SHA比较和remotecompile，launch后实际PID/CWD/命令/日志增长核实。非法来源/代码、冲突、非有限、F state变动或预测缺项保留产物并失败，无自动重试。

精确命令、CWD、来源、路径和矩阵见experiment.json。IQ/权重/物理ID/逐样本预测/原日志/私有机器状态不提交；只提交聚合诊断与实现。用户目标及正在跑的固定模型比较完全保留，没有向另一用户对话发指令。

一次N607普通账号preflight VERIFIED，新root/release不存在，原R03合法、baseline为既有source-only1800；四个scratch/spectral训练进程实际LIVE并全部保留。CPU2线程/无GPU，双既有score.lock串行资源方案固定。正式诊断尚未启动。


## 实际完整源诊断结果

已完成180物理sourceV×七视图×八读出×三原F预测与独立CPU truth-last评分，实际controller PID已退出。全部原F默认头冻结，0训练/拟合/状态更新和0target访问；当前目标最好66.406548%未因此改变。实际TX样本数为{"0": 37, "1": 27, "2": 23, "3": 25, "4": 29, "5": 39}，每RX36；不是按TX平衡随机抽样。

|source环境|原received|原R03|oracle tail24|oracle prefix232|oracle gain|sourceclean参考|
|---|---:|---:|---:|---:|---:|---:|
|clean|99.074074%|99.074074%|99.074074%|99.074074%|99.074074%|99.074074%|
|high|68.148148%|99.259259%|99.629630%|99.074074%|99.259259%|99.074074%|
|mid|64.629630%|97.407407%|98.333333%|98.888889%|97.592593%|99.074074%|
|low suburban|63.333333%|97.407407%|98.703704%|99.074074%|97.777778%|99.074074%|
|high urban|65.925926%|97.777778%|98.888889%|99.074074%|97.592593%|99.074074%|
|mid urban|49.444444%|84.814815%|89.444444%|98.518519%|88.333333%|99.074074%|
|low urban|43.148148%|80.740741%|86.481481%|97.222222%|82.592593%|99.074074%|

|source环境|原R03|oracle低功率64bins|相对R03（百分点）|oracle高功率64bins|相对R03（百分点）|
|---|---:|---:|---:|---:|---:|
|clean|99.074074%|99.074074%|+0.000000|99.074074%|+0.000000|
|high|99.259259%|99.444444%|+0.185185|99.259259%|+0.000000|
|mid|97.407407%|97.407407%|+0.000000|99.259259%|+1.851852|
|low suburban|97.407407%|96.666667%|-0.740741|98.518519%|+1.111111|
|high urban|97.777778%|97.777778%|+0.000000|98.518519%|+0.740741|
|mid urban|84.814815%|84.074074%|-0.740741|93.148148%|+8.333333|
|low urban|80.740741%|80.740741%|+0.000000|90.925926%|+10.185185|

|source low urban|三原F均值±样本标准差|相对原R03（百分点）|修复/损坏/净修复（3头决策合计）|
|---|---:|---:|---:|
|原received|43.148148±3.255258%|-37.592593|22/225/-203|
|原R03|80.740741±1.283001%|+0.000000|0/0/0|
|oracle tail24|86.481481±0.641500%|+5.740741|35/4/31|
|oracle prefix232|97.222222±0.555556%|+16.481481|93/4/89|
|oracle低功率64bins|80.740741±1.603751%|+0.000000|4/4/0|
|oracle高功率64bins|90.925926±1.785861%|+10.185185|66/11/55|
|oracle全帧复gain|82.592593±2.245251%|+1.851852|15/5/10|
|sourceclean参考|99.074074±0.320750%|+18.333333|103/4/99|

|source low urban波形|full256 NMSE|prefix232 NMSE|tail24 NMSE|logFFTpower MSE|
|---|---:|---:|---:|---:|
|原received|2.020478529|2.016105559|2.120791451|0.154007937|
|原R03|0.066418368|0.061922272|0.107627598|0.022102612|
|oracle tail24|0.056441227|0.061922272|0.000000000|0.018835239|
|oracle prefix232|0.009977142|0.000000000|0.107627598|0.003070104|
|oracle低功率64bins|0.065421936|0.061204401|0.103913401|0.021953485|
|oracle高功率64bins|0.022048173|0.019896197|0.045057560|0.004745699|
|oracle全帧复gain|0.061236641|0.056939228|0.099423942|0.022134900|
|sourceclean参考|0.000000000|0.000000000|0.000000000|0.000000000|

本样本上的直接观察：低功率64bins修复净收益为0；高功率64bins为55/540决策、即10.185185个百分点；tail24为31/540、即5.740741个百分点；prefix232为89/540、即16.481481个百分点；全帧复gain为10/540、即1.851852个百分点。三个读出共540次决策仍只有180个物理样本，跨视图/干预重复也不增加独立样本数；跨F标准差不是统计置信区间。该诊断不表明高功率bin就是TX指纹，不证明弱频谱补偿没有作用，也不证明高功率误差能从received恢复。改变不同坐标的oracle信息量与能量不同，且原F为非线性判别器，不能把局部替换收益相加或当作deployable上界。

所有oracle操作仅有助于定位本source panel的残差影响。它不能支持目标low urban达70%、其他环境接近clean或在轨/Phase2性能声明；当前scratch/spectral两组方案在本诊断前已固定并继续健康执行，没有据此热改、重排、停止或选择性重跑。逐F/TX/RX/day、真实TX计数、混淆/净纠错及所有视图的波形统计见results.json。

## 执行与验证

24项聚焦检查及唯一一次独立P0/P1审查PASS，审查者独立24项通过；一次普通N607 preflight、方法commit`14b3f2c0af8fefaa766a7b7664f01d4e73ef81f9`、单归档SHA比较和远端compile/import PASS。实际原R03/三F/四Ls no-query smoke PASS，0TX/V/target/optimizer。CPU资源日志确认同时取得scratch/spectral既有score.lock，全部七view预测后独立CPU子进程评分完成；四条健康GPU训练/预测进程继续保留。七视图推理时间合计27.472s（含各视图8干预/3F，不含smoke和独立scorer）。

首次独立进程探测时诊断已完成，因此没有取得运行中CWD/命令/日志增长快照；已独立核实实际PID退出、匹配代码的launch/完整预测/CPU truth-last产物和全部阶段日志。该运行证据限制如实保留，不为补过程截图而重跑。全部运行日志完整解析无技术错误。0GPU worker、0target输入、0新头或训练状态；原IQ/物理ID/逐样本预测/权重/原日志只保留私有产物。没有向另一用户对话发消息。
