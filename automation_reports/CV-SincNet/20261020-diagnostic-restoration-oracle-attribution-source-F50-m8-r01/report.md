# 原 R03 恢复误差的源域位置归因

状态：PLANNED。仅source机制诊断，不训练模型、不读取目标IQ/真值、不提供可部署oracle方法，不能据此完成low urban≥70%、其余环境接近原clean的用户目标。当前四个GPU训练行保持固定方案。

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
