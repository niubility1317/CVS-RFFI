# 固定 additive As：RX6 确认通过，跨 seed clean 保持失败

本轮源 RX6 的三组 scratch E20 已自然完成，实际执行及独立完整核算均为 VERIFIED。D_As 通过本轮预登记全部 source 条件；同时，独立 RX1 seed02 复核证实 clean 下降 8.25 个百分点，固定 additive 配方的双提升未能跨 seed 重现。该配方停止继续扩展，不据此调整系数、场景概率、RX、seed、阈值或预算救援。此次不访问正式目标数据，不宣称正式测试或联合目标完成。

## 执行与实际基线

运行：`20261016-phase1-FcrossRX-additiveAs-source-RX6-e20-s2026092701-m3-r01`。训练源码 commit：`367b2f945ef65488e6d97eddd2357388d2477100`；完整评分工具 commit：`e1ef20d19da3d8ff1e17b865d7fb6b837b39c0db`，最终 Git 交付另含报告、图表标注与只读质量阈值统计。起点 commit：`c5183a3300c183ce0b1b2cd95debc76a2cdc2a3b`。

基座固定 F＋跨 RX 同 TX SupCon：SIG_TIME_X3、统一 Sinc CENTER_UNITS、NoExplicitCFO、真实 joint160、既有内部融合和默认 6×160 cosine30 分类头。参数量 900,722，骨干与默认头正常联合训练；无新增部署分支、诊断头或单独头训练。保留原 13L＋7L/U、ownEMA、旧伪标签门控、AdamW 与 cosine 日程。

共同目标 `.50R(raw)+.25R(ordinary(raw))+.25R(raw第三对)+w(epoch)SupCon`，R 为同一 IQ 两次随机子网络的 smoothed CE 均值＋双向 KL（LS=.01、KL1）。SupCon 仅作用原 ordinary 对，time_down/t2.pool、temperature=.2、最大权重 .1、warm5/ramp10。D00 第四 raw 对权重0；D_raw 第四 raw 对权重 .25；D_As 第四完整 As(raw) 对权重 .25。三组均8次L前向，总pair系数1/1.25/1.25；共同 raw 绝对系数 .75、ordinary .25 保留，相对份额仍变化。D_As 与 D_raw 才是同总尺度的新增视图对照。未加入新的 A_g 或跨视图对齐损失。

数据：复用原 VALIDATED_ONCE carrier/split392005，源 RX1/3/4/6/8、day1/2/3、6TX；本轮 fit RX1/3/4/8 的 L5040/U45360/V21600。RX6 的原 L1260 仅固定 FINAL20 读出，不用于训练、EMA、校准或选模；held U/V 不参加训练。U TX 真值始终隐藏。实际角色 ID 与匹配旧 RX6 技术 smoke 契约严格一致。保持原静态15域映射；不把 held L1260 与旧四前向实验的 held V5400 当成相同评价集合。

实际导入：`.../releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code/leo_practical/{channel,residual}.py`，完整 __file__ 与实际配置见 practical_runtime_audit.json。声明采样率25MHz、IQ256、fc2.462GHz，原源包RMS归一化后视图生成，As 再用原 normalize_input 与原 AGC，不追加输出RMS。六种原 nominal 场景均匀采样；raw 直接经一次完整 residual/post_sync，EQ关闭即G=1，phase tracking、IQ compensation、质量适配均启用，reference_snr_db=30、boundary=reflect、fractional FIR数值延迟24点。未更改正式测试IQ、信道或SNR，也未剔除样本。

硬件按 RX/day/eq session 稳定，不用 TX 定义；训练 channel namespace 随 epoch/step 刷新，与 final source_eval/RX6/final20 随机流分离。全部有标签 As 参加 CE，没有学生置信度筛选，不反演未知 TX 硬件。

## RX6 逐 row 七视图结果

每格为 Accuracy / 宏平均F1（%），每视图1260，均为联合训练后 checkpoint 的默认学生头。

|视图|D00|D_raw|D_As|As−D00（pp）|
|---|---:|---:|---:|---:|
|clean|78.73 / 78.12|78.81 / 78.23|83.25 / 82.70|+4.52|
|practical_high|43.17 / 37.05|44.29 / 39.11|76.75 / 76.44|+33.57|
|practical_mid|41.51 / 36.21|42.54 / 37.77|72.30 / 72.32|+30.79|
|practical_low_suburban|38.33 / 33.65|38.97 / 35.04|68.25 / 68.34|+29.92|
|practical_high_urban|42.54 / 37.32|43.97 / 39.63|73.49 / 73.19|+30.95|
|practical_mid_urban|33.49 / 30.38|34.05 / 31.09|56.43 / 56.39|+22.94|
|practical_low_urban|30.56 / 28.80|29.84 / 28.52|46.83 / 46.93|+16.27|

Urban三场景均值：D00 35.53%；D_raw 35.95%；D_As 58.92%。相对D_raw，D_As clean +4.44pp、urban +22.96pp、low +16.98pp。因此，本轮收益无法只用额外前向或额外raw监督解释，但不能单独归因于“保留raw绝对系数”。

As 的三个 suburban 与自身 clean 的差距分别为 6.51 / 10.95 / 15.00pp，仍未达到各场景≤2pp的探索目标。low urban 46.83%仍低于60%。本轮只包含held RX6，不提供正式全RX最差值。

全部源规则通过：相对D00 clean及F1不降、urban≥+3pp、low≥+5pp；相对D_raw clean/F1下降≤.5pp、urban≥+3pp、low≥+5pp；各practical与seenV下降≤.5pp；共同base目标≤.5且预算/权限健康。比较采用预先固定1e-10pp舍入容差；它远小于一个样本的0.079365pp。

## 修复、新错与哪类退化得到缓解

|视图|修复D00原错|新增错误|净增正确|
|---|---:|---:|---:|
|clean|75|18|57|
|practical_high|451|28|423|
|practical_mid|427|39|388|
|practical_low_suburban|414|37|377|
|practical_high_urban|419|29|390|
|practical_mid_urban|340|51|289|
|practical_low_urban|280|75|205|

Low urban 的原生状态分层（全部样本保留）：

|维度/区间|观测数|D00准确率%|As准确率%|增益pp|
|---|---:|---:|---:|---:|
|lock/degraded|216|31.94|43.98|+12.04|
|lock/locked|470|42.55|74.47|+31.91|
|lock/unlocked|574|20.21|25.26|+5.05|
|state/BLOCKED|709|21.86|27.50|+5.64|
|state/LOS|374|45.99|76.74|+30.75|
|state/PARTIAL|177|32.77|61.02|+28.25|

收益主要落在 LOS/PARTIAL 和仍 locked 的观测；BLOCKED、unlocked 绝对准确率仍低。Urban三场景合并观测中，LOS/PARTIAL/BLOCKED分别提升32.75/29.19/6.65pp，locked/degraded/unlocked分别提升32.83/11.89/4.64pp。这些区间互相依赖，合并3780是场景观测数，不是独立物理样本数。它们不能证明多径、CFO、噪声或门控的单一因果作用。

固定D00 clean正确集合共有992个样本，low urban保持率由33.97%升至50.60%；同集合的真实类−最强错误类margin/30均值由-0.038458升至0.023307。这是本轮同预算D00的固定集合，不是继承父权重；没有拟合额外头。

仍有不利TX：本轮low urban的TX4下降3.81pp，clean的TX0/3/4分别下降0.95/2.86/0.48pp；完整6TX×7视图和3day变化在source_strata.csv及error_redistribution图。不能把总体提升说成全部TX受益。逐包内部门控值未记录，未以未启用的旧日志占位字段替代实际门控证据。

## 质量、边界与覆盖

Low urban质量代理中位数6.12dB，实测signal_to_added_noise中位数6.00dB；30dB仅为新增噪声参考，不是总SNR。绝对残余CFO中位数157.81Hz，p90 729.78Hz；残余核长度62，FIR数值延迟24，1260包均为synthetic_reflect，AGC限幅0包。

按相同5/15dB阈值比较当前补偿下的质量代理与实测新增噪声SNR分箱，六场景不一致数依次为 1 / 9 / 9 / 7 / 33 / 51（每场景1260）。low urban为51/1260。此统计没有重算接收状态或补偿，也没有估计总SNR；不能当作接收机重算的总效应或固定接收状态的受控效应。EQ关闭未引入G滤波的噪声着色，本轮没有保存噪声序列/完整残余核以估计PSD。有效无人工边界区域、去损伤对照及门控变化仍需独立诊断，不能从常量reflect/24点元数据推断因果。

As训练观测568320，全部432个 TX×fitRX×day×scene 组合覆盖，单组合1199至1415包；LOS/PARTIAL/BLOCKED、locked/degraded/unlocked均出现。三组L/U物理序列逐项相同；L观测568320、唯一5040，U观测396368、唯一45360。U没有As、额外对比或真值使用，teacher额外As调用0。完整coverage cells保留在私有analysis_summary.json，图展示原nominal覆盖而非独立放大物理扰动。

## 完整曲线与训练成本

独立解析每组全部20轮JSONL/CSV、4440条native step、4440条视图step、4440条Sinc单位step与全部stdout；没有技术失败、跳过非有限更新或目标访问。原生未启用项的NaN占位不当作活动训练失败。每组SupCon实际3330更新、13320层/视图项。共同目标随SupCon权重增加而变化，不把辅助项启用后的上升误判成分类CE恶化。

|row|新E/更新数|L前向样本|全部NN样本|原生训练秒|final共同base|final额外pair|fitV%|
|---|---:|---:|---:|---:|---:|---:|---:|
|D00|20 / 4440|4546560|5823318|1392.39|0.315280|0.103930|97.98|
|D_raw|20 / 4440|4546560|5823318|1452.76|0.317042|0.099685|98.02|
|D_As|20 / 4440|4546560|5823318|2282.59|0.322297|0.551673|97.96|

本轮新增3模型共60modelE/13320更新，全部NN=17469956（包含独立dispatcher真实CP smoke2包）。三组并行，从成功预检到矩阵完成墙钟2312.41秒；全部自有worker与dispatcher自然退出，failure不存在。每组额外L前向1136640、ordinary生成568320、held读出8820。实际计数每组student_train4942928、student_eval475202、teacher_train0、teacher_eval396370；扣除EMA继承计数偏移2再加held读出得到5823318，不把重复计数当成真实前向。

训练初始化没有父权重，继承父训练成本0；旧RX6 scratchE20 C0仅用于数据角色匹配的真实checkpoint兼容性smoke，从未初始化学生/EMA或作为性能对照。其历史训练不伪称不存在；本轮新增成本仅该smoke2包。引用的RX1两个历史矩阵各60modelE/13320更新/17469956NN，列为既有研究成本，不是本轮重复训练；不把这几项当项目全部历史成本。工程检查累计10020合成NN/6合成更新，含已修复测试辅助NameError后的重跑；额外读数/绘图/阈值分析均0NN/0更新。E20预算完成不等于收敛，As额外pair目标约0.552，未宣称训练充分。

## 跨 seed 复现与否定结论

RX1两个seed使用相同fit角色ID、held L1260及实际六场景配置，原生参数只差seed及run/candidate/output标识。seed同时改变模型和原普通增强随机流；不是仅模型初始化独变。seed02实际代码05984a1229ebac7025a95081dc5fae4108276b28，三组完整记录与26460个预测已独立重算，没有修改原owner任何产物或作业。RX6是单独留出折，不作第三个seed。

|held RX/seed|As clean%|相对D00 clean pp|urban均值增益pp|low增益pp|As low%|原source条件|
|---|---:|---:|---:|---:|---:|---|
|RX1/2026092701|83.57|+3.33|+9.71|+5.71|33.81|PASS|
|RX1/2026092702|74.37|-8.25|+12.35|+8.73|36.11|FAIL|
|RX6/2026092701|83.25|+4.52|+23.39|+16.27|46.83|PASS|

RX1两seed均值：urban +11.031746pp、low +7.222222pp、clean −2.460317pp，As low平均34.960317%。证实了这两个seed的星地收益方向，否定了“保留raw绝对监督足以稳定保护clean”的当前假设。fitV差异很小，不能替代跨RX保持。RX6局部通过不抵消seed02的clean失败，也不证明新的A_g有效。

## 验证、命令与交付

本地检查：D00与原六前向loss及全部参数梯度逐位一致；相同初始化与CPU/Python随机末状态；三组8L、同IQ成对、仅D_As完整As(raw)；默认头联合更新/export160；20配置负例、6角色负例、U隐藏标签、RX1合法/RX6拒绝、真实physics捕获输出及draw完全相同；精确5pp边界/实质低于边界/权限拒绝检查通过。一轮限定独立P0/P1 PASS及一次数值边界定向复核PASS。真实匹配CP no-query smoke2合成包PASS，单归档比较、远端compile、PID/CWD/argv/GPU/log增长均VERIFIED。

本地实际验证入口：`E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 E:/type10-7/local_artifacts/cvs_FcrossRX_additive_RX6_20261016/run_checks.py`，另有narrow_checks.py、boundary_checks.py；完整只读核算为 `F:/App/miniconda3/python.exe -X utf8 E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_f_additive_rx6/analyze.py --input E:/type10-7/local_artifacts/cvs_FcrossRX_additive_RX6_20261016/final`；质量阈值为同解释器 physics_summary.py --input 同目录；seed02为analyze_seed02.py --input E:/type10-7/local_artifacts/cvs_FcrossRX_additive_RX6_20261016/seed02_final。

N607实际启动命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_additiveAs_RX6_source_E20_20261016_r01/experiments/cvs_f_additive_rx6/runner.py`，CWD即该release。source3row IDs、配置、GPU动态共享容量规则（所有owner≤3/GPU、空闲≥12000MiB、CPU2）、系统性技术失败stop规则见原登记。没有修改健康其他owner任务。独立回收seed02时修正了只读辅助脚本行ID，首次partial归档保留，未重跑模型；查询其preflight.json不存在属于该发布器产物命名差异，执行身份从实际release_commit与行级launch验证。

实际修改源码：design/model/source/runner/dispatch/preflight/publish/checks/registration，以及harvest/analyze/analyze_seed02/physics_summary/plot；均在experiments/cvs_f_additive_rx6。原预登记保存在preregistration.md与analysis/cvs_FcrossRX_additive_RX6_20261016.md。报告JSON、执行与row证据在同run目录；完整预测/物理ID/日志/曲线CSV及图均留私有local_artifacts，不提交IQ、权重或原始运行日志。

实际已核实图表（每项同时有PDF）：

- [source_accuracy](E:/type10-7/local_artifacts/cvs_FcrossRX_additive_RX6_20261016/final/figures/source_accuracy.png)
- [training_curves](E:/type10-7/local_artifacts/cvs_FcrossRX_additive_RX6_20261016/final/figures/training_curves.png)
- [urban_strata](E:/type10-7/local_artifacts/cvs_FcrossRX_additive_RX6_20261016/final/figures/urban_strata.png)
- [fixed_clean_retention](E:/type10-7/local_artifacts/cvs_FcrossRX_additive_RX6_20261016/final/figures/fixed_clean_retention.png)
- [error_redistribution](E:/type10-7/local_artifacts/cvs_FcrossRX_additive_RX6_20261016/final/figures/error_redistribution.png)
- [training_coverage](E:/type10-7/local_artifacts/cvs_FcrossRX_additive_RX6_20261016/final/figures/training_coverage.png)
- [crossseed_RX1](E:/type10-7/local_artifacts/cvs_FcrossRX_additive_RX6_20261016/final/figures/crossseed_RX1.png)

其余可复核产物：`E:/type10-7/local_artifacts/cvs_FcrossRX_additive_RX6_20261016/final/training_curves.csv`、`source_strata.csv`、`physics_strata.csv`、`mechanism_analysis.json`、`quality_threshold_sensitivity.json`、`source_readouts/*.json`。七组PNG/PDF共14文件已回收、格式检查并逐张查看，只有报告数据，无新增模型前向。

## 未完成项与当前决策

固定additive组织不再扩展。已关闭的normalized75/25、课程参数救援、局部观察衰减等负结果保持；此前冻结梯度诊断也未满足预设双模型/双流全模型冲突条件，不据此新增梯度投影。下一阶段须先区分可恢复观测适应与严重遮挡/失锁的信息损失，并为独立A_g建立源域clean证据；不靠叠加对齐损失宣称解决。

尚未证明：有效新A_g、同契约B00/B10/B01/B11交互、正式目标clean＋六practical及最差RX、跨seed联合收益、suburban各场景距clean≤2pp、low urban>60%。没有执行正式target预测或truth-last评分，不能把源读出当正式部署成绩。逐包门控、完整残余核/噪声谱、人工边界有效区域和“移除损伤并重算接收状态/固定接收状态”的受控消融均未由本轮完成。
