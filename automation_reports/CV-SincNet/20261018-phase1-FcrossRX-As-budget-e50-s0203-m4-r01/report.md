# 原完整 practical E50：固定02／03种子的匹配复现

状态：**RUNNING，已实际远端发布并独立验证四行GPU训练。** 唯一owner为codex/root/practical-budget50-replication；本run登记四行，防止与另一对话重复启动。实际实现提交为 `4ea924316d6a5fe84278c81e3db78706ff674f21`，分支codex/cvs-iq-timing-e20-20261016。真实父权重/F检查由launcher首步执行，GPU按剩余容量分配，未把本地验证当成远端运行。

唯一假设：在原配方保持不变时，各自固定FINAL30+20的预算收益可在原定seed02/03复现。依据是01的合法源域预算结果：fullV clean+0.118519pp、mid/low+1.972222/+1.638889pp，两场均值+1.805556pp；As50−NoAs50 fullV clean−0.074074pp。小V1800 clean配对−0.166667pp保留，不能与全V混为同一口径。正式成绩不参与本次候选、种子或参数选择。

四行矩阵以experiment.json为唯一维护位置：02/03各B00 NoAs、B01完整As，全部从自己的replicate49 FINAL30 student/ownEMA和scratch20祖先继承。固定新增20轮/4440更新，总50轮/11100更新，resetAdamW/cos20，即20+10+20，非连续cos50。原01不重训不重测，其已冻结结果与三seed E30历史只作完整配对汇总。

保留F/CROSS_RX/SIG_TIME三路/统一Sinc/joint160/defaultcos联合训练、NoExplicitCFO及内部补偿、IQ256，原L6300/U56700隐藏/V27000，源RX1/3/4/6/8 day1/2/3、split392005。不留训练RX，不拆Ldev，不新增Ag/损失/分支/部署头/U机制。原6L前向/1.25L尺度、64交替As/fullKL/dropout、同IQ RDrop、ordinaryepoch13用自己scratch20args冻结均不变。测试固定原receivedIQ/SNR/信道/全部包、clean+六practical，不作任何修改。

每row固定FINAL50并自动source读出、全部七视图预测后独立exec CPU truth-last评分；源科学负值不跳过评分。全三seed逐seed和配对均值/样本标准差全部报告，clean阶段每seed及配对均值按−0.1绝对百分点衡量；微小均值<0.1pp不直接称稳定。只有三seedclean及low/mid均为正且clean配对均值≥0.1pp时，才称本固定三seed一致双提升；否则如实报告不一致或均值观察，不补选seed救跑。该判据约束结论，不用formal回流选择或调参。

新增4行80model-epochs/17760更新；预期每row native6271922NN、source12600NN、formal1176002NN，总新增29842096NN加sharedsmoke3074NN（真实父CP2＋四行真实F合成3072），预期新增总29,845,170NN；学生/教师/续训/读出/历史成本分列，最终按全日志实测，未完成不写成实际。

公开路径：PROJECT/releases/cvs_FcrossRX_As_budget_E50_s0203_20261018_r01；PROJECT/runs/20261018-phase1-FcrossRX-As-budget-e50-s0203-m4-r01；逐row日志logs/{row_id}.log。普通N607账户CPU2，每GPU所有owner总进程≤3且空余≥12000MB，不修改/停止其他健康任务。一个不可覆盖root、一次preflight/一次release archive SHA比较/一次远端compile、一次真实CP noquery smoke和聚焦负例；启动后独立PID/CWD/argv/GPU/日志增长。仅run-owned契约/父权重/defaulthead/数值/输出冲突/非法预测/无法运行等系统技术失败触发保留现场并停止尚未启动本run行；无自动重跑。

预期闭环：真实parent provenance/args、native完整日志/checkpoint/source_frozen、固定源panel、七视图不可变prediction、独立score、全部逐seed/配对错误/预算交互/成本报告和登记/Git交付。预登记时尚无新模型训练；现已实际训练中，尚无新的冻结正式结果。

父来源静态核实：旧49 release `cvs_FcrossRX_As_replication_E30_s0203_20261018_r02`，提交 `63b7c8cd7bb4139a2d800e6f4c4a7f1f2163fd47`；其固定P43依赖提交 `19aedc3604fed9c193237dd92366164178af0983`。每行直接CP本地epoch10／EMA2220、自己的scratch20祖先epoch20／EMA4440，不能将累计E30计数误当CP本地计数。原stage20从零及source固定10轮均已完整source审计；本次真实加载前再执行每行自身严格配置、role_ids、student/ownEMA、默认头与有限值检查，未改数据不重验。

本地实现已冻结：96项协议负例、4项native seed正例、13个Python文件AST/compile及2段emitted远端脚本AST均PASS，0NN/0CP/0IQ/0target；唯一一次新四行绑定P0/P1独立审查PASS，无P0/P1。详见review.json/verification.json；本地验证时尚未执行真实CP/F检查；当前launcher已完成，见实际启动证据。

## 完整三种子零模型分析

独立分析release `cvs_F_practical_budget50_replication_analysis_20261018_r01`，输出本run的 `analysis_auto_r01`；CPU2/CUDA隐藏，0NN/0IQ/0CP/0更新，不更改训练或模型。先等待真实dispatcher/row句柄与源产物，逐步完整读取4×4440新增step/80epoch，核对实际student/teacher计数与物理/ordinary序列；旧六个E30及01 E50直接复用已完成source full-log audit及实际原生counter，不重扫旧步骤，不重复计源读出。

固定三seed×四逻辑条件共12行；源V1800两个draw与原生fullV27000分开，mid/low不是源urban三场均值。全12×7冻结prediction及独立score完成才读正式truth，独立重算CM/accuracy/F1/worstRX/RXdayTX及四类配对修复/新错、预算交互，逐seed与配对mean/sampleSD全部保留。源科学负值不跳过固定正式收尾。真实句柄消失且缺件则记录准确缺项退出，不无限等/不自动重试。

7项合成定向检查最终一次整套PASS（1.868s），4文件编译及diff检查PASS。分析提交 `5f3a0fc9c128944994f40d4f4858a5f8e7730b40` 已实际发布，远端7项检查和编译PASS；独立核实PID1177231/CWD/argv/CPU2/CUDA隐藏，当前WAIT_SOURCE、truth_opened=false。只回收固定公开JSON/CSV/MD和可选图，不包含IDs/NPZ/IQ/权重/原始日志/私有路径。

## 实际启动证据

观察时间：2026-10-18T04:46:56.376Z。训练release提交 `7bbb5536cb72c9b95c7c2045ba660e1e72e234e7`，dispatcher PID1174889，4行实际GPU/CWD/argv/env一致且日志增长，无failure。96负例/4seed正例/realCP2NN+四行真实F3072NN均PASS，共3074NN共享smoke、0更新、0真实IQ、0query。native训练source-only；完整训练成本尚未产生，不能把首轮局部计数当累计50轮成本。

|row|PID|GPU|累计轮数|新增更新|累计更新|日志增长bytes|
|---|---:|---:|---:|---:|---:|---:|
|B00-s2026092702|1174919|0|32/50|444/4440|7104/11100|+10287|
|B01-s2026092702|1174997|1|31/50|222/4440|6882/11100|+5144|
|B00-s2026092703|1175080|2|32/50|444/4440|7104/11100|+10284|
|B01-s2026092703|1175233|3|31/50|222/4440|6882/11100|+11743|

全部extra_U_NN=0、U_As=0、U_truth_read=false、target_access=false；原01不重训，02/03新冻结结果尚无。analysis PID1177231等待真实源产物，12×7完整前真值关闭。一次N607 preflight/archive比较/compile已执行，未重哈希。详见execution_verification.json/analysis_verification.json。
