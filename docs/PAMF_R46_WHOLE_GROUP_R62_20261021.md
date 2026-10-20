# R62：整组压缩的正式性能与成本

VERIFIED / ANALYZED：两行完整E20/4940、源1800×7、正式168000×7、逐行七预测固定后独立truth-last、全20轮/同卡成本/实际所属进程退出和独立权重CPU严格重载已核实。group_preserve 248828参数，较指定M35七景提高0/下降7，low urban 66.230952%（较M35-1.279167pp）；较R49同增量预算提高0/下降7，low差-0.591071pp；70%=False、其他五景gap≤5pp=False；同卡G B1父/本版48.9053/48.2824ms，G+F B1父/本版59.1257/57.9017ms；regroup_control 248828参数，较指定M35七景提高0/下降7，low urban 66.207738%（较M35-1.302381pp）；较R49同增量预算提高0/下降7，low差-0.614286pp；70%=False、其他五景gap≤5pp=False；同卡G B1父/本版46.9371/47.3193ms，G+F B1父/本版57.3480/57.7004ms。同初始权重/通道/预算下GN3相对GN1七景提高6/下降1，low urban差+0.023214pp；归约和实际成本不同，不称完美计算匹配。直接指定350521初始化/冻结IQ教师不变，全prior保留并更新，原pairedcleanIQ/谱/1→.1IQ教师，全组件联合；0TX/F/V/target优化/新校准/新参数/额外G前向。全部正负结果保留，不把减参或1pp容差当作性能提升或逐细节TX真实性。完整目标ACTIVE。

**VERIFIED / ANALYZED**。两固定行完整 E20/4940、source1800×7 和正式168000×7；每行七景预测固定后独立 truth-last，实际所属 controller/worker 全退出，独立权重严格重载通过。全部正负结果保留，不反馈调参或选择性重跑。

性能优先。两行同248828参数、同选组和初始权重、同训练预算，只改变 context GN3/GN1；prior完整112594参数。权重选择删除一个完整原GN组，无校准IQ。GN3保留原每组通道宽度，GN1重新合组；层级同值等价不代表整个恢复器等价，归约方式和实测成本不同。

直接指定350521父初始化和冻结三阶段IQ教师；全coarse/action/prior联合，pairedclean IQ NMSE+.05谱误差+线性1→.1教师，全63000源记录/20缓存轮/globalmicro256。L126000/U1134000，0TX/F/V/target优化，0新增参数/校准IQ/额外G前向；合法源clean模拟配对的TX无标签自监督。

方法commit `7a8fa7bd836946c3493635cbe0966591ecf14e1e`。完整命令/环境/矩阵/数据权限见[登记](../automation_reports/CV-SincNet/20261021-phase1-restoration-R46-whole-group-s2026092701-m2-r62/experiment.json)，完整曲线、每TX/RX/day、修复损伤、NMSE/FFT/特征描述见[结果](../automation_reports/CV-SincNet/20261021-phase1-restoration-R46-whole-group-s2026092701-m2-r62/results.json)与[CSV](../automation_reports/CV-SincNet/20261021-phase1-restoration-R46-whole-group-s2026092701-m2-r62/row_metrics.csv)。

|行|完整参数|较M35减参|原R17逐景1pp|M35逐景1pp|low urban（%）|其他五景gap≤5pp|
|---|---:|---:|---|---|---:|---|
|group_preserve|248828|29.011956%|True|False|66.230952|False|
|regroup_control|248828|29.011956%|False|False|66.207738|False|

|行|场景|准确率（%）|MacroF1（%）|较M35（pp）|较原R17（pp）|较R49同预算（pp）|净修复|
|---|---|---:|---:|---:|---:|---:|---:|
|group_preserve|clean|84.322024|84.041123|-0.134524|-0.264881|-0.119048|-595|
|group_preserve|practical_high|81.034524|80.513802|-1.417857|-0.284524|-0.847024|+67707|
|group_preserve|practical_mid|80.107143|79.661697|-1.114881|+0.529167|-0.319048|+67520|
|group_preserve|practical_low_suburban|78.679167|78.357848|-0.941071|+0.682738|-0.257738|+66611|
|group_preserve|practical_high_urban|79.963690|79.503637|-1.423214|-0.389881|-0.798214|+67646|
|group_preserve|practical_mid_urban|73.188690|73.162088|-1.258333|-0.496429|-0.542262|+65025|
|group_preserve|practical_low_urban|66.230952|66.686781|-1.279167|-0.985714|-0.591071|+60365|
|regroup_control|clean|84.355357|84.066338|-0.101190|-0.231548|-0.085714|-539|
|regroup_control|practical_high|80.949405|80.469675|-1.502976|-0.369643|-0.932143|+67564|
|regroup_control|practical_mid|79.954167|79.573342|-1.267857|+0.376190|-0.472024|+67263|
|regroup_control|practical_low_suburban|78.632143|78.408469|-0.988095|+0.635714|-0.304762|+66532|
|regroup_control|practical_high_urban|79.841071|79.440278|-1.545833|-0.512500|-0.920833|+67440|
|regroup_control|practical_mid_urban|73.060119|73.125889|-1.386905|-0.625000|-0.670833|+64809|
|regroup_control|practical_low_urban|66.207738|66.759798|-1.302381|-1.008929|-0.614286|+60326|

主消融为GN3相对GN1：相同初始权重、通道和预算，保留全部逐景差；同预算未裁剪R49 retain_refine只读复用，不重复训练/继承其权重。宽度和归一化同时区别于R49，不把全部收益归于单一因素。

|场景|GN3−GN1准确率（pp）|MacroF1（pp）|
|---|---:|---:|
|clean|-0.033333|-0.025215|
|practical_high|+0.085119|+0.044127|
|practical_mid|+0.152976|+0.088355|
|practical_low_suburban|+0.047024|-0.050621|
|practical_high_urban|+0.122619|+0.063359|
|practical_mid_urban|+0.128571|+0.036199|
|practical_low_urban|+0.023214|-0.073017|

|行|coarse/action/prior参数|文件字节|Conv/Linear MAC每IQ|完整训练秒|训练peak allocated字节|
|---|---|---:|---:|---:|---:|
|group_preserve|77796/58438/112594|1275740|31548897|1527.4988|1276847104|
|regroup_control|77796/58438/112594|1275740|31548897|1527.5768|1276503040|

|行|口径|父/本版median ms|本版P90/P95 ms|本版sample/s|父/本版增量峰值字节|
|---|---|---|---|---:|---|
|group_preserve|G_only_B1|48.9053/48.2824|63.1211/64.7224|19.2738|67194880/79946240|
|group_preserve|G_only_B32|92.5700/82.6005|98.6401/100.7848|378.6072|276216832/276216832|
|group_preserve|G_plus_F_B1|59.1257/57.9017|74.1209/76.4633|16.0559|67194880/79946240|
|group_preserve|G_plus_F_B32|96.5823/91.0821|107.6726/112.1233|346.8920|276216832/276216832|
|regroup_control|G_only_B1|46.9371/47.3193|58.3268/62.6557|19.8478|67194880/79946240|
|regroup_control|G_only_B32|91.6933/79.1482|98.6656/100.3612|388.7444|276216832/276216832|
|regroup_control|G_plus_F_B1|57.3480/57.7004|74.2679/77.2018|16.1951|67194880/79946240|
|regroup_control|G_plus_F_B32|96.5593/89.8721|113.4048/114.9340|340.3555|276216832/276216832|

成本使用同卡64条sourceV、G/G+冻结原F、B1/B32、3轮交替次序和预热。同步预加载GPU输入，不含传输/前处理/loader；三G一F共享驻留suite，增量峰值是当前前向，不当作独立部署显存。Conv/Linear MAC不含FFT/CG/norm/activation/F，训练peak包括教师/快照/优化器/激活；不同GPU时段不据绝对数值作速度排名。

group_preserve完整20轮首→末：loss 0.27823031→0.023799172；iq_nmse 0.14429862→0.023163804；spectral_error 0.04088287→0.0059758076；teacher_stage_nmse 0.13188754→0.003365775；每轮中位76.5110s；lowurban70=False；otherfive5pp=False；日志类别[]
regroup_control完整20轮首→末：loss 0.35851386→0.023941636；iq_nmse 0.18364192→0.023276095；spectral_error 0.050839055→0.0060113963；teacher_stage_nmse 0.17232999→0.0036497109；每轮中位76.2030s；lowurban70=False；otherfive5pp=False；日志类别[]

原R17逐景1pp是退化上限，性能提升、lowurban≥70%与其他五受损场景距rawclean84.6761904762%≤5pp分别判断。单seed、相关视图、已暴露且由用户指定父的工程基准；全部正负结果保留，不称新独立盲测。重建、识别率或F几何不能单独证明每个恢复细节来自输入TX。完整目标 ACTIVE。

旧每组件80+本次20=100、组件轮次和300、串行140 pass，原F50训练另计。相对M35增加20轮，另报告同增量R49控制，不能把追加预算收益全部归于减参。

独立权重位于local_artifacts/pamf_restoration/r46_group_drop_r62_20261021/models/<arm>/final.pt；运行experiments.pamf_r46_group_drop.runtime.load_checkpoint后输入float32[B,2,256]原生IQ，输出同形IQ。恢复器无需教师/F/源缓存，识别时同一恢复IQ接原冻结F50默认头与FFT/PA/时域/统计分支，无另训头。


训练期间6次被动采样覆盖约25秒：GPU2整卡利用率均值100%，GPU3为97.66666666666667%。整卡值包含其他owner，不能归于本进程或作为独占加速证明；原始PID/采样留在本地，不写入公开结果。


## 固定方案与历史发布记录

# R62：保留归一化边界的整组压缩

状态：VERIFIED / TRAINING，已发布并实际训练，无新识别性能或 GPU 时延结论。完整 56 项针对性检查、CLI 与一次独立 P0/P1 审查通过；另已严格加载指定真实父权重，用合成 IQ 在 CPU 验证两组全部组件梯度有限且非零、父权重不变。实际 IQ、F、TX 标签和源／目标分数读取均为 0。真实两条 L_s 与 F50 的 no-query smoke 在启动器首步执行，PASS 后自动继续。

本轮继续压缩既有 U-Net 接收链恢复器。直接初始化与冻结 IQ 教师都固定为用户指定 R46 `prune_refine350521`，方法 `bbb8daf817de050b958d217b70d4c33f4aefe341`，源域固定 E20/4940。禁止继承 R48/R49/R53/R57/R58/R60 训练后的学生。父权重原物理划分与 original7/63/30、0TX/F/V/target 训练来源已由既有严格 loader 核对。该父版由用户在已暴露的历史工程基准上指定，不称新独立盲测。

现有按通道删减会改变每个 GN4 组内的统计总体。本方法只在 coarse spectral、coarse temporal 和 action 三个上下文 U-Net 中删除完整的一个原 GN 组，保留三组原通道；完整 prior、三阶段、六次 CG 和原有物理约束均保持。组选择只按父 encoder 两个残差块中四个长卷积的输出通道均方权重归一化后相加，再对原四组求和，稳定选前三组并按原顺序排列。初始化不读取 IQ，不拟合参数，也不依据源验证或目标分数选组。

两行具有相同通道索引、初始参数、参数量和训练预算，仅上下文归一化分组不同：

|行|上下文归一化|上下文参数|prior 参数|完整参数|
|---|---|---:|---:|---:|
|group_preserve|GN3，保留原每组通道宽度|136234|112594|248828|
|regroup_control|GN1，将相同保留通道重新合组|136234|112594|248828|

两行的上下文 temporal/action 宽度均为 9/15/24/33，spectral 为 6/9/15/24；prior 宽度为原 12/20/32/44。完整参数较指定父版减少约 29.01%，较现有 273765 参数图再少 24937 个参数；后者只是参数比较，并非初始化来源。prior 的 28 个共享长卷积权重、各 stage 独立 bias 与归一化 affine 均完整移植，仍参与后续联合优化。

对相同的输入特征值，GN3 保留子组的数学映射与原 GN4 对应组相同；已经用 FP32/FP64 的独立层级检查验证。上游卷积仍因删通道而改变，所以不保证整个恢复器等价，也不假定 CUDA 逐位相同。GN1 对照使该归一化差异可在同参数与同初始化下检验；两行归约方式与实测开销不同，不称完全计算匹配。

训练固定单 model seed 2026092701，E20/4940、global256/micro256、direct float32 buffer；重置 AdamW、LR 2e-4 至 2e-5、247 updates warmup、weight decay 1e-4、梯度裁剪 1、FP32/TF32 off。沿用 R07 已生成的 20 个受损视图，6300 条 L_s 与 56700 条 U_s 共 63000 条原训练记录；每行曝光 L126000/U1134000，父／学生 G 各 1260000 次输入，不访问 TX 标签、F 权重／输出、V 或 target 训练。

损失保持原 paired clean IQ NMSE＋0.05 三尺度相对功率 STFT 误差＋线性 1→0.1 的三阶段冻结父 IQ 教师误差。coarse/action/prior 全部联合优化，不新增可学习参数、校准 IQ、分类头、推理分支或额外 G 前向。旧参数继承 80 轮，本轮追加 20 轮；每组件累计 100 轮、组件和 300、串行累计 140 pass，F50 训练成本另计。该训练使用合法源域 clean 配对，属于 TX 无标签源域自监督。

同增量预算的未裁剪控制只读复用 R49 `retain_refine`，方法 `e96397bc3c7036f085317f9cc0e59074fb6ee24f`，不重复训练或加载为教师。它与新行拥有相同父、E20/4940、micro256、IQ 教师日程和原缓存，但通道、参数及归一化不同；只用于已登记的同预算比较。主对照为两行 GN3/GN1，另完整报告相对原 R17、指定父与 R44 的逐景差。

每行固定最终 E20 后，自动执行源域 1800×7、同卡完整成本测量、正式 168000×7 预测及独立 truth-last。正式七景是 clean、郊区 high/mid/low、urban high/mid/low；每行全部预测固定后才连接真值。源分数只作诊断，不据此过滤任一行。原 F50 使用原 jointly trained 默认头，同一个恢复 IQ 交给其 FFT/PA/时域/统计分支，无另训头。所有正负结果、逐 TX/RX/day、Macro-F1、修复／损伤和完整 20 轮曲线均保留，不反馈选模、参数或选择性重跑。

成本分 G 与 G＋F，B1/B32、64 条固定 source V、三轮交替顺序，报告中位/P90/P95、吞吐、驻留／增量峰值显存、训练时间与文件大小。计时为同步预加载 GPU 输入，不含传输、前处理及 loader。Conv/Linear MAC 不含 FFT、CG、norm、activation 和 F，不从参数减少推定实际加速。

资源为 N607 既有环境、物理 GPU 3/2 各一个本任务 worker，总数最多两张；每卡全 owner 少于 2 个已有 GPU PID 且空闲至少 18 GB 才分配。CPU 控制器与 scorer 为 2 线程，R41 score.lock 保证最多一个本任务 scorer。R60 本任务 GPU 进程已全部退出；其他 owner 的健康任务保持运行。正式发布使用独立不可覆盖的 `releases/pamf_r46_group_drop_20261021_r62` 和 `runs/20261021-phase1-restoration-R46-whole-group-s2026092701-m2-r62`。

固定命令入口为 `python -u -m experiments.pamf_r46_group_drop.runner run`；完整参数、Git method commit、环境/CWD、数据与输出路径写入发布登记的 `experiment.json`。启动器第一步执行真实父权重、F50 和两条 L_s 的 no-query smoke，PASS 后立即继续，无额外审批。预期产物为 resolved/comparison plan、smoke、workers/stages、逐行 initialization/training 20 records、固定最终权重、source/target 七景 predictions 与独立 summary、matched profile、assessment、完整报告及独立权重说明。

技术失败保留受影响行产物，不自动重试、停止或热修改健康行；原父来源／数据权限不匹配、非有限损失或输出、严格权重与分组合同不匹配均为实际技术失败。性能优先：原 R17 每景最多下降 1 个百分点只是退化上限，减参、层级归一化等价、源重建或容差通过不等于性能提升。low urban 70%、其他五受损场景距 raw clean ≤5 个百分点及 TX 细节真实目标均保持 ACTIVE。

流程边界：`REJECTED_EXTRA_GATE`——通用 brainstorming 中额外的设计／计划审批不属于项目最小流程白名单，按已有用户授权继续；不新增设计 approval、签名、成员哈希或收据链。完整流水线实现与必要验证完成前不宣称正式发布。


VERIFIED / TRAINING：固定方法 7a8fa7bd836946c3493635cbe0966591ecf14e1e 已发布，普通账户、单归档 SHA 比对和远端 compile 通过；真实指定父/F50/两条 L_s no-query smoke PASS。两固定行物理 GPU3/2 的实际 PID/CWD/argv/CUDA/CPU2、GPU PID、日志增长和至少一轮有限损失已核实。每卡一条本任务训练，总数2；其他 owner 不变。整组 GN3/GN1 同 248828 参数及初值，全 prior、原波形目标、E20/4940 后两行均自动七景正式测试/独立 truth-last/同卡成本，无源分数筛选，无新识别收益结论，完整目标 ACTIVE。
