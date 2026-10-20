# R17 精简版：目标性能保持约束

状态：ACTIVE。2026-10-20 用户明确要求本对话以后专门优化 R17 精简版，最大程度精简。最新确认的性能口径为：**以目标域测试为准，各场景准确率较原 R17 下降不超过 1 个百分点**（例如 80% 至 79%，不是相对下降 1%）。该要求覆盖本文件此前的目标零下降验收；零下降仍作为额外诊断保留。本文件覆盖本对话此前继续新增独立恢复路线的计划；原长期 low urban≥70% 与其他环境接近 clean 的目标保留，但压缩验收通过不等于原识别目标已经完成。不向“设计星地 IQ 无监督去信道网络”对话发送指令。

## 固定原版与验收

固定原版为 `20261020-phase1-restoration-unrolled-selfsup-s2026092701-m2-r17/unroll_unshared`，原 source-only 选择的最终 E20/4940 权重。原 R17 在目标访问前已固定；用户现指定它为压缩对照，基准已有暴露，后续不宣称新盲测。完整来源与数据权限沿用[原 R17 记录](../PAMF_UNROLLED_RESTORE_R17_20261020.md)，不以失败压缩权重或目标分数重新挑选父模型。

原正式结果使用同一原 F50 seed `2026092701`、其联合训练默认头、168000 个目标物理样本及固定七视图：

|场景|原 R17 准确率（%）|
|---|---:|
|clean|84.58690476190476|
|practical_high|81.31904761904762|
|practical_mid|79.57797619047618|
|practical_low_suburban|77.99642857142857|
|practical_high_urban|80.35357142857143|
|practical_mid_urban|73.68511904761905|
|practical_low_urban|67.21666666666667|

按用户最新确认，各目标场景准确率须≥原 R17 同场景准确率−1 个百分点，边界包含等号。每场景168000条对应最多减少1680条正确样本，使用原始整数正确数验收，不用四舍五入、七场景均值或多 F 平均掩盖超限场景。源域健康筛查和波形损失不能替代目标准确率证据。原始 raw clean 84.67619047619047% 与恢复 clean 分开报告。Macro-F1、逐 TX/RX/day、净修复/损伤和配对 clean 波形差距（全帧/主体/尾部）同步报告，出现冲突如实说明。默认头冻结，不另拟合分类头。

R04 已在最新要求前冻结方法并启动。其训练、候选、源选择、预测和运行时历史零下降字段保持原记录；最终用户验收由独立结果分析按1个百分点计算，同时保留零下降诊断。更新发生在任何新压缩候选目标结果返回前，不停止或热改健康训练，不追加落选目标测试。下文各历史运行段落中的“0下降”描述其原预登记口径，不覆盖本段最新用户要求。

精简以完整推理链参数量、FP32 模型状态大小、实测延迟和峰值显存衡量。分别报告 coarse/action/prior/数值求解成本，不能只报新先验的参数，也不能把去除优化器状态称为网络变小。若参数减少但延迟增加，保留实际取舍；不预先承诺“全局最小”。

## 已核实的压缩起点

本地严格加载原最终 checkpoint，使用一个合成零 IQ 进行静态算量，0 source/V/target 物理样本、0 训练、0 评分：

|组成|参数量|卷积与线性层 MAC/单 IQ|
|---|---:|---:|
|冻结 coarse|1305958|99098624|
|冻结 action estimator|851589|63422592|
|三份独立 prior|2507526|190906368|
|完整恢复链|4665073|353427584|

MAC 不含 FFT、复数 MAP/CG、归一化、激活、插值与访存，不等于实测延迟。三 prior 占约 53.75% 参数；冻结 coarse/action 合计 2157547 参数，单压 prior 存在此成本下限。原训练 checkpoint 为 39291705 字节，模型张量 18660548 字节，优化器张量 20061744 字节；删除优化器属于无损部署打包，参数量与推理计算不变。

首步无损导出已验证：部署文件为 18955769 字节，比原训练 checkpoint 减少 51.7563%。严格复载原 R17 图与全部同 dtype 模型张量，逐张量完全相同；实际原最终权重在 2 个合法 L_s IQ 上的输出逐位一致，最大绝对差为 0，0 TX 标签/V/target 访问，原训练 checkpoint 保留。8 项聚焦检查通过，覆盖输出等价、保留来源、禁止覆盖、非最终/权限不符来源拒绝。实现身份为 commit `f8ccada7bddda8091cb51808ad7cd05e531bc3bd`，本地与远端分支 OID 已独立比较一致。该产物减少训练状态存储，完整网络仍为 4665073 参数，尚未证明网络结构压缩后的目标性能。

## 后续执行范围

先完成无损部署导出与实际原权重等价检查，再研究三步 prior 的参数共享、窄化与 R17 源域蒸馏；只有源域证据支持时才进入已预登记的有限候选测试。保留 R17 三步软物理一致性作为首轮结构控制，避免同时改变先验和求解预算；减少步数或去除 coarse/action 作为后续独立消融，不能假定无损。

每个新候选在目标访问前固定结构、预算、源域选择与停止规则；训练/蒸馏只用合法 L_s/U_s 与其模拟配对，不用 V 反传、不用目标数据拟合。来源/物理 ID/数据契约核实后，按最小实验流程在 N607 发布。全部声明视图的预测固定后才独立 truth-last 评分。失败与下降结果全部保留；目标评分只作固定候选验收，不反馈调参、候选重排或选择性重跑。

已启动的 scratch_guidance、spectral_precision 健康任务保留原配置和自动测试，不追加、停止或热改；完成后只归档其既定结果，不作为新压缩父模型。下一次新增训练或测试发布专用于本 R17 精简目标。当前尚无缩小网络后保持目标性能的证据，目标保持未完成。


## 首轮固定 R17 压缩试验（完整结果已返回）

[首轮登记](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r17-compact-distill-source-s2026092701-m2-r01/report.md)：shared_mean完整2993389参数，三训练后prior逐元素平均并共享；factor_half完整3775105参数，三prior分别按半秩CPU SVD分解长卷积。原R17/source-selected finalE20/4940作为冻结源教师；三步×2CG与原coarse/action不变，各追加E20/4940，原R07全L/U缓存和顺序，不加额外clean拼接。最终IQNMSE+10倍三步teacher stage NMSE，0TX/F/V/target训练。prior累计40轮，完整继承训练80串行pass及蒸馏教师成本单列。

28项聚焦检查与一次独立P0/P1审查、唯一P1修复定点复审通过；方法commit`606b5b20943e9839fe97cbe15a6c19baf6ef0189`，归档比较/远端编译、实际原R17/原F50/2Ls无querysmoke PASS。新GPU4/5等待scratch根终态且实际owned handles退出，随后两source真实PID/CWD/命令/GPU/日志增长已核实，各完成第1轮/247更新；spectral6/7保持，CPU评分等spectral终态再ownlock。源V仅资格，逐七场景不降分+原波形护栏，合格者按最少完整参数固定选一个自动全目标确认，不在失败后补测落选者。当前完整结果：两候选均未通过固定源域资格，0新目标预测；未交付能保持目标性能的结构压缩版本。


### 原版参数冗余的只读证据

在首轮压缩源资格/新目标结果返回前，读取原R17最终权重的28个长卷积/步，0物理IQ、forward或TX/F/V/target访问。共同均值相对卷积权重Frobenius差异28.1663%，每个步间差异矩阵保留99%能量的秩中位61（范围23至121）；秩16的共享基底＋差异假设仍剩47.8833%的差异能量未表示。它仅反驳“步间差异天然极低秩”的直接假定，不能判定识别/恢复退化或排除函数级冗余。当前两个完整输出蒸馏候选继续固定E20，不增加新模型或改变预算/源选择；全部代数成本和边界见[当前run报告](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r17-compact-distill-source-s2026092701-m2-r01/report.md)。


首轮压缩完成结论：两候选均未通过固定源域资格，0新目标预测；未交付能保持目标性能的结构压缩版本。全量20轮日志、source七视图资格、自动选择/目标范围、完整成本和逐场景/逐TX/RX/day证据均见[本轮报告](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r17-compact-distill-source-s2026092701-m2-r01/report.md)。不从目标分数继续调参、替换父模型或补测落选者；完整持续目标保持ACTIVE，尚未证明最大可达精简或70%/接近clean全部要求。


## R02 纯函数保真精简（完整结果已返回）

[R02登记](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r17-fidelity-source-s2026092701-m2-r02/report.md)：仅依据R01源域负结果和原R17权重分析，在任何新压缩目标结果前固定两臂。step_norm_fidelity完整3118001参数（减少33.1629%），仅28长卷积weight跨三步共享原均值，保留各步独立bias/GroupNorm/1×1及其他状态；factor_half_fidelity完整3775105（减少19.0773%），保留R01半秩架构作损失目标匹配参考。全部从原R17教师初始化，0失败R01student初始化；仅10倍三步teacher IQNMSE，配对clean只作detached能量分母和日志诊断。关于原GT损失削弱保真的解释是待检验假设。

预算E20/4940、原完整63000缓存/顺序、3步×2CG/coarse/action均固定；prior累计40轮、完整继承80串行训练pass与教师成本分列，0TX/F/V/target训练、0头拟合。10项聚焦检查和唯一独立P0/P1审查PASS；方法commit `1515d73c5de3e962a822bbaca34182e28a6ae584` 已push/OID一致，归档/远端编译/实际原权重无querysmoke PASS。GPU4/5等待R01终态及实际handles退出后两source实际启动，PID/CWD/命令/GPU/日志增长已核实。CPU2线程/max1 ownscorer；每卡一own，保留其他owner。

源V1800逐七视图不降分与既有波形护栏合格后，按最少完整参数再名称选一个自动全168000×七视图、原F50/defaulthead目标确认；无合格则0target，不回流调参或失败后补测落选者。R02 完整结果：两候选均未通过固定源域资格，0新目标预测；未交付能保持目标性能的结构压缩版本。


R02训练期间的原权重只读SVD分析：84个长卷积的统一半秩成本精确复核3775105完整参数，原长卷积整体相对Frobenius误差.52557；排除输出宽32卷积时为3821809参数/.49508误差。逐核保留99%能量的SVD均不具备正参数节省。0IQ/forward/TX/F/V/target访问，无当前模型/预算/选择变更；只约束权重近似假设，不代表识别性能或函数压缩极限，见[成本与边界](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r17-fidelity-source-s2026092701-m2-r02/report.md#原始卷积低秩存储证据)。


后续R17候选的数学准备（未启动新实验）：原权重SVD误差不等于实际特征输出误差，因此实现了依输入未中心化二阶矩初始化低秩卷积的独立工具。固定正则化二阶矩C，分解W·Cholesky(C)，再逆变换左因子；保持卷积bias与教师权重。5项合成算术检查PASS：同一秩下按实际输出能量分配、流式patch计数/未中心化矩、奇异矩阵正则、解析/直接输出误差相符、拒绝不省参数或无效矩阵。这里只验证数学实现，0真实 IQ、0原 R17 forward、0目标访问，尚无真实R17校准、模型训练或性能结果。后续若采用，校准仅用合法L_s/U_s训练IQ，新增曝光和成本单列，必须在新运行前另行固定方案；不改当前R02。


R02 纯函数保真精简完成结论：两候选均未通过固定源域资格，0新目标预测；未交付能保持目标性能的结构压缩版本。全量20轮日志、source七视图资格、自动选择/目标范围、完整成本和逐场景/逐TX/RX/day证据均见[本轮报告](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r17-fidelity-source-s2026092701-m2-r02/report.md)。不从目标分数继续调参、替换父模型或补测落选者；完整持续目标保持ACTIVE，尚未证明最大可达精简或70%/接近clean全部要求。


## R03 激活校准精简（完整结果已返回）

[R03登记](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r17-calibrated-source-s2026092701-m2-r03/report.md)：半秩3775105（−19.0773%）与2/3秩4241809（−9.0730%），仅原R17教师/初始化，0失败student。固定R07源训练epoch01中seed2026102002选512 IQ、每层8位置未中心化二阶矩/固定ridge，以SVD(W·Cholesky(C))及左因子逆白化初始化；bias/GN/1×1和三步×2CG保持。纯10倍三步教师输出NMSE，E20/4940与原完整L/U缓存顺序；额外512校准曝光/实际L-U/初始化耗时单列，0TX/F/V/target训练或校准。

该科学方案在R02尚运行且未读取任何新压缩模型目标结果时固定。18项聚焦检查与唯一独立P0/P1审查PASS；方法commit `bbccd16408a0d9fc356b48f3579044e2284112dd` 已push/OID一致，归档/远端编译/实际原权重2Ls无querysmoke PASS。等待R02根终态且实际owned GPU handles全部退出后，两source GPU4/5实际PID/CWD/命令/GPU/日志增长及首轮已核实。每view源V准确率不低于原R17和原波形护栏，合格者按最少完整参数只选一个自动完整目标或0target；不依据目标重排/追加/重跑。R03 完整结果：两候选均未通过固定源域资格，0新目标预测；未交付能保持目标性能的结构压缩版本。


## R02 冻结源预测的配对诊断

复用全部已冻结/评分source预测，0原始IQ/权重/模型forward/目标访问。1800个low urban样本中，step_norm_fidelity49次换类（原正确变错20、原错变对17、净−3），factor_half_fidelity69次（31/20、净−11）；两版source clean均0次换类。独立CPU配对计数与此前完整汇总一致，完整TX/RX/day和输出距离见[同row补充](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r17-fidelity-source-s2026092701-m2-r02/report.md#冻结源预测的配对决策诊断)。波形均值误差不能替代决策保持；可在后续独立R17精简候选中检验冻结原默认头的后验/特征保真，但当前R03不变，不以此追溯补测或改变目标选择。分析实现commit `ed82cd7b711c592f7bb58a283c143034ca2a4514`，7项聚焦检查PASS。


## 任务保真数学原型（未启用）

R02已冻结源预测显示，low urban换类样本的原F置信度均值约.617/.652，低于全体.944；丢失正确样本的特征距离高于全体均值。这是源域事后关联，不是因果或目标证据。作为后续独立R17精简候选的准备，实现了教师/学生原默认头后验KL（原生T=1）加单位特征半平方距离；教师输出stop-gradient，已有reader全部冻结eval，仅向学生IQ传梯度，不添加或拟合头。7项合成检查PASS，覆盖一致输出零误差、特征尺度/共同logit偏移不变性、学生梯度/教师与reader不更新、非法shape和非冻结reader拒绝。0真实数据、0真实F/R17权重读取；微型未拟合模块仅为数学测试，未验证真实原F的梯度兼容或性能。

当前R03配置和原0F训练/校准权限保持。未来若使用该原型，必须作为新run显式登记已监督原F知识辅助、核实同源F来源并记录实际F访问/计算成本；恢复器无新增TX标签读取不等于不含F监督知识。尚未启动新的F辅助压缩实验，不以原型检查代替真实checkpoint smoke、源资格或目标测试。


R03 激活校准精简完成结论：两候选均未通过固定源域资格，0新目标预测；未交付能保持目标性能的结构压缩版本。全量20轮日志、source七视图资格、自动选择/目标范围、完整成本和逐场景/逐TX/RX/day证据均见[本轮报告](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r17-calibrated-source-s2026092701-m2-r03/report.md)。不从目标分数继续调参、替换父模型或补测落选者；完整持续目标保持ACTIVE，尚未证明最大可达精简或70%/接近clean全部要求。


## R04 冻结 F 知识约束精简（完整结果已返回）

[R04登记](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r17-task-source-s2026092701-m2-r04/report.md)：step_norm_task3118001（−33.1629%）与activation_task3775105（−19.0773%），仅原R17初始化，不加载失败学生。依据R02已保存源预测配对诊断，加入冻结原F50/defaulthead的T1后验KL及单位160维特征距离，教师参照同输入原R17恢复结果；原10倍三步IQ蒸馏保持。只更新精简prior，0新TX/V/target标签训练，原F/默认头0更新；明确使用源监督F知识，不称F-free无监督。E20/4940/原完整L-U顺序不变，Fteacher/student各1260000前向及输入反传成本单列；stepnorm无校准，activation额外512源LU校准。

本轮源规则改为健康筛查：每视图accuracy≥原R17−1个百分点及原波形护栏。该规则仅适用R04，不追溯重判或补测R01至R03，不将不同规则的合格率当作性能提高。所有源候选完成后按最少参数只选一个自动全目标或0target，无fallback；最终目标仍按七场景原始正确数逐项≥原R17、0下降。

19项检查和唯一独立P0/P1审查PASS；方法commit `a310b3b5ff30e23584e7bc950e8a0ff96ac685fb` 已推送/OID一致，归档SHA/远端编译/真实原权重2Ls smoke PASS，实际F输入梯度有限非零且F/默认头冻结。GPU4/5两source真实PID/CWD/命令/GPU/日志增长与首轮已核实。R04 完整结果：两候选均未通过预登记源健康筛查，0新目标预测；未交付能保持目标性能的结构压缩版本。


R04 冻结F任务精简完成结论：两候选均未通过预登记源健康筛查，0新目标预测；未交付能保持目标性能的结构压缩版本。全量20轮日志、source七视图资格、自动选择/目标范围、完整成本和逐场景/逐TX/RX/day证据均见[本轮报告](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r17-task-source-s2026092701-m2-r04/report.md)。不从目标分数继续调参、替换父模型或补测落选者；完整持续目标保持ACTIVE，尚未证明最大可达精简或70%/接近clean全部要求。


## R05 固定双模型目标域对照

[R05登记](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r17-compact-target-comparison-s2026092701-m2-r05/report.md)：按用户最新“性能测试以目标域为准、最多下降1个百分点”要求，固定评测R04已完成E20/4940的两份最终权重，step_norm_task3118001（−33.1629%）和activation_task3775105（−19.0773%），每版同168000物理样本×七视图、原F50/defaulthead，0新训练或头拟合。两版源准确率均在1个百分点范围内，但原R04波形NMSE护栏未过；旧source_qualified=false/selected=null/0target记录保留，不伪造源资格。波形指标不替代目标身份性能，新建显式comparison_plan作为固定双行目标比较依据。两行和checkpoint在新目标结果前固定，不回流训练、超参、候选重排、追加或选择性重跑；全部结果包括负结果保留，不称新盲测。

28项检查及唯一新适配层P0/P1审查PASS；N607普通账户预检VERIFIED，旧实际GPU进程已退出，GPU4/5空闲。新不可覆盖root、一次归档比较/远端compile、launcher-first实际R04-final/F50/2Ls no-query smoke及实际PID/CWD/cmd/GPU/日志增长核实按最小流程执行。R05 实际发布 VERIFIED：方法commit `81c39cb908373d4e19b70ec8406f4083529da99a` 已推送/OID一致，归档/远端编译/实际两版final+F50+2Ls无query smoke PASS；两个target GPU4/5实际PID/CWD/命令/GPU/日志增长与首个clean视图完成已核实。R05完整目标结果已返回：符合七场景1个百分点容差：step_norm_task, activation_task。


R05完整结果：符合七场景1个百分点容差：step_norm_task, activation_task。两版均完成同168000×七视图、原F50/defaulthead、独立truth-last；整数正确数验收、逐TX/RX/day、clean恢复差距、净修复/损伤与完整成本见[本轮报告](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r17-compact-target-comparison-s2026092701-m2-r05/report.md)。保留原R04源护栏失败记录与两版全部目标结果，不回流训练/调参/重跑。长期70%/接近clean及最大可达精简仍按实际证据核对，完整目标保持ACTIVE。


## R06 共享低秩进一步精简

[R06 登记](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r17-shared-factor-s2026092701-m2-r06/report.md)：固定 shared_half 2821345 / shared_quarter 2582905 完整参数，约减少39.52%/44.63%。从原 R17 源最终权重取长卷积均值后 CPU SVD，跨三步共享左右因子，各步 bias/GN/1×1 和其他状态独立；0校准IQ，3步×2CG不变。只加载原R17teacher/init，不加载R04/R05或失败学生。固定同R04损失、原source63000缓存/顺序与E20/4940预算；原F50/defaulthead冻结，0TX/V/target训练和0头拟合，使用源监督F知识。

两行在新运行前固定，source/V/波形仅诊断；每行冻结后自动source七视图/独立truth-last/成本测量及target168000×七视图/独立truth-last。按原R17逐场景最多下降1个百分点，不以均值或波形覆盖超限，不用目标结果改结构、超参、父权重、预算或重跑。40项检查和唯一独立P0/P1审查PASS，普通N607预检VERIFIED，旧R05实际GPU句柄已退出、GPU4/5空闲；后续一次归档比较/远端编译/launcher-first真实R17+F50+2Ls noquery smoke、实际PID/CWD/cmd/GPU/loggrowth。实际发布 VERIFIED：方法 `1ccc14248d025006b1cf38cdf6779685156ee092` 已推送/OID一致，归档/远端编译及真实原R17+F50+2Ls无query smoke PASS，GPU4/5实际PID/CWD/完整命令/GPU/日志增长与E1/247均已核实。R06两行完整训练与目标测试均已结束，完整结果与两种参考基准差值见本节末及R06报告。

旧R04/R05显存profile包含checkpointpayload，原R17含optimizer而精简final不含，不能解释为纯推理显存收益；数值保留并补明口径。R06排除GPU上的payload/optimizer，分别报告resident/peak/增量peak。目标准确率和参数结论不受此统计口径影响。


## R07 四候选并行扩展

用户最新要求空闲服务器一次发布多个、快速迭代，覆盖此前等R06完成再追加的次序约束。[R07登记](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r17-parallel-compact-s2026092701-m4-r07/report.md)固定四行：shared_three_eighths2702125、shared_eighth2463685、shared_quarter_pointwise2471609、shared_eighth_pointwise2352389。新GPU2/3/6/7各一own worker，R06健康4/5继续，共六路；R06/R07共享R06score.lock、跨六行最多一个CPU scorer。原R17源权重CPU均值/SVD，另两行共享原1×1权重，保留逐步bias/GN；只原R17teacher/init，不加载R04/05/06学生，R06尚无目标结果。固定同损失/数据/顺序/E20/4940；每行source诊断后自动全168000×七目标场景，独立truth-last与逐场景1pp验收，0target训练/参数反馈/重排/重跑。40项检查/唯一P0P1审查PASS，普通N607预检VERIFIED，四卡空闲，新root不存在。R07实际发布 VERIFIED：方法 `f03015d75428f156430cff60633e1c99ec4983b4` 已推送/OID一致，归档/远端编译和四候选实际原R17/F50/2Ls smoke PASS，实际PID/CWD/完整命令/GPU2/3/6/7/日志增长与首轮完成已核实。原R06健康4/5继续，跨六行串行独立评分；尚无完整新目标结果。


## R08 压缩主导成本的上下文

纯结构盘点（随机初始化/0checkpoint/IQ/F/TX/V/target）显示原coarse1305958+action851589=2157547，占最小R07完整2352389的91.7%；仅压prior已接近此路线的参数下限。在R06/R07新target分数尚未读取时，预登记[R08两行](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r17-context-compact-s2026092701-m2-r08/report.md)：context半秩+priorquarter完整1822977，context半秩+prioreighth/原1×1共享完整1592461。只原R17teacher/init；84个context长卷积独立CPU半秩SVD后完全冻结为1397619参数，prior保持原来源共享压缩并优化20轮。context为确定性源权重近似，不能称原context逐位不变。

teacher必须用自身原context，student用自身压缩context；helper及sentinel负向检查保证隔离，额外teachercontext前向1260000单列。其余同source63000/数据顺序/E20/4940/原F50defaulthead冻结/固定蒸馏损失，0新TX/V/target训练或F/head/context更新；全部source仅诊断，逐行自动168000×七target，独立truth-last和1pp整数验收。两行在新结果前固定，不用R06/R07结果改超参、结构、父权重、重排或重跑。

35项检查/唯一独立P0P1审查PASS，普通N607预检VERIFIED，R06实际GPU4/5句柄仍2条。launcher首步实际原R17/F50/2Ls smoke通过即等待R06根终态且实际GPU句柄退出，不读其评分，之后4/5各一own接续；R07健康2/3/6/7继续，共享R06score.lock保持一个own CPU scorer。R08实际发布 VERIFIED：方法 `2e23af45f4a5fc26c3d44c4efa4305ac36ef45db` 已推送/OID一致，归档/远端compile和真实原R17/F50/2Ls独立context smoke PASS；R06完整测试/实际句柄退出后，GPU4/5新实际PID/CWD/命令/GPU/loggrowth及E1已核实。R07健康四卡继续，全部仍共享一个CPU scorer；尚无R08目标准确率。


R06本轮两种结构全部完成：符合七场景1个百分点容差：shared_half, shared_quarter。E20/4940、source七视图诊断及同168000×七目标场景、独立truth-last、实际进程退出均核实；完整训练曲线、精确1pp验收、相对已验收3118001版本差、配对clean差距、TX/RX/day和成本见[本轮报告](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r17-shared-factor-s2026092701-m2-r06/report.md)。两份完整恢复器已取回本地并严格加载验证；本轮未追加其他结构，目标反馈未用于改训练/超参/重跑，长期指标依实际结果报告。


## R09 压缩同时优化恢复与识别

用户最新明确“压缩的同时也要优化”，1个百分点只是逐场景下降上限，不能把参数减少及容差通过称为识别提升。[R09固定四行](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r17-joint-optimize-s2026092701-m4-r09/report.md)为两种既定R08结构（完整1822977、1592461参数）×teacher任务保真/配对source-clean纠正目标。两种目标都联合训练压缩coarse/action/prior，另两行增加1倍最终source-clean IQNMSE，并将冻结原F50/defaulthead的参照从原R17恢复改为对应source-clean。10倍三步原R17IQ保真、1倍T1KL及0.5倍单位160维特征平方距离固定；只合法源训练L/U的配对clean，0新增TX/V/target训练或头拟合，不称F-free无监督。纠正目标同时改变IQ和F参照，不能归因于单一项。

只从原R17源选择权重初始化，不加载已测学生；原teacher独立context且全部冻结，student独立context可反传，直接绕过旧coarse的no_grad路径。E20/4940、source63000缓存/顺序/学习率/三步×2CG不变；继承coarse20+action20+R17prior20与本轮joint20共80串行训练pass，每组件累计40、component-epoch总和120，不能将120当串行轮数。原F50预训练成本与双F/双context前向及输入反传单列。

四行在新R09目标预测前固定，最终E20冻结，不以源V或目标重排、调损失、改秩/预算、换父模型、追加或选择性重跑。R06已暴露目标结果如实保留；本方案依据用户性能优化要求和合法source-clean契约建立匹配对照，不声称新的独立盲测。每行自动source诊断与168000×七目标场景，所有视图预测完成后独立truth-last评分；逐场景1pp原始整数验收，同时报告真实涨分/损伤、lowurban是否70%、配对clean恢复差距与相对已验收3118001版本差。

42项检查/唯一独立P0P1审查PASS；普通N607预检VERIFIED，R07实际四条句柄均退出。launcher-first真实原R17/F50/2Ls smoke通过后确认R07根终态和实际进程退出，再GPU2/3/6/7各一own worker；R08健康4/5继续，共六路，所有own scorer共享R06score.lock。R09实际发布 VERIFIED：方法 `63a8b45b502351ba72aaf80f83b377036ce255fe` 已推送/OID一致，归档/远端compile和真实原R17/F50/2Ls完整联合梯度smoke PASS；R07完整测试/实际句柄退出后，GPU2/3/6/7新实际PID/CWD/命令/GPU/loggrowth及E1已核实。R08健康4/5继续，全部仍共享一个CPU scorer；尚无R09目标准确率。


R07本轮四种结构全部完成：符合七场景1个百分点容差：shared_three_eighths, shared_eighth, shared_quarter_pointwise, shared_eighth_pointwise。E20/4940、source七视图诊断及同168000×七目标场景、独立truth-last、实际进程退出均核实；完整训练曲线、精确1pp验收、相对已验收3118001版本差、配对clean差距、TX/RX/day和成本见[本轮报告](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r17-parallel-compact-s2026092701-m4-r07/report.md)。四份完整恢复器已取回本地并严格加载验证；本轮未追加其他结构，目标反馈未用于改训练/超参/重跑，长期指标依实际结果报告。


R07冻结完整汇总补核：2352389参数版比此前已验收3118001版再减24.554578%，比上一轮最小2582905版再减8.924680%；对两个参考版本均clean正确数相同，六种practical全部提高。相对2582905版lowurban净+820条/+0.488095个百分点，相对3118001版仅+9条/+0.005357个百分点，单seed不称稳定总体提升。原R17七场景1pp仍全部通过；lowurban67.361905%未到70%。0新模型forward/训练/原始IQ/truth访问，完整差值/波形见[R07补充](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r17-parallel-compact-s2026092701-m4-r07/cross_run_comparison.json)。R08/R09健康固定训练保持，目标结果未回流改配方。


## R10 配对源 clean 直接纠正

在未读取R08/R09目标结果前固定[R10两行](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r17-clean-corrective-s2026092701-m2-r10/report.md)：沿用同既定1822977/1592461完整结构、原R17source-final CPU SVD初始化、原source63000缓存/顺序/seed/优化器/E20/4940，全部coarse/action/prior联合优化。相对R09对应clean arm，仅删除10倍三阶段R17波形模仿项；损失为最终sourceclean IQNMSE（clean detached mean energy floor1e-4）+T1KL(Fclean||Frestored)+0.5单位F160距离平方，系数固定，不从目标分数调参。原R17对象训练前删除，0原R17 teacher/context训练前向；原F50/defaulthead冻结，Fclean/student各1260000，source1260000、0新TX/V/target训练/头拟合。不称严格无监督或同训练算量。

继承coarse20/action20/R17prior20+一次joint20=80串行pass，各组件累计40/component sum120；只原R17初始化，0已测学生继承。所有行固定最终E20，source仅诊断，自动全168000×七目标视图/独立truth-last，原R17每场景1pp底线及真实提分/lowurban70/配对clean差距分别报告，全部负结果保留。38项检查和唯一独立P0P1审查PASS，普通N607预检VERIFIED，R08实际GPU句柄仍2条；launcher-first实际原权重/F50/2Ls smoke通过即等待R08根终态及实际句柄退出，再接续GPU4/5，R09健康2/3/6/7继续，max6own GPU/R06score.lock共享max1own CPU scorer。R10实际发布 VERIFIED：方法 `d113672a6e51e75eb3ea0f7ecce0500e67f92966` 已推送/OID一致，归档/远端compile和真实原R17初始化/F50/2Ls直接clean联合梯度smoke PASS；R08实际句柄退出后，GPU4/5新实际PID/CWD/命令/GPU/loggrowth及E1已核实，R09健康四卡继续；尚无R10目标准确率。


R08本轮两种结构全部完成：两版均未通过七场景最多下降1个百分点验收。E20/4940、source七视图诊断及同168000×七目标场景、独立truth-last、实际进程退出均核实；完整训练曲线、精确1pp验收、相对已验收3118001版本差、配对clean差距、TX/RX/day和成本见[本轮报告](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r17-context-compact-s2026092701-m2-r08/report.md)。两份完整恢复器已取回本地并严格加载验证；本轮未追加其他结构，目标反馈未用于改训练/超参/重跑，长期指标依实际结果报告。


R09本轮四个联合优化对照全部完成：符合七场景1个百分点容差：joint_teacher_quarter, joint_clean_eighth_pointwise。E20/4940、source七视图诊断及同168000×七目标场景、独立truth-last、实际进程退出均核实；完整训练曲线、精确1pp验收、相对已验收3118001版本差、配对clean差距、TX/RX/day和成本见[本轮报告](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r17-joint-optimize-s2026092701-m4-r09/report.md)。四份完整恢复器已取回本地并严格加载验证；本轮未追加其他结构，目标反馈未用于改训练/超参/重跑，长期指标依实际结果报告。


## R11 匹配上下文压缩矩阵

[R11固定四行](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r17-context-grid-s2026092701-m4-r11/report.md)沿用R10直接paired source-clean IQNMSE+T1KL(Fclean||Frestored)+0.5单位F160平方距离、相同source缓存/顺序/seed/优化器/E20/4940，补齐full与quarter context×两种固定prior，R10 half context作为匹配对照。完整参数2582905、2352389、1212813、982297，context分别2157547、2157547、787455、787455。full context从原R17逐位复制，quarter context的84长卷积CPU SVD；全部coarse/action/prior联合优化。只原source-selected R17/cd977448 CPU初始化，0已测学生继承；原R17在optimizer前删除，正式R17训练forward0，原F50/defaulthead冻结，无新TX/V/target训练标签。

full/half/quarter静态成本早于R08目标，现补齐固定上下文层级；已暴露R08/R09 benchmark如实披露，不称新盲测，不按目标调秩、损失、预算、父权重、顺序或重跑。每行自动完整168000×七目标预测后独立truth-last，原R17逐场景1pp底线、实际提升/lowurban70/配对clean差距分别报告。继承coarse20/action20/R17prior20+joint20为80串行pass，各组件累计40/component sum120，F50预训练另计。42项合成/协议检查与唯一独立P0/P1审查PASS；普通N607预检VERIFIED，R09实际句柄退出，GPU2/3/6/7空闲，各一own worker，R10健康4/5保持，max6own GPU，R06锁max1own scorer。R11实际发布VERIFIED：方法 `5f8c563e96766fee9f4601fe64bf0da4395892ae` 推送/OID一致，归档/远端compile和真实原R17/F50/2Ls smoke PASS，GPU2/3/6/7四个新实际PID/CWD/cmd/GPU/loggrowth及E1均核实；尚无R11目标性能。


R10本轮两个直接纠正对照全部完成：符合七场景1个百分点容差：clean_only_eighth_pointwise。E20/4940、source七视图诊断及同168000×七目标场景、独立truth-last、实际进程退出均核实；完整训练曲线、精确1pp验收、相对已验收3118001版本差、配对clean差距、TX/RX/day和成本见[本轮报告](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r17-clean-corrective-s2026092701-m2-r10/report.md)。两份完整恢复器已取回本地并严格加载验证；本轮未追加其他结构，目标反馈未用于改训练/超参/重跑，长期指标依实际结果报告。


## R12 八分之一上下文

[R12固定两行](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r17-eighth-context-s2026092701-m2-r12/report.md)补齐早于R08目标的0IQ静态容量审计层级context=.125，两种既定prior对应完整参数907657/677141、context482299。沿用R10/R11固定直接source-clean IQNMSE+T1KL(Fclean||Frestored)+0.5单位F160平方距离，以及相同source缓存/顺序/seed/优化器/E20/4940；全部coarse/action/prior联合优化。R10 half与R11 full/quarter是已登记匹配对照，R10 benchmark已暴露而R11尚无目标结果；不按目标选秩/系数/预算/父权重或排序、选择性重跑。只原source-selected R17/cd977448 CPU初始化，0已测学生继承，R17在optimizer前删除，0R17正式训练forward，原F50/defaulthead冻结/0新TX/V/target训练标签。每行自动完整168000×七目标预测后独立truth-last，原R17逐项1pp、真实升降/lowurban70/paired-clean差距分别报告，所有负结果保留。

继承coarse20/action20/R17prior20+joint20=80串行pass，各组件40/component sum120，F50预训练另计。40项合成/协议检查与唯一独立P0/P1审查PASS；普通N607预检VERIFIED，R10实际句柄退出，GPU4/5空闲，R11健康2/3/6/7保持，max6own GPU/max1own scorer共享R06锁。R12实际发布VERIFIED：方法 `152f11283ba81b81f1a9e58b5b094f656f1c308b` 推送/OID一致，归档/远端compile和真实原R17/F50/2Ls smoke PASS，GPU4/5两个新实际PID/CWD/cmd/GPU/loggrowth及E1均核实；尚无R12目标性能。


已完成R05至R10全部16行的[目标域并列表](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r17-clean-corrective-s2026092701-m2-r10/cross_run_compact_summary.md)：11行通过原R17逐场景1pp，最少1592461参数的R09/R10两版存在场景取舍。逐场景支配关系只作描述，不回流R11/R12训练或权重继承；全部负结果保留，无新增IQ/模型/truth访问。


## 用户指定79万基底：R13

用户最新明确“在79万参数的模型基础上继续进行压缩优化”，后续新工作切换为R38 dense_half完整794529参数、固定source E20/4940、method e0868f0a0c3cf056a1753a5b4e5ce4f0bff84381。该模型训练0TX/F/V/target优化，原R17/cd977448合法源祖先；用户在已暴露benchmark后指定此基底，不称新独立盲测。此明确请求替代后续只原R17初始化的旧方法默认，不追溯改现有R11/R12健康行。

[R13固定六行](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r38-successor-s2026092701-m6-r13/report.md)：原宽度794529、进一步GN4通道收窄442805、204561，各自IQ与task两目标。原宽度逐位copy，窄结构完整coarse/action/prior联合训练、decoder旧宽偏移skip及谱首层投影保持；0IQ校准。全部final pairedclean IQNMSE+1×冻结R38三stage IQ锚，task额外冻结原F50 clean-reference T1KL+0.5单位F160平方距离，IQ控制正式训练不加载/前向F。两目标有同宽同增量预算对照，不按目标筛行或重跑。

只追加E10/2470/L63000/U567000，重置AdamW1e-4/warm247/cos1e-5/WD1e-4/clip1，global256/micro64按真实尾batch归一化。原祖先coarse20/action20/R17prior20+R38joint20+newjoint10=90串行pass，各组件累计50/component sum150，F50E50另计；0新TX/V/target训练、0F/head/teacher更新。每行自动source1800×7及同168000×七目标全部预测后独立truth-last；相对用户R38和原R17逐场景1pp均报告，lowurban70和真实升降/paired-clean差距独立报告，所有负结果保留。

41项检查及唯一P0/P1定向修复复审PASS；E10预测入口已显式传递合法预算，共享旧默认E20保持。普通N607预检VERIFIED，R11/R12六个实际GPU句柄仍LIVE。R13 launcher首步真实R38/F50/2Ls noquery smoke通过即按各卡旧row终态且实际owned句柄退出逐卡派发，GPU2–7各一own/max6、R06锁max1own scorer。R13实际QUEUE_VERIFIED：方法 `593bf5905f50fb7894e475b062bc191a2837a149` 已推送/OID一致，单归档/远端compile/help及真实R38/F50/2Ls smoke PASS，实际CPUcontroller PID/CWD/cmd/commit/等待日志核实。六行按R11/R12相应旧卡句柄退出自动接续；当前排队，不冒称GPU训练或目标性能完成。


R11固定四行完整完成：七场景原R17逐项1pp验收通过：quarter_context_eighth_pointwise。完整源诊断、目标七场景/精确1pp/实际升降、配对clean、TX/RX/day、完整日志和成本及匹配R10 half-context对照见[R11报告](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r17-context-grid-s2026092701-m4-r11/report.md)。四份完整恢复器取回且严格加载VERIFIED；长期目标依实际指标判断，仍保留全部历史已验收模型与负结果。


R12固定两行完整完成：七场景原R17逐项1pp验收通过：eighth_context_quarter。完整源诊断、目标七场景/精确1pp/实际升降、配对clean、TX/RX/day、完整日志和成本及匹配R10 half-context对照见[R12报告](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r17-eighth-context-s2026092701-m2-r12/report.md)。两份完整恢复器取回且严格加载VERIFIED；长期目标依实际指标判断，仍保留全部历史已验收模型与负结果。


R13 六行完整完成 E10/2470、source1800×7 与 target168000×7、逐行独立 truth-last，实际所属进程均退出，六份模型取回并严格加载。通过原 R17 与用户 R38 双参考逐场景 1pp 的行：keep_iq。进一步缩小且通过的 F-free 行：无；IQ 与使用监督 F 知识的 task 行分别报告，不静默替代权限。原宽度 keep_iq 的 low urban 为 68.502976%，参数仍为 794529；压缩保持与长期 70%/5pp/TX 真实性目标分别判断。全部负结果、逐 TX/RX/day、完整 10 轮曲线、全链参数/MAC/存储/实测延迟/显存和独立模型入口见 [R13 完整报告](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r38-successor-s2026092701-m6-r13/report.md)。本轮未追加目标反馈训练或选择性重跑。


## R14：冻结上下文、仅压缩先验

用户要求分析R13失败，并继续从原794529参数R38做F-free压缩和性能优化。完整60轮日志无数值故障；窄版末轮stage IQ拟合误差增大且梯度裁剪频繁，不能据此断言容量极限。冻结原548795参数coarse/action，仅压缩prior，固定794529/689129/613041三版；原R38 source E20/4940作为父，0继承R13目标评分后的状态。最终IQ配对NMSE+原R38三stage IQ锚，0TX标签/F/V/target训练，E10/2470，同宽对照、0IQ校准，coarse/action累计40、prior累计50。每版完成1800×源七视图与168000×目标七场景，逐版所有预测后独立truth-last，0源分数筛选。GPU2/3/6各一own，保持其他owner；15项检查及唯一P0/P1审查PASS。实际结果、完整成本与目标差距待正式评分，未宣布性能成功。


R14 三行完整完成 E10/2470、source1800×7 与 target168000×7、逐行独立 truth-last，实际所属进程均退出，三份模型取回并严格加载。通过原 R17 与用户 R38 双参考逐场景 1pp 的行：prior_keep。进一步缩小且通过的 F-free 行：无；全部三行为F-free，coarse/action累计40，prior累计50。原宽度 prior_keep 的 low urban 为 68.117857%，参数仍为 794529；压缩保持与长期 70%/5pp/TX 真实性目标分别判断。全部负结果、逐 TX/RX/day、完整 10 轮曲线、全链参数/MAC/存储/实测延迟/显存和独立模型入口见 [R14 完整报告](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r38-prior-compact-s2026092701-m3-r14/report.md)。本轮未追加目标反馈训练或选择性重跑。


## R15：保持通道/GN的低秩先验

R14 source完整30轮表明context冻结后裁剪几乎消失，但窄prior末轮teacher-stage NMSE仍为同宽约17/29倍。新两行仅prior内部SVD，保留全部通道/GN/skip/输出尺寸，固定rank fraction .75/.625，仅节省参数卷积才替换，left/right均跨stage共享、bias独立；完整779745/750035参数，原548795context冻结。只原R38 source E20/4940独立父权重，E10/2470，0TX/F/V/target训练；context累计40/prior50，每版全七场景正式测试。同预算control复用R14 prior_keep既有结果，不重训/重测；结构参数依据source拟合与结构事实，0target反馈搜索。14项检查与唯一P0/P1审查PASS。


R15两行均完成E10/2470、source1800×7、target168000×7及独立truth-last，实际进程退出、模型严格取回加载。通过双参考1pp：无。完整目标结果、成本及局限见 [R15报告](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r38-factor-prior-s2026092701-m2-r15/report.md)。0目标反馈调参或重跑，全部负结果保留。


## 当前目标收紧为原79万基底逐场景0.8pp；R16纯教师IQ压缩

用户持续目标：在固定原R38 794529参数基础继续压缩优化，每个七场景最多下降0.8个百分点，即168000样本最多少1344条正确预测（整数500×损失≤4×样本数）。原R17单列参考，不以均值或旧1pp代替。新三版750035/720185/689129，prior为SVD .625/.5或GN4 .75；原548795 context精确冻结。原R38 source E20/4940独立父，不继承已测R14/R15学生。只mean3stage IQ教师NMSE优化，pairedclean detached仅诊断：精确父的教师误差为零而pairedclean梯度会拉离父函数，按该源侧事实检验目标冲突，不依据目标分数选rank/系数/轮次。E10/2470，0TX/F/V/target训练，context累计40/prior50；每版全部source1800×7和target168000×7预测后独立truth-last。18项检查及唯一P0/P1审查PASS。长期70%/5pp另列，本次0.8pp目标尚待证实。


## 本轮R17：同曝光密集更新对照

R16完整source30轮记录显示clip全0，最后五轮教师误差仍降15–16%；本轮在其目标评分前固定，不读其target/F分数。新三行750035/750035/689129，保持纯教师IQ目标与原R38 E20/4940独立父、source630000曝光与10轮不变，B256/micro64的2470更新改B64/micro64的9850更新，optimizer成本不同完整披露；不延长已测学生。factor625两行同结构只峰值LR1e-4对3e-4，warm985/floor1e-5相同；dense750为3e-4。原548795 context精确冻结，prior累计50/context40，0TX/F/V/target训练。每版全部source1800×7与target168000×7，逐行所有预测后truth-last，原79万基底逐场景精确0.8pp/1344条；0源门槛或选择性重跑。19项检查及唯一P0/P1审查PASS，R16健康预测保持，GPU2/3/6实际旧句柄退出后接续，共最多3 GPU workers。


R16三行全部完成纯教师IQ目标E10/2470、source1800×7、target168000×7及独立truth-last，实际所属进程退出，模型取回严格加载。通过当前原79万基底逐场景0.8pp的更小F-free行：无。完整识别、净纠错/TX/RX/day、波形证据及完整成本见 [R16报告](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r38-fidelity-only-s2026092701-m3-r16/report.md)。0目标反馈调参/重跑；70%/5pp长期目标另列。


本轮R17三行全部完成纯教师IQ目标同曝光B64/E10/9850、source1800×7、target168000×7及独立truth-last，实际所属进程退出，模型取回严格加载。通过当前原79万基底逐场景0.8pp的更小F-free行：factor625_b64_lr1, factor625_b64_lr3, dense750_b64_lr3。完整识别、净纠错/TX/RX/day、波形证据及完整成本见 [本轮R17报告](../../automation_reports/CV-SincNet/20261020-phase1-restoration-r38-dense-steps-s2026092701-m3-r17/report.md)。0目标反馈调参/重跑；70%/5pp长期目标另列。
