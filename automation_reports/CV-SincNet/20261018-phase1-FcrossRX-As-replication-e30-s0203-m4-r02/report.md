> RUNNING／VERIFIED：实际代码 `63b7c8cd7bb4139a2d800e6f4c4a7f1f2163fd47` 的r02已通过完整bootstrap导入及3074NN真实CP/四F合成检查，四row在GPU4/5/6/7独立运行；原数据角色、native seed和实际F工厂seed均已核对。B00两行已完成首epoch222更新，B01两行优化日志已增长。尚无训练完成或新正式结果，不把初始化检查当作跨seed收益。

> 技术替代r02：首轮导入前失败，未开始训练。此次仅修复顶层包标记阻断legacy namespace，加入原bootstrap纯导入验证；数据、四行、seed、目标和E30预算不变。保留r01，无原地热改。

# clean 保留式 As：补齐模型 seed02/03 的 E30 基线确认

本轮补齐联合目标中的跨 seed As 基线证据，不提出新 Ag，不扩展已失败的 BG/WHITE/HardLCE/Tukey/AMPM 等候选。已有 P43 的 As 独立源域适应判据通过，新增 Ag 判据失败的结论保持；固定使用 P43 B00/B01 的学习规则及成本作为唯一参照，不根据其正式分数挑选配方。代码基于已发布commit19aedc3604fed9c193237dd92366164178af0983，仅增加真正的02/03模型随机性、独占输出与聚合。

## 唯一矩阵及数据

新增四行：B00/B01 × model seed2026092702/03。seed2026092701的 P43 B00/B01 作为已冻结控制复用，不重训、不重测。每行从零 own20（13L+7U）再自身 own10（10U），累计E30、6660更新；全程L仍更新，自己的student/EMA严格恢复，AdamW重置与原cos20/cos10、lr2e-4/WD1e-4保持。不加载历史E50/E200作训练初始化。实例化F所用的旧AF50配置仅为架构工厂配置，不代表训练50轮或加载其权重。

完整L6300/U56700隐藏/V27000、6TX、源RX[1,3,4,6,8]/day1/2/3、eq1、split392005、IQ256/25MHz、原RMS center=false。没有训练RX留出或L_dev。每段核对原物理角色ID EXACT_MATCH；只允许复用同row自己的stage20父模型，继承来源严格核对。V只验证、不能反向传播或更新EMA/原型/归一化状态。U TX始终隐藏。

固定F/SIG_TIME_X3、统一Sinc CENTER_25K/CENTER_UNITS、NoExplicitCFO、joint160、既有内部门控/参考响应及默认cosine30头，骨干和头联合训练。原U/EMA/门控保持。不新增头、分支、Ag、一致性损失、教师或对抗模块。

每步6×128 L网络包，目标为.5commonRD+.5mixedRD+原SupCon+.25rawRD，总L系数1.25。B00的mixed槽为ordinary64+raw64；B01为ordinary64+原完整practical64，因此.75ordinary/.25As的clean保留组织保持，额外raw槽两行均有。旧普通IQ增强和RDrop保留，B00不能称为完全无增强或纯CE。As直接作用raw，从low_suburban/mid_urban/low_urban采样，原传播—质量—补偿依赖保留，G=1的NoEQ residual不被偷换。session不由TX定义，合法L的全部sat CE保留。

## 种子及实现接点

native model seed同时控制初始化、dropout、旧普通增强与原sampler顺序；已固定物理角色不随其变化。private trainAs20261016421、validation20261016422、receiver20261016423保持；正式receivedIQ本来已经固定。六类seed与适用范围见 experiment.json，不把信道draw当作模型seed。

实际代码发现底层F工厂调用fullsource_concat20.design.base_model_config时固定使用SEED=2026092701；只改上层row/args不能改变F的CompositePool局部投影。新model.factory_seed只在该row构造/install期间临时绑定真实model seed，finally恢复旧工厂。该接点不增加参数或更改结构。首步真实F smoke比较：同seed两行所有初始state相同；不同seed的默认头和局部投影weight确实不同；参数900722/state206、原梯度、768NN/步和64/0As实测计数保持。真实训练args.seed与row/F工厂seed逐段核对。

runner先加载原已发布P43运行环境，然后以新包名复用原source/evaluate/qualification/channel/quality/protocol实现；其实际__file__仍指原P43文件，新增design/model/observer/dispatch/report指本次release。没有复制或热改原release。训练保留actual_modules.json，真实leo_practical继续使用原adv3b02_xuc/code路径，不凭根code默认值推断。

## 预登记判定与评价

每个seed的源V clean、逐RX clean与固定V1800/2draw的mid/lowurban比较B01−B00。As跨seed源资格要求各seed两draw×mid/low四格均至少+5个百分点，整体clean下降不超过.5个百分点、各源RX clean下降不超过1个百分点。全三seed如实报告，通过不是目标泛化证明；失败也不改配方或选seed。固定四行无论科学判据如何都按协议完成正式测试。

每row FINAL30固定后自动source只读诊断，再立即固定全部clean+六种practical预测；每视图168000个原样本，原正式IQ/channel/SNR/困难样本不变。七视图全完成后独立CPU exec的scorer才读truth。按row/seed报告accuracy、macroF1、最差RX、urban3mean/low/mid、clean与suburban绝对值/差距、固定各seed B00 clean正确集合的sat保持率，以及源状态/质量/间隔。三seed配对As效应全部保留并报mean/SD，不挑最好seed，不回流调参/早停/重排/选择性重跑。源/正式逐包门控轨迹若原实现未保存就如实列缺项，不另训诊断头。

新增四模型共120model-E、26640更新，预期L_NN20459520，每行总NN9149286（含源读出12600和正式1176002），加共享CP/合成smoke3074，新增总NN36600218。已有seed01两控制60model-E/13320更新/18298572NN单列复用，不冒充免费父训练或新训练。各行实际更新/teacher/额外前向/墙钟从完整结果核对；预算耗尽不等于收敛。不提前保证clean、60%或suburban≤2目标。

## 发布、停止和产物

唯一run 20261018-phase1-FcrossRX-As-replication-e30-s0203-m4-r02；owner codex/root/As-seed-replication49；CWD=/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_As_replication_E30_s0203_20261018_r02；输出=/home/szu2070436088/2510044040/CV-SincNet/runs/20261018-phase1-FcrossRX-As-replication-e30-s0203-m4-r02；各row独占stage20/source/prediction/scoring。命令 CVS-RFFI/bin/python -u experiments/cvs_As_replicate49/runner.py。总dispatcher CPU，worker按实时capacity选择GPU，所有owner总进程≤3/卡且free≥12000MB，不改其他健康任务。代码/config的实际Git commit写入release_commit.txt和现有登记。

本地8项边界/种子/预算测试及AST，唯一一次直接P0/P1审查；一次普通账户preflight、单archive一次SHA比较、remotecompile。launcher首步一真实合法sourceCP合成2包无query smoke（不作初始化）和4个真实F合成梯度/seed/预算检查3072NN，PASS立即开训练。没有额外审批、签名、成员哈希或数据重验。

仅所属source/CP/角色/config越界、非有限、输出碰撞、无法执行或无法合法predict/scorer时技术停止并保留；dispatcher遇技术失败停止未发row，已健康row继续。低准确率或科学FAIL不是停止/重启依据。启动后独立PID/CWD/argv/GPU/loggrowth核实，观测超时先核实原PID不重发。

预期全部原完整训练/源读出/prediction/scoring、queue/completion、逐row/seed与配对效应CSV、三seed聚合/曲线/分层图/独立复算报告。任何未完成row/seed都列缺项，不以不完整统计冒充三seed。训练权重、原IQ、逐包输出和原始日志不进Git。

本次只补齐As基线再现性，不能因此完成Ag独立价值、Ag/As四因子交互、lowurban>60%或suburban接近clean的完整联合目标。该目标保持ACTIVE。

## 独立分析准备

新增独立分析器 `analysis/cvs_As_seed_replication49_results_20261018.py`、源训练覆盖复算 `analysis/cvs_As_seed_replication49_coverage_20261018.py`，12项聚焦测试通过。当前仅LOCAL_VERIFIED，未对真实当前run执行，没有新指标或图。source模式必须全部六个源模型完成，只读训练完整日志/源V派生预测与状态，拒绝truth参数；final模式先核对全部六行的七视图冻结/评分完成及已保存的source-only决定，再独立连接truth。种子标签以已核实row替换旧reader历史展示常量；三seed配对mean/样本SD、每seed固定B00 clean正确集合、宏F1/最差RX/urban均值、clean/suburban差距及新4模型/复用2模型成本分开报告。覆盖复算只解码L，逐batch比对原状态汇总，不读取IQ、不解封U、不生成新信道波形。命令见脚本帮助，真实执行时填独占analysis输出；未执行不编造产物路径。原训练release/配置保持不变，不热补丁。
