# 固定三成员概率集成：三种子30轮最终确认

目标为clean三primaryseed均值84%，尚未达成。合法源端30轮先导完整核实：个体均值60.6235%，ensemble61.7778%，差1.1543个百分点；4/5RX非负，最差RX下降0.1852个百分点，全部满足预登记。相对原CE30单成员源均值61.5222%仅高0.2556个百分点，不夸大它。目标数据未用于此次方差先导，资格来源见同run source_decision/report。

primary2026092701/02/03各自形成3成员ensemble；内部初始化和dropout种子固定primary+10000m（m=0,1,2），L采样primary+101。成员0复用对应已核实合法数据契约、真实物理角色、无显式CFO、pureCE、fixedfinal30的原CE30权重，只作自身推理成员，不给其他成员初始化、蒸馏或teacher。新成员1/2各自完全随机初始化、只用原L_s clean6300训练30x222=6660，label_smoothing0。V27000只epoch10/20/30健康诊断，不选epoch。无U/EMA/教师、半监督、星地或其他训练增强/辅助loss。

既定方法为三softmax(logits)的固定等权平均，温度1，类别共享公式；不拟合权重、校准、挑选成员、不读query真值/真实类别构成/配额，不进行跨query重排或更新。所有成员fixedfinal30后写零拟合ensemble引用bundle；加载前逐成员核对实际数据角色、config、ancestry和冻结来源。原源码/model/config通过一次source候选P0/P1及一次原初始化问题定点复核；此阶段沿用固定方法和原seven-view独立truth-last闭环，不增加runner、报告或重复完整审查。

每row训练两个新成员后立即自动预测clean及六practical，各168000；全七视图ID/数组/类型/范围/来源完整后独立新进程才能打开truth。既有VALIDATED_ONCE接收视图复用，不重新生成、按成绩选视图或重复验证。三row都完成后汇总3.528百万ensemble预测、378指标记录，三seed/RX/day/TX/场景及混淆保留，与对应已完成CE30只读配对比较。主判定固定clean三primaryseed均值>=84%，不依据逐row分数改变其他训练或选择性重跑。

成本明确：每成员177025可训练参数，每ensemble531075（3倍）；6个新成员39960更新/180modelepochs，3复用成员历史19980更新/90modelepochs，总59940更新/270modelepochs，相比原3单成员为3倍训练和推理。不是相同计算量的机制优越性、物理不变/纯RX解耦。真实总参数、时间、显存和日志按产物报告；源pilot成本单独保留，不与正式训练成本混写。

单新不可覆盖run/release/output，唯一owner，每GPU最多3个总实验进程含其他owner；不热改/停止/重启健康任务。一次实际Git代码/config提交、聚焦权限负测及合成seven-view truth闭合、一次N607preflight、一次归档SHA本地/远端比对、一次compile、真实匹配checkpoint无querysmoke PASS立即接续、启动PID/CWD/cmdline/GPU/log增长独立核对。只按已登记的系统技术/数据权限/输出碰撞故障停止所属run，不按低准确率终止。精确矩阵、环境/CWD/命令/路径/stoprule在experiment.json；实际执行commit在release_commit.txt和启动后登记。

现有目标基准已被多轮观察，本次是开发基准的冻结确认，不是新的独立盲测；地面代理及practical模拟不能称真实卫星部署、Phase2适应/注册或Phase3协同。84%待完整预测和独立评分证明。
