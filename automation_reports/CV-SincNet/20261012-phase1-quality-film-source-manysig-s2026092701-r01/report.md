# 浅中层质量条件调制：源端30轮预登记

继续完成用户84%clean三seed目标，预算从头30轮。此前波形商空间、有限间隔、软度量与普通原型源确认均未过冻结稳定性条件；30轮最终matched对照已完整七视图score，CE30clean79.4562%、SELF_SSL30clean79.4310%。保留负结果，不按target调尺度、挑老师或重跑。新机制来自用户优化报告共同条件坐标假设以及源端跨RX不稳定证据；它不是针对具体目标RX/TX错误的分支。

## 核心结构与控制

原时频/PA/参考/身份内部门控与cos头完整保留。只在实际time_down（48channel×128time）和t2.pool（96×32）插入通道FiLM，所有TX共享同一调制公式，覆盖浅中层而非仅最终特征。scale=exp(.2tanh(g))处于0.8187至1.2214，shift=.1tanh(b)处于±0.1activation units；没有整方向投掉。generator最后层0初始化，原模型函数严格保持；记录实际层名、形状和梯度。

QUALITY_FILM使用同一包IQ的logRMS、FFTpowerflatness、abs lag16coherence三个代理，四RX fitL独立计算固定mean/populationstd、std地板1e-4、clip4。没有CFO数值，不将lag16称已验证重复前导，不将q当纯RX参数。q(X)没有增加观测信息，仅作为条件归纳偏置；不使用独立target支持、不声称完全RX解耦或全模型严格可逆。

FREE_FILM从对应实际层channelmean/std学习三维context，再用同一有界FiLM；CE30没有maps或quality前向。参数分别177025/182913/182049（free与quality差864，约0.47%），不把普通新增自适应容量误归为quality机制，也不声称完全等容量。三个控制使用相同modelseed、fitL遍历、30epoch、optimizer和原数据角色，无U/teacher/EMA/星地或其他辅助loss。

## 真正整源RX留出

单seed2701，五源RX各三控制共15模型，每个从零30×222=6660。四RX L5040只训练和quality统计，heldRX完全不见；heldRX原V5400仅最后30轮冻结后预测score，不拆新V角色。全部15TX类对、day、混淆和worstRX保留。scale/shift和criterion预先固定，不据target反馈调节。

QUALITY晋级需源mean相对CE至少+1个百分点，相对FREE至少+0.5个百分点；各比较至少4/5RX非负；worstRX相对各控制均不下降超过1个百分点。全过才进入原物理契约三seed全源30和自动clean+六practical独立truthlast；未过保留，不放宽门槛。当前source-only不读取target。

## 执行与保护

矩阵、commit、精确命令、CWD/环境、路径、GPU、owner和产物在experiment.json。单新不可覆盖run、一次P0/P1、一次真实checkpoint无querysmoke、archive一次比对与compile后继续，reuseVALIDATED_ONCE数据，不做额外seal/hash层。每GPU最多3个总CUDA进程。只系统技术/权限/输出等确定故障停止所属run，不因低性能停止，不影响其他健康任务。原权重只作smoke，实际新身份和FiLM全部随机初始化。84%目标未完成。

本次恢复核实远端尚无此run/release，无重复启动。本地CUDA检查PASS：原模型零初始化等价、浅中层梯度、实际层形状、有界scale/shift及四项协议负测。一次独立P0/P1审查结论见review.md。实际执行commit由发布归档内release_commit.txt记录，启动后独立读回。
