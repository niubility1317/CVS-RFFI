# 加权普通硬伪标签：只加入监督端标签平滑

状态：VERIFIED_QUEUED。166 项相关检查通过，一次独立 P0/P1 审查 PASS、无 P0/P1。真实原 F21 E30 source-only smoke 待首卡位；没有新训练或目标成绩读取。

用户要求基于高纯度普通 PL 提高权重并引入监督机制。本版以 HPL2 为匹配项，只在 U 硬伪标签交叉熵加入与监督端相同的标签平滑 0.01；保留阈值 0.95、权重 2、当前学生 eval 硬标签、完整 U 批分母、raw IQ 和一次 U 学生前向。原 L 六前向、As、RDrop、跨 RX 对比与联合默认头保持。没有新增 U 视图、KL、对比、EMA、原型、G 或另训头。

## 源域依据及局限

完整源 V 各 27000 条的原 F21 E30 伪标签接受精度为 99.8868%/99.8366%/99.9027%，覆盖 94.9037%/95.1815%/95.1296%；U 真值保持隐藏，不称为 U 实测正确率。源 V 仅作无更新诊断与门控冻结。

HPL2 三模型完整新增日志各 4440 条，权重为 2 的 U 损失占每步总损失比例的均值为 2.6703%/2.7309%/2.6931%。直接检查冻结原 F21 E30 的 24 个 L/U 源批次（每 seed8），U/L 梯度范数比中位数为 0.298518，因此低损失比例不证明 U 训练信号微弱。24 批有 13 批方向相反；同一批学生输出上，仅把 U CE 标签平滑从 0 改为 0.01，冲突降至 8 批，19 批余弦提高。这个探索性源诊断在新 HPL/UVB 目标分数读取前完成，是选择单因素消融的依据，不能视为目标性能或 Adam 更新保证。

诊断全部使用合法 L/U，U TX 为 -1，真实原 F21 E20/E30 来源与默认头核实；每次恢复原参数，无 optimizer/EMA/原型或头拟合，临时训练前向 buffer 丢弃，不修改健康进程。L 输入为受控 raw，内部原六分支保持，不冒称原生普通增强训练轨迹重放。完整记录见 source_hard_PL_gradients.json、source_paired_smoothing_gradients.json 与 source_loss_utilization_initial.json；损失文件的 HPL2 是完整三模型，HPLT1U 仅当时种子2702的136条前缀，未混成完整结果。

## 固定训练与测试

HPL_LS2 三 seed2026092701/02/03，均从原 F21 同 seed E30 独立追加20轮/4440步至E50/11100、13L＋37LU。HPL2 是比较项，不是权重父模型；不使用新 HPL/UVB 测试结果选模或调参。每模型累计 U2097900、学生 U2097900，新增20轮 U1134000、4440调用，U physical及raw-As均0，unique U56700；默认头联合训练，固定最终学生，成本单列。

GPU5/6/7 一 seed 一卡，新行等待同 seed HPL_NO_UAS 全部14项 GPU 预测完成、HPLT1U 已派发，再接替释放卡位；每卡最多2条本任务、含其他 owner 总数最多3条、空闲显存≥10000MiB。smoke/正式派发把前序 HPLT1U 已启动未注册 CUDA 的 PID 纳入注册/active PID 并集。CPU scorer 等待整个 noUAs 终态后串行。不停止、重启、热改健康任务。

命令/CWD/环境/路径见 experiment.json；N607 普通账户、固定 CVS-RFFI 环境、FP32/TF32off，不可覆盖 PROJECT/releases/20261024-phase1-PAMF-HardPL-LS-m3-r01 及 PROJECT/runs/20261024-phase1-PAMF-HardPL-LS-m3-r01。一次代码 Git commit、一次 archive 本地远端 SHA 比较、一次远端 compile。首卡真实原 F21 source-only smoke 检查 smoothing0.01、门控0.95/权重2、L768/U256、一次U学生前向、Uphysical0、原LPractical64、有限默认头梯度；PASS 即丢弃 probe、正式三 seed 重载原父权重继续。两个相同技术失败停止新派发并保留健康进程/产物；无性能停止、重排或选择性重跑。

每 row 源域固定最终 checkpoint 后自动 raw PAMF 与同一冻结 DURA v1.0 各 clean＋六 practical，每景168000条，共42个正式cell。全部预测固定后独立 truth-last；完整矩阵后报告逐景/seed/TX/RX/day、修复损伤、F20/F21 配对差和训练成本，两个完整矩阵均评分后再列 HPL2 单因素差。DURA 单 G seed 仅冻结参考，不用于 F 训练；G＋F20 不称全流程仅 L。

主目标仍为 raw clean 同时超过 F20/F21 至少1个百分点，六景均值同时严格超过1.5个百分点；DURA 不能替代。目标 ACTIVE，尚无达标方法；历史测试暴露如实披露，不称全新独立盲测。预期产物为 source/state/smoke、E50默认头checkpoint、全部step记录、hard_pl_state、源V诊断、source_execution_check、双七景预测/独立scores/完整汇总。排队、本地通过、源梯度改善均不视为训练成功或目标达成。


## 发布后实际排队证据

执行 commit `b981bad253b144791579421ad18e3f4cfa480408`；一次 archive 本地／远端 SHA 对照及远端 compile PASS。独立读回实际 CPU 控制器 PID／CWD／完整命令、存活和同版本 state，三 row 均 QUEUED、GPU 子进程 0。等待同 seed noUAs 完成全部 GPU 预测后执行真实原 F21 source-only smoke，PASS 立即正式重载对应父权重继续；前序 HPLT1U 启动中 PID 计入容量。现有六条健康训练及其他 owner 保留。源机制和矩阵在本轮 UVB 目标成绩读取前已冻结；排队不等于训练成功或目标提升。
