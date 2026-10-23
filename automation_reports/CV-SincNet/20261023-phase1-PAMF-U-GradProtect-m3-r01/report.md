# PAMF 有标签梯度保护：固定源域假设

状态：VERIFIED_RUNNING。80 项检查、一次独立 P0/P1 审查及真实 E30 source-only 保护梯度检查均 PASS，三个正式 seed 已按原固定配方续训。原目标仍 ACTIVE。

源依据：同原 RXPSM E30 的 24 个受控源域梯度 probe 中，整体 13 个、中层 t2 和默认分类头各 14 个出现 L/U 负内积；最大 U/L 范数比在 t2 为 1.171557、分类头为 1.022838。源类别覆盖、原型覆盖和教师 source-V 诊断没有要求更换原父来源。原始证据与局限见九模型 reference 的 source_gradient_diagnostic.md／e30_source_gradient_diagnostic.json。

保留原 RXPSM E30 学生、自身 EMA0.996、高斯0.999、置信低限0.7、目标锐化0.5、跨 RX 的 L-only 原型与现有权重、U 半普通噪声／半 Practical 输入，以及 T1 六次 L 前向、全部原监督损失和默认分类头联合训练。只更改同一步 L/U 梯度合并；不叠加 UCRIC、统计混合、PAIR、G 或新教师。

固定三个互斥参数组：名称含 `.t2.` 的参数；其余名称含 `.cls_head.` 的默认分类头；所有其余可训练参数。每组以原 T1 L 损失梯度为 gL，原已加权 U 损失梯度为 gU。若 ||gL||²<=1e-12，当前组 U 梯度置零；否则令 gU'=gU-min(<gL,gU>,0)/||gL||²*gL，再令 a=min(1,||gL||/||gU'||)（gU'=0 时 a=1），实际梯度 gL+a*gU'。L 保留，冲突投影仅作用 U，最后限制 U 范数不超过当前 L 范数；没有随机任务顺序或参数搜索。loss 标量仍报告原 L+U，参数梯度以显式替代方式进入原反向、clip 和 optimizer；无高阶梯度或新前向。epsilon 只定义近零 L 分支，不用来近似非零分母。

该规则仅约束当前欧氏梯度的内积和范数，不能保证经过原 Adam 动量／预条件后的真实 L 下降，更不能保证 clean 或目标收益。实现必须验证投影数学、正常/近零/缺失梯度、默认头梯度、native loss 接入和 EMA/曝光状态；真实 E30 no-query smoke 临时计算一次，丢弃该 probe，正式三个 seed 重新载原 E30。

每 seed 固定追加 20 轮／4440 步，累计 E50／11100、13L＋37LU；teacher/student U2097900、physical1048950，新增引导4440步。记录各组冲突／范数限制／实际保留 U 范数和梯度计算成本，不称同 FLOP。默认固定最终学生，source V 仅诊断，不选模或早停。

GPU5／6／7 对应 seed2026092701／02／03。等待同 seed UCRIC 的 raw＋DURA 全 14 项 GPU 预测结束，UMIX 已派发后占空出的第二名额；每卡最多两个本任务 worker，含外部最多三个计算进程，派发 free>=10000MiB，单 CPU scorer 等 UMIX 终态。不可覆盖 root `PROJECT/runs/20261023-phase1-PAMF-U-GradProtect-m3-r01`；唯一系统停止规则为同一技术失败两次停止新派发并保留健康任务，成功 smoke 立即续行。

每行冻结后自动完成 PAMF raw 和用户固定 DURA v1.0 同一 F 各 clean＋六 Practical，每景168000；全部14项决策固定后独立 truth-last。沿用同 readout 的 F20 监督／F21 普通 PL 三 seed 基线，逐 seed／景／RX／TX／day、SD、配对纠错损伤、曝光及增量／累计成本全部报告。原 raw clean 至少 +1pp、六景严格超过 +1.5pp 对两个基线同时成立才满足 goal，DURA 不能替代。

投影思想参考 [Gradient Surgery for Multi-Task Learning（NeurIPS 2020）](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)。本版为有标签优先的单侧投影加固定范数上限，是本项目待检验改动，不称原 PCGrad 完整复现。固定规则、父来源和矩阵在 UCRIC／UMIX 目标成绩读取前登记；不由其测试结果修改或选择性重跑，不声称新独立盲测。


80 项检查覆盖投影及范数上限的解析例子、近零 L、组间隔离、缺失／非有限梯度、native identity_loss＋correction 抵消与上游缩放、隐藏 U 真值拒绝、原 EMA／曝光状态往返、无新增网络参数、原始父配置及双 owner GPU 等待、单 CPU scorer、raw＋同一 DURA 实际方法名，以及此前 PSM／PAIR／VIEW／UCRIC／UMIX 回归。显式梯度桥接只改变合并梯度，标量仍为 L＋U。真实 E30 source-only smoke 将逐参数核对普通 backward 的 .grad 与预制保护梯度，并核对每组约束、原默认头／教师权限；probe 丢弃，正式三 seed 重新加载原 E30。数学性质不等于 Adam 或目标准确率保证。


## 已核实实际保护梯度训练

实际执行 commit `0aba8a89c8b74e3058a1bec727fef86805d75f6c`；一次发布包 SHA 本地／远端对照和远端编译 PASS。独立核实控制器、三 worker 的 PID／CWD／完整命令、实际 GPU 与 step 日志增长。真实 E30 probe 在 768 条 L／256 条 U 下，普通 backward 的每个参数 .grad 与预制保护梯度相符，三个参数组均实际发生负内积并投影到数值零附近，范数上限通过；默认头梯度有限、教师无梯度、EMA 仅一次。临时 probe 丢弃，正式三 seed 重新加载原 E30。已逐条核对完整刷盘的 step 前缀，U曝光、调用、各组冲突／限制累计数及每步约束符合预定，gzip尚未关闭，不宣称完整训练或目标收益。详见source_activation.json。

UCRIC 三 seed 全部预测结束后，UGP 接续 GPU5／6／7 每卡第二名额，与原 UMIX 各一个本任务 worker，含其他 owner 总计算进程最多三个；没有停止、热改或覆盖健康任务。冻结最终 E50 后自动执行 raw／固定DURA各七景，单 CPU评分器在UMIX终态之后接续，全部结果和负结果均保留。


## 三seed源训练完整，正式评测继续

VERIFIED_SOURCE_FINAL：三行各新增20轮/4440步，累计E50/11100；逐条读取全部13320条新增闭合step记录，源U曝光2097900、物理U曝光1048950、学生U视图2097900和自身EMA11100次均符合预定，父前6660步复用原完整核验记录。无U真值/目标训练或另训头。完整源证据及已发生父阶段/增量/累计时间见source_training_complete.json；程序经过时间来自共享GPU，不能等同独占GPU小时。

模型已按固定最终学生冻结，并已自动进入raw与固定DURA各七景预测；这是源训练完成，正式预测/独立评分尚未全部齐，不宣称整体实验完成或目标达成。部分目标成绩仍不读取，三seed全矩阵完成后统一分析。


## 完整目标测试结果

ANALYZED：三个 seed 的 raw 与固定 DURA 各七景、共 42 项读出／场景均完成独立 truth-last 评分，六个匹配监督／普通 PL 对照模型完整复用。逐景／逐 seed／SD／Macro-F1／RX／TX／day／配对修复损伤、实际曝光与成本见 target_results.md、target_summary.json、target_per_seed.csv、target_strata.csv、target_paired_raw.csv、restoration_effects.json、evaluation_completion.json。原 raw 双门槛尚未满足，研发目标仍 ACTIVE。

raw clean：84.295436508% ± 0.360883355%（三个模型 seed 样本 SD）。
DURA-LU clean：84.366468254% ± 0.411546919%（三个模型 seed 样本 SD）。
raw six_mean：71.818253968% ± 1.018445988%（三个模型 seed 样本 SD）。
DURA-LU six_mean：77.420006614% ± 0.260469899%（三个模型 seed 样本 SD）。

不依据本结果热改健康旧任务或替换父来源。
