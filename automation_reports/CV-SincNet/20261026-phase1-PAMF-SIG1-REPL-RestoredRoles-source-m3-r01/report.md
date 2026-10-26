# 单视图 REPL：L/U 恢复视图三角色消融

用户要求同时提高原始clean泛化和同DURA性能，保留同TX跨RX关系，并比较L/U是否采用相同机制。本run只训练三个新候选，复用已完成的NoStyleNoRXE及旧PL-X3/REPL-SIG1 source结果；不训练或重复预测对照。

| 新候选 | L的新G视图 | U的新G视图 | 原L/U训练 |
| --- | --- | --- | --- |
| REPL_G_U | 无 | 有 | 全部保留 |
| REPL_G_L | 有，全部真标签监督 | 无 | 全部保留 |
| REPL_G_LU | 有，全部真标签监督 | 有 | 全部保留 |

名称L/U只描述恢复视图放在哪个分支，三行都保持半监督L/U。完整PAMF单SIG_TIME18976参数、原6×160默认头联合训练；L768/AsKL0/ordinary和rawRDrop/LS.01/浅中层crossRX，U6/weight6/raw95/guard1/joint.05保持，风格与整特征外推关闭。原分类与KL权重和归一化保持，不把新G视图分类loss再叠加一份。

仅替换原两次mixed前向中ordinary半批：L为每次64条rawL经H_a/G和H_b/G，原As64保持；U为每次半U批的rawU经H_a/G和H_b/G，原raw半批保持。原ordinary首对的浅/中/joint跨RX对齐仍执行，raw对保留。L真标签从不受伪标签门控；U分类仍使用原始IQ当前学生.95伪标签，不按G置信度挑恢复视图。这里暂不新增G专属跨RXloss或低置信U损失。

每对a/b采用不同场景和独立随机实现：三scene为lowSuburban/midUrban/lowUrban，a=(globalstep−1)%3，b=globalstep%3。namespace包含role、累计epoch和a/b及物理ID，跨epoch更新；原RX/day硬件保持。固定G/DURA原F0-LU，noGrad，InferenceEngine reuse/graph/nativeFalse、FP32/TF32off，不优化G。额外高斯噪声另作后续消融，首轮不叠加。

原F0 VALIDATED_ONCE L6300/U56700/V27000不变，U TX始终隐藏。每行只有模型seed2026092701，复用自己已完成m201单视图ordinaryPL E20/4440，从相同父独立+10+20至E50/11100，累计13L37LU。各新增6660更新，三行90模型轮/19980更新；不是三seed。每行L/U骨干前向数不增加，新增G处理分别1701000/852480/2553480包（U/L/LU，warm每段2包另计），G和原E20成本/耗时分开报告。

每行final后自动sourceV27000×8：raw/Graw/三H/三G，原默认头、F/G冻结，逐TX/RX/day保存。旧三个source参照直接复用完整结果，无新前向。三行均完成后按逐指标max参照固定判断：rawCorrect≥max−27，H3correct≥max−81，G3correct≥max+243，对应−.1/−.1/+.3pp。通过者G、raw、H正确数依次最高，完全平分选G成本低的L/U/LU；无通过则不扩种子或target，不按新结果降低门槛。X3参照的结构差异单列，source已见RX不冒充目标泛化。目标仍是同一候选三seed原168000/seed raw/同G clean+六practical实际性能及独立truth-last，严格新单视图基线和原历史目标比较不变。

唯一owner codex-pamf-sig1-restored-roles-source-20261026。Git执行commit写runtime，N607既有Phase1 Python/Torch2.1、FP32/TF32off/CPU2，无安装；不可覆盖run/release与每行命令见experiment.json。GPU0..7派发前计入全部owner的CUDA+pending<2且free≥10000MiB，不改其他健康进程。首个输入/来源/角色/shape/finite/冻结G/smoke/预算/输出技术错误停止新派发，健康行完成，保留产物，不自动重试；科学低分保留。

真实E20 L128/U256联合backward smoke作为第一步：检验L768/AsKL0/U1536、实际L替换128或0/U替换256或0、a/b恢复IQ确实不同、原physical半未变、U梯度到单SIG/head/body非零、F梯度有限、G所有参数/buffers未变；0optimizer。PASS即继续。一份archive比较、remotecompile及实际PID/CWD/cmd/GPU/log增长核实。

当前LOCAL_VERIFIED：10项聚焦检查和15协议子检查、compile及唯一P0P1审查PASS；真实checkpoint smoke为发布第一步。该实现的参数在读取新匹配基线的目标分数前固定。


## 实际启动VERIFIED

执行commit `ac131ec8cdbd378377a126a62f0e096f2c92bd00`。一次archive传输比较VERIFIED、remotecompile PASS，三个真实E20联合smoke全PASS：U256/六视图1536、L768/AsKL0，角色实际替换L128或0/U256或0；独立a/b恢复IQ相对差L约.211973/U约.193988，实际U梯度到singleSIG/head/body有限非零，G参数与buffers不变，0optimizer。原physical半保持、无新增F骨干前向。

controller及三个worker的PID/CWD/完整命令/物理GPU与跨读step日志增长已独立核实。读取时正在E21起的新30轮，最新全部可读2652条更新逐条验证L768/U6、原crossRX/AsKL0/六权重/style0、角色G包数/调用数、非有限跳步0和U曝光。仅为可读前缀，不是完整训练或三seed结果。六基线正式结果已另行核验，配方未按其target分数改动。[启动与可读记录证据](launch_evidence.json)。


## 当前部分源结果（全矩阵尚未结束）

U-only和L-only训练/源V27000×8已完成、原进程退出；LU保持原配方继续。下面为两个单seed源结果；相对三份预登记参照各列最大值，不能作为目标集或三seed成绩。逐行source完整预测/冻结标志已读回，全部家族结束后才运行一次完整独立数组/checkpoint/逐步核算，不提前选中。

| 新学生 | raw clean | 未恢复H三景 | 同G三景 | raw差值pp | H差值pp | G差值pp | 源门槛 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| U-only | 97.788889% | 74.022222% | 91.013580% | −0.633333 | −1.149383 | +0.340741 | 未通过 |
| L-only | 98.018519% | 73.396296% | 91.206173% | −0.403704 | −1.775309 | +0.533333 | 未通过 |

它们支持“G训练压力和原始IQ保持存在取舍”，不能据此降低门槛。原始L-only/U-only配方不热改、不重训；新GH两行是另行登记的同预算源消融。LU未结束时不把本run标ANALYZED，也不启动本家族三seed或target。见[部分源证据](partial_source_evidence.json)。
