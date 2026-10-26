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
