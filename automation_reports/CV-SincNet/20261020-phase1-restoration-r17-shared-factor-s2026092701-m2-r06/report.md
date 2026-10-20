# 共享低秩先验的 R17 进一步精简 R06

状态：PLANNED。预先固定两个源权重代数构造：shared_half 完整恢复器 2,821,345 参数（比原 4,665,073 减少 39.5220%），shared_quarter 为 2,582,905（减少 44.6331%）。这只是结构成本，目标性能尚未测定。用户标准是同一目标域七场景各自准确率最多下降 1 个百分点，例如 80% 至 79%；168000 样本/场景允许最多少 1680 条正确，按整数判定，不能以均值或舍入掩盖单场景超限。low urban≥70% 及其他环境接近 clean 的长期目标另报，“接近”未指定数字容差。

每个原 R17 三步长卷积先在 CPU 取权重均值，再用平衡截断 SVD 分解成共享的空间卷积和 1×1 卷积。秩固定为 floor(min(Cout,Cin×kernel)×1/2 或 1/4)，最少为 1，仅在减少权重时分解。28 个长卷积/步的左右因子共享；各步 bias、GroupNorm、原 1×1 和其他状态分别保留。每步仍有独立状态，原 coarse/action/nativevariance、三步各两次 CG 和当前 IQ 输入不变。新 prior 参数 663798/425358，冻结上下文 2157547。只加载原 source-selected R17/cd977448/E20/4940，不加载 R01 至 R05 学生；构造依据为原源权重代数及此前源共享/低秩机制，不使用目标分数调整结构、损失、秩、超参、父权重、轮次或重跑。0 初始化 IQ/F/V/target/优化更新，CPU SVD 时间另记。

固定训练与 R04 一致：10×三步学生/原 R17 IQNMSE 均值 + 原 F50 默认头 T1 posterior KL + .5×单位 F160 特征距离平方，后两项系数 1。教师参照是同一 received IQ 经原 R17 恢复的输出，clean 仅 detached 能量分母 floor1e-4/日志诊断。R07 完整 63000（L6300/U56700）缓存 epochs1..20、15%clean/85%residual；batch256、model2701/data2026102001、AdamW2e-4/warm247/cos2e-5/WD1e-4/clip1/FP32/noTF32、E20/4940。每行 L126000/U1134000，R17/Fteacher/Fstudent 各1260000前向，Fstudent 路径仅输入反传；0新TX/V/target训练、F/context/teacher参数更新及头拟合。使用源监督 F 知识，不能称 F-free 无监督。prior 继承20+新20=40，coarse20/action20合计80继承串行pass，F50预训练50/11100与教师成本另列。

按用户“目标域性能为准”，两行及最终预算在新运行前固定。source V 七视图只诊断，不作性能资格门槛/早停/候选选择。每行 E20 冻结后自动 source 七视图预测、独立 CPU truth-last 评分、source 成本测量，再自动同168000×七目标场景预测及独立 truth-last 评分；不等待另一行完成，不读取目标结果改健康行。每行全部七视图 IQ/logit/feature 和 IDs 固定后才连接 truth/clean。报告逐场景 accuracy/MacroF1/TX/RX/day/confusion、raw repair/damage/net、相对原 R17 精确正确数和1pp验收，以及full256/prefix232/tail24波形、配对clean F特征差距；0pp诊断独立报告。已暴露基准，不称新盲测、全局最简或 Phase2/在轨验证。

40项合成与协议检查PASS，覆盖共享因子/独立状态/梯度/严格复载/alias冲突拒绝、继承与0TX/V/target训练、最终预算、固定矩阵、truth-last/CPU评分、自动逐行测试与失败隔离。唯一独立P0/P1发布审查完成后发布。launcher第一步实际原R17/F50/2Ls CPU no-query smoke，验证F输入梯度有限非零、prior梯度和teacher/context/F/head冻结、严格复载输出逐位一致；0TX/V/target/optimizer更新，PASS立即进入固定队列。实际smoke尚未运行，合成检查不证明目标性能。

N607普通账户，GPU4/5每卡一个own worker，source启动free≥18GB/allowner<2，共享capacity lock；先核实R05旧实际GPU/评分进程已退出，保留其他owner。CPU2线程/max1ownscorer/ownscore.lock。一个新不可覆盖root，一次预检、一次归档SHA比较/远端compile、PID/CWD/完整命令/GPU/日志增长核实。契约/nonfinite/OOM/其他技术错误停止受影响行并保留产物，健康行继续；两行系统性失败则run停止，无自动重试/回退，性能负结果不作重启理由。实际命令、环境/CWD、路径、GPU和产物见experiment.json。

成本测量口径：source V固定16条/batch1，先CPU加载并删除检查点payload（含optimizer）再放入GPU，报告模型完整参数、resident_before_forward、peakallocated及两者之差。仍受同卡其他owner及运行顺序影响，不宣称部署加速。旧R04/R05 peakallocated包含检查点payload，原R17含optimizer而精简final不含，旧数值保留但不能解释为纯推理显存收益。

发布前证据：唯一独立 P0/P1 审查 PASS，无确定 P0/P1；40 项合成/协议检查独立复跑 PASS，未访问真实数据/权重/远端。普通账户 N607 预检 VERIFIED，原数据与继承句柄一致，R05 实际旧 GPU 进程 0，GPU4/5 各 24124 MB 空闲，新输出/release 不存在。预检首次读取旧分析产物路径名称错误，修正为实际 paired_prediction_analysis 后只读预检通过，未发生远端写入或运行重启。实际 checkpoint smoke 将由 launcher 首步执行。
