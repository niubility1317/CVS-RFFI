# PAMF 有标签梯度保护：固定源域假设

状态：PLANNED。只固定源域依据、算法与三 seed 矩阵，尚未实现、审查、发布、smoke、训练或访问本版目标。原 UCRIC／UMIX 健康任务保持。

源依据：同原 RXPSM E30 的 24 个受控源域梯度 probe 中，整体 13 个、中层 t2 和默认分类头各 14 个出现 L/U 负内积；最大 U/L 范数比在 t2 为 1.171557、分类头为 1.022838。源类别覆盖、原型覆盖和教师 source-V 诊断没有要求更换原父来源。原始证据与局限见九模型 reference 的 source_gradient_diagnostic.md／e30_source_gradient_diagnostic.json。

保留原 RXPSM E30 学生、自身 EMA0.996、高斯0.999、置信低限0.7、目标锐化0.5、跨 RX 的 L-only 原型与现有权重、U 半普通噪声／半 Practical 输入，以及 T1 六次 L 前向、全部原监督损失和默认分类头联合训练。只更改同一步 L/U 梯度合并；不叠加 UCRIC、统计混合、PAIR、G 或新教师。

固定三个互斥参数组：名称含 `.t2.` 的参数；其余名称含 `.cls_head.` 的默认分类头；所有其余可训练参数。每组以原 T1 L 损失梯度为 gL，原已加权 U 损失梯度为 gU。若 ||gL||²<=1e-12，当前组 U 梯度置零；否则令 gU'=gU-min(<gL,gU>,0)/||gL||²*gL，再令 a=min(1,||gL||/||gU'||)（gU'=0 时 a=1），实际梯度 gL+a*gU'。L 保留，冲突投影仅作用 U，最后限制 U 范数不超过当前 L 范数；没有随机任务顺序或参数搜索。loss 标量仍报告原 L+U，参数梯度以显式替代方式进入原反向、clip 和 optimizer；无高阶梯度或新前向。epsilon 只定义近零 L 分支，不用来近似非零分母。

该规则仅约束当前欧氏梯度的内积和范数，不能保证经过原 Adam 动量／预条件后的真实 L 下降，更不能保证 clean 或目标收益。实现必须验证投影数学、正常/近零/缺失梯度、默认头梯度、native loss 接入和 EMA/曝光状态；真实 E30 no-query smoke 临时计算一次，丢弃该 probe，正式三个 seed 重新载原 E30。

每 seed 固定追加 20 轮／4440 步，累计 E50／11100、13L＋37LU；teacher/student U2097900、physical1048950，新增引导4440步。记录各组冲突／范数限制／实际保留 U 范数和梯度计算成本，不称同 FLOP。默认固定最终学生，source V 仅诊断，不选模或早停。

GPU5／6／7 对应 seed2026092701／02／03。等待同 seed UCRIC 的 raw＋DURA 全 14 项 GPU 预测结束，UMIX 已派发后占空出的第二名额；每卡最多两个本任务 worker，含外部最多三个计算进程，派发 free>=10000MiB，单 CPU scorer 等 UMIX 终态。不可覆盖 root `PROJECT/runs/20261023-phase1-PAMF-U-GradProtect-m3-r01`；唯一系统停止规则为同一技术失败两次停止新派发并保留健康任务，成功 smoke 立即续行。

每行冻结后自动完成 PAMF raw 和用户固定 DURA v1.0 同一 F 各 clean＋六 Practical，每景168000；全部14项决策固定后独立 truth-last。沿用同 readout 的 F20 监督／F21 普通 PL 三 seed 基线，逐 seed／景／RX／TX／day、SD、配对纠错损伤、曝光及增量／累计成本全部报告。原 raw clean 至少 +1pp、六景严格超过 +1.5pp 对两个基线同时成立才满足 goal，DURA 不能替代。

投影思想参考 [Gradient Surgery for Multi-Task Learning（NeurIPS 2020）](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)。本版为有标签优先的单侧投影加固定范数上限，是本项目待检验改动，不称原 PCGrad 完整复现。固定规则、父来源和矩阵在 UCRIC／UMIX 目标成绩读取前登记；不由其测试结果修改或选择性重跑，不声称新独立盲测。
