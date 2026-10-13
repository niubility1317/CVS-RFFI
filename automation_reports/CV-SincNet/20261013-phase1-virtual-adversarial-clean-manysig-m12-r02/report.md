# 虚拟对抗一致性：预登记

主目标为固定2026092701/02/03三seed的VAT_FEATURE clean均值达到84%。四臂CE50/RANDOM_INPUT/VAT_INPUT/VAT_FEATURE共12模型全部从零50epoch/11100updates，普通cos160头，固定末轮。每row自动冻结、168000clean预测、独立truth-last评分；无源V选模、候选晋级门槛或目标驱动重排/重跑。与另一活动对话的低通/多相采样结构实验分开。

方法依据[Miyato等VAT原论文](https://arxiv.org/abs/1704.03976)：不需要标签的局部分布平滑。这里只用合法U56700的IQ和非TX源metadata；其TX真值不进入训练接口，L6300仍计算纯CE；V27000仅epoch10/30/50健康。划分、RX1/3/4/6/8、day1/2/3、6TX、split392005及既有VALIDATED_ONCE物理ID保持。

前5轮不迭代/前向U，第6轮开始每步128U，5轮ramp到0.25 consistency KL。每步先当前student eval无梯度q，再随机单位方向上相对tensorL2的xi0.001 probe，计算KL对probe梯度，detach方向后以epsilon0.02扰动；一轮power iteration。RANDOM_INPUT计算相同probe梯度后丢弃，使用初始随机方向，同半径和计算量。VAT_INPUT作用于IQ；VAT_FEATURE交替time_down48/t2.pool96，一次只加一层扰动，U弱前向获取该层范数。所有U前向均eval无dropout但最终一致性前向保留身份参数梯度；L正常训练dropout与采样不受影响。无EMA、教师copy、隐藏标签门控、卫星增强、新参数或推理辅助网络。

网络前向fullFP32，q/logsoftmax/KL用float64处理小probe的数值消减。逐包L2半径固定，不按batch聚合；零搜索梯度沿确定初始随机方向回退并记录，不把科学退化当技术停止。扰动界限不是身份不变性保证，也不声称精确复现原论文图像设置。

本地CUDA负测、输入/特征双层梯度、detach搜索、模式和CPU/CUDA RNG恢复、精确半径、逐包推理和eval无hook检查通过。检查中VAT重复KL绝对差2.14e-8来自GPU反向数值差异；已在正式fullFP32策略下以rtol1e-5/atol1e-8验证，不要求位级一致。没有模型状态或随机数泄漏。唯一独立P0/P1审查及真实checkpoint无querysmoke按最小流程执行。

全部12模型合计133200optimizerupdates，L17049600包；9regularized模型合计U11508480包，每模型9990弱/搜索/一致性前向；CE不打开U。每GPU总实验进程最多3，计入其他owner；共享容量锁，>=12000MiB空闲。CWD独占N607release root，环境现有CVS-RFFI Python，code/config由一个实际Gitcommit标识。精确路径/命令/六类seed/矩阵见experiment.json。

预期source checkpoint/freeze/config/完整step与epoch日志、prediction/complete、scores accuracy/F1/混淆和RX/day/TX、analysis、finalreport及completion。首个run自身系统故障停止新调度，健康row完成，保留partial；不性能早停、不盲重试、不热改/停止其他任务。固定prediction全部完成后独立scorer才连接truth；不声明新盲测、Phase2注册或真实卫星部署。

用户在发布前授权由Agent自主选择epoch，但每模型最多50。此轮固定选择50，cosine日程分母49，末轮为epoch50，累计11100updates；数据/机制/seed/半径与源/目标权限不变。已完成旧run保留原预算。

独立P0/P1机制与训练/预测/评分衔接审查通过，没有缺失发布依赖。用户随后授权预算最多50，发布前仅更新预算并完成11100/9990计数、50上限与cosine日程验算；不重复完整审查。

修复 r02：替代未产生预测的 r01 技术失败。统一 expected_counts 和 final_artifacts，producer→source_proof→own_model 回归通过并拒绝旧30轮字段；VAT机制、数据、seed、半径和50轮预算保持。12模型全部从零，旧产物保留且不加载，失败成本另列。该问题只进行一次定点P0/P1复审。

原问题一次定点P0/P1复审PASS：预算闭合与产物处理无残留P0/P1。旧30字段负测和完整producer→proof→checkpoint恢复回归通过。

发布 VERIFIED：执行 commit 79f51363a9027b39d1f107c749025aa4f5d5f966；单一归档传输比对、远端编译、普通账户预检及 real-checkpoint no-query smoke 均通过，dispatcher PID 1503330，独占 r02 release/run 根。训练完全从零；不继承 r01 权重。首批 PID/CWD/argv/GPU 及 epoch/log 增长已核对。

用户 2026-10-13 进一步明确：84% 目标通过提升特征提取网络追求，分类头形式保持固定。当前 VAT 保留 ordinary cosine 6×160 头，输入/浅中层特征一致性更新骨干；后续结构探索必须继续与其他对话路线区分。

完整收尾 VERIFIED / ANALYZED：12×50 轮、133200 更新、2016000 clean 预测、216 同 row 指标。全日志、预测文件、checkpoint 和混淆指标核验通过。主方案三种子 79.0274%，84% 未达到；所有负结果及失败尝试成本见 final_analysis.md。

用户基线更正（2026-10-13）：正式对比名称为“旧门控＋cos”。本轮原机器 row/arm 标识 CE50 保留，其显示名称更正为“旧门控＋cos（CE 训练，50 轮匹配对照）”。三个真实保存的 checkpoint 已严格重载，使用 2 条零 IQ 无查询前向；FreqBandGate1d 与 EnvelopeGate1d 均实际调用一次，cosine 权重 6×160、scale 30，身份优化器范围 177025 参数。指定历史 E20 的本地构造具有相同门控、头类型和参数形状。CE 指损失目标，不表示删去旧门控。

本轮仍是 NoExplicitCFO 的从零 50 轮对照；历史 E20 保留原 20 轮及其原输入策略，不将本轮冒充完全同配方的 E20，也不追溯改动历史权重、预测或评分。后续特征提取候选以旧门控＋cos 为明确网络基线，保持分类头形式；训练预算、输入策略、采样和执行路径均显式匹配或披露。无查询核验证据在 local_artifacts/vat_delivery_r02/old_gate_baseline_verified.json。

基线语义更正：用户指历史伪标签门控＋EMA＋cos；本 run 的 CE50 只保留内部门控，U/EMA 实际关闭，属于纯 CE 历史控制，不冒充指定主基线。后续新矩阵另行登记。
