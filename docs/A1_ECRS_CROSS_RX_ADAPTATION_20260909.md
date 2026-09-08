# A1结合ECRS跨接收机判别机制

用户于本轮V2发布时补充：加入“制定ADV3B02-ECRS落地计划”中的相关有效创新机制。已读取该任务及`C:/Users/lh594/Downloads/ECRS_V2_optimization_spec.md`。本文件在改代码前冻结适配范围。已有V2运行release`89758ce1`不修改。

## 选择依据与限制

采用规范第5节的同TX跨RX判别约束，以及第8节的已有前向复用、二次规模配对计算。已有ECRS响应支路在融合关闭时不影响raw决策；其独立性能收益尚无证明。因此本轮把相同判别数学直接用于A1实际预测的`z_id`，明确是新适配，不宣称完整ECRS物理响应/融合移植，也不把历史target指标用于选择系数或挑选候选。

原A1的TX监督和DAOT clean/LEO EMA一致性保留，不叠加重复的无标签目标。仅L标签进入新约束，U真值永不进入；没有新数据角色、采样器、参数化模块、checkpoint或额外主干前向。

## 数学与匹配矩阵

对anchor i，正例为同TX不同RX；负例为不同TX且与anchor同RX、day、view。余弦距离，margin=0.2；遍历全部合法正负组合，先每anchor平均，再按有效anchor平均。权重0.05来自原提案，不根据target调整。采用排序负距离和前缀和，保持全部triplet hinge均值，存储O(B²)，不构造B³张量。

|行|设置|用途|
|---|---|---|
|X0_A1_RUNTIME|A1-V2的F1；新loss关闭|匹配从零控制|
|X1_CROSS_RX_CLEAN|X0+clean身份约束，权重0.05|验证直接跨RX身份结构|
|X2_CROSS_RX_VIEWS|X1+复用实际执行的LEO身份表示|只改变约束视图范围|

以上为首组三个基础对照。X2把clean和实际LEO表示放在一个集合，view用于匹配负例；不额外加倍系数。跳过增强时只用clean；不把clean副本算作LEO。首三行共同保留A1-V2的EMA缓存修复与执行优化，关闭EMA启动平均和coverage新加权，以隔离ECRS适配。

用户随后明确改为每卡两个训练进程，并要求结合其他机制。因此本轮扩展为以下8行，每卡新增一行，保留现有任务。无空位时等待，最大并行占用为每卡两个训练；不再等待前序R3评分完成才使用第二名额。

|行|GPU|相对X2的预登记变化|解释|
|---|---:|---|---|
|X0_A1_RUNTIME|0|关闭跨RX|共同控制|
|X1_CROSS_RX_CLEAN|1|仅clean参与|跨RX基本效应|
|X2_CROSS_RX_VIEWS|2|无|clean/实际LEO身份判别|
|X3_WARM_EMA|3|EMA启动期平均|减少从零教师的初期滞后|
|X4_COVERAGE_KL|4|归一化后共识覆盖率加权|低覆盖蒸馏贡献不会被EMA归一化抵消|
|X5_BALANCED_L|5|4TX×4RX/day单元×8包=128，使用既有平衡采样器|增加合法跨RX正例及同RX/day异TX负例；U采样不变|
|X6_BOUNDED_DOMAIN|6|GRL-CE改为有界域混淆|身份目标为KL(p(domain|z)||Uniform)，限制于[0,log D]，domain判别器梯度分离|
|X7_COMBINED|7|合并X3/X4/X5/X6|预登记组合检验，不能把组合差分归因于单一机制|

所有行seed392005、随机初始化、E200、L128/U256、source数据角色和final-only一致。X5/X7会改变L批次组织和样本重复轨迹，须报告覆盖率和配对数量，不称与随机采样轨迹等价。4个RX/day单元至少来自两个RX（每RX只有3个source day）；4个TX从全部6个TX中轮换选择。先验证完整格点，缺格点不能默默空转。

有界域混淆复用[既有设计](plans/2026-08-24-fasttrust-qb3-bounded-confusion-design.md)和已实现入口：L为`cur_w[adv]×(0.5×discriminator+0.5×confusion)`；U遵循既有RC4默认回退，分别为`0.16×discriminator+0.16×confusion`，不声称L/U统一权重。新方法只增加小型域头计算，不增加身份主干前向。有限目标并不保证准确率或域不变性，效果待匹配训练。

增加这些行基于源域训练数学、元数据与可运行实现，不根据已见target分数重排机制。所有改动在新结果出现前冻结；不存在“已证实最优组合”的声明。

## 追溯

|ID|来源|要求|目标|状态|验证|
|---|---|---|---|---|---|
|EX1|规范5、6|匹配RX/day/view负例、跨RX同TX正例、anchor均值|a1_ecrs_cross_rx.py|verified|字面triplet loss与梯度CPU/CUDA对照通过|
|EX2|规范2.2、9|新loss必须到达实际身份表征|train_ssdg.py|verified|真实模型8种梯度组合及实际X7入口4次成功更新|
|EX3|规范8|复用已有forward，O(B²)配对，FP32稳定计算|helper/训练入口|verified|helper不接收模型、不新建forward；CPU/CUDA/AMP验证；整体耗时待正式训练|
|EX4|规范2.1、5|空合法集合、U隐藏标签、跳过LEO语义|helper/入口|verified|无效标签、缺metadata、空集断图和副本排除通过|
|EX5|用户明确要求|无历史checkpoint、单因素，资源上限由EX10覆盖|runner/报告|verified|八行真实parser与随机模型/heads/RNG一致|
|EX6|规范9|记录configured/executed/count/gradient/成功更新|训练日志|implemented|源码接线和实际入口非零梯度通过，正式日志在run报告追加|
|EX7|规范3、4、7|物理估计/复杂锚点/固定融合|本轮不实现|deferred|增加求解与推理成本，互补性未证明；不在加速主线中整树合并|
|EX8|用户追加/规范6|固定batch128的TX/RX-day结构化采样|既有sampler/新矩阵|verified|真实subset取值，合成完整90格点、每批128合法anchor；运行guard阻止缺格点空转|
|EX9|用户追加/QB3设计|有界域混淆与梯度隔离|既有bounded_domain_confusion/入口|verified|7项边界/梯度测试、L/U实际权重核对、X7主入口通过|
|EX10|用户追加|每卡两个训练、8行预登记组合|runner/报告|implemented|容量检测及detach参数/防重派测试通过，remote读回待run报告追加|

本轮只检验新适配是否有效，不能预先称为已有效创新。最终按同row clean、三LEO、弱RX、耗时、显存和跳步报告；单seed不晋级默认。最终prediction固定后独立评分，不反馈调参或重跑。
