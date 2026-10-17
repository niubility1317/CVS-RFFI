# L 六步重排：完整元数据依据与训练组织原型

2026-10-17。状态：ANALYZED_METADATA_ONLY；合成接入检查 PASS。完整联合目标仍 ACTIVE。当前只研究训练组织，不称为 IQ 泛化增强 A_g 或星地增强 A_s，不把批次统计改善当作身份识别收益。现有 F＋跨 RX 同 TX 对比学习、SIG_TIME／joint160／默认 cosine 头、NoExplicitCFO、原 U 规则和 E20 不变。

## 已核实事实与未得到支持的解释

当前 Git 起点 10e6f062da5a4acb94bccc99c5df1a3cdcbe2bbb。上一固定 As 对 KL 移除已经完整 E20 失败并关闭，不重开损失系数、场景、RX 或种子救援。另一对话“开发 CVS Phase1 星地增强训练”已读取，当前在检验共享传播 session；此处不改变传播随机流或场景，不与其重复。

读取实际无 As 控制 B00 的 actual_native_args、完整 training_sequence 和全部 satmix_steps：use_tx_rx_balanced_sampler=false。L 采用 RandomSampler，DataLoader 未提供专有 generator；盲目提前取六批会改变原全局 RNG 的消耗时机，因此真实比较不能复用旧训练结果当作匹配控制。

来源：20261017-phase1-F-crossRX-sceneCycle-sourceRX1-e20-s2026092701-m3-r01／B00-RX1-s2026092701，实际训练 commit c9a74172e5f81bb4f04ce12cd4386ad4dd953ee9；固定 E20／4440 更新。只使用已保存的合法 fit L5040 标签、原 L 序列与统计，TX 标签来自既有可见 L 预测文件；根据实际 native 索引构造连接 base_index、RX/day/eq/sig 后与保存 L 标签核对，绝不按同 sig_i 猜 TX。U 记录在访问其 ids 之前跳过，0 U 标签解析／0 IQ／0 checkpoint／0 NN／0 更新／0 target。

完整复算 568320 次 L 观察、4440 批和3330个有效对比更新的13320个已有层／前向正对统计，全部与原运行记录一致。所有有效更新均128个valid anchor，故“正对缺失”解释被否定。正对数量和批内域比例仍波动；这支持一个可检验的训练组织问题，不能证明波动有害。

|批次统计|原序列|同物理观察的六步重排|
|---|---:|---:|
|每批最少跨 RX 正对数：全程最小／中位数|4／10|10／13|
|最少正对数＜8 的更新数|506|0|
|正对数批内方差：全程均值|13.288121|2.392999|
|每批 RX 数量极差：全程中位数|11|5|
|批内 TX/RX 互信息：全程均值 nats|0.060258|0.010236|

这些是原序列与固定组合学反事实的统计。批内互信息没有被加入损失，也不是部署泛化指标。原 SupCon 的浅／中层 mean/std 描述子会按批标准化；因此批次组成可能改变已有对比目标中的暂时域偏置。这是待检验假设，未证明其影响真实身份判决。

## 固定重排算子与匹配控制

固定6个更新／768个已读L包为一块，不选包、不重抽、不删尾、不改IQ。对每个可见 TX×RX cell，原块N个包按每批floor(N/6)与余数分配，6批间该cell数量极差至多1；余数分配保持每批128包。cell内包依次分配、行内用专有seed202610175101＋block索引打散。没有TX专属系数或权重。原块每个物理包的多重集及全程每包出现次数完全一致；原全程曝光范围104至120次也完全保留。

完整740块均通过物理多重集核对。该候选改变L包在相邻更新中的组织，不改变监督份额、视图数量、CE/KL/SupCon公式、网络、分类头、U伪标签规则或部署计算。所有旧ordinary开关保持共同配方，包括已披露的历史PA/CFO等，不宣称其已被验证为身份保持增强；没有新增硬件变换。

真实最小比较必须两条新scratch E20：O0和O1都使用相同六步预取、拼接/gather、私有L RandomSampler及DataLoader generator（202610175201/202610175202，后继固定seed按同一偏移规则）；两组都计算相同重排配额，O0按原顺序交付，O1交付重排顺序。该共同设置用于控制预取及RNG效应，不把旧B00当作对应控制。原U加载器不修改，原13L＋7U/oldgate/ownEMA/隐藏TX保持。

四对既有L IQ前向、原raw／ordinary／RDrop／浅中层跨RX同TX SupCon保持；不训练As，额外raw对权重.25仍保留，总对权重1.25。每行8L前向、4546560 L NN、完整4440更新；原U学生、EMA教师、V、七个源held视图另计，预计每行总5823318 NN，再另计一次真实CP smoke2。两行是40新model-epoch／8880更新，不冒称只有20总epoch；每模型预算仍E20。缓冲/gather增加的CPU时间与最大768包逻辑缓冲单独计，不称零计算成本。

固定源契约仍fit RX3/4/6/8、day1/2/3的L5040/U45360(hiddenTX)/V21600，RX1 L1260仅final20后读取七视图。不是完整L6300或L_train5580/L_dev720。新行从零初始化，没有父权重；必要真实checkpoint仅匹配契约no-query兼容smoke后丢弃。无正式target访问。

## 识别假设、最小实验与放弃条件

唯一假设：在上述共同缓冲控制下，仅均衡相邻六步的TX/RX批次组成，能改善F＋跨RX同TX的clean跨RX身份提取，并保护完整practical读出。不是IQ A_g；通过后也不能填入B10代替真正A_g证据。

拟固定同seed2026092701的O0/O1两匹配E20，均只读final20，源RX1七视图逐行自动预测并独立复算。O1相对O0：clean accuracy至少＋1pp、clean Macro-F1不下降，六个practical场景accuracy各最多下降0.5pp，seen V accuracy最多下降0.5pp；每行完整预算/defaulthead/hiddenU/no target且final base L objective≤.5。同时报告全部TX/day、修复/新错、绝对suburban及与自身clean差距、完整20轮曲线、真实NN/教师/更新与CPU成本。批次统计不作科学通过条件。

任一上述识别条件失败，关闭固定六步L重排，不改块长、L私有seed、分配方式、SupCon/CE权重、RX、模型seed或预算救跑。PASS只支持另行固定独立源seed确认，不能自动访问target或宣称新A_g/稳定As/四因子联合目标完成。

目前尚未实现完整source observer／dispatcher／七视图收尾的本候选入口，未登记或启动上述E20；不将此文档当作发布审批或新增门槛。下一实际工作是完成该匹配入口及现有最小流程内的必要检查，再执行授权内源实验。

## 已执行验证与限制

reorder元数据检查通过：精确物理多重集、每cell余数配额、固定私有seed、错误批长/标签/held RX拒绝、U ids毒化对象未访问、无正对反例不伪造valid anchor。完整历史复算结果和逐4440批CSV保存在local_artifacts/F_L_batch_order_20261017（不入Git）。

pairing原型已在实际普通N607 CPU2／CUDA隐藏环境通过合成检查：O0 IQ逐值相同；O1每个唯一合成sample_id对应的IQ/y/domain及所有额外metadata逐值相同；222步／37块，原函数退出后恢复，没有U batch，Torch/Python全局RNG不变，真实RandomSampler的私有L顺序匹配。0模型／CP／真实IQ／NN／优化更新。该检查只证明原型数据接线与接口，不是完整冻结F训练调用验证。

本地SSR Torch2.1与NumPy2桥接在RandomSampler.numpy转换处失败，前面的合成gather检查已经执行；记录保留，未修改环境或模型训练来绕过。相同代码在N607实际环境运行至全部断言PASS（Torch2.1.0+cu121、NumPy2.2.5）。没有把本地失败写成PASS，也没有安装包。

本轮另核查[WiSig原论文IV-B](https://arxiv.org/html/2112.15363)：均衡阶段25→20Msps后会重采样回25Msps再保留256点；因此不能用该处理说明擅改现有采样率或重开已关闭周期增强。此文献只支持数据处理限制，不为六步重排的识别收益背书。


后续接入：两条新匹配E20的完整source observer／runner／dispatcher／七视图收尾已实现并登记 20261017-phase1-F-crossRX-LbatchOrder-sourceRX1-e20-s2026092701-m2-r01；本地26反例、两新模型与原F206状态/初始logits逐值一致（6合成NN/0真实IQ/CP/更新/target）及唯一P0/P1审查PASS。当前尚未发布，完整训练调用和真实识别仍待实际运行，不把本依据分析当实验完成。


2026-10-17：20261017-phase1-F-crossRX-LbatchOrder-sourceRX1-e20-s2026092701-m2-r01 已完整 ANALYZED／VERIFIED，FAIL_FIXED_SIX_STEP_L_ORDER_CLOSED。两条新 scratch E20／各4440更新，17640源判决、每行740块物理多重集及全部实际交付输入、13320 SupCon mask term、20轮loss/U/教师预算独立复算。O0→O1 clean80.238095%→78.412698%（−1.825397pp），Macro-F1−2.012662pp，low urban27.936508%→27.380952%（−.555556pp）；mid suburban−.555556pp。原正对统计改善但身份收益失败，关闭固定6步重排，不改块/分配/私有Lseed/SupCon/CE/RX/modelseed/预算救跑。40新modelE／8880更新／11646638实际NN含dispatcher2；0target/新IQAg/As训练/父权重。原进程自然完成，4图PNG/PDF已查看，完整goal ACTIVE。报告 automation_reports/CV-SincNet/20261017-phase1-F-crossRX-LbatchOrder-sourceRX1-e20-s2026092701-m2-r01/report.md。
