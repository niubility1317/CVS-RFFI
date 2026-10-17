# 完整源普通星地拼接：E20 固定参照

状态：已实现并登记，尚未启动训练或产生新预测。训练不会留出一个源RX。一次限定P0/P1审查PASS（一个发布路径P1已修复并仅做该问题的定向复核PASS）；本地语法与33个配置／协议反例PASS，真实F合成检查安排在发布后、任何真实训练之前执行。

用户于2026-10-17指出近期探索偏离星地识别、部分新方法不如普通拼接，并要求不再减少源训练域。当前唯一问题是：固定F＋跨RX同TX和完整源覆盖下，普通源／practical配对监督的真实clean与星地收益如何。先建立该参照，后续新机制必须直接比较，不把源统计风险下降算作性能。

底座：三路SIG_TIME、完整统一Sinc／CENTER_25K+CENTER_UNITS、真实joint160、默认cosine30头联合训练、NoExplicitCFO及必要内部逐包补偿、原ordinary、RDrop1／LS.01／浅中层CROSS_RX（time_down/t2.pool、T=.2、最大.1、warm5/ramp10）。不新增部署参数、头、对齐损失、课程、门控或U增强。

数据：原完整source RX1/3/4/6/8、day1/2/3、全部6TX，L6300/U56700隐藏TX/V27000，split392005，原physical IDs。无源训练RX留出、无5580/720替代；V仅原source健康诊断，无反传／EMA／状态更新。全部行从零，不加载历史训练父权重。完整source_contract元数据已核对L_s6300/U_s56700/V27000。既有VALIDATED_ONCE数据复用，无新数据验证或额外hash。

|row|有标签目标|每物理L前向|固定L前向包量|完整practical训练包量|
|---|---|---:|---:|---:|
|NATIVE|原RD(ordinary)＋原SupCon|2|1136640|0|
|REPEAT|.5RD(同ordinary对1)＋.5RD(同ordinary对2)＋原SupCon(对1)|4|2273280|0|
|CONCAT|.5RD(ordinary)＋.5RD(practical(raw))＋原SupCon(ordinary)|4|2273280|568320|

所有模型seed2026092701，各scratchE20=13L＋7U、222步/轮、4440更新，固定FINAL20学生默认头。合计3新模型／60model-epoch／13320更新，0续训。REPEAT和CONCAT共有ordinary对权重.5，另一个槽显式替换；不隐瞒native ordinary总权重由1到.5的差异，也不声称保留了native完整clean损失。每对是同一已生成IQ两次dropout，REPEAT不重抽ordinary。原SupCon只作用共有第一对，不扩展到As或U。三行保留原L/U采样，新增视图不改训练集合。

As沿用旧最朴素C1：均匀low_suburban/mid_urban/low_urban，完整residual/post_sync、EQ off/G=1、25MHz、2.462GHz、referenceSNR30、原质量/噪声/传播/补偿/AGC/reflect耦合；硬件按RX/day/eq稳定且不按TX定义。直接处理原规范化IQ，不把ordinary PA/AWGN/multipath预先叠在practical上。每个合法有标签satellite进入监督，没有高置信筛除、删难包或重复完整信道。实际导入、完整Config和普通增强resolved args会保存。正式测试保留原固定2.45GHz观测等配置，不提高SNR或改received IQ。

一份匹配完整源数据契约的固定final50 CROSS_RX checkpoint只用于兼容性no-query smoke（2合成包），不初始化训练或teacher。已核实它从零、无祖先／父CP、sourceV27000、无heldRX、target_contact=false、固定final50；发布首步再执行一次实际strict-load/no-query前向。每新模型native自身scratchCP另有2包smoke，计入实际计数。历史E50训练成本不算新训练，也不参与成绩对照。

比较固定CONCAT−REPEAT（完整practical视图增量）、REPEAT−NATIVE（额外前向效应）、CONCAT−NATIVE（整配方差）。普通拼接没有净星地收益、或只以clean下跌换差距缩小，均不能称本假设成立；负结果完整保留，不扫描配比／seed／RX救援。这里建立参照，不是新Ag或联合成功声明，不根据目标结果晋级或调参。

每行训练完按预定final20冻结，即自动逐行预测原clean＋六种practical：high、mid、low_suburban、high_urban、mid_urban、low_urban；全部168000×7预测与logp保存后，以exec启动独立CUDA为空CPU scorer连接truth，默认头准确率／宏F1／全部CM以第二实现复算。原正式观测manifest已核实VALIDATED_ONCE／168000／六场景齐备；不生成新正式IQ。全部3行都完成，不按前一行分数停止其他训练。报告每RX/day/TX、最差RX、urban均值、成对修复/新错、各suburban绝对值和与自身clean差。suburban探索差≤2pp和lowurban>60%预先量化，均非承诺；单seed不能证明稳定联合收益。

运行ID：20261017-phase1-F-crossRX-fullsource-concat-e20-s2026092701-m3-r01。配置/代码身份为实际Git提交（release_commit及resolved_config记录HEAD）；源码experiments/cvs_fullsource_concat20。CWD PROJECT/releases/cvs_F_fullsource_concat_e20_20261017_r01。命令：`CVS-RFFI/bin/python -u experiments/cvs_fullsource_concat20/runner.py`。N607普通账户、CPU2 dispatcher／worker、共享每GPU最多3个总进程（计入其他owner）。唯一owner codex/root/F-fullsource-concat-20261017。一个不可覆盖root PROJECT/runs/20261017-phase1-F-crossRX-fullsource-concat-e20-s2026092701-m3-r01，逐行source/prediction/scores和logs。

仅协议越界、角色／输出冲突、非有限更新、实现无法执行或不能生成合法prediction触发系统技术停止；保存失败／已有产物，停止pending，健康旧任务和已启动健康行继续，无自动重试或按分数早停。预期产物见experiment.json，含完整20轮／逐步日志、实际L/U序列、Sinc更新、As覆盖、默认头checkpoint、固定7视图预测／独立scores、实际成本、同row分析图与登记。IQ／权重／原始日志和预测不提交Git。额外真实训练、合成检查3个optimizer步骤、教师、U、target前向及墙钟分别实测，不能把计划数当已执行。完整Ag／As／四因子及交互／跨seed联合目标ACTIVE。


## r01 技术终态

FAILED／VERIFIED：发布归档解包和编译后，合成检查在评分反例测试桩中失败。测试桩把读取仓库内reference_recipe.json误判为数据越界；不是真实数据泄露或性能失败。远端release保留，实际run root／submit均不存在，所属进程0；真实训练更新／IQ／checkpoint／target均0。执行提交151a7eb75d68ff3b80c429750837fa29be79de23。

网络三行的合成部分已到达评分反例之前；从固定执行路径可推算合成optimizer3步、1292个NN包及158个physics包，未写出完整PASS结果，不冒充正式训练或完整检查通过。唯一测试桩修复：允许读取该仓库配置，路径比较用Path.name/parent.name；本地复现原异常后六个未完成预测门槛均正确拒绝，0真实数据／NN。保留r01，不复用release或覆盖输出；仅以新r02恢复同矩阵/数据/seed/E20，科学配方不变。
