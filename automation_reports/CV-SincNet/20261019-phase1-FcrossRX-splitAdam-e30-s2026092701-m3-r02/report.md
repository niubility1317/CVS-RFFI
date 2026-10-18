# F＋跨 RX：ordinary／As 分离 Adam 状态 E30

状态：PLANNED；技术修复 r02，未训练、未评分。

R01三行因逐坐标梯度相等检查误拦退出：N01已成功16次更新，C01/S01为0。原产物保留；失败成本26684NN/44VJP/16更新，独立诊断1282NN/4VJP/0更新。诊断证实重复同一total-loss VJP也有Sinc近零坐标误差，真实clip正确。r02只将断言改为原容差下的逐参数RMS范数检查，记录旧逐点失败而不阻断；不改变梯度、loss、clip或Adam配方。若同指纹再次出现，停止盲目重发。

## 判断与唯一假设

完整 practical 是应保留的基础。已完成的条件分层、U 路由和公共传播组织均有固定版本的源负结果；不原样扩展。PCGrad 已实际训练；viewMoment 只向同一个 AdamW 二阶矩加入视图分歧平方项，不能代替独立状态实验。P41／mixedContrast 已让实际 As 参与标签 SupCon，不能换名重做。

本轮仅检验：先把 ordinary/raw/SupCon/U 与 As 梯度相加后作 AdamW 归一化，是否限制了可用的 As 身份更新。已有导数差异不证明这一原因。候选将两组独立的一、二阶矩归一化后以固定 0.8/0.2 合成一次参数更新。0.8/0.2 来自原有标签监督名义质量 1.0/.25 归一化；不声称实际 As 参数位移仍占 20%。

## 最小匹配矩阵

逐行配置唯一维护在 experiment.json。N01：自身 NoAs20＋10，共享 AdamW；C01：自身 As20＋10，共享 AdamW；S01：同一 As20＋10，分离状态。三行均新增 2220 次更新，两次 VJP／更新，无新增网络前向、训练损失、部署参数或分类头。MEASURE 不改变原生 loss/backward/step，候选一次 decay／step／EMA。

原父运行：20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01，实际 release commit 19aedc3604fed9c193237dd92366164178af0983。使用其原自身 scratch20 的 B00／B01 student 及 ownEMA；完整继承及实际 source role 由现有 parent loader 核对，禁止关闭候选／目标选父。

完整 L6300/U56700（TX 隐藏）/V27000；RX1/3/4/6/8 全部训练，day1/2/3、split392005；没有训练 RX 留出或内部 Ldev。IQ256、Fs25MHz，真实 joint160、SIG_TIME_X3／统一 Sinc、默认 cosine6×160 联合训练。保留必要补偿和 NoExplicitCFO。

源条件在正式结果读取前固定：S01−C01 的固定两 draw×mid/low 平均准确率至少 +0.5 个百分点，S01−N01 的完整源 V clean 不低于 −0.1 个百分点。任一失败则关闭，不扫描 0.8/0.2、不挑 seed／父状态补救、不扩 E50。即使源条件通过，也只是未来预算资格，不证明未知 RX clean 或最终目标。所有三行均自动完成固定七视图评分，科学负结果不触发停止。

## 执行与成本

实际代码身份为发布时 Git HEAD，写入 release_commit.txt 和独立启动证据；不是设计文件 SHA。命令：CVS-RFFI/python -u experiments/cvs_split_adam30/runner.py。环境：现有 N607 普通账户／PyTorch；CWD PROJECT/releases/cvs_FcrossRX_splitAdam_E30_20261019_r02；输出 PROJECT/runs/20261019-phase1-FcrossRX-splitAdam-e30-s2026092701-m3-r02；log 子目录／逐 row 独占。CPU2；每 GPU 所有 owner 合计最多 3 个计算进程且空闲至少 12000MB；不干预健康旧任务。

三条新尾段合计 30 model-epochs、6660 次新更新、13320 次 VJP（各 VJP 图的实际成本单列，次数相同不等于耗时相同）。两份唯一父模型累计 40 model-epochs 为既有成本，不重复计为新训练。每条尾段 L NN1704960、原 U student567000、teacher567000；实际源健康 eval、源面板12600、正式每 row1176002、真实 checkpoint synthetic smoke另计。记录 masked RD 额外张量运算、分离状态内存与墙钟，不预告速度收益。

仅停止准确归属本 run 的系统技术错误：角色／query／父权重／头或输出碰撞、非有限、不能产生合法 prediction、至少两行同指纹的 pre-prediction 故障。保留产物并核实进程后处理，不因低性能重跑。

## 测试与交付

固定 clean＋practical_high／practical_mid／practical_low_suburban／practical_high_urban／practical_mid_urban／practical_low_urban；每视图168000 原有物理 ID／received IQ。每 row 全部 prediction 固定后单独 exec scorer 才连接 truth；不回流训练、选模、比例、seed 或 E50 决策。默认头直接预测。

报告七视图准确率、macro F1、最差 RX、RX/day/TX 分层、修复／新增错误、urban 三场景均值、suburban 绝对差距、完整成本和未完成项。只有一个模型 seed2026092701；两次信道 draw 不当作额外 seed。不足 0.1 个百分点的微增和单 seed 双涨均不能宣称稳定双提升。

预期产物：experiment.json／events.jsonl、源与配置 metadata、final checkpoint、逐步完整日志、source_matrix_decision、固定 predictions、独立 scores、automatic_summary、最终分析和同 row 判定。
