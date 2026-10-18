# 星地识别优先：E30 探索与 E50 匹配预算验证

2026-10-18 用户最新要求：提高固定六种 practical 的识别性能，重点 low／mid urban；阶段 clean 相对同预算、同种子、同数据角色的 NoAs 下降不超过 0.1 个绝对百分点。最终仍需取得可重复的星地与 clean 双提升。单种子或不足 0.1 pp 的上涨只能作为初步观察。无需必须构造 Ag＋As，也不设独立额外跨 RX 泛化门槛；RX／day 分层继续报告。历史判据与结果保留。

固定 F＋CROSS_RX、三路 SIG_TIME／统一 Sinc／joint160／默认 cosine 头随骨干联合训练、NoExplicitCFO 与内部补偿；原 L6300／U56700（TX 隐藏）／V27000，源 RX=[1,3,4,6,8]、day=[1,2,3]、split392005、现有 IQ256。全部训练 RX 保留。正式 target RX=[0,2,5,7,9,10,11]、day=[0,1,2,3]，原 received IQ／信道／SNR／所有包不变。source 检查聚焦真实标签破坏、数值异常和明显性能损伤。目标数据不参与设计、选模、晋级、选种或救跑。

## 已完成证据

原完整 practical 配方仍有最清楚的星地收益。三模型种子相对匹配 NoAs：low urban +17.2849±3.4169 pp，mid urban +21.9448±5.1968 pp，urban 三场均值 +22.3079±5.5014 pp；clean −0.3210±0.5189 pp。clean 逐种子为 +0.0935／−0.1536／−0.9030 pp；未满足三种子的阶段要求，也未取得稳定双提升。完整逐种子、六场、F1／RX／成本在 As-replication-r02 报告，不把不同研究增益相加。

最新完成 `20261018-phase1-FcrossRX-satelliteDropout-e30-s2026092701-m2-r01`：新 AS_CE／AS_DET 两行各 scratch20＋10，复用 P43 同种子 NoAs／原 As 两行，L 前向和原 U 匹配。AS_CE 仅去 satellite KL；AS_DET 再仅对实际 As 行绕过 feature dropout，完整生成原 mask，未修改部署／分类头或信道。

该候选四项预登记源域条件均 PASS，不能改写成 source 失败。正式单种子 AS_DET 相对 NoAs 的 clean −0.126190 pp、mid urban +26.505357 pp、low urban +20.466667 pp、urban 三场均值 +27.117063 pp；clean 超过 0.1 pp 代价。相对原完整 As，clean／mid／low／urban 均值分别 −0.219643／−1.168452／−0.637500／−1.251389 pp，全部七视图都低于原 As 与 AS_CE。此固定单种子没有新增机制收益；不据正式成绩微调参数或选择性救跑，不以 n=1 宣称机制全局被否定。源通过未转化成正式收益。完整证据见该 run 的 final_analysis.md。

## 当前唯一假设与最小实验

用户明确授权“对于表现好的，可以跑 E50 更多预算，看看有没有提升”。常规探索默认仍为 E30，当前仅对原完整 As 做 E50 预算验证。候选在读取上述 dropout 新正式结果前，已由原完整 As 的固定三种子合法 source 证据选定：full V clean 配对均值 −0.080247±0.020398 pp；源 mid／low 增益均值 +24.972222／+19.842593 pp。01 的源 clean −0.103704 pp 略超过 0.1 pp，如实保留，不用源均值冒充正式通过。两 draw 来自同一 V1800，不算两个模型种子。

唯一假设：保持原完整 As 配方不变，各自固定 FINAL30 再追加 20 轮，能否提高星地表现，并使 As50 相对同预算 NoAs50 的 clean 下降不超过 0.1 pp。最小矩阵固定 seed2026092701：原 NoAs 与原 As 各从自己的 FINAL30 student＋own EMA 追加 20 轮。既定首种子顺序不变，不从成绩挑最好种子。保留既有 U 规则、默认联合分类头、同视图数、同损失尺度和同训练预算。

轨迹为 20＋10＋20，总 E50；新增段重置 AdamW／cosine20，不称连续 cosine50。记录父训练／新增／累计成本，以及学生／教师、额外前向与读出成本。无新 Ag、分支、头、损失或门控；不是 dropout 方案的救跑。若追加预算没有源域 practical 正收益，或 clean 明显受损，则预算假设不获支持；全部固定正式评分仍按计划完成，不调参救跑或自动扩种。

run `20261018-phase1-FcrossRX-As-budget-e50-s2026092701-m2-r01` 已实际启动，执行提交 `0f399c910324e21a8b1a60346d879be5c2964a9c`。独立核实 dispatcher 1143533 和两 worker 的 PID／CWD／argv／GPU／日志增长；观察时 NoAs 累计 E35、As 累计 E33，无 failure，U 真值未读、target_access=false。远端 30 项负例、真实父 checkpoint 2 NN smoke、双真实 F 合成 1,536 NN 均 PASS。预登记、命令、路径、停止规则、实际证据见该 run 的 report.md／execution_verification.json。

两个 row 固定 FINAL50，分别自动完成 source 读出与 clean＋六 practical 全部预测，再独立 truth-last 评分。报告各自 E50−E30、As50−NoAs50 以及预算交互，准确率／宏 F1／最差 RX／修复和新增错误／全部种子与成本。源或正式结果不反向修改本轮配方。当前 E50 未完成，无性能结论，无阶段通过或稳定双提升证据。

另一对话已放弃重复 E50 实现并复用本侧；它运行的是连续 cosine30 的单 As 技术补跑，与本次 20＋10＋20 不同。健康任务不修改。automatic_seed_extension=false。
