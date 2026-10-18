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

run `20261018-phase1-FcrossRX-As-budget-e50-s2026092701-m2-r01` 已完成训练、源读出、七视图预测、独立评分和完整分析。执行提交 `0f399c910324e21a8b1a60346d879be5c2964a9c`；独立分析提交 `ab16c3848794124f8c7fdee6013e01b2a46aae73`。原训练／分析PID正常退出，无failure。504正式CM、600源CM、84分区合并验证；8880新增step与40个新增epoch全日志完成，匹配物理序列及普通增强MSE mismatch均0。详见该run的final_analysis.md与analysis_verification.json。

两个 row 固定 FINAL50，分别自动完成 source 读出与 clean＋六 practical 全部预测，再独立 truth-last 评分。报告各自 E50−E30、As50−NoAs50 以及预算交互，准确率／宏 F1／最差 RX／修复和新增错误／全部种子与成本。源或正式结果不反向修改本轮配方。E50 单种子正式 As−NoAs clean +0.365476 pp、mid／low +29.864881／+22.508929 pp、urban均值+30.707738 pp，本种子阶段通过。相对As自己的E30，clean／mid／low分别+0.319643／+1.095238／+1.143452 pp；NoAs自身新增预算clean+0.047619 pp、mid／low−1.095833／−0.261310 pp，不能把全部匹配差值归给新增预算。

另一对话已放弃重复 E50 实现并复用本侧；它运行的是连续 cosine30 的单 As 技术补跑，与本次 20＋10＋20 不同。健康任务不修改。automatic_seed_extension=false。

## E50 已完成结果与尚未解决的问题

源 full V27000 As50−NoAs50 clean −0.074074 pp；As50−As30 full V clean +0.118519 pp、V1800两draw mid／low +1.972222／+1.638889 pp，源两场均值+1.805556 pp、预算交互+1.472222 pp。固定预算假设获得该种子源域身份性能支持；不以正式分数作为晋级或调参依据。V1800 clean As50−NoAs50 −0.166667 pp 单列，不将panel与全V混为同一口径。

As50 clean85.660714%、mid urban63.901786%、low urban52.387500%、urban均值65.010913%。三个suburban与clean差距3.421429／6.104167／8.884524 pp；平均准确率较As30均上升，但low suburban最差RX下降0.116667 pp。新增两行40model-epochs／8880更新，全部新增训练／读出＋sharedsmoke实测14,922,586 NN；历史E30两行18,298,572 NN另列。训练是20+10+20/resetcos20，非连续cos50。

本次技术交付完整，但只有seed2026092701，尚无可重复双提升证据。下一唯一未检验命题是固定预算收益在原定其他模型种子能否复现；当前run仍automatic_seed_extension=false，未启动额外种子。多种子须保持原顺序、参数／数据／默认头／固定FINAL50及匹配NoAs，不挑最好seed或据正式分数救跑。原E30三种子负clean均值保留，目标继续active。

## 固定02／03复现：已登记待发布

已登记新run `20261018-phase1-FcrossRX-As-budget-e50-s0203-m4-r01`，唯一owner `codex/root/practical-budget50-replication`。只包含原定seed02/03各B00 NoAs与B01完整As，共4新row；各自合法FINAL30+20，总E50，固定参数、全部原数据与默认头/U/physics不变。依据仅为01的源域预算收益；不按formal分数选seed/调参数，01旧预测及所有E30历史原样复用。新实现提交 `4ea924316d6a5fe84278c81e3db78706ff674f21`、96负例/4seed正例/13py AST+compile/2emittedAST和唯一P0/P1均PASS。尚未远端发布或训练。

本轮新增80model-epochs/17760更新，预期29,845,170NN含shared3074，实际学生/教师/读出/父训练成本分列。逐行source→固定七视图预测→独立truthlast评分，正式分数不回流其他row。全部3seed逐seed/配对mean与sampleSD保留；每seed与mean阶段clean按−.1pp判定。只有三seedclean及low/mid都为正且clean配对mean≥.1pp时才称本固定三seed一致双提升，否则报告实际不一致/均值观察，不选择性补种。详细矩阵/命令/路径/停止规则在该run report.md。
