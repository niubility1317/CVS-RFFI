# SARD 与无增强 F50 接 DURA：技术恢复三种子七景

状态：ANALYZED（VERIFIED）；实际代码／配置 commit `4f85eee2444c31f83981a736e45d6022be34cbe9`。继承 r01 在新目标访问前固定的同一科学矩阵；只修复原 native SARD source smoke 的推理模式兼容性，不改变模型、权重、星地视图、数据、种子或性能选择。

用户要求原 SARD（T1 E30＋20）和无星地增强最佳三 seed 原 F＋CROSS_RX E50 都加入固定 DURA v1.0 查看测试性能。NoAs 原 run `20261016-phase1-F-supcon-e50-clean-manysig-m9-r02`，clean 三seed 85.113889%±样本SD0.379881pp；SARD种子01复用原E50续训，02／03复用既有T1三seed续训。所有默认分类头均按原训练流程联合训练，本次严格冻结直接读出，不换头／另训头。

完整矩阵唯一保存在 experiment.json：两F家族×seed2026092701／02／03×原IQ／DURA输出×clean＋六practical；每单元168000原物理样本，split392005，L6300／U56700隐藏／V27000，原RX/day/TX和VALIDATED_ONCE数据不变。NoAs33L＋17U与SARD13L＋37U均累计50轮／11100更新，但不是纯As/KL因果消融。F和G来源都按原完整角色、祖先、固定FINAL和默认头检查。

DURA固定R69 stage2的E20／4940、266951参数和原checkpoint；执行commit47c264286ca689f060a3f36aafb763fe209a6f06，FP32／TF32off，reuse=True／graph=True／native=False。只复用原`20261021-phase1-restoration-stage2-graph-s2026092701-r01`完整七景恢复IQ，全部F读取同一固定G输出，opaque physical IDs与T1原索引逐项对应。G本次不重新运行、不训练；所有F/头/EMA也零更新。

r01三条NoAs原输入＋DURA全部十四单元及独立truth-last评分均已完成，原目录只读复用；本r02不再派发NoAs，不重复其预测／评分。r01的SARD01在首步source-Ls native对照的CenteredSinc读取_version时抛出Inference tensors do not track version counter，SARD目标预测0，02／03未派发。旧所属PID均已自然退出，无需停止任何任务。所有旧产物和失败开销保留。

本次单一修复仅使native源检查使用inference_mode(False)+no_grad，使原Sinc可读取普通中间张量版本；实际目标读出仍使用原FrozenCenteredSinc和固定权重。回归先本地复现同一RuntimeError，修复后5项测试及7文件AST PASS；原唯一P0/P1审查后，本次仅一次限定问题复核，均无遗留阻塞。无额外审批或数据重验证。

新r02仅启动SARD三行，新增21个DURA-F单元＝3528000个F目标读出；SARD原输入七景继续复用。连同r01已完成26个NoAs新单元，整链仍只有47个唯一新增F目标单元，37个更早的预测单元复用。r02导出F source smoke6NN＋native对照6NN＝12；r01成功NoAs source6NN保留，失败SARD的2包导出F及部分native执行另记，无目标前向。

每条SARD实际checkpoint和2个合法L_s无query smoke为首步；原native与导出默认头一致且冻结状态保持后立即预测。每row原输入／DURA全部十四单元固定后，独立CPU scorer才连接truth；NoAs沿原已完成评分不再执行。三旧NoAs＋三新SARD都完成后汇总Accuracy／Macro-F1、逐seed／RX／day／TX、repaired／damaged／net、同seed增量、均值／样本SD(ddof1)。G仅一个固定seed，不声明G跨seed稳健性；不把缓存读出时间当在线G+F时延。

普通N607 CVS-RFFI Python3.10／Torch2.1、CPU2、FP32／TF32off。CWD `PROJECT/releases/cvs_SARD_NoAs_DURA_E50_m6_20261021_r02`，命令 `CVS-RFFI/python -u experiments/cvs_sard_dura_eval/runner.py`；不可覆盖root `PROJECT/runs/20261021-phase1-SARD-NoAs-DURA-e50-m6-r02`，owner codex/root/SARD-DURA。八卡按所有owner总进程数最少和空闲显存最多排序，free≥12000MB，每卡总进程≤3、本run每卡1、同时最多2卡，CPUscorer1。原任务与完整输出保持。沿本任务已完成的普通账户直连preflight，另作本r02一次归档SHA比较／远端编译及启动后PID/CWD/argv/GPU/log-growth核实。

只因自有协议、来源、物理ID、种子、默认头、非有限、输出碰撞、无法启动或无法生成合法预测等系统技术失败取消未启动行并保留partial，不自动重试，不以分数停止。已暴露工程基准，结果不反馈调参、选模、预算或选择性重跑。预期新launch／workers／queue／reused_NOAS、三个SARD smoke／prediction／scoring／complete、含三旧NoAs的results和complete；缺任一row/view不称完整完成。

独立启动核验 VERIFIED：新归档比对、7文件编译通过；SARD01／02分别在GPU2／5，所属PID/CWD/完整命令、实际checkpoint、2包source-Ls smoke及原native默认头等价均PASS，实际预测日志已增长。NoAs三行从r01只读复用，未重复启动；SARD03由同队列随后执行，无新训练。

最终独立核验 VERIFIED／ANALYZED：六行／两组各三seed原输入及DURA全部七景齐全。全部source checkpoint smoke通过、SARD原默认头native等价通过；1512个CM和756个配对单元独立复算一致，全部完整日志无系统技术错误。整条执行链37个单元复用、47个新增F读出（r01新增26个，r02新增21个），零训练／零新G执行。完整逐seed、均值SD、净纠错和边界见 [正式测试结果](test_results.md)。
