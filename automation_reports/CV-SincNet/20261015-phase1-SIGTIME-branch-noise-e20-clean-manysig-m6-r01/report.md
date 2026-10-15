# SIG_TIME 训练噪声位置：正式默认头三种子验证

状态 **RUNNING / 启动 VERIFIED**。下文保留预登记。完整源seed02三条件已ANALYZED/VERIFIED，固定资格{'CLEAN_SIG_primary_qualified': True, 'NONE_independent_confirmation_passed': False, 'all_source_healthy': True, 'main_run_launched': False, 'target_access': False}。本次固定arm NONE / CLEAN_SIG，主指标arm为CLEAN_SIG，每arm三个原定seed2026092701/02/03，所有row scratch20/13L+7U/4440真实更新；总6模型/120epoch/26640更新。CLEAN_SIG通过时必须保留NONE正式匹配控制，既保留正路线，也检验签名取点的独立增量；不在目标结果出来后决定要不要补控制。源资格证据见source_selection_evidence.json。本轮不继承四RX或旧全5RXcheckpoint，所有model从零。

实际L6300/U56700隐藏TX/V27000，RX1/3/4/6/8、day1/2/3、原六TX与物理角色/NoExplicitCFO及内部补偿不变。所有train/U/V/calibration都是原完整五RX，不调用fold helper。每row原U7轮396900重复观察，0额外训练IQ前向/新loss/参数/头拟合。原oldgate/ownEMA/Gaussian U/AdamW2e-4/wd1e-4/clip5/cosmin1e-6/LS0保持，无普通L或星地增强。默认头随骨干正常联合训练，不另训任何诊断头。固定FINAL学生，不用EMA或尾平均选读出。

原匹配BASE_SSL三个E20父只复用预测/评分（504000决定/54指标），原训练60epoch/13320步单列；新目标决定1008000、指标108。前置源探索两个seed各60epoch独立披露，不冒充本次训练成本。只有已完成源矩阵按固定规则确定arm，目标分数不回流选结构/父/seed/停止或重跑。

首seed所有必要arm完整训练冻结后，在源V健康上与同seed原BASE_SSL比较：final训练LCE≤.5，V drop≤1个百分点，最差源RX drop≤2个百分点；任一失败则全批0target，不改门槛。通过后逐row自动clean168000默认头预测，完整后独立新CPU scorer连接truth；第二/三个seed源健康仅报告不筛seed，训练预算/配置始终固定。最终汇总另启新CPU进程，报告每arm三seed mean/sampleSD/MacroF1/最差RX/day、逐seed修正及新增错误，若有CLEAN_SIG再与NONE逐包配对。所有结果保留，主指标不按target重排。

真实feature-only和主前向joint160相同；对原W做扰动也不改z，所有3seed/两支持mode的200初始state与原BASE_SSL逐位匹配。只证明导出边界/初始化，不替代性能证据。CLEAN_SIG只让签名读取t2.drop前、原pool后输入，不detach；主干mask/RNG保留，上游随机性仍在。签名池化和关闭失活有既有先例，当前是CVS训练噪声位置的待验机制；不宣称全球首创、未见TX或卫星/Phase2能力。最终正式目标仍是固定主arm三seed clean均值84%，尚未完成。

owner codex/root/SIGTIME-branch-noise-train20-20261015，actual Git commit以release_commit.txt固定。CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_branch_noise_train20_20261015_r01`，命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u experiments/cvs_sigtime_noise_train20/runner.py experiments/cvs_sigtime_noise_train20/configs/matrix.json`，不可覆盖root `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-SIGTIME-branch-noise-e20-clean-manysig-m6-r01`。普通N607/CVS-RFFI Python，一GPU顺序pipeline，所有owner每GPU≤3，最少进程/最多空闲且>3072MiB，不停止或热改健康旧任务。一preflight/一archive SHA比较/remotecompile，每row初始真实scratch checkpoint no-query smoke PASS后直接继续；12聚焦检查、实际模型边界检查及本代码唯一独立P0/P1审查PASS，实际固定arm配置也须与生成函数逐项吻合。启动后独立PID/CWD/argv/GPU/log核验。技术停止、矩阵和路径以experiment.json为准，无额外审批或hash链。

启动独立核实：commit `0fe97397defa9c72cb238556f0facc8d56c55513`，ordinary GPU0所属PID/CWD/argv/GPU一致，所有owner进程1/3，日志增长、真实own scratch checkpoint no-query smoke PASS。首row NONE是必要对照，实际seed2026092701/LS0/13L+7U/t2p0；原完整5RX L6300/U56700hidden/V27000，无fold mask/旧权重。源资格固定主arm CLEAN_SIG，NONE独立确认失败的事实明确保留。首seed两arm健康通过前不访问target，后续自动逐row预测/新CPU评分；尚无正式结果或84%结论。

<!-- RESULTS_START -->


完整源训练1/6行，clean已预测评分0/6行；独立解析4440条连续step和20条epoch。其余健康任务按冻结计划继续。


| 源训练完成row | final训练LCE | V准确率 | 最差源RX | 与原baseline初始200state相等数 |
|---|---|---|---|---|
| NONE-s2026092701 | 0.02131456 | 98.303704% | 95.277778% | 200 |

尚无完成的本轮target预测/评分；源健康和权重更新不替代正式识别性能。

NONE-s2026092701：U总396900重复观察、被选观察6477.0，所有loss有限/Utruth0；F71初末参数位移10.63857，默认W位移0.60803516，正常联合训练。全部20epoch与两阶段数值保存在CSV/JSON；多次U观察不冒充不同U包覆盖。

NONE-s2026092701 单独的固定合成IQ检查：PASS，训练前后实际joint160输出L2变化20.102078；feature-only与部署joint160逐位相等=True，扰动默认W后joint160逐位不变=True。本检查仅作8个合成packet特征前向，实际IQ读取0、truth读取0、优化/拟合头/改写checkpoint均0；证明计算路径已学习及特征/头边界，不代表识别率或未见TX泛化。

完整产物分析读取全部训练日志、合法同契约checkpoint和已经冻结的truth-side混淆矩阵/正确性；重算18分层CM指标、分层合计和逐包配对，不新增IQ、truth读取、推理或训练。固定合成IQ特征检查另列实际前向量；不参与源健康门槛或训练选择。部分进度中的训练成本只统计已完成row，未含正在运行row的已消耗量。完整分组指标保存在results_analysis.json；默认头联合学习的变化不能直接称未见TX或丢弃W后的迁移提升。
