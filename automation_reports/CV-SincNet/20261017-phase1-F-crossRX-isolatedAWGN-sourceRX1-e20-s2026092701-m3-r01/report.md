# F＋跨 RX 同 TX：孤立 AWGN 的三组匹配源 E20

PLANNED，尚未启动。唯一矩阵：[experiment.json](experiment.json)。B00／C_ordinary／B10 分别增加权重 .25 的 raw／同一已生成 ordinary／native AWGN(raw) 配对视图。公共原 F＋跨 RX 同 TX、三 SIG_TIME／统一 Sinc／joint160／NoExplicitCFO／cosine30 默认头保持正常联合训练；scratch seed2026092701，E20／13L＋7U／4440更新每row，三row合计60 model-epoch／13320更新，无父权重或续训。

共同完整base保留raw.75＋旧ordinary.25＋原shallow/mid CROSS_RX SupCon；extra .25R(view)，R=原smoothedCE.01均值＋同IQ RDrop1KL，均8L前向／总pair权重1.25。raw绝对监督份额稳定，相对占比变化如实记录。C_ordinary复用同一实际ordinary tensor，不重抽；每个新IQ只生成一次供两次RDrop网络前向。没有新增loss/部署分支/头。

本唯一假设：在已通过双默认头源身份保持（50402NN／0更新）的基础上，额外孤立轻噪声是否比同成本raw/ordinary视图更利于held RX身份提取。G0只是PASS_SOURCE_IDENTITY_ONLY，不是DG或星地收益。方向区别另一对话完整practical内部正负噪声配对，已有失败路线保持关闭。

B10在原raw调用native proper complex AWGN，各包uniform[20,36]dB、按包功率；范围来自旧ordinary，非目标或源幅度搜索。private seed202610172301＋optimizer_step，fork CPU及当前CUDA状态后恢复。CPU/GPU同native分布，不声称抽样相同。算子不读标签/RX/day/置信度，无MP/PA/CFO/phase/shift/DAC/量化/clip/后RMS/重抽。名义SNR有界，高斯幅度不严格有界，所有实际能量和尾部保留。added-noise SNR不是原录音总SNR，数字IQ变换不是更换真实RX。

实际sourcefold为fit RX3/4/6/8、L5040/U45360/V21600、day1/2/3、6TX和原物理ID权限；held sourceRX1 L1260仅final20读出，不是目标数据。不混同fullL6300或L_train5580/L_dev720。原U隐藏TX真值、13L+7U/旧gate/自有EMA/原增强保持；新增噪声不扩展U，不增加教师前向。复用原VALIDATED_ONCE source handles，只核对实际role/RXfilter。

每row固定final20冻结后立即自动生成heldRX1 clean＋六practical的原默认头预测，各1260／8820每row／26460总。场景high、mid、low_suburban、high_urban、mid_urban、low_urban；相同源物理观察和既定channel/session/eval streams，不改received IQ、传播、SNR或补偿。此明确source-only首轮不构造target或正式truth评分；完整契约四格／跨seed／独立truth-last仍待完成。

固定科学条件：B10相对BOTH控制 clean accuracy≥+1pp、clean macro-F1≥0、各六practical≥−.5pp、seenV≥−.5pp；健康需完整预算、SupCon3330次、final base L objective≤.5。任一FAIL关闭固定20至36dB／全覆盖方案，不抬SNR、改范围/系数、clip/删尾/重抽、按RX/TX/day条件或换seed/RX/预算救跑；全部3row及读出仍完成后统一判断。PASS仅支持独立源seed确认，不等于DG/As/联合成功。

成本：每row4546560 L NN，其中1136640额外；旧ordinary568320每row，B10noise568320，控制noise0，训练As0。原U/EMA/源V、held8820、own2NNscratch smoke和墙钟分别实际记录。dispatcher先做真实匹配CP合成2包no-query smoke，绝不作为训练初始化，PASS立即继续，无token。保存全部13320native/view/Sincunitsteps、L/U序列、全noise SNR/RMS/功率交叉项/72格覆盖。

本地28负例、4114合成NN、3合成联合更新、640noise／128ordinary PASS，0realIQ/CP/target；nativecanonical、三组scratch初始化/CPU/Python RNG、B00 loss/headgrad精确匹配旧additive raw、sameIQ8前向、原U args、export160均核实。一次独立P0/P1审查PASS见review.json；旧Torch NumPy桥未调用，不安装环境。

唯一owner codex/root/F-isolated-AWGN-E20-20261017；普通N607现有CVS-RFFI Python，dispatcherCPU2、workersFP32/loader0/CPU2，原共享容量锁计入全部owner，每GPU最多3进程且free≥12000MiB，不干预健康旧任务。CWD PROJECT/releases/cvs_F_isolated_AWGN_sourceRX1_e20_20261017_r01；入口CVS-RFFI/bin/python -u experiments/cvs_isolated_awgn20/runner.py，逐row命令见矩阵。不可变输出PROJECT/runs/20261017-phase1-F-crossRX-isolatedAWGN-sourceRX1-e20-s2026092701-m3-r01，row日志在logs/，release/diagnostic.stdout.log。

发布使用已验证本地Torch Python E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 -m experiments.cvs_isolated_awgn20.publish，CWD当前managed WT。Git commit为代码身份，实际写release_commit.txt/resolved；一个普通账户预检、一份archive/一次传输SHA比较、remotecompile、启动后PID/CWD/argv/GPU/loggrowth读回。系统性protocol/role/init/budget/output/nonfinite/启动或非法预测错误保留failure/partial、取消pending、健康worker继续，无自动重试。

期望：实际checkpoint/args/角色/初始化/冻结、每row默认头7views、全native/view/unitsteps和完整noise统计、实际计数／全曲线／分层／suburban绝对与ownclean gap／修复新错／独立复算报告登记/Git。IQ/权重/逐包ID/预测/日志不提交Git，只提交聚合。有效Ag、稳定As、完整四格交互及正式联合目标仍ACTIVE。


实际发布：VERIFIED RUNNING，执行commit 06ebf579c87c834d0ebc14856212e3fa97619d30。一个archive传输比较／remotecompile／真实匹配CP两合成包smoke PASS；三row真实进程CWD/argv/GPU及存活期loggrowth已独立比对，GPU0/1/2，未修改旧任务。详细状态见[launch_summary.json](launch_summary.json)。当前已开始训练，尚无最终heldRX读出／科学判定／正式评分，不能声称新增Ag有效或联合成功。
