# P66：完整 practical U 教学的观测条件分配

本轮唯一假设：在保留全部有标签 As CE 与原 U 训练的前提下，将额外 U 默认头 KL 按本次学生实际星地观测的新增噪声能量比分配，比相同总教学权重的均匀减弱更有效。先训练两个匹配 E30 行，不按正式测试成绩调参或晋级。

上一轮 P65 已完成并关闭。额外 U 教学相对原 As 有小幅 urban 收益，但 clean 代价超标；固定 NoAs 教师相对固定 As 教师未显示足够来源优势。物理质量 q 已用于旧 HQ/L margin 且源失败；本次只检验此前未实训的 U 默认头 KL 作用位置，不重开质量代理修正、状态筛选课程或教师来源路线。另一聊天“审计并设计 CVS 联合增强”只读核查：完整 N0 随机化、延后 As、场景配额均已完成或关闭，本轮不重复启动。

增强前基座固定 F＋CROSS_RX。模型为 SIG_TIME_X3、统一 Sinc、CENTER_UNITS、joint160、默认 cosine30；900722 参数、206 个部署状态键。保持原联合训练、六 L 前向、L 损失规模 1.25、SupCon .1、RDrop 和旧 U 规则，不新增部署模块、诊断头或一致性家族。

## 数据、运行基线与来源

代码起点 5ca93a7e8f9692971b892ec19a3bb68cc888454d；实际发布代码以本记录 code.commit 为准。继承已核实 P43 As20 的 student 和 ownEMA，同 seed 2026092701。父模型从零训练 20 轮、4440 更新，13 L＋7 U，无外部父权重、无目标选模，final_only。新尾段各 10 轮、2220 更新，重置 AdamW/cos10/cache，学生累计 E30/6660 更新。唯一冻结 raw-U 教师是同一 As20 checkpoint 的 ema_model；无新增教师训练，也无第二份 NoAs20 历史前提。

数据完整 L6300/U56700（TX 始终隐藏为 -1）/V27000，源 RX1/3/4/6/8，day1/2/3，原六 TX，split392005。无 held RX、无 L_dev。既有 VALIDATED_ONCE source handles 和正式观测复用；Phase1 的 p2 capsule 不适用。

实际路径由 launcher 继承 P43 固定运行时，leo_practical 来自 evidence release 下 experiments/adv3b02_xuc/code，不能以根 code 默认值替代。训练启动日志打印 __file__ 和配置。IQ 为 B×2×256、25 MHz；进入 practical 前原 per-packet RMS、center=false。完整 residual/post_sync、EQ=false（G=1）、质量适配/相位跟踪/注入 RX IQ 补偿开启，reference_snr_db=30、reflect、训练 fc2.462 GHz。正式 received IQ/fc2.45 GHz、clean 和六 practical 均不更改。

L/U 场景均为 low suburban、mid urban、low urban，各 1/3。传播、遮挡、噪声、接收质量和补偿依赖保留。硬件由物理 RX/day/eq session 稳定定义，不含 TX；实现按 epoch/step/sample 命名流刷新，源验证和正式观测流分开。

## 唯一匹配矩阵

两行都从同 As20 独立分叉。每个原 U256（尾124）只取首 min(128,n)，额外生成一次完整 practical 学生视图，并执行一次固定 As20 raw-U 教师前向。没有 U 标签、学生 urban 置信度门、padding 或困难包删除。额外 dropout 私有随机流恢复，原 L/native U 随机流不被扰动。

- MASS_UNIFORM：qT×sum(qT qN)/sum(qT)。同权重质量均匀减弱控制，不作为晋级候选。
- OBSERVATION_WEIGHTED：qT×qN。唯一主候选。

qT=clip((softmax(teacher).pmax−.95)/.05,0,1)。qN=sigmoid((signal_to_added_noise_db−5)/5)，沿用既有物理映射；全部三个训练场景应用。None 显式 qN=0 并计数，不删除样本。sum(qT)=0 时两行均零。两组 q 总和同质量，每批记录误差与状态分配。KL 为 T1、教师 stop-gradient、whole-subset mean(q×KL)，系数本地前五轮爬坡至 .16；不按正 q 数量归一化。

signal_to_added_noise_db 是实际全 256 点信号分量与本次新增噪声分量的能量比，含原录音未知噪声和 reflect 边界；不是总 SNR，也不是身份可恢复概率。quality_snr_db 是白输入功率代理，lock/residual_scale 同源。本轮不改变这些状态或 IQ。AGC_limited 表示增益上限，此实现没有波形 clipping。EQoff 时新增噪声不经过 H；当前仍含注入 IQ 的广义线性作用。

## 预登记判定与自动收尾

唯一候选在完整源 V clean 与固定两 draw 源 practical 面板判定：clean−NoAs30≥−0.1 绝对百分点；mid/low 均值−As30≥+0.5；low−As30≥0；mid/low−MASS_UNIFORM≥+0.25；low−MASS_UNIFORM≥0。只读源结果作出并固化判定。PASS 可按最新授权开展总 E50（另补 E20），同时保留同种子/同预算/同数据角色的必要控制；FAIL 关闭本路线，不用正式分数换 q、教师、系数、种子或选择性重跑。源两 draw 不是两个模型 seed。

每行训练冻结后立即自动 clean＋六 practical 预测，全部七视图完成后单独 CPU truth-last 评分。源 FAIL 仍完成正式评价。最后 CPU 分析复算全部行/全部视图、逐 seed accuracy/F1/最差 RX、urban 均值、配对修复/新增错误、源状态/质量/分类间隔、训练覆盖/曲线及实际成本。固定 NoAs30 clean 正确集合的保持率明确为参照集合，不冒充 As20 父模型正确集合。不得把物理状态相关性写成损伤因果证据。

## 预算、命令、环境与停止规则

新训练 2×10=20 模型轮、4440 更新；各学生累计 E30。同一既有 As20 兼作父和教师不重复计训练成本。每新行预期 native3162962＋studentU284120＋frozenU284120=3731202 NN。两行加 source25200、formal2352004、前置 smoke4＝9839612 远端 NN 预期；实际计数为准。本地一次真实 F 合成预期3072 NN，真实 IQ/CP/更新0，单列。父成本、共享 NoAs30/As30 历史成本与新成本分别报告，不把旧参照称作全部前向成本匹配。

普通非特权 N607，既有 CVS-RFFI Python/Torch，full FP32、CPU2。CWD=PROJECT/releases/cvs_FcrossRX_U_quality_E30_20261018_r01。执行 CVS-RFFI/python -u experiments/cvs_u_quality66/runner.py。不可变根 PROJECT/runs/20261018-phase1-FcrossRX-Uquality-e30-s2026092701-m2-r01；日志 logs/ROW.log。两行按资源队列运行，每 GPU 最多2个总进程，计其他 owner，free≥12000 MB。健康实验不停止、不热改、不重启。

只有数据/父来源错误、运行时错误、非有限实际 loss/gradient、输出冲突等预登记技术故障停止本 run 所属行；保留全部产物，不自动技术重跑。其他健康行继续。预算用尽不是收敛。新协议负例、一次真实 checkpoint 无 query smoke、一个 P0/P1 评审、一份 archive 比较/编译和启动 PID/CWD/cmdline/GPU/日志增长核验按最小工作流执行，不新增签名或哈希链。

预期产物：每行 source/resolved_config、actual_modules/channel、source_contract、initialization/EMA/optimizer、step/epoch/training sequence、u_quality_actual/progress/steps/physics、source_frozen/completion、source_qualification；全部七 prediction/scoring；source_matrix_decision、results、analysis_r01/complete、CSV、PNG/PDF、完整负结果/缺项。权重、IQ、原始日志与私有运行信息不入 Git。
