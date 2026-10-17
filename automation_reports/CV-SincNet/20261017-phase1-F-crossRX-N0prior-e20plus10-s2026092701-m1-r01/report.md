# 固定 N0 先验训练：首个 E30 反证

状态：ANALYZED／VERIFIED；固定 N0 训练配方未通过预定识别收益条件。执行代码 commit=a47ffaf90888347fc3474d43ed365c7e51b1c1a9，远端 OID VERIFIED；此前源输入实验仍为失败。

基线为 F（三路 SIG_TIME、统一 Sinc 修正、joint160）＋跨 RX 同 TX SupCon，原默认 cos6×160/scale30 头与骨干联合训练。全部 RX [1,3,4,6,8]、day[1,2,3]、L6300/U56700（TX 真值隐藏）/V27000、split392005 保持；不留出训练 RX，不混入 L5580/L_dev720。

唯一假设：固定等先验线性 N0 倍率 [0.5,1.5] 的训练观测，可使同一个模型适应原 practical residual，并保持 clean。仅改变一个有标签 As 的噪声参考方差；由原实现重新计算接收质量、锁定、残余尺度、fine CFO/相位跟踪与 AGC。NoEQ，原传播、场景、几何、状态、shadow/scatter、标准噪声、15 个 session 硬件/校准随机流保持。一包只调用一次完整 residual，不额外注入 AWGN、不重归一化输出、不按置信度筛除有标签 As。期望线性 N0=.001，实际均值和次数如实记录；不声称平均 SNR/dB 或实测接收机温度保持。

前一源研究 20261017-diagnostic-F-N0prior-fullL-s2026092701-m1-r01 在冻结父模型下失败：mid 增益 .492063492pp 未达 .5pp。此结果与阈值保留。它没有训练，因而不能否定训练后收益；上一轮将其扩大为永不训练的停止条件过强。本次单独登记首个训练检验，不重跑源筛查，不改倍率、概率、种子、父权重或源门槛，不宣称源通过。它是 As 配方检验，new_Ag=false。

同 seed2026092701 的普通 As source-final20 与 ownEMA 严格加载，再重置 AdamW/native cosine10，新增 10 轮、2220 更新，累计 E30/6660 更新；不是 scratchE30 或恢复原优化轨迹。父路径为 PROJECT/runs/20261017-phase1-F-crossRX-fullsource-concat-e20-s2026092701-m3-r02/CONCAT-s2026092701/source。匹配普通拼接式控制复用 20261017-phase1-F-crossRX-clock-e30-score-recovery-s2026092701-m5-r01/FULL_AS-s2026092701。父训练20与新增10、NN与物理处理/教师/验证成本分别报告。

每物理 L128 保持 4L 前向：普通 IQ 的 RDrop pair .5，As 的 RDrop pair .5，既有 SupCon .1。每个 pair 使用同一个已生成 IQ；所有 As 标签均受监督。原始监督不稀释，没有新 loss、参数、U 视图、教师、门控或部署分支。独立 N0 selector20261017961 只使用物理 ID 与累计 epoch/step，不依赖 TX 或全局 RNG。

固定 final10/cumulative30。预定训练反证条件同时满足：相对普通拼接式 low urban≥+.5pp；clean≥0；mid urban≥0；各 suburban≥−.25pp。失败关闭该固定训练配方，不以改 N0/SNR、概率、seed、父权重或延长预算救跑。无自动扩种。报告同预算 B00 clean 净收益、low>60%、各 suburban 绝对准确率/距 own clean（探索2pp），不把单 seed 或这条 As 比较当成完整 Ag/As 四因子结论。

发布命令：CVS-RFFI/bin/python -u experiments/cvs_n0_prior30/runner.py；CWD=PROJECT/releases/cvs_F_N0prior_E30_20261017_r01；环境为现有普通 N607 CVS-RFFI/FP32/CPU2。GPU 遵守全 owner 每卡总进程≤3和原容量规则。唯一 owner=codex/root/F-N0prior-E30。immutable root=PROJECT/runs/20261017-phase1-F-crossRX-N0prior-e20plus10-s2026092701-m1-r01；预期 source/checkpoint/完整逐步日志、training_result、7view predictions/logp、独立 scores、同 row 统计/图与登记。

训练后自动固定原 clean＋六 practical 共7×168000条预测及 logp，再 exec 独立 CPU truth-last scorer。原 received IQ/SNR/信道/样本不变。目标结果不参与参数、晋级、排序或选择性重跑。

仅遇数据/父来源不合法、非有限、不可启动、输出碰撞或系统性技术失败停本 run，保留产物；不停止、重启或修改健康旧任务，不自动重试。启动后核实 PID/CWD/cmdline/GPU/log增长。

本地 focused checks：PASS，33项负例、30个适配器案例、15个原核心精确回放（误差0）、输入不变/批次次序不影响/全局 RNG 不变、900722参数与默认头。只有2条合成 NN，无真实 IQ、checkpoint、训练更新或目标读取。已知本地旧 Torch/NumPy ABI 告警存在，检查 exit0/全部断言通过；正式运行使用既有远端环境。

独立 P0/P1 审查：PASS，仅检查本次新候选实际训练接线；18文件语法检查、预算复算通过，无真实 IQ/CP/目标读取，无 NN/远端动作。未增加其他审批层。

发布后独立核实：archive_transfer/remote_compile VERIFIED，远端33项负例和15次精确回放 PASS，真实父 checkpoint no-query smoke PASS。dispatcher613750、worker613775/GPU0均 alive，CWD/argv匹配；参数900722、RX[1,3,4,6,8]、holdout=null，逐步日志与采样序列已增长。当前未完成训练/正式评分，不宣称成功。

## 实际闭环结果

状态：ANALYZED，训练、七环境预测/独立 truth-last 评分、同物理样本分析均完成。只运行1个seed，不声明跨seed复现。

| 场景 | 普通拼接式 E30 | N0 先验 E30 | 差值（百分点） |
|---|---:|---:|---:|
| clean | 82.963690% | 82.867857% | -0.095833 |
| practical_high | 81.620238% | 81.695238% | +0.075000 |
| practical_mid | 79.329762% | 79.389286% | +0.059524 |
| practical_low_suburban | 76.977381% | 77.014881% | +0.037500 |
| practical_high_urban | 78.189286% | 78.288690% | +0.099405 |
| practical_mid_urban | 64.371429% | 64.438690% | +0.067262 |
| practical_low_urban | 53.447024% | 53.475000% | +0.027976 |

固定判据：{'low_urban_gain_at_least_0_5pp': False, 'clean_not_lower': False, 'mid_urban_not_lower': True, 'each_suburban_no_more_than_0_25pp_lower': True}；科学结论：CLOSED_FAILED_PREREGISTERED_IDENTITY_BENEFIT。未改倍率、概率、门槛、父权重、seed或预算。前一冻结输入失败保留；本次首次训练证据不能与它混成源筛查通过。

low urban 修复 934 个、产生 887 个新错误，净变化 47 个；clean且low共同正确数量变化为 -0.024405pp。全部7view的修复/新增错误、宏F1、最差RX见CSV，不能用锁定更多或辅助loss代替识别成绩。

各suburban距模型自身clean：{'practical_high': 1.172619047619039, 'practical_mid': 3.4785714285714198, 'practical_low_suburban': 5.8529761904761894}pp。相对同预算无新增增强B00 clean净收益 -2.400595pp；联合目标核对={'clean_positive_vs_matched_B00': False, 'low_urban_above_60': False, 'each_suburban_gap_within_exploratory2pp': False, 'Ag_tested': False, 'joint_factorial_tested': False, 'model_seeds': 1, 'goal_complete': False}。这里仅As配方一行，无新Ag或四因子证据，不计算不存在的交互项。

实际新增2220/累计6660更新；L前向1136640，As观测284160，学生训练/验证前向1703640/324002，教师训练/验证前向0/567000。新增训练墙钟1320.279301秒，其中channel 555.668759秒（包含关系，不重复相加）。父20与新增10分开，原nominal控制复用不重复训练；相同更新/NN预算不等于相同墙钟或物理处理FLOPs。

实际倍率计数 .5=142311、1.5=141849；期望线性N0=0.001，实现均值=0.000999187077699、倍率均值=0.999187077699。未声称经验均值精确等于期望或平均SNR/dB保持。284160次完整residual、0额外传播预运行、NoEQ；15个session硬件/校准保持；原quality依赖随改变N0重算。

完整日志复算：同nominal的4440条L/U采样记录完全相同，68项实际共有参数一致，2220个监督目标最大归一化误差1.0728836064e-07；252个混淆矩阵、42组分区求和与7组配对计数独立核验。无真实IQ/CP/NN额外分析读取；只复用固定scores与源日志。继承控制曾有清理阶段技术缺项，缺失的瞬时前向计数仍标缺，不补造。

正式预测NN包数1176002、wall=42.290075秒，独立CPU评分wall=3.283799秒，单独记录，不纳入训练更新。

原始 result.json 中 MMSE_mix 说明文本系复制的旧报告字段，描述错误；它不参与训练、配置、预测或评分。真实 channel/noise_policy/计数记录N0变体，本报告明确更正该描述，保留原产物，不热改健康任务。

图：训练曲线包含共享父20轮与新增10轮所有2220步的逐epoch均值；七环境/差值/各suburban-gap图仅引用已闭环数值。失败不扩种或改参救跑。全局Ag/As联合目标未完成。

闭环分析入口首尝试因独立脚本导入 experiments 路径失败，移除已有内置函数的重复导入后完成汇总；没有重训、重复NN、修改prediction或重开truth。图PNG/PDF已实际生成、下载并查看；完整source曲线与逐view配对数字见同目录CSV。
