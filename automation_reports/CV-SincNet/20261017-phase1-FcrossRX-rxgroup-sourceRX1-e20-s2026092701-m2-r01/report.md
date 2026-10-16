# 原普通增强按物理 RX 分组：最小源域对照

状态：ANALYZED。执行 VERIFIED；科学 FAIL：源域 clean 增益＋0.1587 pp，低于预登记＋1 pp。两组 E20 和七视图评价／独立复算／图表均已完成。下方设计与预算保留发布前预登记，实际结果见后半部。

起点提交 f68a9f9642a7975a34792beafc7b278c0251d238。实际代码／配置身份为本次发布的 release_commit.txt；后续分析提交另记。

## 已有证据和唯一假设

固定基座 F＋跨 RX 同 TX SupCon（SIG_TIME_X3、统一 Sinc CENTER_UNITS、NoExplicitCFO、joint160、默认 cosine scale30），网络及头按原流程联合训练。此前 ordinary As 可改善 urban，但跨 RX clean 代价不稳定；coherentmix、timegrid、跨 RX donor、固定课程、LA16 等既定候选未通过各自预登记条件。最新窄带谱线诊断的 RX/bin 跨折稳定性仅 14/18，低于 15/18；不能据此制造 RX 去谱线增强。上述候选保持关闭，不用改名重试。

原普通增强部分触发开关以及 taper／多径核由整批共享。当前只检验：让物理 RX 之间独立抽取同一套原有条件，是否改善跨 RX SupCon 的 clean 泛化。既有单包算子、概率、范围、顺序、边界及归一化保持，实际事件数不要求相等。此假设没有识别收益保证。

## 固定矩阵与预算

| row | 唯一区别 | 预算 |
|---|---|---|
| BATCH-RX1-s2026092701 | 原整批共享条件；分组结果只作 shadow | scratch E20／4440 updates |
| RXGROUP-RX1-s2026092701 | anti-shortcut／general-channel 按实际 RX 分组；整批结果只作 shadow | scratch E20／4440 updates |

两组均为原 .25 RD(ordinary)＋.75 RD(raw) 六次 L 前向，原两次 ordinary 前向复用同一个已生成 IQ，SupCon 只复用这两次；四次 raw 前向、原损失系数、AdamW／cosine、13L＋7U／旧门控／独立 EMA 不改。总计 40 model-epochs／8880 updates，绝非单模型 E40。每组 L NN 3,409,920；原 U／教师／V 及 FINAL 8820 留出样本前向按实际完整日志计数，预计每组 4,686,678 NN packets；真实 checkpoint 兼容 smoke 额外 2 个合成 NN。新增 stage shadow 只产生 IQ，不增加网络前向。

normal PA／DAC 原调用保持在分组阶段之外，class signature 关闭；不添加新的 PA／DAC，不声称反演或替换 TX／RX 硬件。只有显式真实 L 上下文的 normal view 分组；U 包括非负伪标签和所有 defect 路径均直通原实现。private RNG 按 step／stage／物理 RX 刷新，seed20261017831，不使用 TX／day／质量／损失，恢复原 CPU／Python／CUDA 随机流。

## 数据、模型与评价边界

复用原 VALIDATED_ONCE source_contract／split392005，无重划分或新增标签。fit 为 RX3/4/6/8：L5040、U45360（TX 真值隐藏）、V21600。source RX1 L1260 仅固定 FINAL20 评价；原 carrier 加载和 metadata 校验保留，原角色不改变。两个学生／EMA／优化器均从零，无父权重；已核实旧 B00 同有效契约 E20 默认头只严格加载作 2 个合成输入兼容 smoke，随后丢弃，绝不初始化或教师。

IQ2×256／25 MHz／原 RMS 位置不改。FINAL 默认头自动预测 clean 和六种原 practical：high、mid、low_suburban、high_urban、mid_urban、low_urban。保持原 residual／post_sync／EQ off（G=1）、reference_snr_db30 和 reflect 边界、传播／接收处理与 eval 随机流；不弱化测试。这里只是合法源域留出读数，不接触正式 target、query 或 truth，不能代替正式 truth-last 测试。

每 row 固定20 epoch 后冻结，逐 row 自动保存七视图预测、margin/30、完整物理状态与物理 ID。七视图完成后独立离线重算 accuracy／macro F1／状态分层／固定 BATCH clean 正确集合的卫星保持率、repair／break、训练曲线、门控和覆盖。源成绩不改变另一健康 row 或预算。

## 预先固定的判断与停止规则

RXGROUP 相对本次新 BATCH：held clean 至少＋1 pp，clean macro F1 不下降，seen-V 下降不超过 0.5 pp，最终 L CE≤0.5；均需20 epoch／4440 updates／3,409,920 L NN／SupCon3330 steps／4440 coupling calls／8880 stages／568320 L view packets／完整72 TX×RX×day cells，finite、合法、无新增 U／As／head fit／目标访问。全部六种 practical 必须报告，但不作为本次独立 clean 候选晋级门槛。

FAIL 关闭此固定 RX 分组候选，不改变算子子集、分组、RX、seed、阈值或预算救援。PASS 也只允许另行预登记第二 seed 源确认，不构成已验证 Ag、联合突破或 low urban>60%。

系统技术失败仅限本 run 的角色／来源／非有限值／错误预算／不可启动／输出碰撞；保留局部产物，无自动重试，不停止或热改任何健康旧任务。性能差属于科学负结果。

## 执行和最小验证

环境：普通 N607 CVS-RFFI Python，原 full FP32；CWD /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_rxgroup_sourceRX1_E20_20261017_r01。单一 owner codex/root/FcrossRX-rxgroup-20261017；容量锁按全部 owner 每 GPU≤3、可用≥12000 MiB，CPU threads2；不使用管理员账号。

命令：/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_rxgroup_sourceRX1_E20_20261017_r01/experiments/cvs_f_rxgroup20/runner.py

run root：/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-rxgroup-sourceRX1-e20-s2026092701-m2-r01

日志：/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-rxgroup-sourceRX1-e20-s2026092701-m2-r01/logs/<row>.stdout.log

实际模块路径、native args／resolved config／augmentation scalar 与 channel 配置由 run 保存。源码依赖固定 backend 7f398321cd2ffd2c2d2cb11f52f67bfd8c44987b；新增代码位于 experiments/cvs_f_rxgroup20。

本地合成验证：20 epoch native parity、12 项协议负例、U 非负伪标签直通、TX 标签置换不影响生成 IQ、原 CPU／Python 流及 PA 流保持均 PASS（8448 合成 IQ call packets，0 NN／真实 IQ）。真实 CVS 合成训练检查 PASS：206 state 相同；BATCH 与原完整 L loss／全部梯度／RNG 逐位相同；RXGROUP 使68个参数梯度变化；六次 L 前向／默认头联合更新／export160 确认；10 项模型负例；2322 合成 NN／3 合成 optimizer updates。梯度变化不能替代识别提升。

一次独立限定 P0/P1 静态 review PASS，无直接发布阻断；未读真实 IQ 或 checkpoint。一次 N607 只读 preflight VERIFIED。发布只使用一次 archive SHA 本地／远端比较及一次远端 compile，smoke PASS 立即继续训练，不增加签名／receipt／额外审批门槛。

预期产物：完整20 epoch及4440 step logs、L/U物理序列、coupling trace及72cell覆盖、原门控及教师计数、七视图预测与状态、独立逐 row 指标及全曲线／分层图、真实 NN／墙钟时间、失败项和最终提交。此处为发布前记录；实际成绩见后半部。

发布状态：VERIFIED，code commit 82975411f152e2be7d80ebc79b7caa850d2ee858；真实 checkpoint 合成2 NN smoke PASS，dispatcher98932／BATCH99020 GPU0／RXGROUP99095 GPU1，独立 PID／CWD／argv／GPU／首 epoch 日志增长核对通过。两组健康运行，尚无最终成绩。


---

# F＋跨 RX：原普通增强的 RX 分组随机条件

两组均完成 E20 及七视图源域读数；固定 RX 分组候选未通过预登记条件，保留负结果并关闭。

训练代码 82975411f152e2be7d80ebc79b7caa850d2ee858；离线复算实现 experiments/cvs_f_rxgroup20/analyze.py 随结果提交。结果均为 source held RX1 L1260／seed2026092701／FINAL20／原默认 cosine 分类头；不能作为正式 target、跨 seed 或联合目标完成的证据。

## 逐 row、逐 seed 结果

| view | BATCH accuracy % | RXGROUP accuracy % | 差值 pp | BATCH macro F1 % | RXGROUP macro F1 % |
|---|---:|---:|---:|---:|---:|
| clean | 81.6667 | 81.8254 | 0.1587 | 81.3306 | 81.6280 |
| practical_high | 38.3333 | 41.5873 | 3.2540 | 28.2533 | 31.5110 |
| practical_mid | 38.5714 | 40.8730 | 2.3016 | 29.3862 | 32.0933 |
| practical_low_suburban | 35.3968 | 36.6667 | 1.2698 | 28.0449 | 29.1865 |
| practical_high_urban | 38.4921 | 40.2381 | 1.7460 | 29.6476 | 31.2468 |
| practical_mid_urban | 31.5873 | 32.3016 | 0.7143 | 26.7252 | 27.3718 |
| practical_low_urban | 28.3333 | 28.8889 | 0.5556 | 24.8072 | 25.2339 |

urban 三场景均值差 1.0053 pp；clean macro F1 差 0.2974 pp；已见 RX V 差 0.0417 pp。

| row | clean % | urban mean % | low urban % | 各 suburban 与自身 clean 的差 pp（high/mid/low） |
|---|---:|---:|---:|---|
| BATCH | 81.6667 | 32.8042 | 28.3333 | 43.3333, 43.0952, 46.2698 |
| RXGROUP | 81.8254 | 33.8095 | 28.8889 | 40.2381, 40.9524, 45.1587 |

## 固定判断与收益归因

固定阈值：clean≥＋1 pp，clean macro F1≥0 pp，seen V≥−0.5 pp，两组 final L CE≤0.5，完整 E20/4440 updates/六次 L 前向/3330 SupCon steps/合法数据和完整覆盖。六种 practical 全部报告，本次独立 clean 资格不以 practical 升降为门槛。结果：{"clean": false, "macroF1": true, "seenV": true, "healthy": true}。

FAIL 关闭这组固定 RX 分组条件；不改算子子集、分组、RX、seed、阈值或预算救援。

两组均从零，网络／参数量／默认头／原 SupCon／六次 L 前向／raw 监督权重／U 规则匹配。唯一训练差别是原 anti-shortcut 和 general-channel 条件在不同物理 RX 之间的相关性，含原整批 taper、多径核及触发条件；PA／DAC 保持原调用。不能将任何差值单独归因于某一个 gate 或某一种物理损伤。

| low urban 分层 | n | BATCH % | RXGROUP % | 差 pp | BATCH margin/30 | RXGROUP margin/30 |
|---|---:|---:|---:|---:|---:|---:|
| state:LOS | 384 | 41.6667 | 44.2708 | 2.6042 | -0.0286 | -0.0268 |
| state:BLOCKED | 714 | 19.3277 | 19.4678 | 0.1401 | -0.0603 | -0.0599 |
| state:PARTIAL | 162 | 36.4198 | 33.9506 | -2.4691 | -0.0320 | -0.0322 |
| lock:locked | 467 | 41.7559 | 44.1113 | 2.3555 | -0.0270 | -0.0253 |
| lock:degraded | 228 | 24.5614 | 21.0526 | -3.5088 | -0.0571 | -0.0581 |
| lock:unlocked | 565 | 18.7611 | 19.4690 | 0.7080 | -0.0595 | -0.0588 |
| quality:[20,30) | 396 | 42.4242 | 44.9495 | 2.5253 | -0.0275 | -0.0258 |
| quality:[5,10) | 157 | 22.2930 | 19.7452 | -2.5478 | -0.0636 | -0.0642 |
| quality:[0,5) | 385 | 18.9610 | 20.7792 | 1.8182 | -0.0582 | -0.0567 |
| quality:<0 | 180 | 18.3333 | 16.6667 | -1.6667 | -0.0622 | -0.0631 |
| quality:[10,20) | 142 | 33.8028 | 31.6901 | -2.1127 | -0.0334 | -0.0335 |

上述按固定同批 IQ／同信道状态分层，是表现定位，不是去除噪声／多径后的受控因果消融。本轮不训练 As，不能证明 practical 适应机制改善或与 As 存在正交互。added-noise SNR 不是总 SNR，quality 为既有白输入功率代理；保留原 boundary、数值 FIR 延迟和 RX 补偿。

## 固定 BATCH clean 正确集合与错误变化

固定集合 n=1029；所有场景复用这一集合，不按新模型重新挑选。

| view | BATCH 保持率 % | RXGROUP 保持率 % | 修复 | 损坏 | 净正确样本数 |
|---|---:|---:|---:|---:|---:|
| clean | 100.0000 | 98.9310 | 13 | 11 | 2 |
| practical_high | 36.6375 | 39.8445 | 49 | 8 | 41 |
| practical_mid | 37.4150 | 39.4558 | 42 | 13 | 29 |
| practical_low_suburban | 34.5967 | 36.1516 | 35 | 19 | 16 |
| practical_high_urban | 37.4150 | 39.2614 | 38 | 16 | 22 |
| practical_mid_urban | 30.1263 | 31.0982 | 29 | 20 | 9 |
| practical_low_urban | 28.4742 | 28.9602 | 37 | 30 | 7 |

## 完整覆盖、训练健康和成本

| row | updates | L NN | 总 NN packets | stage shadow 时间 s | 原训练墙钟 s | 到最终结果墙钟 s |
|---|---:|---:|---:|---:|---:|---:|
| BATCH | 4440 | 3409920 | 4686678 | 25.8575 | 1241.2048 | 1258.8211 |
| RXGROUP | 4440 | 3409920 | 4686678 | 26.2825 | 1246.5982 | 1264.2291 |

新训练总计 40 model-epochs／8880 updates；9373358 NN packets（包括 dispatcher 兼容 smoke2）。每组学生／EMA／optimizer 均从零，父训练成本0；旧 B00 仅 smoke 载入后丢弃，不继承。合成验证另计2322 NN／3 updates／8448 IQ call packets，离线评分及出图0 NN。NN 计数扣除了 EMA 复制时继承的2个评估计数，保留实际学生初始化 smoke；墙钟已由进程 start ticks 与产物 mtime 独立核对，近似精度1 s；E20 耗尽不能称为收敛。

两组各有4440 coupling calls、8880 stages、568320普通L视图、5040唯一L样本、完整72个TX×RX×day cell，完整4440训练步和20轮已检查；原U真值隐藏且未分组，默认头联合训练、无额外头。L/U 完整5994条序列相同：True（L=True，U=True）。各批 operator 实际触发数量可不同，不能把保持边际分布误写成实际计数完全相等。

| operator | BATCH 触发 L packets % | RXGROUP 触发 L packets % | RXGROUP 正样本对条件不同 % |
|---|---:|---:|---:|
| p_dc_offset | 6.3063 | 5.7278 | 10.6922 |
| p_bandedge_taper | 4.6396 | 4.7957 | 9.1110 |
| p_time_shift | 6.3964 | 6.8018 | 12.6804 |
| p_amp_scale | 8.9414 | 8.2366 | 14.9187 |
| p_phase_rot | 9.0991 | 8.6912 | 15.8239 |
| p_cfo | 6.7568 | 6.7905 | 12.5398 |
| p_phase_noise | 5.4955 | 5.7842 | 10.7859 |
| p_multipath | 2.8378 | 3.5755 | 6.8694 |
| p_awgn | 7.6351 | 7.3837 | 13.5663 |

## 图表与证据

图表均由完整已冻结源预测／完整日志生成。

- [source_scene_results PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_rxgroup_20261017/final/figures_v2/source_scene_results.png)；[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_rxgroup_20261017/final/figures_v2/source_scene_results.pdf)
- [complete_training_curves PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_rxgroup_20261017/final/figures_v2/complete_training_curves.png)；[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_rxgroup_20261017/final/figures_v2/complete_training_curves.pdf)
- [ordinary_coupling_exposure PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_rxgroup_20261017/final/figures_v2/ordinary_coupling_exposure.png)；[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_rxgroup_20261017/final/figures_v2/ordinary_coupling_exposure.pdf)
- [low_urban_strata PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_rxgroup_20261017/final/figures_v2/low_urban_strata.png)；[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_rxgroup_20261017/final/figures_v2/low_urban_strata.pdf)
- [fixed_clean_retention_and_errors PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_rxgroup_20261017/final/figures_v2/fixed_clean_retention_and_errors.png)；[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_rxgroup_20261017/final/figures_v2/fixed_clean_retention_and_errors.pdf)

独立结果：[analysis_summary.json](E:/type10-7/local_artifacts/cvs_FcrossRX_rxgroup_20261017/final/analysis_summary.json)；完整分层和曲线同目录。原始 IQ／checkpoint 留在不可变 N607 run root，未提交或拉取。执行命令／环境／模块文件路径／数据角色／停止规则见同 run 预登记。

## 最终问题的回答与未完成项

当前只有一个源 RX／一个模型 seed。正式目标域默认头提升、clean 代价、最差目标 RX、low urban>60% 和跨 seed 重现均未证明；尚无已验证新 Ag、B00/B10/B01/B11 因子矩阵或 Ag×As 正交互。本轮 source practical 的状态表现变化如上表，不能声称固定正式测试下某类退化已获缓解。

已见源 RX 的 V 最差项均为 RX3：95.4259%→95.5556%（＋0.1296 pp），与正式目标域最差 RX 不同。历史 B00 兼容 smoke 来源为 scratch E20／4440 updates，未继承其模型或教师状态，其历史成本不计入新训练预算。

状态归因：执行与训练健康均 VERIFIED；科学候选 FAIL，固定 RX 分组路线关闭。跨 seed／正式目标域／Ag×As 因子矩阵仍未完成，活动研究目标保持。


## 实际执行的验证与分析命令

以下本地命令 CWD 均为 E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014；模型合成检查 wrapper 自行设置已核实的原生模块路径。

- E:/type10-7/local_envs/ssr-gpu/python.exe -m experiments.cvs_f_rxgroup20.checks_runtime --native E:/type10-7/code/worktrees/cvs_iq_timing_20261016/experiments/adv3b02_xuc/code/DataAugmentation.py --schedule E:/type10-7/code/worktrees/cvs_iq_timing_20261016/experiments/adv3b02_xuc/code/cvsrffi/schedule.py --actual E:/type10-7/local_artifacts/cvs_FcrossRX_timegrid20_20261017/final/TIMEGRID-RX1-s2026092701/source/ordinary_augmentation_actual.json
- E:/type10-7/local_envs/ssr-gpu/python.exe E:/type10-7/local_artifacts/cvs_FcrossRX_rxgroup_20261017/run_model_checks.py
- F:/App/miniconda3/python.exe -m experiments.cvs_f_rxgroup20.checks_analysis
- F:/App/miniconda3/python.exe -m experiments.cvs_f_rxgroup20.harvest --output E:/type10-7/local_artifacts/cvs_FcrossRX_rxgroup_20261017/final
- F:/App/miniconda3/python.exe -m experiments.cvs_f_rxgroup20.analyze --input E:/type10-7/local_artifacts/cvs_FcrossRX_rxgroup_20261017/final
- F:/App/miniconda3/python.exe E:/type10-7/local_artifacts/cvs_FcrossRX_rxgroup_20261017/plot_local_v2.py

runtime 12＋模型10＋离线评分10项反例检查均 PASS；真实 checkpoint 兼容 smoke2 PASS；独立完整记录复算 exit0。图表初版的 U 接受量图注曾误写每步，已用全部 step 日志求和确认每轮总数并生成 figures_v2，初版保留、不作最终交付图。更正未改变任何训练、预测、评分或阈值。
