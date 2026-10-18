# P68 最终分析：固定六次前向中的独立完整 practical 观测

状态：ANALYZED；本轮全部七环境完成，missing=[]。固定 INDEP_CE6 源域晋级失败，不延长 E50；整体研发目标仍 ACTIVE。所有结果仅模型种子 2026092701，数值单位为准确率百分比或绝对百分点（pp）。

## 结论及本次唯一假设

在同父、同预算、同物理样本顺序的 SAME_CE6 控制上，把第二次 As 前向从重复同一信道实现换成独立完整 practical 实现，urban 均值仅提高 0.114286 pp，mid urban 提高 0.087500 pp，low urban 提高 0.075595 pp，clean 提高 0.019643 pp。主比较属于单种子微增，不能宣称可重复改善。

相对同种子、同总 E30、相同数据角色的无星地增强整配方 NOAS30，urban 均值提高 28.302778 pp，clean 下降 0.013690 pp，满足阶段性 clean 下降不超过 0.1 pp。相对原完整 As30，urban 均值下降 0.065675 pp，clean 下降 0.107143 pp；low urban 只净多对 3 个样本。原 As30 前 20 轮保留 satellite KL，而本轮和 SAME 使用 AS_CE20，因此该比较包含父训练历史差异。NOAS30 是整配方预算对照，不是与 INDEP 共用同一 As 父的局部机制消融。

最值得保留的仍是完整 practical 真实标签 CE、稳定 clean 监督份额和既有原头联合训练。当前矛盾是：完整 As 能显著提高受损 IQ 识别，但增加观测独立性并未在源域稳定解决 low urban 的信息缺失；更丰富的扰动本身不保证 clean 与星地共同提升。正式 low urban 为 51.245833%，距 60% 尚差 8.754167 pp。

## 审计、实现和可归因范围

实际训练代码 commit：f520ae98791a80775b6b5923183fcffba7e91c73；新增实际父集合诊断代码 commit：54fe00dce90ae24fa80ce0ed427348e18a825051。父集合读出已由 c46cc1c6b44321f07c6d4d1b7464fdc7735d1842 提交并独立核实远端。起点和既有流程保留在 report.md／experiment.json。

固定 F＋CROSS_RX、SIG_TIME_X3／统一 Sinc／CENTER_UNITS／joint160／默认 cosine30，900722 参数、206 个状态键。网络、内部门控、参考响应和分类头结构不变，正常联合训练；没有新增头、部署模块或诊断分类器。完整 L6300／隐藏 TX 的 U56700／V27000，RX1/3/4/6/8、day1/2/3、六 TX、split392005，未留出源 RX。沿用已验证合法数据句柄；未访问 U 真值或用目标成绩选模。

INDEP 与 SAME 都严格加载 SAT18 的 AS_CE20 student＋ownEMA（epoch20、4440 更新、source-only scratch／final_only），重置 AdamW 并运行相同原 cosine10 尾段；总 E30、6660 更新。P43 原 As20 不能替代该父。实际 native args 相对 SAME 仅 candidate_id、run_id、output_dir 不同。

主改动位于 experiments/cvs_full_channel_pair68/channel.py、observer.py、source.py；design.py／protocol.py 固定方案和数据边界，runner.py／evaluate.py／qualification.py 串联训练、源冻结、七视图预测和独立评分；其余 adapter 与 checks.py 保留原运行契约。跨 IQ 的 KL 不参与损失；原 ordinary/raw KL、dropout、ordinary128 的 SupCon、U 原 batch_neighbor 门控与 ownEMA 保持。不是 P67 epoch_bank。

两个 satellite CE 系数分别 0.125，总 As 权重 0.25；混合组 0.5、总 L 尺度 1.25。每步仍为 6×128＝768 个 L 前向样本。全部有标签卫星样本参与 CE，不先经过学生置信度筛选。第一观测保留 FcrossRX/train/e{epoch}/step{step}/L_S 命名流；第二观测追加 /view1-full-independent，从相同 raw L IQ 重新生成完整信道。两个实现共享实际 RX/day/eq 硬件 session 对象；session 不按 TX 定义，各自重新计算几何、遮挡、传播、噪声、接收质量、状态与补偿残差，不把两条信道串联。

训练物理流种子 20261016421、RX 硬件种子 20261016423，ordinary/dropout 沿原模型随机流；训练、源诊断和固定正式测试流分离。实际导入 leo_practical 的 channel.py／residual.py 位于 PROJECT/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code/leo_practical/，不是顶层 code/ 实现；完整 __file__ 与 resolved config 见 runtime_configuration.json。

采样率 25 MHz，IQ 为 2×256；逐包 RMS、center=false 在 practical 前，CENTER_UNITS 在原网络内。训练中心频率 2.462 GHz，full practical residual／post_sync；EQ=false，所以 G=1，不能称为已均衡弱残余。质量适配、相位跟踪、IQ 补偿、AGC 开启；reference_snr_db=30、接收质量参考 20、reflect 边界、AGC ±30 dB。正式固定 IQ／2.45 GHz／全部六 practical 完全不变，没有提高测试 SNR、删除难样本或减弱信道。新增噪声比不是总 SNR；状态仍按原接收机代理计算。

历史 MC2 已训练独立完整信道，但采用 10L、heldRX1 和每实现配对 RDrop，源结果为负；本次关键改变是原 6L 中的观测分配与 fullsource，不能称为首次独立信道算法。NoisePair、HQ、SAT192、另一对话 LRaw／tailNoU 已核查，不重复启动。

## 源域决定：先关闭方案，再读取正式指标

预登记只有两项晋级条件：full V clean 相对 NOAS30 ≥−0.1 pp；V1800、两个信道 draw 的 pooled mid/low urban 相对 max(AS30,SAME) ≥＋0.5 pp。两个 draw 是重复信道观测，不是两个模型种子。

实测 full V clean 为 98.329630%，相对 NOAS30 −0.029630 pp，通过；pooled mid/low 为 66.263889%，相对 SAME ＋0.069444 pp，未达 ＋0.5 pp，失败。source low 相对 SAME −0.027778 pp，两 draw 分别 −0.111111／＋0.055556 pp，方向不一致。源决定文件先于新行 prediction/scoring 完成写出；主 owner 先保存 source_closure 再读正式分数。固定 SOURCE_FAIL 不因正式微增改判，未启动 E50、扩种子或改 namespace／剂量救跑。

源分层中，mid 的 LOS／PARTIAL／BLOCKED 主差分别 ＋0.052659／＋0.544959／＋0.224888 pp；low 分别 0／−0.178253／0 pp。mid locked／degraded／unlocked 为 ＋0.136054／−0.139860／＋0.588235 pp；low 为 −0.073153／−0.449775／＋0.191571 pp。low added-noise <3、3–8、8–15、≥15 dB 分别 ＋0.266193／−0.251889／−0.330033／−0.072674 pp。小幅正负交错，不足以断言噪声、多径、残余 CFO 或特定锁定状态已稳定改善。完整质量、状态、锁定和噪声区间计数保留在 source_strata.csv／source_paired_effects.csv；正式缓存没有这些状态标签，不能把源分层冒充正式分层。

## 正式七环境逐行结果

每个逻辑条件、每个环境 168000 样本；三个历史 E30 直接复用其已固定预测／独立评分，未重预测、重评分或重训。全部 504 个混淆矩阵与 84 个 RX/day 分区重新独立核对；四条件七视图均符合 prediction 完成后接 truth 的顺序，missing=[]。

| 环境 | NOAS30 | 原 AS30 | SAME_CE6 | INDEP_CE6 |
|---|---:|---:|---:|---:|
| clean | 85.247619 | 85.341071 | 85.214286 | 85.233929 |
| practical_high | 42.685119 | 81.405357 | 81.019643 | 81.167857 |
| practical_mid | 40.995238 | 78.661310 | 78.345238 | 78.512500 |
| practical_low_suburban | 39.472619 | 75.861905 | 75.639286 | 75.767262 |
| practical_high_urban | 41.439881 | 77.767262 | 77.452976 | 77.632738 |
| practical_mid_urban | 35.132738 | 62.806548 | 62.654762 | 62.742262 |
| practical_low_urban | 30.139881 | 51.244048 | 51.170238 | 51.245833 |
| urban 三场景均值 | 35.570833 | 63.939286 | 63.759325 | 63.873611 |

下表报告新行全部宏 F1、最差 RX 和同父主比较。最差 RX 是各场景分别计算，不能把均值称作同一个物理 RX。

| 环境 | 宏平均 F1（%） | 最差 RX（%） | RX ID | 相对 SAME Δ准确率（pp） | 修复／新增错误 |
|---|---:|---:|---|---:|---:|
| clean | 84.962832 | 78.858333 | 14-7 | +0.019643 | 306／273 |
| practical_high | 80.905497 | 73.100000 | 14-7 | +0.148214 | 755／506 |
| practical_mid | 78.270483 | 71.470833 | 14-7 | +0.167262 | 846／565 |
| practical_low_suburban | 75.523351 | 69.025000 | 14-7 | +0.127976 | 856／641 |
| practical_high_urban | 77.369666 | 70.370833 | 14-7 | +0.179762 | 840／538 |
| practical_mid_urban | 62.428159 | 57.350000 | 14-7 | +0.087500 | 1028／881 |
| practical_low_urban | 50.851475 | 47.100000 | 14-7 | +0.075595 | 1166／1039 |

主比较 mid 在 5/7 个 RX、4/4 个日期正向，low 在 6/7 个 RX、4/4 个日期正向；这些是同一个模型的分层观察，不是跨 seed 复现。clean 修复 306／新增 273，mid 1028／881，low 1166／1039。相对原 As30，low 只多 3/168000 个正确预测，不能解释为突破。完整四行逐环境 F1／最差 RX、全部三组配对变化及 RX/day 错误见聚合文件和 CSV。n=1，不估计跨 seed 标准差、不挑最好种子。

## 真正 AS_CE20 父模型 clean 正确集合

旧 P43 As20 的预测不能替代实际父。本轮补了独立 source-only 默认头诊断：原 V1800 clean＋同一两个 draw 三 practical，12602 NN（含 2 NN checkpoint smoke）、0 更新／0 教师／0 新信道生成／0 target。父 clean 正确 1767/1800，satellite 分母为 1767×2＝3534。随后仅 CPU 对齐同一有序物理 ID／TX／RX／day，以下结果未用于晋级。

| 行 | clean 保持 | low suburban 保持 | mid urban 保持 | low urban 保持 |
|---|---:|---:|---:|---:|
| PARENT_ASCE20 | 1767/1767（100.000000%） | 3171/3534（89.728353%） | 2584/3534（73.118280%） | 2073/3534（58.658744%） |
| NOAS30 | 1763/1767（99.773628%） | 1769/3534（50.056593%） | 1548/3534（43.803056%） | 1270/3534（35.936616%） |
| AS30 | 1765/1767（99.886814%） | 3179/3534（89.954726%） | 2611/3534（73.882286%） | 2111/3534（59.734012%） |
| SAME_CE6 | 1765/1767（99.886814%） | 3194/3534（90.379174%） | 2616/3534（74.023769%） | 2122/3534（60.045274%） |
| INDEP_CE6 | 1765/1767（99.886814%） | 3204/3534（90.662139%） | 2619/3534（74.108659%） | 2121/3534（60.016978%） |

新行相对 SAME 的该集合 mid 只净多 3/3534、low 净少 1/3534，clean 相同。这里 low 保持率 60.016978% 是源域固定子集指标，绝不是正式 low urban 准确率，更不能拿它宣布达到 60% 目标。默认 cosine scale 30，报告 logit margin/30；该集合新行 low 间隔 0.064624809，SAME 为 0.064706266，未显示间隔改善。

本轮没有对全部物理损伤做固定接收状态／重算状态的反事实分解，也没有新增全部分支／门控的梯度因果诊断；head＋Sinc 联合反传 smoke 只能验证训练可用。现有证据支持完整 As 缓解固定复合 practical 失配，不支持把新增独立观测的微增归因于某一种物理损伤。

## 训练覆盖、曲线与实际成本

尾段 2220 更新，每视图生成 142080 个完整 As，两视图合计 284160。view0、view1 各覆盖 90/90 个 TX×RX×scene 单元；合计 LOS／PARTIAL／BLOCKED＝143606／45475／95079，locked／degraded／unlocked＝171947／47744／64469。实际 15 个稳定硬件 session；2220 步两观测硬件对象相同。531/810 个 TX×RX×scene×state×lock 联合格有观察，其余受物理依赖约束，不能把未出现的物理组合强行独立采样。

4440 条 L/U 序列记录核对表明主两行物理样本顺序相同；每步完整物理记录保存为原始 gzip。As IQ pair MSE 均值 0.575578，只说明观测不同，不单独证明统计独立；完整条件随机流与共享硬件的代码路径已核实。两次 CE 均值 0.744626／0.743820，L 分解最大误差 1.416e−7。As-only KL 仅记录诊断，不回传；ordinary 半组 KL 均值 0.033599 保留。

原 U 观察 567000，接受 8822，SAME 为 8813；接受率不是伪标签精度，U TX 仍隐藏。完整可用曲线涵盖父 E1–20 与新增 E21–30，不从最后几行推断收敛；预算耗尽不等于收敛。

| 成本项 | 实测 |
|---|---:|
| 复用自身 AS_CE20 父 | E20／4440 更新／4797722 native NN／1827.538 s |
| 本次新增尾段 | E10／2220 更新／3162962 native NN／1362.876 s |
| 新尾段学生训练／学生 eval／教师 eval | 2271960／324002／567000 NN |
| 新行源诊断／正式七环境／共享真实父 smoke | 12600／1176002／772 NN |
| P68 主流程新增总 NN | 4352336 |
| 实际父 source-only 诊断 | 12602 NN／0 更新／2.686 s |
| 本轮含实际父读出的新增总 NN | 4364938 |
| 本地实际 CVS NN／分析 NN | 0／0 |
| 源评估／正式评估／CPU 分析 | 1.826／47.313／43.090 s |
| 主流程提交至分析完成墙钟 | 1471.654 s（约 24.528 min） |

新学生累计 E30／6660 更新。三个历史 E30 合计 90 model-epochs／19980 更新均复用；将共享 AS_CE20 只计一次，四条件实际已发生训练合计 100 model-epochs／22200 更新。NN 是模型处理的样本数，不是调用次数。物理模拟翻倍但 NN 没增加，不是零成本：新尾段比 SAME 多 293.892 s（27.493%），这也包含记录与资源条件差异，不能全归因于模拟算法。父诊断与主流程有重叠，墙钟不能相加。原物理记录约 116.64 MB，全部检回 86 文件约 136.51 MB；权重／真实 IQ／日志不进入 Git。

## 执行核实、图表与交付

本地 32 个协议负例及纯张量梯度／路由／硬件／命名流／异常恢复检查 PASS；分析 8 项加字段对齐后 1 项定向检查 PASS；18 个 Python 文件 compile PASS。首次本地依赖路径失败发生在执行检查前，修正到已验证路径后通过，未触发训练补跑。唯一 P0/P1 review PASS。远端一次归档比较 SHA ca5482c667f9e39ccd9626c1b52536f8cb996184b6345295120f09ea7409e8b1、17 Python compile 和 772 NN 真实父＋合成 6L smoke PASS。一次普通账户 N607 preflight、GPU0 所属 PID/CWD/argv/GPU/log 增长已核实；完成后主 owner／训练／父诊断所属 PID 均消失且无 failure，主流程分析终态 ANALYZED。

父 source-only 诊断自身 7 项纯 toy／4 文件 compile 与唯一独立 P0/P1 PASS；归档、加载来源、独立预测重数和终态核实。该 2.686 s 短任务在第一次实时 /proc 观察前结束，因此没有独立的运行中 PID/CWD/GPU 快照；准确披露该缺项，不为快照重跑。其终态和 12602 NN 输出 VERIFIED，详见独立父诊断报告。

训练曲线、源质量分层、正式配对与物理覆盖 4 张 PNG 已逐图查看，可读且非空；对应 PDF 均存在。正式图中 NoAs 的大效应会压缩新增机制的小变化，精确表格优先。

- [完整训练曲线](E:/type10-7/local_artifacts/cvs_FcrossRX_full_As_pair6_E30_20261018_r01/retrieved_r01/analysis_r01/training_curves.png)，[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_full_As_pair6_E30_20261018_r01/retrieved_r01/analysis_r01/training_curves.pdf)
- [源域状态／质量／噪声分层](E:/type10-7/local_artifacts/cvs_FcrossRX_full_As_pair6_E30_20261018_r01/retrieved_r01/analysis_r01/source_quality_state_lock_noise.png)，[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_full_As_pair6_E30_20261018_r01/retrieved_r01/analysis_r01/source_quality_state_lock_noise.pdf)
- [正式配对差](E:/type10-7/local_artifacts/cvs_FcrossRX_full_As_pair6_E30_20261018_r01/retrieved_r01/analysis_r01/formal_paired.png)，[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_full_As_pair6_E30_20261018_r01/retrieved_r01/analysis_r01/formal_paired.pdf)
- [物理训练覆盖](E:/type10-7/local_artifacts/cvs_FcrossRX_full_As_pair6_E30_20261018_r01/retrieved_r01/analysis_r01/pair_physical_coverage.png)，[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_full_As_pair6_E30_20261018_r01/retrieved_r01/analysis_r01/pair_physical_coverage.pdf)
- [全部四条件正式指标](E:/type10-7/local_artifacts/cvs_FcrossRX_full_As_pair6_E30_20261018_r01/retrieved_r01/analysis_r01/formal_metrics.csv)、[全部配对与修复错误](E:/type10-7/local_artifacts/cvs_FcrossRX_full_As_pair6_E30_20261018_r01/retrieved_r01/analysis_r01/formal_paired_effects.csv)
- [完整源分层](E:/type10-7/local_artifacts/cvs_FcrossRX_full_As_pair6_E30_20261018_r01/retrieved_r01/analysis_r01/source_strata.csv)、[源配对变化](E:/type10-7/local_artifacts/cvs_FcrossRX_full_As_pair6_E30_20261018_r01/retrieved_r01/analysis_r01/source_paired_effects.csv)
- [完整训练曲线数据](E:/type10-7/local_artifacts/cvs_FcrossRX_full_As_pair6_E30_20261018_r01/retrieved_r01/analysis_r01/training_curves.csv)、[训练物理联合覆盖](E:/type10-7/local_artifacts/cvs_FcrossRX_full_As_pair6_E30_20261018_r01/retrieved_r01/analysis_r01/pair_TX_RX_scene_state_lock.csv)
- [实际父集合完整读出](E:/type10-7/local_artifacts/cvs_FcrossRX_full_As_pair6_E30_20261018_r01/actual_ASCE20_cohort_r01/cohort_metrics.csv)、[全部聚合结果](E:/type10-7/local_artifacts/cvs_FcrossRX_full_As_pair6_E30_20261018_r01/result_numbers.json)

执行命令（远端已实际完成，不重复发布）：在 PROJECT/releases/cvs_FcrossRX_full_As_pair6_E30_20261018_r01 用原 CVS-RFFI/python -u experiments/cvs_full_channel_pair68/runner.py；评估由原 dispatcher 串联，CPU 分析使用 PROJECT/releases/cvs_FcrossRX_full_As_pair6_E30_20261018_r01/analysis/cvs_full_channel_pair68_analysis_20261018.py（实际 argv 见本地运行记录）。唯一不可变输出根为 PROJECT/runs/20261018-phase1-FcrossRX-fullAsPair6-e30-s2026092701-m1-r01。本地 canonical 报告位于 automation_reports/CV-SincNet/20261018-phase1-FcrossRX-fullAsPair6-e30-s2026092701-m1-r01/，源码由上述 code commit 固定，未热改健康训练。

未完成项：E50（源 FAIL 未晋级）、跨模型种子重复验证、low urban ≥60%、相对匹配 NoAs 的可重复 clean 正收益、物理损伤因果归因。正式七环境和本轮分析无缺项；不把这些未完成目标写成成功。后续可以给源合格的配方总 E50，不能凭此次正式微增晋级。
