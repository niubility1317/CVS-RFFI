# P59：完整 practical 的 urban 重点分配 E30 结案

**VERIFIED／ANALYZED。** 固定 seed 2026092701、同预算 E30、原 CVS／默认联合训练 cosine 头和正式 IQ 条件下，新行相对匹配 No-As30：urban 三场景均值 +28.550992 个百分点，low urban +21.325000、mid urban +27.871429，clean +0.098214。该单种子满足正式 clean 下降不超过 0.1 个百分点的阶段要求；clean 增幅不足 0.1，不能宣称可重复双提升。low urban 仍仅 51.464881%，未达到 60%。

新增机制的归因必须比较原 As30：urban 均值仅 +0.182540 个百分点，low +0.220833、mid +0.197619、clean +0.004762。大部分星地收益来自已有完整 As 配方，不能归给此次 urban 比例变化。

源域预定晋级条件未通过，本固定比例方案关闭，不扩 seed 或 E50、不扫比例救结果。正式成绩没有改变该决定。用户允许好方案达到总 E50；原 As 的三种子 E50 已另行完成，其预算收益与 clean 代价见下文。

## 唯一变化与对照

只在精确原 As20 student＋own EMA 父模型的 E21–30，将 low suburban／mid urban／low urban 从各 1/3 改为 .2／.4／.4；完整信道、全部状态、原六次 L 前向／clean 槽位、同 IQ RDrop、跨 RX SupCon、旧 U 规则与损失尺度 1.25 不变。没有新增 Ag／loss／头／结构或标签门控。硬件 session 按 RX／day／eq 稳定，非 TX；As 从 raw IQ 生成，未先叠普通 PA／噪声／多径。

复用原 P43 B00 No-As30 和 B01 As30 的全部固定预测／独立评分，不重训、不重预测、不重评分。三者 all5 源 RX1/3/4/6/8、3day、L6300／隐藏 U56700／V27000、6TX、split392005、默认头与总 30 轮一致；不留源 RX 或 L_dev。新行与 As30 共用同一原 As20 父权重。

实际 leo_practical 来自旧 evidence-responsibility release 下 experiments/adv3b02_xuc/code/leo_practical 的 channel.py／residual.py，已打印 __file__；非根 code。fs25MHz／IQ256，residual／post_sync／EQ=false 即 G=1、reference_snr_db30、quality_adaptation／phase_tracking／IQcompensation=true、reflect。训练 fc2.462GHz／正式固定 fc2.45GHz，RMS 位置和所有正式 IQ／信道／SNR／样本保持。详情见 [原登记](<E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-urban-prior-e30-s2026092701-m1-r01/experiment.json>)、[实际普通增强审计](<E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-urban-prior-e30-s2026092701-m1-r01/ordinary_augmentation_audit.md>)。

## 正式逐行结果

全部值为百分数，差值为绝对百分点。每视图 168000 个固定物理样本。七视图、三个逻辑 row 的冻结配置／ID／logits／argmax／独立评分完成后才做 truth-side 配对分析；378 个混淆矩阵重算与 63 次分区对账全部通过。

| 视图 | No-As30 | 原 As30 | 新 URBAN | 新−原 As | 新−No-As | 新宏 F1 | 新最差 RX |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 85.247619 | 85.341071 | 85.345833 | +0.004762 | +0.098214 | 85.050403 | 78.362500 |
| practical_high | 42.685119 | 81.405357 | 81.480357 | +0.075000 | +38.795238 | 81.177696 | 72.016667 |
| practical_mid | 40.995238 | 78.661310 | 78.749405 | +0.088095 | +37.754167 | 78.459531 | 70.354167 |
| practical_low_suburban | 39.472619 | 75.861905 | 75.945833 | +0.083929 | +36.473214 | 75.662098 | 67.850000 |
| practical_high_urban | 41.439881 | 77.767262 | 77.896429 | +0.129167 | +36.456548 | 77.568779 | 69.208333 |
| practical_mid_urban | 35.132738 | 62.806548 | 63.004167 | +0.197619 | +27.871429 | 62.632936 | 56.708333 |
| practical_low_urban | 30.139881 | 51.244048 | 51.464881 | +0.220833 | +21.325000 | 51.061025 | 46.670833 |

urban 三场景均值：No-As30 35.570833%、原 As30 63.939286%、新 URBAN 64.121825%。新 clean 最差 RX 为 1-1；全部 practical 的新最差 RX 均为 14-7。新 suburban 与自身 clean 的差距为 high 3.865476／mid 6.596429／low 9.400000 个百分点，仍有明显差距。全部控制行的宏 F1、最差 RX、逐 RX／day／TX 结果见 [正式指标 CSV](<E:/type10-7/local_artifacts/cvs_FcrossRX_urban_prior_E30_20261018_r01/retrieved_r02/analysis_r02/formal_metrics.csv>) 和 [完整混淆矩阵](<E:/type10-7/local_artifacts/cvs_FcrossRX_urban_prior_E30_20261018_r01/retrieved_r02/analysis_r02/formal_confusions.json>)。

相对原 As30，clean 修复 259／新增错误 251（净 +8）；mid urban 950／618（净 +332）；low urban 1199／828（净 +371）。相对 No-As30，clean 3868／3703（净 +165）；mid 53594／6770；low 44620／8794。不是只统计被修复样本。固定 No-As30 clean 正确集合为 143216 个；new low 保持率 55.872249%，原 As30 为 55.632751%，No-As30 为 32.807089%。

间隔统一除以固定 cosine scale30：low urban 从原 As30 .028075527 到新 .028696325；mid 从 .053809803 到 .054256000；clean 从 .146124229 到 .145850390，clean 间隔略降。不能把准确率的微小上涨解释为所有身份证据或门控更好。实际匹配错误与间隔见 [配对 CSV](<E:/type10-7/local_artifacts/cvs_FcrossRX_urban_prior_E30_20261018_r01/retrieved_r02/analysis_r02/formal_paired_effects.csv>)。

![正式七视图对比](E:/type10-7/local_artifacts/cvs_FcrossRX_urban_prior_E30_20261018_r01/retrieved_r02/analysis_r02/formal_accuracy.png)

## 源域判定与退化类别

full-V 原生 final student：No-As30 98.359259%、原 As30 98.255556%、新 98.251852%。新相对 No-As30 −0.107407 个百分点，超过预定 −0.1 上限；只差 2 个样本，不宣称显著 clean 损伤。固定 V1800×2 信道抽样：新相对 As30 的 mid／low 均值 +0.027778 个百分点，远低于预定 +0.5；low +0.111111。两个信道抽样是同一模型的重复观测，不是两个模型种子。

| 源域 full 条件／状态 | 观测数 | 新−原 As30（百分点） | 新−No-As30（百分点） |
|---|---:|---:|---:|
| mid urban BLOCKED/degraded | 656 | 0.000000 | +11.280488 |
| mid urban BLOCKED/unlocked | 678 | −0.147493 | +11.799410 |
| low urban BLOCKED/degraded | 414 | +0.966184 | +12.801932 |
| low urban BLOCKED/unlocked | 1565 | +0.383387 | +10.223642 |

已有完整 As 主要缓解受损观测的识别失败，BLOCKED/degraded 与 BLOCKED/unlocked 中都有相对 No-As 的明显改善；本轮场景重点分配仅对 low 的这些组产生小增量，mid 并未一致改善。它没有证明哪一个物理损伤是原因，也未发现新的稳定突破。正式 IQ 无逐包状态标签，不能把源状态分层冒称目标状态结果。

分层使用 signal_to_added_noise_db，即信号对新增噪声比，不是总 SNR。此轮未更改单场景传播与接收状态联合规律，也没有新做固定补偿参数的物理干预；不据此声称分离出噪声／多径／CFO 的因果效应。空 cell、小 cell、全部状态与质量区间保留。见 [源分层](<E:/type10-7/local_artifacts/cvs_FcrossRX_urban_prior_E30_20261018_r01/retrieved_r02/analysis_r02/source_strata.csv>)、[源配对](<E:/type10-7/local_artifacts/cvs_FcrossRX_urban_prior_E30_20261018_r01/retrieved_r02/analysis_r02/source_paired_effects.csv>)。

![mid urban 源状态与新增噪声比](E:/type10-7/local_artifacts/cvs_FcrossRX_urban_prior_E30_20261018_r01/retrieved_r02/analysis_r02/practical_mid_urban_source_states.png)

![low urban 源状态与新增噪声比](E:/type10-7/local_artifacts/cvs_FcrossRX_urban_prior_E30_20261018_r01/retrieved_r02/analysis_r02/practical_low_urban_source_states.png)

## 实际训练覆盖、曲线与成本

完整 2220 步／10 轮、6 次 L 前向每步、实际 1704960 个 L 前向样本、142080 个生成 As 已逐步核对。实际 low suburban 28829／mid urban 56529／low urban 56722，覆盖全部 TX×RX×场景；15 个 RX/day/eq session 稳定，全部状态支持保留，dropped=0。4440 条 L/U 物理序列记录与原 As30 尾段完全匹配；ordinary MSE 不一致数 0。原隐藏 U 共处理 567000 个，原门控选中 8713；这不是对有标签 As 的筛选。

原 As20：4440 更新／4797722 个原生学生＋教师＋验证前向，墙钟 1810.249 秒；新增 E10：2220 更新／3162962 个原生前向，墙钟 1054.587 秒；累计模型 E30／6660 更新，原生训练墙钟合计 2864.837 秒。新增源读出 12600、正式预测 1176002、远端 smoke770，新增远端总前向 4352334；本地合成验证1536另列。正式预测墙钟48.186秒，源读出墙钟未单独保存。

新段原生前向分解：student_train2271960／student_eval324002／teacher_eval567000／teacher_train0。复用两条 E30 对照历史训练共 E60／13320 更新；算上本次新增 E10，唯一训练共 E70／15540 更新，共享 As20 只计一次，已知远端总前向22650906。历史六行共享 smoke 尚未分摊，不能把该已知成本称为全部历史开销。对照无重新预测／评分。

实际 E21–30 的 train loss 约 .64 至 .62，存在波动；梯度有限且非零，源准确率在98%左右，不据此宣称训练已收敛。原两条对照完整 E1–30 与新尾段 E21–30 均使用全部记录绘图，没有补造 E20 连接点或平滑。见 [完整曲线 CSV](<E:/type10-7/local_artifacts/cvs_FcrossRX_urban_prior_E30_20261018_r01/retrieved_r02/analysis_r02/training_curves.csv>)、[完整覆盖 CSV](<E:/type10-7/local_artifacts/cvs_FcrossRX_urban_prior_E30_20261018_r01/retrieved_r02/analysis_r02/training_coverage.csv>)、[成本 JSON](<E:/type10-7/local_artifacts/cvs_FcrossRX_urban_prior_E30_20261018_r01/retrieved_r02/analysis_r02/costs.json>)。

![完整训练曲线](E:/type10-7/local_artifacts/cvs_FcrossRX_urban_prior_E30_20261018_r01/retrieved_r02/analysis_r02/training_curves.png)

## 技术恢复与验证

候选代码33a7046ee6e15898f30eca05f7bbc2c3249ce7e3；9项定向检查、一次独立P0/P1审查（0/0）、一次真实As20 no-query smoke、单归档传输与远端编译、启动后PID/CWD/argv/GPU/日志增长均已核实。原模型／own EMA严格加载，optimizer／cos10／temporal cache重置；没有停止或热修改其他健康实验。

源资格预测全部完成后，source_decision 的 numpy.bool_ 写JSON失败，自动 all7 正式收尾仍完成。修复81f7b178b34cefda2710a1d60df7180058a2537c仅给三条既定比较加 bool 转换；先复现后4项检查PASS，随后0NN从原源文件恢复原判定。恢复发生在当前分析读取正式分数之前；原failure保留，科学FAIL未改变。

首次 CPU 分析也在 NumPy 布尔写JSON时退出，保留 analysis_r01。分析修复a4cea81447f015a54585c3e5a7e561bab193dfe7支持 np.generic.item 且继续拒绝NaN／未知类型；原11项合成检查及本次5项定向检查PASS。新 analysis_r02 完成，20.073秒，NN／IQ／checkpoint／更新均0，missing=[]。首次失败 CPU 墙钟未独立保存。四张PNG均已实际打开检查；同名PDF可导出。两次修复未重训、未重做源资格或正式预测／scorer，未更改阈值、seed、标签或正式条件。

## E50授权与未完成目标

原 As 三种子 E30→E50：low urban平均再升1.156746／mid .954167／urban均值 .895635 个百分点，clean变化−.017659；跨seed星地预算增益方向一致。但与匹配No-As50相比，clean平均−.150198（配对SD .955701）仍超.1，low urban均值51.926587%，不是稳定双提升。全部逐seed结果与成本见 [E50三种子报告](<E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-continuous-recovery-e30-s2026092701-m1-r01/E50_three_seed_readout.md>)。

本轮固定比例源判定FAIL，因此不以正式小涨为理由扩到E50。总E50授权保留给后续有源证据的好方案，并继续使用匹配无As对照。不能自动扩成E70。

尚未完成：本轮跨模型seed复现、low urban>60%、稳定clean正收益。此轮保持率使用固定No-As30正确集合；未额外计算As20父模型正确集合，也未新增每分支／内部门控全量诊断。物理因果干预与此前诊断分开，不把当前状态关联当作因果答案。当前row全部预定训练、源读出、七视图预测评分和报告已完成；全局研发目标仍未完成。

## 可执行入口与交付

训练发布CWD为 PROJECT/releases/cvs_FcrossRX_urban_prior_E30_20261018_r01，使用既有CVS-RFFI Python运行：

```text
python -u experiments/cvs_urban_prior59/runner.py
```

该run已存在，禁止再次执行训练命令覆盖或重复启动。已执行CPU收尾的CWD为 PROJECT/releases/cvs_FcrossRX_urban_prior59_analysis_20261018_r02，命令为：

```text
python -u analysis/cvs_urban_prior59_analysis_20261018.py --project PROJECT --out PROJECT/runs/20261018-phase1-FcrossRX-urban-prior-e30-s2026092701-m1-r01/analysis_r02 --truth PROJECT/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json --code-commit a4cea81447f015a54585c3e5a7e561bab193dfe7
```

实际新增实现位于 [experiments/cvs_urban_prior59](<E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_urban_prior59>)，方法配置在 [design.py](<E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_urban_prior59/design.py>)；分析在 [analysis脚本](<E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/analysis/cvs_urban_prior59_analysis_20261018.py>)。运行命令和私有后状态在本机 local_artifacts/cvs_FcrossRX_urban_prior_E30_20261018_r01 保存；未把权重、IQ、NPZ、日志或生成图加入Git。
