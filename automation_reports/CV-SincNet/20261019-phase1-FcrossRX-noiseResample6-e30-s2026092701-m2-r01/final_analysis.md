# P75 完成报告：固定传播条件，仅重抽第二次原生噪声

本轮两条匹配 E20＋E10、各七环境冻结预测、独立 truth-last 评分和完整日志分析均完成，ANALYZED／VERIFIED，技术缺项为 0。模型种子只有 2026092701。源域假设未通过；不扩 E50、不扩种子、不扫噪声强度。单种子不足 0.1 pp 的微涨不构成可重复双提升，low urban 仍未达到 60%。

保留完整 As 物理引擎、raw→practical、ordinary/raw 监督份额与既有网络。P74 冻结源诊断支持噪声有影响，却没有单一分支足以独自恢复。本轮唯一假设是：既有两次 As CE 中，仅重抽第二份原生噪声比重复同一受损 IQ 更有效。源 FAIL 和正式微增共同表明这一固定方法没有足够增量；正式分数不用于后续晋级。

## 实际配置审计

- 起点 commit：7b476dc78c348a8dd9c87e1dcc04d1349b0eeeeb；实际代码 commit：ecdefc9bb562535bcc8087b5d59e38af25ab817a。
- 固定 F＋跨 RX 同 TX 对比、SIG_TIME_X3、统一 Sinc、joint160、默认 cosine30 分类头及正常联合更新。无额外诊断头或部署模块。
- 实际模块 __file__ 位于 PROJECT/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code/leo_practical/{channel,residual}.py，DataAugmentation 与 SSDG.train_ssdg 也来自该 pinned 实现，不是顶层 code/。
- IQ 2×256，fs=25 MHz，eq1，原逐包 RMS、center=false；N 与第一视图使用同一 normalize_input，再进入原生 residual。训练 fc=2.462 GHz，正式固定 received IQ 为 2.45 GHz，未改变。
- 实际 residual／post_sync／EQ=false，即 G=1；reference_snr_db=30。接收质量适配、相位跟踪、IQ 补偿、AGC 开启，reflect，fractional_delay_half_length=24。配置中 equalizer_delay=64、method=mmse 在 EQ 关闭时不代表已均衡。
- 实际 hook 场景是 practical_low_suburban／practical_mid_urban／practical_low_urban，channel_seed=20261016421、receiver_seed=20261016423。原生 args 中的旧场景和 2027 seed 字段不是 hook 的实际抽样值；全部物理记录已验证实际覆盖。
- L=6300、U=56700（TX 真值隐藏）、V=27000，源 RX=1/3/4/6/8，day=1/2/3，6 TX，split=392005，不留出训练 RX 或 L_dev。session 按 RX/day/eq 稳定，不由 TX 定义，不反演未知原录音 TX 硬件。
- 严格加载同契约 source-only FINAL20 student 和 ownEMA；重置 AdamW lr=0.0002、weight_decay=0.0001、cos10，固定 FINAL30。源面板 1800 个合法 V、2 个 draw，不计为两个模型 seed。
- N 只替换 latent.rngs[noise]，几何/H/遮挡/质量/锁定/相位/IQ/补偿重放；N 的 AGC 按新观测原生重算。C 复制第一 IQ，没有第二次物理生成或 AGC 计算。原生噪声经过 IQ／相位／G；不在输出加白噪声、不精确归一化噪声能量，不调弱正式条件。

## 实验结果与判定

# P75：同一完整 practical 的独立新增噪声 CE

两个新模型均从同一 SAT18 AS_CE20 的 student 与 ownEMA 分叉，追加10轮，总E30。C复用第一 satellite IQ；N仅替换第二 native noise RNG，几何/H/状态/相位/RX IQ/补偿与原 seed 保持，AGC按各view信号加噪声重新计算。六次L前向、两次As CE各0.125、原ordinary/raw KL、ORD128 SupCon和native U保留。无新增一致性损失、Ag、分类头或部署分支。

主比较 N−C。旧P68独立完整信道为同父来源的范围参照；原P43 As20前20轮包含As KL，与新共同父历史不同，P43 NoAs30是同预算、同seed、同数据角色整配方参照，并非同As父NoAs控制。三旧行全部复用，未重预测或重评分。

| 环境 | NoAs % | As % | 独立完整 % | C % | N % | N−C pp |
|---|---:|---:|---:|---:|---:|---:|
| clean | 85.247619 | 85.341071 | 85.233929 | 85.186310 | 85.222619 | +0.036310 |
| practical_high | 42.685119 | 81.405357 | 81.167857 | 81.116071 | 81.110714 | -0.005357 |
| practical_mid | 40.995238 | 78.661310 | 78.512500 | 78.408929 | 78.432143 | +0.023214 |
| practical_low_suburban | 39.472619 | 75.861905 | 75.767262 | 75.689286 | 75.751190 | +0.061905 |
| practical_high_urban | 41.439881 | 77.767262 | 77.632738 | 77.534524 | 77.550000 | +0.015476 |
| practical_mid_urban | 35.132738 | 62.806548 | 62.742262 | 62.664881 | 62.725595 | +0.060714 |
| practical_low_urban | 30.139881 | 51.244048 | 51.245833 | 51.194048 | 51.242857 | +0.048810 |

N相对C：low urban +0.048810 pp，mid urban +0.060714 pp，urban均值 +0.041667 pp，clean +0.036310 pp。
N相对P43 NoAs30：clean -0.025000 pp，urban均值 +28.268651 pp；clean下降不超过0.1绝对百分点：True。
只有一个模型seed；两个channel draw不是两个模型seed。不能宣称可重复双提升。全部七环境acc/F1/最差RX、逐RX/day/TX、修复与新增错误、默认头margin/30、固定P43 NoAs30 clean正确集合保持率见CSV。该集合不是共同AS_CE20父模型集合。

源面板分层包括LOS/PARTIAL/BLOCKED、lock、接收质量代理和新增噪声比。新增噪声SNR不是总SNR；相关分层和保存的native元数据重放检查不单独证明物理因果或随机独立性。EQ关闭保持G=1，但RX IQ校准残差仍可能产生improper noise。N未在输出加白噪声，也未精确归一化噪声能量；AGC允许随新噪声改变。

新增20 model-epochs／4440更新；每row总E30/6660更新。新增native NN=6325924、source=25200、formal=2352004、共享真实CP smoke=1540，合计=8704668，本地NN=0。C真实训练快照142080，N284160；每row非噪声latent实现142080，两As前向槽位284160，不能把C复制槽位再算新信道生成。共享smoke物理快照192。训练过程秒数合计=2728.003；相同NN不意味着相同墙钟。父训练成本单列且复用，不计新增。预算耗尽不等于收敛。

已存源判定：source_qualified=false。full V clean N−NoAs=−0.018519 pp，通过；源 pooled mid／low N−max(C,原As)=−0.055556 pp，未达预登记 +0.5 pp。源判定先保存并在本地观察，再读取新正式指标，没有根据正式微涨改变决定。

固定判据：N fullV clean相对原NoAs下降不超过0.1pp，源两draw pooled mid/low相对max(C,原As)至少+0.5pp。源失败仍完成all7，关闭该固定方案，不扫描噪声强度/seed或自动扩大E50。正式分数只用于独立报告。

技术缺项：[]

实际图表：
- E:/type10-7/local_artifacts/cvs_FcrossRX_noise_resample6_E30_20261019_r01/retrieved_r01/analysis_r01/training_curves.png
- E:/type10-7/local_artifacts/cvs_FcrossRX_noise_resample6_E30_20261019_r01/retrieved_r01/analysis_r01/training_curves.pdf
- E:/type10-7/local_artifacts/cvs_FcrossRX_noise_resample6_E30_20261019_r01/retrieved_r01/analysis_r01/source_quality_state_lock_noise.png
- E:/type10-7/local_artifacts/cvs_FcrossRX_noise_resample6_E30_20261019_r01/retrieved_r01/analysis_r01/source_quality_state_lock_noise.pdf
- E:/type10-7/local_artifacts/cvs_FcrossRX_noise_resample6_E30_20261019_r01/retrieved_r01/analysis_r01/formal_paired.png
- E:/type10-7/local_artifacts/cvs_FcrossRX_noise_resample6_E30_20261019_r01/retrieved_r01/analysis_r01/formal_paired.pdf
- E:/type10-7/local_artifacts/cvs_FcrossRX_noise_resample6_E30_20261019_r01/retrieved_r01/analysis_r01/pair_physical_coverage.png
- E:/type10-7/local_artifacts/cvs_FcrossRX_noise_resample6_E30_20261019_r01/retrieved_r01/analysis_r01/pair_physical_coverage.pdf

本离线分析新增NN、训练更新、IQ/checkpoint读取、重推理与重新评分均为0。

## 宏 F1、最差 RX 与净纠错

| 环境 | C 宏 F1 % | N 宏 F1 % | C 最差 RX % | N 最差 RX % |
|---|---:|---:|---:|---:|
| clean | 84.913674 | 84.948345 | 78.954167 | 78.916667 |
| practical_high | 80.824404 | 80.831189 | 73.312500 | 72.995833 |
| practical_mid | 78.137996 | 78.170842 | 71.454167 | 71.250000 |
| practical_low_suburban | 75.423810 | 75.491950 | 69.108333 | 68.979167 |
| practical_high_urban | 77.241452 | 77.265717 | 70.387500 | 70.225000 |
| practical_mid_urban | 62.335648 | 62.397247 | 57.366667 | 57.250000 |
| practical_low_urban | 50.778972 | 50.819822 | 47.058333 | 47.079167 |

| N−C | 修复旧错误 | 新增错误 | 净多正确 | margin/30 均值变化 |
|---|---:|---:|---:|---:|
| clean | 265 | 204 | +61 | +0.000461772 |
| practical_mid_urban | 748 | 646 | +102 | -0.000100367 |
| practical_low_urban | 899 | 817 | +82 | -0.000009228 |

low urban 仅净多 82 个正确，margin/30 均值没有随准确率微涨而改善。mid urban 和多数 practical 的最差 RX 未随整体微涨改善。固定 NoAs30 clean 正确集合下的 N low urban 保持率为 55.541280%，这不是共同 AS_CE20 父集合。

## 源域 low urban 分层

两 draw 合计 3600 次观察，N−C 修复 15／新增 18，净少 3 个正确。以下是不同分层口径，不能相加解释总量。

| 分层 | n | C % | N % | N−C（百分点） |
|---|---:|---:|---:|---:|
| state: LOS | 1060 | 95.283019 | 95.188679 | -0.094340 |
| state: PARTIAL | 561 | 84.848485 | 84.491979 | -0.356506 |
| state: BLOCKED | 1979 | 32.693279 | 32.693279 | +0.000000 |
| state_lock: BLOCKED/degraded | 414 | 45.169082 | 44.444444 | -0.724638 |
| state_lock: BLOCKED/unlocked | 1565 | 29.392971 | 29.584665 | +0.191693 |
| signal_to_added_noise_db: <3 | 1127 | 26.530612 | 26.441881 | -0.088731 |
| signal_to_added_noise_db: 3-8 | 794 | 39.798489 | 39.924433 | +0.125945 |

没有证明哪类 urban 退化获得明确缓解：BLOCKED 总正确数完全相同，BLOCKED/unlocked 只净多 3 个、BLOCKED/degraded 净少 3 个，低新增噪声比组无一致改善。正式 IQ 没有逐包 state/lock 标签，不能将源分层套给正式样本。P74 固定状态去噪是冻结诊断，不是本轮训练收益或可部署上限。

## 完整成本、验证和执行

| 行 | 父 E20 NN（复用） | 尾 E10 NN | 尾 E10 秒 | source NN | formal NN | 新原生快照 |
|---|---:|---:|---:|---:|---:|---:|
| SAME_NOISE_CE6 | 4797722 | 3162962 | 1233.277 | 12600 | 1176002 | 142080 |
| INDEP_NOISE_CE6 | 4797722 | 3162962 | 1494.726 | 12600 | 1176002 | 284160 |

共同父 E20：4440 更新、4797722 次样本前向、1827.538 秒，只复用一次；不是免费，也不重复计入新增。N 尾段墙钟比 C 增加 21.200%，含本次资源和记录条件差异。端到端 1649.610 秒，依据未改动 dispatcher 至 analysis complete 文件 mtime，不是 profiler。

三条历史 E30 参照的训练内实际 NN（父＋尾）分别为：NOAS30=7960684；AS30=7960684；INDEP_FULL_CE6=7960684。同 E30 不等于同前向成本，不能把预算差全部归因于新机制。

已执行：协议 6 组／22 个负例／5 个源判据样例；信道窄检查；真实模型 hook 的同 IQ loss／梯度／Torch RNG 对齐和作用域恢复；分析 10 个 fixtures；19 个新增 Python 语法检查；一次独立 P0/P1 PASS；一次 N607 preflight、单 archive 本地/远端 SHA 比较、23 文件远端编译；1540 NN 真实父权重 smoke，source IQ=0、target query=false、训练/头拟合更新=0、原生快照192；PID/CWD/argv/GPU/日志增长 VERIFIED。

离线分析从既存预测独立复算 truth-last 指标。本地另核对 630 个正式、1785 个源混淆矩阵和 1134 条正式配对错误行，计数及准确率一致；150 条 epoch 曲线，L/U 样本顺序与普通增强流匹配。四张 PNG 已查看，四份 PDF 头有效。没有重新推理旧参考。本地 legacy factory 因不完整 worktree 同名包遮蔽未导入，真实 pinned 远端 CP 验证已覆盖实际图；没有因此修改历史包或新增 blocker。

实际环境：N607 既有普通账号 CVS-RFFI Python3.10／Torch2.1／NumPy2.2，full FP32 CENTER_UNITS，GPU0=C、GPU1=N。CWD 为 PROJECT/releases/cvs_FcrossRX_noise_resample6_E30_20261019_r01。PROJECT 是远端项目根的报告别名。

执行命令：

    CVS-RFFI/python -u experiments/cvs_noise_resample75/runner.py

    CVS-RFFI/python -u analysis/cvs_noise_resample75_analysis_20261019.py --project PROJECT --out PROJECT/runs/20261019-phase1-FcrossRX-noiseResample6-e30-s2026092701-m2-r01/analysis_r01 --local-nn 0 --smoke-nn 1540 --code-commit ecdefc9bb562535bcc8087b5d59e38af25ab817a

第一条 launcher 自动衔接 source→qualification→all7 prediction→独立 exec scorer→完整分析。输出根 PROJECT/runs/20261019-phase1-FcrossRX-noiseResample6-e30-s2026092701-m2-r01，owner=codex/root/noise-resample75。两个 row 为 TRAINED_AND_SCORED，analysis=ANALYZED，driver 已退出，独立后态 VERIFIED。

实际新增文件：experiments/cvs_noise_resample75/ 下 17 个 Python 文件，以及 analysis/cvs_noise_resample75_analysis_20261019.py 和对应 checks。原冻结 release、默认头和正式 IQ／信道保持。

本地可查产物：

- [全部逐行／逐种子／七环境和 RX/day/TX 指标](E:/type10-7/local_artifacts/cvs_FcrossRX_noise_resample6_E30_20261019_r01/retrieved_r01/analysis_r01/formal_metrics.csv)
- [配对 F1／最差 RX／margin／纠错](E:/type10-7/local_artifacts/cvs_FcrossRX_noise_resample6_E30_20261019_r01/retrieved_r01/analysis_r01/formal_pair_summary.csv)
- [源状态、锁定、质量及新增噪声比分层](E:/type10-7/local_artifacts/cvs_FcrossRX_noise_resample6_E30_20261019_r01/retrieved_r01/analysis_r01/source_strata.csv)
- [源配对效果](E:/type10-7/local_artifacts/cvs_FcrossRX_noise_resample6_E30_20261019_r01/retrieved_r01/analysis_r01/source_paired_effects.csv)
- [150 条完整 epoch 曲线](E:/type10-7/local_artifacts/cvs_FcrossRX_noise_resample6_E30_20261019_r01/retrieved_r01/analysis_r01/training_curves.csv)
- [训练 TX×RX×信道状态覆盖](E:/type10-7/local_artifacts/cvs_FcrossRX_noise_resample6_E30_20261019_r01/retrieved_r01/analysis_r01/pair_TX_RX_scene_state_lock.csv)
- [父、增量、教师、验证和正式成本](E:/type10-7/local_artifacts/cvs_FcrossRX_noise_resample6_E30_20261019_r01/retrieved_r01/analysis_r01/costs.json)

科学未完成项：跨种子可重复双提升、low urban 60% 和稳定 clean 正收益均未达成；本固定候选不扩预算。正式逐包状态归因、当前新模型内部 gate 完整变化未新增测量。技术训练、七环境测试和完整日志分析均已完成。另一对话正在研究 rest／As 独立优化器状态，本轮没有重复该路线或修改其健康实验。

共同父 checkpoint：PROJECT/runs/20261018-phase1-FcrossRX-satelliteDropout-e30-s2026092701-m2-r01/AS_CE-s2026092701/stage20/final_ssdg.pth。

实际代码文件：

- [__init__.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_noise_resample75/__init__.py)
- [channel.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_noise_resample75/channel.py)
- [channel_checks.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_noise_resample75/channel_checks.py)
- [checks.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_noise_resample75/checks.py)
- [design.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_noise_resample75/design.py)
- [dispatch.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_noise_resample75/dispatch.py)
- [evaluate.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_noise_resample75/evaluate.py)
- [model.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_noise_resample75/model.py)
- [model_checks.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_noise_resample75/model_checks.py)
- [observer.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_noise_resample75/observer.py)
- [preflight.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_noise_resample75/preflight.py)
- [protocol.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_noise_resample75/protocol.py)
- [qualification.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_noise_resample75/qualification.py)
- [quality.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_noise_resample75/quality.py)
- [report.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_noise_resample75/report.py)
- [runner.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_noise_resample75/runner.py)
- [source.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_noise_resample75/source.py)
- [cvs_noise_resample75_analysis_20261019.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/analysis/cvs_noise_resample75_analysis_20261019.py)
- [cvs_noise_resample75_analysis_checks_20261019.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/analysis/cvs_noise_resample75_analysis_checks_20261019.py)
