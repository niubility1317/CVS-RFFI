# P72 完整 practical 下 raw—As 掩码相关性：执行与负结果

状态：ANALYZED；两行训练、固定 clean＋六 practical 预测、独立 truth-last 评分、源与正式配对分析全部完成，missing=[]。模型 seed 仅 2026092701。代码/config commit：833c2b5694914a2923c491701b8b7d47cf624205；起点：7c9271cb1f1375b4064830aa136131820d552c2b。

共享掩码没有新增识别收益，关闭固定候选，不扩 E50、不扫位置、p 或随机种子。相对同父、同 E30、同前向成本的独立掩码 I，S 的 clean −0.032143、mid urban −0.007143、low urban −0.042262、urban 均值 −0.025992 个百分点；七环境准确率、宏 F1、最差 RX 差值均为负。

相对同预算同 seed NoAs，S 的 clean +0.025000、mid urban +27.661905、low urban +21.100595、urban 均值 +28.350992 个百分点。这个单 seed 正式结果满足 clean 下降不超过 0.1 个百分点的阶段要求，但大幅星地收益来自已有完整 As，不能归给共享掩码。clean +0.025 不构成可重复双提升。low urban 51.240476%，未达 60%。本轮没有证明任何特定严重退化被新机制缓解。

## 基线、审计与唯一假设

固定 F＋跨 RX 同 TX SupCon、SIG_TIME_X3、joint160、默认 scale30 cosine 分类头，900722 参数／206 state，骨干与默认头正常联合训练。推理直接用最终 checkpoint 的 model 学生键，未单训、重训或替换头，未用 EMA 头替代正式结果。L=6300、U=56700（TX 真值隐藏）、V=27000，源 RX=1/3/4/6/8，day=1/2/3，split=392005；未留出训练 RX、未拆 Ldev。原 P43 源域 scratchAs20 学生及自身 EMA 经严格同契约加载，尾段独立重置 AdamW（lr=2e-4，wd=1e-4）、cos10、CENTER_UNITS。各新增行追加 E10／2220 更新，总 E30／6660 更新。原 NoAs30 来源是自身 NoAs20，不能称四行同父；S−I 才是掩码机制的主比较。

实际 leo_practical 导入来自 PROJECT/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code/leo_practical/channel.py 与 residual.py，模块 __file__ 已保存，运行的不是根 code 副本。25 MHz、float32 B×2×256、逐包 RMS center=false 后从 raw 构造完整 practical；residual/post_sync、equalization_enabled=false，G=1，reference_snr_db=30，reflect 边界，原接收质量适配、相位跟踪、IQ 补偿、AGC 保持。训练 fc=2.462 GHz、固定正式 fc=2.45 GHz 沿用原条件。正式 received IQ、信道配置、SNR 和困难样本未改。signal_to_added_noise_db 仅为新增噪声比，不当作总 SNR。

目标层仅 id_backbone.encoder.core.id_backbone.fuse.2（nn.Dropout p=.45、base160），位于 PA/reference/SIG_TIME 残差相加之前，不是整个 joint 表示。I 用独立私有 raw/As mask；S 只令同物理包的 raw0↔As0、raw1↔As1 mask 相同，RDrop 两次仍独立。所有原生随机调用保留，用 pre-dropout 输入乘私有 scaled mask 覆盖目标行，未做双重 Dropout。两行同 private seed20261019072／step／pair／role，不按 TX 定义随机流。U、EMA、普通视图及其他层不变；无新参数、部署模块、损失或网络前向。

原目标保留 .5 RDordinary128＋.5 RDmixed128＋.25 RDraw128＋.1 ORD128 SupCon，CE 总系数1.25，其中 As CE=.25、As 同 IQ KL=.25；每步 L 六前向共768包。未将不同信道输出强行相等。训练信道仅 low suburban、mid urban、low urban，完整内部依赖保留。

P70 nominal 池142080条只读按原生成顺序各消费一次，每步校验 raw/As 五项物理 key、标签、domain、origin step/slot；不做质量加权、筛选、重画或乱序。池已包含跨 epoch/view 刷新，RX 硬件 session 为 RX/day/eq，不由 TX 定义；训练、源评价、正式随机流分离。所有合法带标签 As 都参加 CE。

本轮只检验随机掩码相关结构，不声称提供新增物理或身份信息、保证梯度降噪。旧 ordinary↔As ANTITHETIC 已涉及13个 encoder 随机层，包含该层；本轮关键差异是单 base160 层、raw 配对、IDENTICAL 正相关。不能把它包装成首次跨视图耦合。P71 固定风险重分配已源FAIL；PROP_GROUP 共享传播单元已训练并源FAIL；另一对话“审计并设计 CVS 联合增强”的三行 U practical 正在独立执行，本侧未重复、未干扰。

## 逐行、逐 seed 正式结果

以下均为同一 seed2026092701、每环境168000包的准确率 %。原 NoAs/As 冻结预测与评分复用，未重训或重新选模；所有新增行 FINAL30/final_only。四行是两个历史参照加一个两行机制对照，不是2×2因子设计。

| 环境 | NoAs E30 | 原完整 As E30 | 独立掩码 I | 共享掩码 S |
|---|---:|---:|---:|---:|
| clean | 85.247619 | 85.341071 | 85.304762 | 85.272619 |
| practical_high | 42.685119 | 81.405357 | 81.387500 | 81.370238 |
| practical_mid | 40.995238 | 78.661310 | 78.632143 | 78.611310 |
| practical_low_suburban | 39.472619 | 75.861905 | 75.865476 | 75.860714 |
| practical_high_urban | 41.439881 | 77.767262 | 77.758929 | 77.730357 |
| practical_mid_urban | 35.132738 | 62.806548 | 62.801786 | 62.794643 |
| practical_low_urban | 30.139881 | 51.244048 | 51.282738 | 51.240476 |

| 环境 | S 准确率 % | S 宏 F1 % | S 最差 RX % | S−I 准确率 pp | 修复／新增错误 |
|---|---:|---:|---:|---:|---:|
| clean | 85.272619 | 84.975029 | 78.262500 | -0.032143 | 97／151 |
| practical_high | 81.370238 | 81.054944 | 71.675000 | -0.017262 | 164／193 |
| practical_mid | 78.611310 | 78.314541 | 70.070833 | -0.020833 | 177／212 |
| practical_low_suburban | 75.860714 | 75.572104 | 67.625000 | -0.004762 | 202／210 |
| practical_high_urban | 77.730357 | 77.390198 | 68.833333 | -0.028571 | 168／216 |
| practical_mid_urban | 62.794643 | 62.431777 | 56.370833 | -0.007143 | 228／240 |
| practical_low_urban | 51.240476 | 50.873310 | 46.375000 | -0.042262 | 256／327 |

S 的 urban 三场景均值63.921825%。clean 最差 RX 是1-1，其余六环境均为14-7；这里采用既有 RX group 标识。S−I 的平均 margin÷30 虽有0.000022至0.000146的微小增加，七环境准确率仍下降，不能用均值间隔代替识别效果。low 修复256次、引入327次错误，净少71次正确；mid 修复228次、引入240次，净少12次正确。全部配对结果、宏F1、最差RX及 RX/day/TX 结果见完整 CSV，不只报告有利分层。

CPU 分析在连接 truth 前核验全部四逻辑行、每行七预测，独立复计504个混淆矩阵与84个分区汇总。训练冻结后逐 row 自动 all7，独立 scorer 才连接 truth；无 target 选模、调参、晋级或选择性重跑。

## 源域判定与机制读数

源判定先于 owner 读取正式分数固定：全V27000 S−NoAs clean=−0.118519 pp，未达到≥−0.1；两固定draw midlow S−I=−0.027778 pp，未达到≥+0.5。两个条件均FAIL。sourceFAIL仍完成预登记 all7，但不扩E50、不扩seed或改写判定。

每个源场景是1800物理样本×2draw=3600观察，不是两个模型seed。S−I：mid净−3／3600，low净+1／3600，无跨困难状态一致改善。固定原As20 clean正确1765集合，I→S的 mid 保持2613→2611／3530、low2112→2114／3530。该条件保持率不等于正式 low urban，也不能弥补全V代价。原NoAs20自身父集合和正式父20集合未新增读取，不伪称完备。完整噪声／LOS-PARTIAL-BLOCKED／locked-degraded-unlocked分层见 mechanism_readout.md。正式IQ没有逐包state/lock标签，不能将源关联结果当作正式退化因果归因。

2220步完整trace：S raw/As位相同率100%，I约50.499934%；两次RDrop位相同率仍约50.5%，私有draw与保留计数逐步匹配，原生RNG调用保留。原序142080包物理配对及每条单次消费通过，全部五源RX和原场景/状态支持保留。这证明机制实施正确，不证明身份收益。完整训练梯度及loss曲线已检查，无非数值或缺轮；预算完成不等于收敛。

## 成本与验证

新训练仅两个E10尾段，共20模型epoch／4440更新；每行native3162962 NN、source12600 NN、formal1176002 NN。一次共享真实父学生/自身EMA无query smoke为4NN，总新增8703132样本前向。实际训练墙钟I=922.272764秒、S=922.989794秒；两行并行，和1845.262558秒是训练进程时间合计。提交至分析完成的实测经过1034.568583秒，CPU完整分析37.923822秒。source墙钟I=2.127250／S=1.973129秒；formal I=52.118129／S=50.136629秒；池加载I=.173721／S=.024237秒。

共用As20父训练4797722NN／1810.249311秒复用未重复；每新增模型累计native7960684NN。去重共同As20父后，本轮父＋新增native为11123646NN；父＋所有新增NN为13500854。含两个不同历史父和四个E10尾段，四逻辑行唯一模型epoch总80；此次新增只20。历史池生成142080次、287.656秒已发生，新模拟0；每行私有mask抽样90931200位，原生范围227328000位。mask操作不算NN，未隔离测其算子墙钟，不能声称零时间或精确加速。

已执行：20协议负例、8mask CPU toy、2原序池／六前向hook toy、8分析CPU toy、21个新Python语法；一次独立P0/P1审查PASS。远端归档一次SHA比较VERIFIED、24py compilePASS；真实父checkpoint严格student/ownEMA noquery smokePASS4NN。启动PID/CWD/argv/GPU/log增长独立核对VERIFIED，真实GPU运行每行2220更新；结束后两worker和driver均已自然退出，analysis status=ANALYZED/missing=[]独立回读VERIFIED。4PNG已实际查看，4PDF头与非空文件确认。未重复无变化审查或测试；未停止其他健康任务。

## 文件、命令与图件

执行入口：experiments/cvs_raw_as_mask72/runner.py；关键实现masks.py、ordered_pool.py、observer.py、design.py，另有CPU分析analysis/cvs_raw_as_mask72_analysis_20261019.py和对应checks。真实配置、模块路径、初始化、完整step/epoch/mask/ordered trace及预测评分元数据已取回至本机retrieved_r01；IQ、权重和原始log不入Git。

远端CWD=PROJECT/releases/cvs_FcrossRX_rawAsMask_E30_20261019_r01，命令：

```text
CVS-RFFI/python -u experiments/cvs_raw_as_mask72/runner.py
```

launcher首先真实父smoke，PASS后立即训练；环境普通非特权CVS-RFFI Python3.10.19/Torch2.1.0+cu121/NumPy2.2.5，FP32/CPU2，I用GPU0、S用GPU1，遵守每GPU最多2总进程和空闲内存要求。immutable run=20261019-phase1-FcrossRX-rawAsMask-e30-s2026092701-m2-r01，唯一owner=codex/root/raw-as-mask72。技术停止规则和逐row矩阵保留在experiment.json及原预登记report.md。

- [全部逐row/逐seed指标](E:/type10-7/local_artifacts/cvs_FcrossRX_rawAsMask_E30_20261019_r01/retrieved_r01/analysis_r01/formal_metrics.csv)，[完整配对变化](E:/type10-7/local_artifacts/cvs_FcrossRX_rawAsMask_E30_20261019_r01/retrieved_r01/analysis_r01/formal_pair_summary.csv)，[逐分层纠错](E:/type10-7/local_artifacts/cvs_FcrossRX_rawAsMask_E30_20261019_r01/retrieved_r01/analysis_r01/formal_paired_effects.csv)。
- [源分层](E:/type10-7/local_artifacts/cvs_FcrossRX_rawAsMask_E30_20261019_r01/retrieved_r01/analysis_r01/source_strata.csv)，[源配对纠错](E:/type10-7/local_artifacts/cvs_FcrossRX_rawAsMask_E30_20261019_r01/retrieved_r01/analysis_r01/source_paired_effects.csv)，[原父正确集合](E:/type10-7/local_artifacts/cvs_FcrossRX_rawAsMask_E30_20261019_r01/retrieved_r01/analysis_r01/P43_As20_clean_cohort.csv)。
- [训练曲线数据](E:/type10-7/local_artifacts/cvs_FcrossRX_rawAsMask_E30_20261019_r01/retrieved_r01/analysis_r01/training_curves.csv)，[实际覆盖](E:/type10-7/local_artifacts/cvs_FcrossRX_rawAsMask_E30_20261019_r01/retrieved_r01/analysis_r01/training_As_coverage.csv)，[mask逐步数据](E:/type10-7/local_artifacts/cvs_FcrossRX_rawAsMask_E30_20261019_r01/retrieved_r01/analysis_r01/mask_steps.csv)，[成本](E:/type10-7/local_artifacts/cvs_FcrossRX_rawAsMask_E30_20261019_r01/retrieved_r01/analysis_r01/costs.json)。
- [完整训练PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_rawAsMask_E30_20261019_r01/retrieved_r01/analysis_r01/full_training.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_rawAsMask_E30_20261019_r01/retrieved_r01/analysis_r01/full_training.pdf)，[源分层PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_rawAsMask_E30_20261019_r01/retrieved_r01/analysis_r01/source_state_quality_lock_noise.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_rawAsMask_E30_20261019_r01/retrieved_r01/analysis_r01/source_state_quality_lock_noise.pdf)。
- [正式配对PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_rawAsMask_E30_20261019_r01/retrieved_r01/analysis_r01/formal_paired.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_rawAsMask_E30_20261019_r01/retrieved_r01/analysis_r01/formal_paired.pdf)，[实际maskPNG](E:/type10-7/local_artifacts/cvs_FcrossRX_rawAsMask_E30_20261019_r01/retrieved_r01/analysis_r01/raw_As_mask_readout.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_rawAsMask_E30_20261019_r01/retrieved_r01/analysis_r01/raw_As_mask_readout.pdf)。

未完成的科学目标：low urban>60%、跨seed稳定clean代价≤0.1pp、可重复星地与clean双提升。本轮没有缺失执行项；固定负候选不以扩预算补救。当前goal继续ACTIVE。
