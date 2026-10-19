# 完整 As 后前缀屏蔽：固定 E30 配对结果

**结论：前缀屏蔽配方失败，不保留，不扩 E50／种子。** 相对原完整 As，正式 mid urban −1.420833、low urban −1.077381、urban 均值 −1.315873 个百分点；clean −0.191071 个百分点。随机等长屏蔽也损害全部六种 practical。不能把它们相对 NoAs 的大幅提升算成屏蔽机制收益。

本次只有模型 seed 2026092701。每个新行 E10／2220 更新，严格从同一个原 P43 B01 As20 student＋自身 EMA 独立开始，累计 E30／6660。原 NoAs30 保留自身 NoAs20 父轨迹，原 As30 与两个新行共享 As20 来源；两旧行只复用既有 prediction／评分，未重新预测或评分。

## 固定条件与实际代码

运行 20261019-phase1-FcrossRX-asPrefix-e30-s2026092701-m2-r01。训练代码 commit 14dbca26d47fdbe180bc5983cea1ed336593c29e；独立源分层工具 commit eff2547bd6cb29338ca10b521f0672e80691d0f5。起点 2467287bec9a9d9d078fafd30df526ce3bade930。实际命令为 CVS-RFFI/python -u experiments/cvs_as_prefix81/runner.py，CWD 为对应不可变 release。

F＋跨 RX 同 TX SupCon、SIG_TIME_X3、统一 Sinc、joint160、默认 cosine6×160 scale30，900722 参数／206 state；骨干、内部门控和默认头按原流程联合训练，无额外部署头。AdamW 重置，lr0.0002／weight_decay0.0001／cos10。原六次 L 前向、1.25 总系数、普通/raw/U/EMA 和同 IQ RDrop 不变。

L6300／U56700 TX隐藏／V27000，split392005，源 RX1/3/4/6/8、day1/2/3，全源无留RX。数据和正式 IQ 复用原 VALIDATED_ONCE。源单包256点、25MHz、eq1原逐包RMS，As从raw经native normalize_input再完整传播与接收处理。实际模块来自固定 r02/experiments/adv3b02_xuc/code/leo_practical；不是顶层code。训练fc2.462GHz，residual/post_sync，EQfalse即G1，reflect/referenceSNR30，质量适配／相位跟踪／IQ补偿／AGC均true，AGC限30dB，fractional-delay半长24。源固定诊断cache fc2.45GHz差异保留。正式测试字节、SNR、信道和样本未改。

PREFIX 在完整 practical＋AGC 后，以概率.5将[0,61)两路IQ归零；RANDOM 同概率／同启用流，在[0,19]∪[160,195]均匀选等长段起点。61来自实际分配核62−1，是统一历史上界，不是每状态精确影响宽度。参考[80,160)直接样本保留，但GroupNorm／功率比仍变。无重新归一化或状态重算；全部合法As仍参与原CE，不以学生置信度过滤。

## 默认头正式结果

每格准确率为%；每环境168000个固定物理样本。urban均值为三个urban场景的等权均值。

| 环境 | NoAs30 | 原 As30 | 随机屏蔽 | 前缀屏蔽 | 前缀−原As（百分点） |
|---|---:|---:|---:|---:|---:|
| clean | 85.247619 | 85.341071 | 85.431548 | 85.150000 | -0.191071 |
| practical_high | 42.685119 | 81.405357 | 80.492262 | 79.878571 | -1.526786 |
| practical_mid | 40.995238 | 78.661310 | 77.735119 | 77.173214 | -1.488095 |
| practical_low_suburban | 39.472619 | 75.861905 | 75.152976 | 74.517262 | -1.344643 |
| practical_high_urban | 41.439881 | 77.767262 | 76.919048 | 76.317857 | -1.449405 |
| practical_mid_urban | 35.132738 | 62.806548 | 61.896429 | 61.385714 | -1.420833 |
| practical_low_urban | 30.139881 | 51.244048 | 50.545238 | 50.166667 | -1.077381 |
| urban_mean | 35.570833 | 63.939286 | 63.120238 | 62.623413 | -1.315873 |

前缀clean85.150000%，相对匹配NoAs85.247619%下降0.097619个百分点，**正式数值上满足≤0.1个百分点阶段限制**，仅余0.002381个百分点。相对NoAs，mid/low/urban均值仍有+26.252976／+20.026786／+27.052579个百分点，但这是已有完整As适应的主要收益；原As同时有更好的clean与六种practical。

随机屏蔽clean85.431548%，相对原As仅+0.090476个百分点、相对NoAs+0.183929个百分点；其urban均值相对原As−0.819048、low−0.698810。该对照不能因正式clean上涨被改选成下一候选。没有跨模型seed的可重复双提升证据。

### F1 与最差 RX

以下均为%；urban的macro-F1为三场景F1均值，最差RX为三urban合并后每RX准确率的最小值；pooled F1另存完整JSON。

| 行 | clean F1 | clean最差RX | mid F1 | mid最差RX | low F1 | low最差RX | urban F1 | urban最差RX |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NOAS30 | 84.979636 | 78.450000 | 33.446603 | 29.495833 | 29.184589 | 25.750000 | 33.704074 | 29.748611 |
| AS30 | 85.044474 | 78.362500 | 62.450463 | 56.458333 | 50.887035 | 46.379167 | 63.586902 | 57.315278 |
| RANDOM30 | 85.128082 | 78.591667 | 61.377084 | 55.016667 | 49.976711 | 45.370833 | 62.597643 | 55.976389 |
| PREFIX30 | 84.822389 | 77.920833 | 60.691340 | 53.929167 | 49.221627 | 44.937500 | 61.886790 | 54.929167 |

全部7环境×4逻辑行的accuracy／macro-F1／最差RX见 [完整指标CSV](E:/type10-7/local_artifacts/cvs_FcrossRX_asPrefix_E30_20261019_r01/retrieved_r01/analysis_r01/formal_all7_metrics.csv)；逐RX/day/TX评分保存在两新行scoring/scores.json。

### 修复与新增错误

| 前缀相对原As | 修复 | 新增错误 | 净正确数变化 |
|---|---:|---:|---:|
| clean | 1028 | 1349 | -321 |
| practical_mid_urban | 2513 | 4900 | -2387 |
| practical_low_urban | 3792 | 5602 | -1810 |
| urban_mean | 7827 | 14459 | -6632 |

固定NoAs30 clean正确集合共143216个正式样本。low urban保持率：原As55.632751%、前缀54.953357%；这是固定NoAs30集合，**不是As20父模型clean正确集合**。后者无现成预测cache，本轮未增加父模型前向来补齐。

## 源证据与归因

读取新正式汇总前已经观察并保存源判定：三项均FAIL。前缀 fullV clean−NoAs −0.133333个百分点，panel midlow−原As −0.888889，midlow−随机 −0.125000；因此固定配方已关闭。正式clean刚好过线不推翻源失败或触发晋级。

源面板1800个独立物理样本、两次重复信道draw，绝非两个模型seed。前缀−原As：mid两个draw均−1.000000个百分点；low两draw为−0.888889／−0.666667。

| 源前缀−原As，合并两draw | mid百分点 | low百分点 |
|---|---:|---:|
| state:LOS | -0.473934 | -0.660377 |
| state:PARTIAL | -0.544959 | -0.178253 |
| state:BLOCKED | -1.874063 | -1.010611 |
| lock:locked | -0.544218 | -0.438917 |
| lock:degraded | -1.678322 | -1.199400 |
| lock:unlocked | -1.764706 | -0.893997 |
| signal_to_added_noise_db:<3 | -2.872063 | -0.266193 |
| signal_to_added_noise_db:3-8 | -1.253482 | -1.763224 |
| signal_to_added_noise_db:8-15 | -1.365188 | -1.650165 |
| signal_to_added_noise_db:>=15 | -0.543971 | -0.436047 |

主要状态／锁定／新增噪声区间都没有总体净修复。mid BLOCKED净−25/1334，low BLOCKED净−20/1979；不能认为人工前缀屏蔽缓解了BLOCKED或unlocked。子区间间隔略升也没有替代实际准确率改善。质量分层使用原完整观测条件，signal_to_added_noise_db不是总SNR；状态/质量组彼此重叠，不能把这些效应相加。

两行各生成142080个完整As，启用屏蔽70906个（49.905687%），TX×RX×场景×状态270格全部覆盖，每格至少75。两组启用数及物理覆盖一致；整体移除能量占比随机12.495101%、前缀12.177425%，所以等长并不等能量。前缀损失更大不能解释为“只是移除更多能量”，也不能据此作唯一因果归因。

这轮否定的是固定p=.5/61点/post-AGC配方。继续保留完整As；不把源P79边界敏感性当作应删除这段观测的证明。此前质量风险、U路由、共享传播、独立视图及对齐固定配方已有负结果，当前局部屏蔽同样未提供新增星地收益。

## 成本与验证

新增两E10共4440次更新，native前向样本6325924，源读出25200、正式预测2352004、共享真实CP/6L smoke1540，合计8704668。两个新行各累计E30/6660；未追加E50或模型seed。共享As20父历史成本4797722个native前向、1810.249311秒，仅复用一次；每个新模型路径父+尾native共7960684。旧NoAs30/As30及既有预测评分没有重做。

新训练过程耗时随机1163.827704秒、前缀1163.052572秒，进程时间之和2326.880277秒；两GPU并行，launch→自动分析完成1260.977196秒（21.016287分钟），不能将进程时间之和当elapsed。自动分析13.295411秒，额外源分层2.558151秒／0NN。源12600读出独立墙钟原artifact未记录。预算耗尽不等于收敛。

必要验证：mask首跑7PASS/1测试桥接ABI错误，frombuffer修复后3项相关PASS；17协议负测试与AST PASS；唯一独立P0/P1 PASS；远端真实父CP/no-query及两种真实6L路径、E21/E30信道调用PASS1540NN/0更新；archive传输与compile及PID/CWD/argv/GPU/log增长VERIFIED。两行七prediction固定后独立truth-last评分；1120个最终指标／配对记录由正确数和混淆矩阵复核PASS，260个空分层保留；五PNG已查看、五PDF头有效，运行进程均已退出。

精确测试与CLI记录在本地工件目录，发布包下runner.py自动串联训练→冻结→源读出→七预测→独立scorer→分析。

## 图表与原始聚合表

- [正式七环境配对差](E:/type10-7/local_artifacts/cvs_FcrossRX_asPrefix_E30_20261019_r01/retrieved_r01/analysis_r01/formal_all7_paired.png)，同目录有PDF。
- [全量训练曲线](E:/type10-7/local_artifacts/cvs_FcrossRX_asPrefix_E30_20261019_r01/retrieved_r01/analysis_r01/training_full_curves.png)，同目录有PDF。
- [屏蔽覆盖与能量](E:/type10-7/local_artifacts/cvs_FcrossRX_asPrefix_E30_20261019_r01/retrieved_r01/analysis_r01/mask_coverage_energy.png)，同目录有PDF。
- [源RX分层](E:/type10-7/local_artifacts/cvs_FcrossRX_asPrefix_E30_20261019_r01/retrieved_r01/analysis_r01/source_mid_low_RX.png)，同目录有PDF。
- [源状态及噪声分层](E:/type10-7/local_artifacts/cvs_FcrossRX_asPrefix_E30_20261019_r01/retrieved_r01/source_strata_r01/source_state_quality.png)，同目录有PDF。

完整表包含120个epoch曲线记录、4440个逐更新reference记录、源RX/day/TX指标600行和配对900行；新增状态指标1300行/配对650行。全部落在本地retrieved_r01与远端原run，未下载IQ、权重或原始logits。

## 未完成与明确答案

训练、评分及计划图表均已完成。测量缺项：源读出独立墙钟未记录。科学缺项：仅一模型seed，未证明可重复双提升；low urban未超过60%；suburban与clean缺口仍在；As20父clean-correct集合cache未补，新屏蔽的分支/内部门控梯度未另作前向诊断；正式逐包物理状态元数据不在现有输出。源缓存对训练载频差异及源读出独立墙钟缺失均如实保留。

固定CVS和固定practical测试下，本轮没有缓解可稳定归因的主要urban退化。默认头相对原As的low/mid/urban均值分别下降1.077381/1.420833/1.315873个百分点。clean相对NoAs下降0.097619个百分点，满足本seed的0.1阶段数值要求；相对原As下降0.191071个百分点。没有可重复“星地增强同时改善clean”的新证据，目标仍ACTIVE。
