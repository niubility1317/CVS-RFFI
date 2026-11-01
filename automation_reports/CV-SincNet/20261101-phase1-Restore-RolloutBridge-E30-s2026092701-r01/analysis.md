# RolloutBridge：原 Main 七景性能结果

**六景达标 0/6，合并 clean 保持。** 原地面完整 Tri-SIG-TIME SSL/default head/Main split，raw clean 86.538690%（145385/168000），各景至少 83.538690%（140345条正确）。

cross_rx clean 净变化 +41 条，cross_both 净变化 +20 条。两个总体分别报告。

六景中 6 景高于 StateBridge，0 景高于 DURA；低档城区距3pp线仍差 36942 条正确预测。

本版尚未达到主性能门槛，目标保持 ACTIVE。默认DURA不变，源端均值和计算成本不替代真实身份性能。

![主性能](E:/type10-7/automation_reports/CV-SincNet/20261101-phase1-Restore-RolloutBridge-E30-s2026092701-r01/primary_performance.png)

| 场景 | DURA | StateBridge | RolloutBridge | 距raw clean | 相对对照 | 相对DURA |
|---|---:|---:|---:|---:|---:|---:|
| clean | 86.615476% | 86.107143% | 86.575000% | -0.036310 pp | +0.467857 pp | -0.040476 pp |
| 高档郊区 | 84.343452% | 71.752976% | 75.960119% | 10.578571 pp | +4.207143 pp | -8.383333 pp |
| 中档郊区 | 83.016071% | 70.942262% | 73.929762% | 12.608929 pp | +2.987500 pp | -9.086310 pp |
| 低档郊区 | 81.322024% | 69.885714% | 71.875595% | 14.663095 pp | +1.989881 pp | -9.446429 pp |
| 高档城区 | 83.049405% | 70.691667% | 74.461905% | 12.076786 pp | +3.770238 pp | -8.587500 pp |
| 中档城区 | 75.705952% | 65.321429% | 67.470238% | 19.068452 pp | +2.148810 pp | -8.235714 pp |
| 低档城区 | 68.801190% | 59.635119% | 61.549405% | 24.989286 pp | +1.914286 pp | -7.251786 pp |

## 两个总体与 clean

| 总体 | 达标景数 | raw clean正确数 | 新G clean正确数 | clean净变化 |
|---|---:|---:|---:|---:|
| cross_rx | 0/6 | 111019/126000 | 111060/126000 | +41 |
| cross_both | 0/6 | 34366/42000 | 34386/42000 | +20 |
| combined | 0/6 | 145385/168000 | 145446/168000 | +61 |

all_score_groups.csv包含全部TX/RX/day/RXday、准确率与macro-F1（比例0至1）；combined_group_metrics.csv按互斥总体混淆计数合并，未平均百分比。paired_changes.csv列DURA/对照净纠错，paired_clean_changes.csv列相对同记录raw clean的纠错/损伤/判决保持。完整数据均来自冻结后独立评分，不用于改选checkpoint或超参数。

## 同条样本修复与损伤

| 对照 | 场景 | 修复错误 | 损伤原正确 | 净变化 |
|---|---|---:|---:|---:|
| DURA_v1 | clean | 596 | 664 | -68 |
| DURA_v1 | 高档郊区 | 4544 | 18628 | -14084 |
| DURA_v1 | 中档郊区 | 5485 | 20750 | -15265 |
| DURA_v1 | 低档郊区 | 6682 | 22552 | -15870 |
| DURA_v1 | 高档城区 | 5118 | 19545 | -14427 |
| DURA_v1 | 中档城区 | 8739 | 22575 | -13836 |
| DURA_v1 | 低档城区 | 12003 | 24186 | -12183 |
| MSAT_StateBridge | clean | 1527 | 741 | +786 |
| MSAT_StateBridge | 高档郊区 | 13548 | 6480 | +7068 |
| MSAT_StateBridge | 中档郊区 | 12650 | 7631 | +5019 |
| MSAT_StateBridge | 低档郊区 | 12034 | 8691 | +3343 |
| MSAT_StateBridge | 高档城区 | 13707 | 7373 | +6334 |
| MSAT_StateBridge | 中档城区 | 14838 | 11228 | +3610 |
| MSAT_StateBridge | 低档城区 | 17458 | 14242 | +3216 |

## 方法检验与边界

本版保持StateBridge/c50的4676898参数、全部从零随机初值、原Ls6300/Us56700/R07、sourceorder/draw、优化器/LR/损失权重/30-7410及四步部署公式。唯一研究机制改为：训练时也从received起点真实展开四步，用最终输出的损失反传贯穿所有状态，不再用clean/received混合状态作网络输入，也不再随机抽取time。这改变训练状态分布、梯度轨迹和计算量，是系统比较，不能称相同训练成本的单因素结果。

源依据来自固定512×20轨迹诊断：StateBridge源低档城区F(clean)一致率86.350975%→81.615599%，IQ NMSE .07612306→.11645158。后续迭代损伤已在源端出现；这不是TX准确率，也未证明某个Main变化的原因。新方法检验训练/部署状态不一致的假设，不据Main误分类改变参数或选模。[原源轨迹](../20261101-diagnostic-Restore-bridge-trajectory-source-s2026092701-r01/analysis.md)。

输入为当前FP32 IQ2×256，估计器只读state/time/坐标；received仅初始化state。共享网络调用4次，remaining4/3/2/1，z_next=z+(predictedclean−z)/remaining，最后直接predictedclean；训练与部署算术相同，四步不detach。无clean/原F0/真实信道/SNR/身份或额外观测作为部署输入，无跨query统计。每次预测单位复功率，中间state不另投影。

最终一步只计算一次IQ NMSE+.05STFT+.1(outputF0cos+源方差标准化MSE)+.1(lastlatentcos+同尺度MSE)+.1(raw/同条clean RMS−1)²。clean、连续160源特征、方差及RMS只作loss，teacher/defaulthead冻结。没有TX/logits/CE/原型/新分类头，V/OE/novel/目标不训练、早停或选模，OE只冻结诊断；方法为F-guided，G无旧G/coarse/DURA/IQ教师祖先。

模型seed2026092701、数据顺序2026110102、R07 20draw+前10、B256tail24/247步、AdamW/.001-warm247-cos2e-5/WD1e-4/globalclip1保持。CUDAseed2026110103为复现重置并保存，实际无bridge时间抽样。固定final30后自动两个Main总体各七景，全部prediction固定后独立CPU truth-last/共享原锁，raw/DURA同IQ-ID判决精确复用，新G全量四步。单seed、已曝光基准、未知设备保真仍是证据边界，默认DURA不变。

![完整训练曲线](E:/type10-7/automation_reports/CV-SincNet/20261101-phase1-Restore-RolloutBridge-E30-s2026092701-r01/full_training_curves.png)

曲线的State对照是t=1的单core随机子集，Rollout新G是全样本真实四步最终输出，不能把二者直接视作同一冻结部署测量。source_training_metrics.csv同时保留对照整体mixed-state目标与t1子集，新G只有actual-final目标；源均值不代替Main真实TX验收。

## 完整训练与成本

| 项目 | StateBridge | RolloutBridge |
|---|---:|---:|
| 唯一参数 | 4676898.000000 | 4676898.000000 |
| 训练秒数 | 2066.574040 | 6110.924673 |
| worker秒数 | 2075.069424 | 6117.282148 |
| 轮时中位数（秒） | 67.825463 | 202.923442 |
| 峰值显存（MiB） | 2435.224609 | 7710.527832 |

部署结构与4×260633600=1042534400实数Conv/Linear/attention MAC完全相同、0FFT/CG；复用State同函数的静态计算，不是本次新的隔离延迟测量。F0、归一化、激活、FiLM逐点、softmax、投影、更新和内存未计。新G训练4core/记录，共7560000 core行；对照1core/记录，均1890000在线F0输出特征及63000源clean参考。四倍core不等于总墙钟恰为四倍，30轮不等于同训练计算量或系统全部历史成本。

独立检查84单元/2604组/1736raw配对/1302额外配对、868相对raw clean配对；最大指标误差1.2e-14，raw/DURA判决差0。覆盖先于truth；60完整训练记录/stdout、实际E0与State逐tensor相同、sourceorder/draw/LR/weightedloss、实际4core/最终loss协议、30技术快照及E30-final一致性核实。复算0新神经前向/更新。

## 识别增强与恢复比例

| 模型 | 场景 | 修复clean原错误 | 损伤clean原正确 | 净变化 | Fclean判决一致率 |
|---|---|---:|---:|---:|---:|
| MSAT_StateBridge | clean | 735 | 1460 | -725 | 98.561905% |
| MSAT_StateBridge | 高档郊区 | 6225 | 31065 | -24840 | 76.923214% |
| MSAT_StateBridge | 中档郊区 | 6284 | 32486 | -26202 | 75.942262% |
| MSAT_StateBridge | 低档郊区 | 6388 | 34365 | -27977 | 74.617262% |
| MSAT_StateBridge | 高档城区 | 6186 | 32809 | -26623 | 75.753571% |
| MSAT_StateBridge | 中档城区 | 6483 | 42128 | -35645 | 69.294643% |
| MSAT_StateBridge | 低档城区 | 6585 | 51783 | -45198 | 62.698810% |
| MSAT_RolloutBridge | clean | 574 | 513 | +61 | 99.269643% |
| MSAT_RolloutBridge | 高档郊区 | 5312 | 23084 | -17772 | 82.339881% |
| MSAT_RolloutBridge | 中档郊区 | 5507 | 26690 | -21183 | 79.908333% |
| MSAT_RolloutBridge | 低档郊区 | 5591 | 30225 | -24634 | 77.528571% |
| MSAT_RolloutBridge | 高档城区 | 5376 | 25665 | -20289 | 80.602976% |
| MSAT_RolloutBridge | 中档城区 | 5839 | 37874 | -32035 | 72.327976% |
| MSAT_RolloutBridge | 低档城区 | 6043 | 48025 | -41982 | 65.350595% |

低档城区准确率61.549405%，相当于raw clean准确率的71.123568%。这是准确率保留比例，不是IQ或发射机信息恢复率，也不是可达上限。

恢复IQ可改变识别线索，修复原clean错误和损伤原正确必须同时报告。源Fclean一致、波形NMSE和真实TX准确率含义不同；不能用闭集收益单项证明未知设备身份被保留。[理论边界](../20261101-phase1-Restore-FeatureCond-E30-s2026092701-r01/theoretical_limits.md)。
