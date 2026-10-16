# F＋跨 RX 同 TX：独立普通 IQ 观察与复用控制

三组 scratch E20 和全部七视图源域读出完成；固定独立普通 IQ 观察未通过预设条件，保留负结果并关闭。

训练代码 c72c359581cee36e9fc70440844803288ba851ef；离线复算 6443685b6bd14ed0d895be843d867db57f2fad2f（之后如有离线修复另记）。全部成绩来自 source held RX1 原 L1260／seed2026092701／FINAL20／默认 cosine 头，不是正式 target、跨 seed 或 Ag×As 结果。三组控制均为新训练，不用有未解释判决漂移的旧同 seed 控制代替。

## 逐 row／seed 七环境结果

| view | RAW % | REPLAY % | REFRESH % | REFRESH−RAW pp | REFRESH−REPLAY pp |
|---|---:|---:|---:|---:|---:|
| clean | 80.3175 | 81.4286 | 82.0635 | 1.7460 | 0.6349 |
| practical_high | 39.1270 | 36.6667 | 37.0635 | -2.0635 | 0.3968 |
| practical_mid | 39.6032 | 36.7460 | 37.2222 | -2.3810 | 0.4762 |
| practical_low_suburban | 36.2698 | 34.2063 | 34.2857 | -1.9841 | 0.0794 |
| practical_high_urban | 38.4921 | 36.5873 | 36.8254 | -1.6667 | 0.2381 |
| practical_mid_urban | 32.4603 | 31.0317 | 31.1111 | -1.3492 | 0.0794 |
| practical_low_urban | 27.9365 | 27.3016 | 27.2222 | -0.7143 | -0.0794 |

| view | RAW macro F1 % | REPLAY macro F1 % | REFRESH macro F1 % |
|---|---:|---:|---:|
| clean | 79.9414 | 81.0801 | 81.7507 |
| practical_high | 28.3045 | 28.2520 | 29.2606 |
| practical_mid | 29.5865 | 29.0866 | 29.8858 |
| practical_low_suburban | 28.3534 | 28.6393 | 28.5131 |
| practical_high_urban | 29.0724 | 29.0923 | 29.4290 |
| practical_mid_urban | 27.0797 | 26.8646 | 26.9864 |
| practical_low_urban | 24.6109 | 24.4648 | 24.5581 |

| row | urban mean % | low urban % | high/mid/low suburban 与自身 clean 的差 pp |
|---|---:|---:|---|
| RAW | 32.9630 | 27.9365 | 41.1905, 40.7143, 44.0476 |
| REPLAY | 31.6402 | 27.3016 | 44.7619, 44.6825, 47.2222 |
| REFRESH | 31.7196 | 27.2222 | 45.0000, 44.8413, 47.7778 |

## 固定条件与收益归因

原完整 base 保留：raw.75／原 ordinary.25／原 SupCon；三组同加 .25 RD(第四视图)，总 pair 权重1.25、每个 L 包8次前向。RAW 第四视图是 raw，REPLAY 是第一份原 ordinary 在 PA/DAC 前的缓存，REFRESH 是从同一 raw 独立抽取的原 anti-shortcut→general-channel。独立观察使用原算子、概率、范围、顺序、边界和 E13 截断日程，不新增 PA/DAC；共用 base 内原 PA 仍在。各对内部同 IQ RDrop，新增对没有 SupCon／跨视图对齐／新头或部署参数。原 U／EMA／旧门控保持。

raw 的原绝对.75权重保留，但总量1.25后相对占比变化已披露；RAW加额外raw后总raw1.00，另两组.75。所有组均构造相同私有 fresh shadow、恢复原 RNG；本轮区分额外监督预算与独立观察，不能将相对旧6L的任何增益直接归于新Ag。

固定要求：REFRESH 相对 RAW 和 REPLAY 均 clean≥＋1 pp、macro F1≥0 pp、seen V下降≤.5 pp；三组 final native L CE≤.5。完整 E20／4440更新／4546560 L NN／3330 SupCon steps／4440生成和消费／568320 fresh包／5040唯一L／72格、合法角色及有限更新必须满足。全部六 practical 必报，但不参与本次独立clean资格判断。

实际条件：{"RAW_clean": true, "RAW_macroF1": true, "RAW_seenV": true, "REPLAY_clean": false, "REPLAY_macroF1": true, "REPLAY_seenV": true, "healthy": true}。
相对 RAW：urban均值 -1.2434 pp；clean macro F1 1.8094 pp；seen V 0.0741 pp。
相对 REPLAY：urban均值 0.0794 pp；clean macro F1 0.6706 pp；seen V 0.0000 pp。

FAIL 关闭固定独立 ordinary 观察；不调整子集、强度、概率、权重、RX、seed、阈值或预算救援，不把控制组事后晋级。

## low urban 分层与默认头间隔

| 分层 | n | RAW % | REPLAY % | REFRESH % | REFRESH−RAW pp | REFRESH−REPLAY pp | REFRESH margin/30 |
|---|---:|---:|---:|---:|---:|---:|---:|
| state:LOS | 384 | 41.9271 | 40.3646 | 39.8438 | -2.0833 | -0.5208 | -0.0311 |
| state:BLOCKED | 714 | 18.9076 | 19.8880 | 19.7479 | 0.8403 | -0.1401 | -0.0581 |
| state:PARTIAL | 162 | 34.5679 | 29.0123 | 30.2469 | -4.3210 | 1.2346 | -0.0322 |
| lock:locked | 467 | 42.1842 | 39.8287 | 40.0428 | -2.1413 | 0.2141 | -0.0295 |
| lock:degraded | 228 | 21.4912 | 20.1754 | 19.7368 | -1.7544 | -0.4386 | -0.0540 |
| lock:unlocked | 565 | 18.7611 | 19.8230 | 19.6460 | 0.8850 | -0.1770 | -0.0576 |
| quality:[20,30) | 396 | 42.6768 | 40.6566 | 40.4040 | -2.2727 | -0.2525 | -0.0301 |
| quality:[5,10) | 157 | 19.7452 | 21.0191 | 21.0191 | 1.2739 | 0.0000 | -0.0597 |
| quality:[0,5) | 385 | 19.4805 | 20.2597 | 20.0000 | 0.5195 | -0.2597 | -0.0561 |
| quality:<0 | 180 | 17.2222 | 18.8889 | 18.8889 | 1.6667 | 0.0000 | -0.0608 |
| quality:[10,20) | 142 | 32.3944 | 26.7606 | 27.4648 | -4.9296 | 0.7042 | -0.0338 |

这是固定同批源 IQ／同物理状态的表现定位，不能唯一归因于噪声、多径、残余 CFO、RX IQ 或内部门控。六 practical 的实际传播、状态与补偿逐包完全相同；EQ关闭时G=1，residual不等于已均衡；quality为原白输入功率代理，新增噪声SNR不等于总SNR。没有重新生成弱测试或删除困难样本。本轮不训练As，未检验与As的组合交互。

## 固定 RAW clean 正确集合与错误变化

集合 n=1012，所有视图复用，不按 REFRESH 重选。

| view | RAW 保持率 % | REPLAY 保持率 % | REFRESH 保持率 % | 对 RAW 修复/损坏 | 对 REPLAY 修复/损坏 |
|---|---:|---:|---:|---|---|
| clean | 100.0000 | 99.3083 | 99.4071 | 28/6 | 15/7 |
| practical_high | 36.9565 | 34.7826 | 35.1779 | 26/52 | 25/20 |
| practical_mid | 37.3518 | 36.2648 | 36.4625 | 24/54 | 23/17 |
| practical_low_suburban | 35.0791 | 34.2885 | 34.0909 | 26/51 | 24/23 |
| practical_high_urban | 36.8577 | 35.7708 | 36.0672 | 17/38 | 18/15 |
| practical_mid_urban | 31.2253 | 29.9407 | 29.9407 | 33/50 | 20/19 |
| practical_low_urban | 27.6680 | 27.3715 | 27.3715 | 37/46 | 18/19 |

## 全量执行、覆盖与实际成本

| row | updates | L NN packets | 总 NN packets | fresh 构造/审计 s | native 训练 s | 到最终结果 s |
|---|---:|---:|---:|---:|---:|---:|
| RAW | 4440 | 4546560 | 5823318 | 3.1111 | 1416.1719 | 1435.1568 |
| REPLAY | 4440 | 4546560 | 5823318 | 3.1454 | 1413.6264 | 1432.3398 |
| REFRESH | 4440 | 4546560 | 5823318 | 3.3004 | 1534.9699 | 1553.9418 |

总计60 model-epochs／13320更新／17469956 NN包（含dispatcher真实CP兼容smoke2）。每个学生／ownEMA／optimizer均scratch，父训练成本0；旧B00 E20／4440更新只载入后作2个合成包smoke再丢弃，没有继承。NN扣除EMA复制时继承的2个初始化计数，保留实际初始化前向，held每组8820单列。矩阵首worker启动至dispatcher完成 1563.0698 s，进程start ticks及产物mtime独立核对，近似1 s精度。预算耗尽不等于收敛。

本地合成native runtime：10496原normal＋7680 fresh包、13负例、0NN／更新；真实CVS合成：4120NN／4更新、206初始state、RAW与原8L完整loss/grad/RNG逐位一致、10配置负例；离线18协议负例0NN。一次P0/P1审查PASS、N607单preflight、单archive传输及remotecompile通过。合成梯度变化不是识别收益。

| row | 最终原生 L CE | grad norm 均值/最大 | fresh 非同一 IQ 的 step 数 | 唯一 L/覆盖格 |
|---|---:|---|---:|---|
| RAW | 0.2641 | 0.9464/10.7528 | 2916/4440 | 5040/72 |
| REPLAY | 0.2469 | 1.1333/10.7528 | 2916/4440 | 5040/72 |
| REFRESH | 0.2403 | 1.0755/10.7528 | 2916/4440 | 5040/72 |

全部20轮／4440步／5994原L-U序列及全stdout复算，所有有效更新有限，U TX真值隐藏，没有新增U／teacher增强。原L/U序列与RAW逐一相同：{"REPLAY": true, "REFRESH": true}；原／fresh shadow全量保存统计相同：{"REPLAY": true, "REFRESH": true}。私有流只依赖固定seed＋step，不依赖TX/RX/day/quality/loss。

| REFRESH 原算子 | 原normal前PA触发% | fresh触发% | 原/fresh条件不同% |
|---|---:|---:|---:|
| p_dc_offset | 6.3063 | 6.3288 | 11.7793 |
| p_bandedge_taper | 4.6396 | 5.0901 | 9.2342 |
| p_time_shift | 6.3964 | 6.3288 | 11.9144 |
| p_amp_scale | 8.9414 | 8.7162 | 16.2162 |
| p_phase_rot | 9.0991 | 8.3784 | 15.5856 |
| p_cfo | 6.7568 | 6.3288 | 11.8243 |
| p_phase_noise | 5.4955 | 5.0901 | 10.0901 |
| p_multipath | 2.8378 | 3.8288 | 6.3514 |
| p_awgn | 7.6351 | 7.5901 | 13.9189 |

实际训练入口：原SSDG与DataAugmentation及leo_practical均来自已记录的adv3b02_xuc/code；新wrapper来自本run不可变release。完整actual args、resolved、默认头/Sinc、原RMS位置、普通增强实际配置与六practical配置随每row保留，默认值不能代替实际运行值。

## 图表、文件与边界

离线出图只读冻结汇总，0 IQ／CP／NN／更新。
- [source_scene_results PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_viewrefresh20_20261017/final/figures_v2/source_scene_results.png)；[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_viewrefresh20_20261017/final/figures_v2/source_scene_results.pdf)
- [complete_training_curves PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_viewrefresh20_20261017/final/figures_v2/complete_training_curves.png)；[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_viewrefresh20_20261017/final/figures_v2/complete_training_curves.pdf)
- [ordinary_observation_exposure PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_viewrefresh20_20261017/final/figures_v2/ordinary_observation_exposure.png)；[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_viewrefresh20_20261017/final/figures_v2/ordinary_observation_exposure.pdf)
- [low_urban_strata PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_viewrefresh20_20261017/final/figures_v2/low_urban_strata.png)；[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_viewrefresh20_20261017/final/figures_v2/low_urban_strata.pdf)
- [fixed_clean_retention_and_errors PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_viewrefresh20_20261017/final/figures_v2/fixed_clean_retention_and_errors.png)；[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_viewrefresh20_20261017/final/figures_v2/fixed_clean_retention_and_errors.pdf)

[独立汇总](E:/type10-7/local_artifacts/cvs_FcrossRX_viewrefresh20_20261017/final/analysis_summary.json)；完整training_curves.json、source_strata.json与逐row原记录同目录。raw IQ／权重未拉取或提交。run_id：20261017-phase1-FcrossRX-viewrefresh-sourceRX1-e20-s2026092701-m3-r01。

本轮只检验一个已暴露源 RX 和一个 model seed，无法报告最差目标 RX、正式默认头提升、正式clean代价或跨seed重现。固定正式 practical 条件保持，但尚无新的正式target结果；low urban>60%、suburban接近clean、新Ag确认和完整B00/B10/B01/B11交互均未完成。内部融合／门控权重未另作本轮读出，不能凭margin或梯度推断。活动目标继续。


NN计数分解（raw计数保留，EMA复制时继承的2个初始化eval计数在总成本中扣除；不是两次新前向）：

| row | 学生 train | 学生 eval | EMA train | EMA eval | held源评估 | 额外第四对L |
|---|---:|---:|---:|---:|---:|---:|
| RAW | 4942928 | 475202 | 0 | 396370 | 8820 | 1136640 |
| REPLAY | 4942928 | 475202 | 0 | 396370 | 8820 | 1136640 |
| REFRESH | 4942928 | 475202 | 0 | 396370 | 8820 | 1136640 |

完整曲线诊断：下列best seen V仅描述曲线，不改变FINAL20选模。

| row | 末5轮总L损失范围 | 末5轮seen V范围% | 末5轮V首末差pp | 诊断best V epoch |
|---|---|---|---:|---:|
| RAW | 0.3328 至 0.3644 | 98.0000 至 98.0370 | -0.0185 | 12 |
| REPLAY | 0.3710 至 0.4168 | 98.0509 至 98.0880 | 0.0324 | 20 |
| REFRESH | 0.3708 至 0.4085 | 98.0694 至 98.1111 | -0.0046 | 12 |

REFRESH相对两控制的最早训练分歧（完整step记录）：{"RAW": {"first_extra_CE_difference_step": 2, "first_total_L_difference_step": 3, "first_gradient_preclip_difference_step": 1}, "REPLAY": {"first_extra_CE_difference_step": 2, "first_total_L_difference_step": 3, "first_gradient_preclip_difference_step": 1}}。完整stdout中的warning行、禁用指标NaN提及数和epoch marker保留在analysis_summary.json；有效loss/gradient有限性另行核验，不把禁用项NaN视为训练崩溃。


## 实际执行补充审计

工作起点 commit b927f01718eb8b4777ec4748f5e27dbb63f657b2。本轮训练 commit c72c359581cee36e9fc70440844803288ba851ef；F／SIG_TIME_X3／统一 Sinc CENTER_UNITS／真实 joint160／NoExplicitCFO／defaultcos30与原CROSS_RX保持。没有父权重初始化；旧匹配B00仅用于兼容smoke，继承链与source角色已核对。

实际模块不是仓库顶层code，而是固定release中的experiments/adv3b02_xuc/code/SSDG/train_ssdg.py、DataAugmentation.py、leo_practical/channel.py及residual.py；完整__file__见每row actual_modules.json和refresh_actual_modules.json。25MHz、IQ2×256、原loader RMS位置保留；新增视图没有后置RMS、重叠完整practical或新增PA/DAC。原normal调用仍包含原PA，DAC与class_signature实际关闭。ordinary_augmentation_actual.json内no_second_pair_redraw指原有同IQ RDrop pair；本轮第四对是否刷新由refresh_steps与refresh_audit单独完整记录。

六种practical逐row实际配置与逐包传播／接收状态完全相同：residual／post_sync／EQ关闭G=1、reference_snr_db30、quality_adaptation／phase_tracking／IQ_compensation开启、AGC上限30dB、reflect边界。quality locked/unlocked阈值15/5dB、quality reference20dB、max residual scale10；接收机CFO std200Hz／residual CFO std100Hz、phase noise linewidth1Hz／tracking residual1deg，IQ注入上限0.2dB/1deg及估计误差0.05dB/0.2deg。EQ配置中的mmse／length129／delay64属于关闭功能，不能据此声称已均衡。观察原点／FIR群延迟与有效观察范围没有另做本轮移除控制；保留full physics中的kernel/boundary与原固定数值实现，不推断它们为唯一原因。

原普通增强base probabilities不是实际每步概率。E13截断日程下，实际p_dc_offset3.00%至7.5180%、taper2.50%至6.2650%、time_shift/CFO3.50%至8.7710%、amp/phase4.50%至11.2770%、phase_noise3.00%至7.5180%、multipath1.80%至4.5108%、AWGN4.00%至10.0240%；逐步值和触发次数全量保存。新观察只使用这些原native阶段。

首步三组记录的grad norm存在约1e-8量级差异，原／fresh保存IQ统计和L/U序列则全程相同。本轮没有在真实GPU训练宣称所有梯度逐位复现；此前逐位loss/grad/RNG一致是合成RAW控制检查的结论。默认原数值策略cudnn_deterministic=false，跨独立run的微小漂移来源未被本轮唯一定位，不能由一个source seed宣称重现。

最终图使用figures_v2，负增益标注留足边距；初版图保留。末段V汇总已纠正原生百分比单位（98.xx即%，不再乘100），这不改变固定判断、全部source预测或七视图分数。出图及离线修复均0IQ／CP／NN／更新；训练与预测没有重跑。


## 已执行命令

训练使用预登记的 N607 dispatcher，CWD 与不可变 release 路径见预登记；它先真实CP smoke，再独立启动三组，完成后逐row自动七视图源评价。以下离线命令的 CWD 为 E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014：

```text
F:/App/miniconda3/python.exe -m experiments.cvs_f_viewrefresh20.checks_analysis
F:/App/miniconda3/python.exe -m experiments.cvs_f_viewrefresh20.harvest --output E:/type10-7/local_artifacts/cvs_FcrossRX_viewrefresh20_20261017/final
F:/App/miniconda3/python.exe -m experiments.cvs_f_viewrefresh20.analyze --input E:/type10-7/local_artifacts/cvs_FcrossRX_viewrefresh20_20261017/final
F:/App/miniconda3/python.exe E:/type10-7/local_artifacts/cvs_FcrossRX_viewrefresh20_20261017/plot_local_v2.py
```

既有 run/output 保持不可变；这些是实际执行记录，不表示对同一 root 重新启动训练。运行代码/config与离线实现分别提交，最终Git报告提交另记。
