# F＋跨 RX 当前底座：冻结 practical 源诊断

**ANALYZED_SOURCE_ONLY / VERIFIED**。实际代码 `398897a01ba161f02f9535ad8e6db597420ab4a2`。两个既有 own-student final20 默认头，模型 seed2026092701；诊断原源 RX1 L1260，均排除在两个父训练／V 角色外。worker／dispatcher 自然结束，全部预测完成后独立新进程汇总，无目标数据或 truth 服务。

C0 为匹配四前向 ordinary 参照；C1 为 .5 ordinary＋.5 原 practical 配对 CE。二者共同保留 F、跨 RX 同 TX 对比、RDrop、旧 U 门控／own EMA、cos 和 joint160／cosine30 默认头。这个源 L1260 诊断的样本及信道流与首折源 V5400 评价不同，不能混报同一成绩。

| 原视图 | C0 accuracy | C1 accuracy | 差 pp | C0 macro-F1 | C1 macro-F1 |
|---|---:|---:|---:|---:|---:|
| clean | 82.6190% | 86.1111% | +3.4921 | 82.3415% | 85.9331% |
| practical_high | 38.5714% | 63.7302% | +25.1587 | 30.2355% | 61.8733% |
| practical_mid | 36.2698% | 61.5079% | +25.2381 | 30.1710% | 59.6536% |
| practical_low_suburban | 33.9683% | 58.7302% | +24.7619 | 28.8245% | 57.1964% |
| practical_high_urban | 35.8730% | 59.3651% | +23.4921 | 28.9582% | 57.7254% |
| practical_mid_urban | 29.5238% | 46.5873% | +17.0635 | 26.0188% | 45.8086% |
| practical_low_urban | 26.3492% | 38.3333% | +11.9841 | 23.7713% | 37.8258% |

urban 三场景均值 C0=30.5820%，C1=48.0952%，差 +17.5132 pp。这是单源折／单模型 seed 的源域读数，不是目标提升或跨 seed 重现。

| 模型／相对 full 的效应（pp） | high urban | mid urban | low urban |
|---|---:|---:|---:|
| C0 no_noise/fixed | -0.0794 | +0.1587 | +0.9524 |
| C0 no_noise/total | +0.3175 | +1.5873 | +0.7937 |
| C0 flat_H/fixed | +0.1587 | +1.9841 | +0.1587 |
| C0 flat_H/total | +0.0794 | +1.6667 | +0.0794 |
| C0 no_carrier/fixed | +0.1587 | +1.1111 | -0.3175 |
| C0 no_carrier/total | +0.1587 | +1.1111 | -0.3175 |
| C0 no_RX_IQ/fixed | +0.0794 | +0.1587 | +0.1587 |
| C0 no_RX_IQ/total | +0.1587 | +0.1587 | +0.1587 |
| C0 actual_power_quality | -0.0794 | +0.1587 | -0.5556 |
| C0 edge_boundary | -8.2540 | -4.2063 | -2.5397 |
| C1 no_noise/fixed | +1.1905 | +3.3333 | +5.1587 |
| C1 no_noise/total | +1.3492 | +2.6190 | +4.5238 |
| C1 flat_H/fixed | +0.4762 | +0.7937 | +0.4762 |
| C1 flat_H/total | +0.6349 | +0.4762 | +0.3968 |
| C1 no_carrier/fixed | -0.1587 | +1.5873 | +0.4762 |
| C1 no_carrier/total | -0.1587 | +1.5873 | +0.4762 |
| C1 no_RX_IQ/fixed | -0.1587 | -0.0794 | +0.3175 |
| C1 no_RX_IQ/total | -0.0794 | -0.0794 | +0.3175 |
| C1 actual_power_quality | -0.0794 | +0.0794 | +0.0000 |
| C1 edge_boundary | -8.8889 | -2.9365 | -3.3333 |

fixed 保留 full 接收状态、补偿、EQ 与 AGC；total 移除该损伤后重算。full 队列固定，各条件不重新分组。移除后仍保留的损伤不同，效应不可相加；C0 与 C1 的效应大小差异也不等于训练的唯一因果机制。flat_H 保留数值延迟和共同核相位；no_carrier 是联合频率／相位移除。no_noise 仍有原录音未知噪声，新增噪声 SNR 不等于总 SNR。

| low urban full 队列 | n | C0 | C1 | 差 pp |
|---|---:|---:|---:|---:|
| state/LOS | 396 | 37.8788% | 64.3939% | +26.5152 |
| state/PARTIAL | 169 | 26.6272% | 44.3787% | +17.7515 |
| state/BLOCKED | 695 | 19.7122% | 22.0144% | +2.3022 |
| lock/locked | 494 | 36.4372% | 61.7409% | +25.3036 |
| lock/degraded | 226 | 19.0265% | 26.5487% | +7.5221 |
| lock/unlocked | 540 | 20.1852% | 21.8519% | +1.6667 |
| quality/<5 | 540 | 20.1852% | 21.8519% | +1.6667 |
| quality/5..15 | 226 | 19.0265% | 26.5487% | +7.5221 |
| quality/>=15 | 494 | 36.4372% | 61.7409% | +25.3036 |

这组固定源数据中，paired practical 的收益主要集中在可用性较好的 LOS／locked，BLOCKED／unlocked 的净改善较小。这是分层关联，尚不能定位唯一物理因果。C1 low urban 移除新增噪声后仍远低于其 clean，其他单项控制也未恢复 clean；不据此把 AWGN、更强多径、Doppler 或 gate 当作唯一主因。

| low urban full / TX | n | C0 | C1 | 差 pp |
|---|---:|---:|---:|---:|
| 0 | 210 | 12.8571% | 25.2381% | +12.3810 |
| 1 | 210 | 2.8571% | 53.3333% | +50.4762 |
| 2 | 210 | 39.0476% | 18.5714% | -20.4762 |
| 3 | 210 | 33.8095% | 30.4762% | -3.3333 |
| 4 | 210 | 50.9524% | 60.0000% | +9.0476 |
| 5 | 210 | 18.5714% | 42.3810% | +23.8095 |

固定 C0 clean 正确集合 n=1041，low urban 保持率由 26.6090%（277 包）变为 43.4198%（452 包）。各模型及共同 clean 正确集合、每 TX／day、quality、CFO 与 AGC 分层完整保留在私有 summary.json。TX2／TX3 的下降没有被总体正收益掩盖。

| low urban 移除／重算 | C0 修复／新错 | C1 修复／新错 |
|---|---:|---:|
| no_noise/fixed | 115／103 | 180／115 |
| no_noise/total | 118／108 | 177／120 |
| flat_H/fixed | 97／95 | 94／88 |
| flat_H/total | 97／96 | 96／91 |
| no_carrier/fixed | 52／56 | 68／62 |
| no_carrier/total | 52／56 | 68／62 |
| no_RX_IQ/fixed | 6／4 | 4／0 |
| no_RX_IQ/total | 6／4 | 4／0 |
| actual_power_quality | 1／8 | 5／5 |
| edge_boundary | 103／135 | 142／184 |

实际 kernel lengths=[62]、numerical delays=[24]、interior counts=[195]；前缀无关内部区最大边界误差 0.0。full low urban AGC 增益触顶 1/1260。white-input 与实际波形功率差 {'minimum': -1.9414767956207664, 'maximum': 2.088195020285518, 'mean': 0.011315907608893285} dB；actual_power_quality 状态变化见 channel_transitions.json。边界控制只用于诊断，正式 reflect／IQ／SNR／困难样本均保持。

真实 time／frequency／PA／reference／三个 SIG 描述及现有动态门控构成的 joint160 与原 logits 重建通过。默认头 logits／30 统一尺度；每包激活 norm、与自身 clean 的余弦、间隔，以及原 14 参数 CE 导数均有完整记录。相似度、间隔或梯度非零不替代身份准确率，静态 learned gain 不冒充随场景变化的 gate。原网络参数和 buffer 前后逐值相等，无 optimizer 或另训头。

新增 0 模型／0 epoch／0 optimizer、EMA、head-fit 更新。两模型各 84422 backbone 网络包（含 2 个真实 checkpoint no-query smoke），合计 168844；2680 次 autograd.grad；1260 个唯一源物理包、7560 个场景 draw。每个 traced packet 另有一次只读 classify 重建，共 168840 次，不重复骨干。worker 墙钟 253.775 s；独立后处理另计。既有两个父各 E20／4440 更新，父训练全曲线见原报告，不计为本次新训练。

一次独立只读 P0/P1、12 协议负例／14 父来源负例、合成 12 网络包／2 导数／6 物理 draw66 波形、隔离发布导入与指标 fixture、compile／登记、一次普通账户 preflight、单 archive 传输校验／remote compile、真实 smoke／PID／CWD／argv／GPU／日志增长及自然终态均核实。未动健康的五源折确认或其他 owner 进程。

联合研究尚未完成：本诊断不能证明新 Ag 的独立训练收益、B00/B10/B01/B11 交互、正式 clean＋六 practical 或跨模型 seed 重现；low urban>60% 与 suburban 接近 clean 的目标仍待实际训练／冻结测试证明。下一训练只由合法源证据设计，不能用目标反馈调参或选择性重跑。

实际矩阵／命令／环境／CWD／路径／GPU／stop 见 experiment.json；代码与启动核实见 launch_execution.json。原逐包 gzip／导数日志保存在不可变远端 run。私有实际产物：

- [analysis/metrics.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_probe20_20261016/final/analysis/metrics.csv)
- [analysis/counterfactuals.csv](E:/type10-7/local_artifacts/cvs_FcrossRX_probe20_20261016/final/analysis/counterfactuals.csv)
- [analysis/summary.json](E:/type10-7/local_artifacts/cvs_FcrossRX_probe20_20261016/final/analysis/summary.json)
- [analysis/channel_transitions.json](E:/type10-7/local_artifacts/cvs_FcrossRX_probe20_20261016/final/analysis/channel_transitions.json)
- [urban_counterfactual_effects.png](E:/type10-7/local_artifacts/cvs_FcrossRX_probe20_20261016/final/urban_counterfactual_effects.png)
- [urban_state_strata.png](E:/type10-7/local_artifacts/cvs_FcrossRX_probe20_20261016/final/urban_state_strata.png)
- [actual_branch_changes.png](E:/type10-7/local_artifacts/cvs_FcrossRX_probe20_20261016/final/actual_branch_changes.png)
- [head_and_gradients.png](E:/type10-7/local_artifacts/cvs_FcrossRX_probe20_20261016/final/head_and_gradients.png)
