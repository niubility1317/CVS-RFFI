# 既有源域损伤控制的证据范围（P60解释用）

本次只读复核，不新增训练、IQ/checkpoint读取或网络前向，也不改变P60的源晋级标准。

既有P24已实际完成V1800×2draw×3scene的11条件控制与三个冻结E20默认头读出，包含No-As B00、As B01、RAW_PRAC，不是仅NoAs20。P33又在同一缓存上读出P32 matched scratchE30两模型；P40 L900为P39 B0 own20+10的噪声梯度观察。它们都不是当前P60的新默认头。

| 条件 | 受控fixed | 总效应total |
|---|---|---|
| no_noise | 固定full锁定状态、残余scale、频率估计、相位追踪、EQ/G、AGC，只去本次新增噪声 | 质量代理设200并重算状态/补偿/AGC；200不是总SNR |
| flat_H | 保持白输入核能量、传播公共复相位与数值延迟，固定接收状态/补偿 | 重算状态/AGC；白输入质量通常不变，不保证状态切换 |
| no_carrier | 固定接收参数，去补偿后主项/镜像项的残余频率与相位旋转 | 重算接收状态/补偿；不去传播核公共相位 |
| no_RX_IQ | 固定接收参数，令注入widely-linear残余α=1/β=0 | 重算接收参数；不反演原录音未知TX硬件 |

另有actual_power_quality：使用有限实际波形mean(|H*x|²)/σ²替代白输入mean(|x|²)sum(|h|²)/σ²，保持其他latent draw，重算状态；edge_boundary则在固定状态/补偿/AGC下reflect→edge，是边界压力控制。native/full/frozen replay逐包一致性以及fixed参数不变均实际检查。

旧P24 As B01去新增噪声fixed时，mid/low源恢复+5.278/+10.083个百分点；total额外+0.222/+1.278。该旧结果不支持把主要损失解释成状态切换；不能挪用为P60的因果效应。actual功率控制确有mid87/3600、low84/3600锁定状态变化，而该旧B01准确率变化−.250/0pp；不能说proxy无影响，也不能据此解释主要缺口。

signal_to_added_noise_db只表示信号对本次新增噪声的比值，原录音噪声未知，去新增噪声不等于无噪声上界。实际EQ关闭/G=1；这些完成结果没有验证EQ开启后的有色噪声分支。

FIR核62taps、数值延迟24、前61输出可能依赖人工前缀、共同内部区域195点的影响已诊断。P34还在平坦核上做延迟24/0×原传播相位/零相位四格，固定噪声、状态、补偿与AGC。纯时间位置和前缀边界仍未完全拆开，不能推广为full色散信道的唯一原因，更不能改正式条件来提升成绩。

P60只预登记full条件源状态/质量分层及全部七种固定formal评价。因此可报告新默认头在BLOCKED/degraded/unlocked等组的识别差和关联；尚不能声称停止ordinary PA注入具体修复了practical噪声/多径/CFO/RXIQ中的哪一项。不追加重复P24诊断作为当前训练或发布门槛。

证据：[P24观察完成](<E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_residual_diag_20261017/evidence/observations/completion.json>)、[总完成](<E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_residual_diag_20261017/evidence/completion.json>)、[物理统计](<E:/type10-7/local_artifacts/cvs_FcrossRX_fullsource_residual_diag_20261017/evidence/readout.json>)；[控制实现](<E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_fullsource_residual_diag24/controls.py>)、[逐包参数检查](<E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_fullsource_residual_diag24/probe.py>)；[P34延迟/相位报告](<E:/type10-7/automation_reports/CV-SincNet/20261017-diagnostic-FcrossRX-delay-phase-e30-s2026092701-m2-r01/report.md>)。
