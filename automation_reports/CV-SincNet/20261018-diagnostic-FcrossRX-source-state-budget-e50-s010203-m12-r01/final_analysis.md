# E30→E50：三种子源域 practical 状态分层

**结论：完整 practical 增强的源域适应收益在三个原定模型种子上重现；追加预算使 mid／low urban 继续提高，但没有得到源域 clean 与星地同时改善的匹配证据。** As50 相对同预算 No-As50，源 clean 面板平均下降 0.259±0.085 个百分点，三个种子均下降；mid urban 提高 28.157±3.325 个百分点，low urban 提高 22.769±1.738 个百分点。As50 相对自身 As30 的 mid／low urban 增量分别为 1.889±0.274、2.083±0.431 个百分点。

这里全部是合法源域 V 面板结果。源 low urban 的 As50 三种子准确率均超过 60%，不能据此声称正式 low urban 已超过 60%，也不能代替正式 clean 下降不超过 0.1 个百分点的验收。报告未读取正式文件或其他 owner 的正式分析。

## 范围与实际执行

- Run：`20261018-diagnostic-FcrossRX-source-state-budget-e50-s010203-m12-r01`。状态：`THREE_SEED_SOURCE_STATE_ANALYZED`。
- 实际分析代码 commit：`b217a2554ea89d148c4b21ab9ea556381599260f`。
- 三个模型种子：2026092701、2026092702、2026092703。每种子四个逻辑行 NOAS30／AS30／NOAS50／AS50，共 12 行；模型、默认 cosine 分类头及已有预测均未改变。
- 使用同一源 V 面板的 1800 个物理样本，源 RX 为 1／3／4／6／8，day 为 1／2／3。只读取保存的源预测及物理元数据；U 的 TX 真值未打开。
- 每个 satellite 场景包含同一批 1800 个样本的两个固定信道 draw，pooled n=3600。两次 draw 不是两个模型种子，也不是 3600 个独立物理样本。
- 只分析 `full` practical residual 条件。源场景为 low suburban、mid urban、low urban；没有 high urban，因此不报告源 urban 三场景均值或全部六种 practical。
- seed01：E30 来自原 P43，E50 来自原 P50；seed02／03：E30 来自原 E30 复现，E50 来自原 E50 四行复现。每个种子使用自身 No-As30 clean 正确集合，未跨种子替换父集合。

四行的 IDs／TX／RX／day 与有序 source panel、JSONL 的 draw／scene／index／physical_id／条件 full 逐项一致。每种子核对 10800 条物理元数据记录。每种子的 4 行×4 个源视图×accuracy／宏 F1 共 32 个值与原已保存 source summary 一致，三个种子 **96 项核对全部一致**。本报告没有重算完整 V27000 的逐样本表现。

## 逐种子总体结果

准确率单位为 %；satellite 为两个固定 draw 合并后的准确率。

| 模型种子 | 行 | clean | low suburban | mid urban | low urban |
|---|---|---:|---:|---:|---:|
| 2026092701 | NOAS30 | 98.444 | 49.861 | 43.556 | 35.611 |
| 2026092701 | AS30 | 98.333 | 88.667 | 72.806 | 59.083 |
| 2026092701 | NOAS50 | 98.611 | 49.361 | 43.667 | 36.167 |
| 2026092701 | AS50 | 98.444 | 90.111 | 74.778 | 60.722 |
| 2026092702 | NOAS30 | 98.611 | 62.472 | 52.194 | 42.000 |
| 2026092702 | AS30 | 98.500 | 88.833 | 72.833 | 58.722 |
| 2026092702 | NOAS50 | 98.778 | 60.167 | 50.389 | 39.750 |
| 2026092702 | AS50 | 98.500 | 90.222 | 74.944 | 60.833 |
| 2026092703 | NOAS30 | 98.611 | 57.250 | 48.972 | 39.694 |
| 2026092703 | AS30 | 98.278 | 89.333 | 74.000 | 59.028 |
| 2026092703 | NOAS50 | 98.833 | 55.250 | 46.778 | 38.861 |
| 2026092703 | AS50 | 98.500 | 90.667 | 75.583 | 61.528 |

三种子配对变化如下。± 为三个模型种子配对差的样本标准差（ddof=1），不使用物理包或信道 draw 伪造模型重复数。单位均为绝对准确率百分点。

| 源视图 | As50−No-As50 | As50−As30 | No-As50−No-As30 | 预算×As 交互 |
|---|---:|---:|---:|---:|
| clean | -0.259±0.085 | +0.111±0.111 | +0.185±0.032 | -0.074±0.085 |
| low suburban | +35.407±5.347 | +1.389±0.056 | -1.602±0.966 | +2.991±0.924 |
| mid urban | +28.157±3.325 | +1.889±0.274 | -1.296±1.234 | +3.185±1.149 |
| low urban | +22.769±1.738 | +2.083±0.431 | -0.843±1.403 | +2.926±1.676 |

交互定义为 `(As50−No-As50)−(As30−No-As30)`，属于预算与 As 的交互，不是 Ag×As 因子交互。No-As 的 mid／low urban 随预算平均退步，因而交互高于 As 自身的增量；不能把全部交互都算成 As 的绝对提高。

As50 的三种子平均宏 F1 为 clean 98.479%、low suburban 90.283%、mid urban 75.079%、low urban 60.781%。相对 As30，mid／low urban 宏 F1 提高 1.869±0.283、2.004±0.360 个百分点；相对 No-As50，提高 30.835±3.535、24.395±1.575 个百分点。

**clean 代价不能省略：** As50−No-As50 的逐种子 clean 差为 −0.167、−0.278、−0.333 个百分点，均为负值，也均超过 0.1 个百分点。这是 V1800 面板口径，不是完整 V27000 或正式 clean 的结论。As50 相对 As30 的 clean 小幅提高或持平（+0.111／0／+0.222 个百分点），仍未追平同预算 No-As50。

## 状态与质量：哪些观测区间得到改善

以下为 As50−As30 的准确率增量。n 是每个模型种子在同一场景两个 draw 合并后的观测数。只列有足够覆盖的主要状态；全部九种 state×lock 组合，包括空单元，均在 CSV／JSON 中保留。

| 场景 | state／lock | n | seed01 | seed02 | seed03 | 配对均值±样本 SD |
|---|---|---:|---:|---:|---:|---:|
| mid urban | LOS/locked | 1899 | +1.106 | +1.632 | +1.106 | +1.281±0.304 |
| mid urban | PARTIAL/locked | 306 | +1.634 | +0.980 | +2.288 | +1.634±0.654 |
| mid urban | PARTIAL/degraded | 59 | +1.695 | 0.000 | 0.000 | +0.565±0.979 |
| mid urban | BLOCKED/degraded | 656 | +5.640 | +3.506 | +3.354 | +4.167±1.278 |
| mid urban | BLOCKED/unlocked | 678 | +1.032 | +2.802 | +1.032 | +1.622±1.022 |
| low urban | LOS/locked | 1060 | +1.604 | +0.566 | +0.849 | +1.006±0.536 |
| low urban | PARTIAL/locked | 307 | 0.000 | +2.280 | +3.583 | +1.954±1.814 |
| low urban | PARTIAL/degraded | 253 | +2.372 | +1.976 | +5.929 | +3.426±2.177 |
| low urban | BLOCKED/degraded | 414 | +2.899 | +6.280 | +1.449 | +3.543±2.479 |
| low urban | BLOCKED/unlocked | 1565 | +1.534 | +2.045 | +3.131 | +2.236±0.816 |

追加预算最明显的平均增量出现在 mid urban 的 BLOCKED／degraded（+4.167 个百分点），以及 low urban 的 PARTIAL／degraded、BLOCKED／degraded（+3.426、+3.543 个百分点）。BLOCKED／unlocked 的 mid／low 增量为 +1.622、+2.236 个百分点，三个种子均为正。相对 No-As50，BLOCKED／unlocked 的 mid／low 增益仍有 +10.964±0.426、+10.607±0.664 个百分点。LOS／locked 也有收益，因此不能将改进只归给单一遮挡状态。

按 signal_to_added_noise_db 的固定分箱，As50−As30 如下。该比值只描述信号与新增噪声，不是原录音加增强后的总 SNR。

| 场景 | 新增噪声比区间（dB） | n | 配对增量均值±样本 SD（百分点） |
|---|---|---:|---:|
| mid urban | <3 | 383 | +1.480±1.574 |
| mid urban | 3-8 | 718 | +2.786±0.975 |
| mid urban | 8-15 | 293 | +4.437±1.024 |
| mid urban | >=15 | 2206 | +1.330±0.189 |
| low urban | <3 | 1127 | +2.041±0.407 |
| low urban | 3-8 | 794 | +3.233±1.769 |
| low urban | 8-15 | 303 | +3.190±1.661 |
| low urban | >=15 | 1376 | +1.211±0.255 |

3 至 15 dB 区间的平均提高通常大于 ≥15 dB；low urban <3 dB 区间也在三个种子上提高。实际波形功率质量分箱得到相近方向，但不能据此把噪声认定为唯一原因：传播、接收状态与补偿残差在 full practical 中仍有关联。

白输入功率代理与实际波形质量落入不同粗分箱的观测，在 mid urban 为 116／3600（3.222%），low urban 为 133／3600（3.694%）。这些分组用于描述已有观测，没有重新计算接收状态，也没有估计改变质量代理所导致的总效应或固定补偿效应。

并非所有分层都改善。mid urban 的 waveform_minus_white_power_db∈[1,3) dB 分组 n=40，As50−As30 为 −2.5／−5.0／0 个百分点，均值 −2.5±2.5；另有个别种子在其他质量差分组下降。low urban 唯一 AGC limited 观测的变化为 0／0／−100 个百分点，均值 −33.333±57.735；n=1 不支持总体机制推断。

## 固定父集合保持与错误变化

NOAS30 clean 正确集合的物理样本数依次为 1772／1775／1775。每个种子在所有行中使用同一集合；satellite pooled 分母依次为 3544／3550／3550 个重复 draw 观测。该集合仅用于读出，没有成为训练样本的置信度门控。

| 模型种子 | mid urban 保持率 As30→As50 | low urban 保持率 As30→As50 | mid urban 修复／新增错误／净修复 | low urban 修复／新增错误／净修复 |
|---|---:|---:|---:|---:|
| 2026092701 | 73.674%→75.564% | 59.594%→61.287% | 123／52／+71 | 136／77／+59 |
| 2026092702 | 73.549%→75.634% | 59.183%→61.211% | 122／46／+76 | 152／76／+76 |
| 2026092703 | 74.704%→76.338% | 59.690%→62.254% | 111／54／+57 | 170／80／+90 |

表中修复／新增错误按全部 3600 个 satellite 观测计算；保持率仅在固定父集合上计算，分母不同。两者都表明改善包含净修复，同时存在新增错误。默认头间隔同时保留原始 logit margin 和除以固定尺度 30 的值；间隔读出不替代身份准确率。

## 稀疏分层、物理边界与成本

全部 **2178 条三种子汇总记录**均保留，其中 660 条为空单元记录、114 条为非空且 n<30 的小单元记录。这些记录包含不同对比、draw 和分组，不能当成独立样本数量。mid／low urban 的 PARTIAL／unlocked 分别只有 n=2／1；LOS／degraded、LOS／unlocked、BLOCKED／locked 为 n=0。空单元的均值与 SD 保留为空，不补零。

实际源元数据中 EQ 全部关闭，噪声着色分组全部落在 flat；数值 FIR 延迟为 24 点，残余核长度为 62，有效 interior_count 全部为 195／256，边界模式为 reflect。由于这些条件在本面板中没有充分变化，不能将预算收益归因为修复 EQ 噪声着色或边界效应。相位跟踪存在启用与关闭，但本任务没有重生成 IQ 或执行反事实干预，也没有记录新的门控／分支训练轨迹。

**本次是源域相关分层分析，不是物理因果归因。** 它没有单独控制噪声、多径、残余 CFO、相位、IQ 残差或补偿状态，不能宣称某一类物理损伤被唯一识别或消除。

本次新增 NN=0、IQ 读取=0、checkpoint 读取=0、optimizer 更新=0、训练=0、正式评分=0；仅 CPU 两线程读取保存源 logits 和元数据并重算描述统计。CUDA 隐藏，CPU 时间 18.7596 s，墙钟 **18.7916 s**。`formal_files_read=false`、`no_promotion=true`；不作为其他 owner 完成流程或下一实验晋级的门槛。历史模型训练成本不计入这笔零前向分析成本。


执行检查：三种子 wrapper 的 6 项定向 fixture 和编译通过，一次独立限定 P0／P1 审查通过；N607 普通账户预检、单个发布包本地／远端 SHA 比较、3 文件远端编译及 6 项远端 fixture 通过。独立后状态确认 PID1197720 自然退出、结果完整，无技术缺项。详见 [验证记录](<E:/type10-7/automation_reports/CV-SincNet/20261018-diagnostic-FcrossRX-source-state-budget-e50-s010203-m12-r01/verification.json>) 和 [实际命令／环境](<E:/type10-7/automation_reports/CV-SincNet/20261018-diagnostic-FcrossRX-source-state-budget-e50-s010203-m12-r01/experiment.json>)。这些检查没有额外模型前向。

## 可核查产物

- [实际执行与成本 report.json](E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_s010203_20261018_r01/retrieved/result/report.json)；[全部逐种子总体结果 CSV](E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_s010203_20261018_r01/retrieved/result/all_seed_overall.csv)。
- [全部逐种子修复／新增错误与保持率 CSV](E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_s010203_20261018_r01/retrieved/result/all_seed_paired_effects.csv)；[全部逐种子预算交互 CSV](E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_s010203_20261018_r01/retrieved/result/all_seed_budget_interaction.csv)。
- [2178 条汇总 CSV](E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_s010203_20261018_r01/retrieved/result/three_seed_paired_summary.csv)；[同口径 JSON](E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_s010203_20261018_r01/retrieved/result/three_seed_paired_summary.json)。
- [跨种子状态增量 PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_s010203_20261018_r01/retrieved/result/three_seed_state_budget_deltas.png)；[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_s010203_20261018_r01/retrieved/result/three_seed_state_budget_deltas.pdf)。

| 模型种子 | 全部分层 CSV | 混淆矩阵 JSON | mid urban 图 | low urban 图 |
|---|---|---|---|---|
| 2026092701 | [CSV](E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_s010203_20261018_r01/retrieved/result/s2026092701/source_strata.csv) | [CM](E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_s010203_20261018_r01/retrieved/result/s2026092701/source_confusions.json) | [PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_s010203_20261018_r01/retrieved/result/s2026092701/practical_mid_urban_source_states.png) | [PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_s010203_20261018_r01/retrieved/result/s2026092701/practical_low_urban_source_states.png) |
| 2026092702 | [CSV](E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_s010203_20261018_r01/retrieved/result/s2026092702/source_strata.csv) | [CM](E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_s010203_20261018_r01/retrieved/result/s2026092702/source_confusions.json) | [PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_s010203_20261018_r01/retrieved/result/s2026092702/practical_mid_urban_source_states.png) | [PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_s010203_20261018_r01/retrieved/result/s2026092702/practical_low_urban_source_states.png) |
| 2026092703 | [CSV](E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_s010203_20261018_r01/retrieved/result/s2026092703/source_strata.csv) | [CM](E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_s010203_20261018_r01/retrieved/result/s2026092703/source_confusions.json) | [PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_s010203_20261018_r01/retrieved/result/s2026092703/practical_mid_urban_source_states.png) | [PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_s010203_20261018_r01/retrieved/result/s2026092703/practical_low_urban_source_states.png) |

本报告证明三个固定模型种子上的源 practical 适应收益和 As 追加预算收益；也保留了源 clean 的负代价与局部退步。它没有证明正式跨 RX 双提升、正式 low urban 60% 或完整六场景验收，未据任何正式分数调参、选种子或重跑。
