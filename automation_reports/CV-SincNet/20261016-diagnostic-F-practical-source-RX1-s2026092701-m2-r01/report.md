# F practical 源域冻结诊断：完整结果

状态 **ANALYZED_SOURCE_ONLY / VERIFIED**。实际执行代码 `cf91b1be7c04ed9656bc2113f3b42ffdf69f2e20`，两父 default-head final20、同 model seed2026092701、held source RX1 原 L1260，0新训练。worker和dispatcher自然结束，prediction全矩阵完成后独立新进程汇总；没有目标查询或目标truth服务。

以下全部为源域冻结诊断，不能当作正式 clean/urban 性能或训练增益。WHITE30原源训练FAIL保持，不晋级、不扫强度/seed。

| 原视图 | A0 accuracy | W30 accuracy | W30−A0 pp | A0 macro F1 | W30 macro F1 |
|---|---:|---:|---:|---:|---:|
| clean | 81.1905% | 80.0000% | -1.1905 | 80.7618% | 79.3449% |
| practical_high | 45.1587% | 44.7619% | -0.3968 | 32.4596% | 32.5514% |
| practical_mid | 44.4444% | 44.2063% | -0.2381 | 33.0387% | 33.7427% |
| practical_low_suburban | 39.8413% | 38.3333% | -1.5079 | 29.5916% | 28.6457% |
| practical_high_urban | 44.2063% | 43.3333% | -0.8730 | 33.3238% | 32.7540% |
| practical_mid_urban | 33.4921% | 34.1270% | +0.6349 | 28.4221% | 28.5508% |
| practical_low_urban | 29.4444% | 28.7302% | -0.7143 | 26.1961% | 25.5587% |

urban三场景均值 A0=35.7143%，W30=35.3968%，差-0.3175pp。W30 clean修复9/破坏24；low urban修复28/破坏37。不是新的Ag收益或跨seed证据。

| A0反事实相对full（pp） | high urban | mid urban | low urban |
|---|---:|---:|---:|
| no_noise/fixed | +0.9524 | +1.9048 | +0.9524 |
| no_noise/total | +0.7937 | +0.7937 | +1.5079 |
| flat_H/fixed | +1.1111 | +1.7460 | +0.7937 |
| flat_H/total | +1.1111 | +1.6667 | +0.8730 |
| no_carrier/fixed | -0.0794 | +0.2381 | -1.0317 |
| no_carrier/total | -0.0794 | +0.2381 | -1.0317 |
| no_RX_IQ/fixed | +0.0794 | +0.1587 | -0.1587 |
| no_RX_IQ/total | +0.0794 | +0.1587 | -0.0794 |
| actual_power_quality | +0.0000 | +0.0000 | -0.2381 |
| edge_boundary | -6.7460 | -2.6984 | -3.2540 |

移除新增噪声并未大幅恢复A0：low urban固定状态修复102/破坏90，净12/1260=+0.9524pp；重算状态修复116/破坏97，净19/1260=+1.5079pp。flat_H固定状态仅+0.7937pp，移除carrier反而−1.0317pp。各项保留的因素不同，效应不可相加；no_noise仍包含原录音未知噪声。不能据此宣称噪声/多径/载频扰动均无害或已找到唯一主因。

## 状态、功率和边界

A0 low urban的full分层：LOS384包48.6979%，PARTIAL196包35.2041%，BLOCKED680包16.9118%；locked516包45.1550%，degraded212包22.6415%，unlocked532包16.9173%。这些是固定full队列，其他条件不重新分组。强LOS/locked仍明显低于该源clean81.1905%，故单独增加AWGN强度缺乏依据。

white-input质量代理与实际波形差在low urban为−1.9035至+2.2520dB，均值+0.01055dB；替换代理使33/1260包跨锁定状态（15 degraded→unlocked、14 unlocked→degraded、2 degraded→locked、2 locked→degraded）。A0分类仅净−3包，W30净−5包，不能夸大为主要失效原因；原测试代理不改。

实际核长度全为62，FIR数值延迟24点，前61输出受人工前缀影响，保证与前缀无关的观察区[61,256)共195点。reflect改edge后，该内部区域误差严格0，但A0 high/mid/low urban准确率分别−6.7460/−2.6984/−3.2540pp；这说明前缀敏感，不证明edge更物理或允许改正式测试。所有本轮flat_H控制仍保留24点delay和原核复系数相位，no_carrier也不是去除全部核相位。未单独隔离这些共有因素，不能由当前结果把问题定为SIG/门控。

EQ真实关闭，G=1，新增噪声的equalizer着色动态范围0dB。full六场景无AGC gain触顶包；此处是增益限幅，不是伪造逐点硬剪裁。low urban实际signal_to_added_noise范围−9.5797至28.7840dB，均值11.1093dB；该指标不是总SNR。残余CFO范围−1528.57至1441.47Hz，不能与未补偿Doppler混称。

## 原分支、默认头与导数

真实F捕获路径重建joint160及默认头logits通过。low urban A0默认头/30平均间隔−0.062571，源clean+0.114570。joint160与自身clean平均余弦0.86073；SIG加性激活0.07932、三局部descriptor0.24595/0.31706/0.29877，reference0.50995，time0.89849、frequency0.98043、PA0.76712。激活相似度不是分类效益，更不能用三路descriptor另造三个分类器。静态PA/reference/SIG learnedgain保持参数，动态Envelope/FreqBand gate直接记录；未把冻结gain随场景变化作为伪机制。

每批对原14个参数张量复用图求CE(LS.01)导数，共2680次，无.grad积累、optimizer或headfit。Sinc导数为原CENTER_25K坐标；统计保留在私有summary.json，不冒称物理Hz梯度或收敛。所有模型参数/buffer前后逐值不变。

## 执行与费用

两模型各84422网络包前向（含2真实CPsmoke），合计168844；2680次导数；1260独立物理包、7560场景物理draw，不是两seed。每个trace附加一次原默认头readout-only重建，共168840次；未重复骨干前向。CPU物理渲染canonical7560次、受控renderer83160次。新增optimizer/EMA/headfit/epoch=0；既有父成本各E20/4440updates单列。worker墙钟255.346s，独立后处理另行执行。没有新训练曲线；两父完整20epoch曲线已在各自报告保存。

一次最小P0/P1发现发布包多余design依赖，修复后一次定向隔离导入复核PASS。累计本地合成56网络包/8导数/12物理draw132波形，0真实IQ/CP/optimizer。单archive校验/remotecompile、普通账户preflight、真实CPsmoke、运行中PID/CWD/argv/GPU/log增长与自然终态均独立核实，无foreign进程操作。

实际模块为原release的experiments/adv3b02_xuc/code/leo_practical/{channel,residual}.py。fs25MHz/IQ256/residual/post_sync/reference30/reflect、quality adaptation/phase tracking/IQ compensation真实配置保存。说明更正：execution.json的“model normalization afterward”为错误的描述字段；实际clean为既有nativeRMS缓存，practical在传播前normalize_input并执行原AGC，F.forward未新增packetRMS。只修正文案，不改原产物、不重跑；特征GroupNorm/LayerNorm照常。

## 产物和未完成项

实际运行命令及注册矩阵见experiment.json，实际代码与启动证据见launch_execution.json。逐模型/逐场景/逐条件/源TX/day/state/lock/quality/CFO/AGC的accuracy、macroF1、保持率、修复/新错与全部梯度统计见下列私有汇总，原逐包gzip和worker全日志保存在同一不可变远端run。全部正式clean+6practical测试、跨目标RX最差性能、多seed训练及混合收益尚未产生，不能宣称lowurban>60%或双突破。下一项唯一假设见新设计文件，尚未实现或运行，不自动晋级W30。

- [完整源指标CSV](E:/type10-7/local_artifacts/cvs_f_practical_source20_20261016/final/analysis/metrics.csv)
- [完整机制汇总](E:/type10-7/local_artifacts/cvs_f_practical_source20_20261016/final/analysis/summary.json)
- [状态转换明细](E:/type10-7/local_artifacts/cvs_f_practical_source20_20261016/final/analysis/channel_transitions.json)
- [urban_counterfactual_effects.png](E:/type10-7/local_artifacts/cvs_f_practical_source20_20261016/final/urban_counterfactual_effects.png)
- [urban_state_strata.png](E:/type10-7/local_artifacts/cvs_f_practical_source20_20261016/final/urban_state_strata.png)
- [actual_branch_changes.png](E:/type10-7/local_artifacts/cvs_f_practical_source20_20261016/final/actual_branch_changes.png)
