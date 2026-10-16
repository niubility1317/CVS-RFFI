# F＋跨 RX：原普通增强的 RX 分组随机条件

两组均完成 E20 及七视图源域读数；固定 RX 分组候选未通过预登记条件，保留负结果并关闭。

训练代码 82975411f152e2be7d80ebc79b7caa850d2ee858；离线复算实现 experiments/cvs_f_rxgroup20/analyze.py 随结果提交。结果均为 source held RX1 L1260／seed2026092701／FINAL20／原默认 cosine 分类头；不能作为正式 target、跨 seed 或联合目标完成的证据。

## 逐 row、逐 seed 结果

| view | BATCH accuracy % | RXGROUP accuracy % | 差值 pp | BATCH macro F1 % | RXGROUP macro F1 % |
|---|---:|---:|---:|---:|---:|
| clean | 81.6667 | 81.8254 | 0.1587 | 81.3306 | 81.6280 |
| practical_high | 38.3333 | 41.5873 | 3.2540 | 28.2533 | 31.5110 |
| practical_mid | 38.5714 | 40.8730 | 2.3016 | 29.3862 | 32.0933 |
| practical_low_suburban | 35.3968 | 36.6667 | 1.2698 | 28.0449 | 29.1865 |
| practical_high_urban | 38.4921 | 40.2381 | 1.7460 | 29.6476 | 31.2468 |
| practical_mid_urban | 31.5873 | 32.3016 | 0.7143 | 26.7252 | 27.3718 |
| practical_low_urban | 28.3333 | 28.8889 | 0.5556 | 24.8072 | 25.2339 |

urban 三场景均值差 1.0053 pp；clean macro F1 差 0.2974 pp；已见 RX V 差 0.0417 pp。

| row | clean % | urban mean % | low urban % | 各 suburban 与自身 clean 的差 pp（high/mid/low） |
|---|---:|---:|---:|---|
| BATCH | 81.6667 | 32.8042 | 28.3333 | 43.3333, 43.0952, 46.2698 |
| RXGROUP | 81.8254 | 33.8095 | 28.8889 | 40.2381, 40.9524, 45.1587 |

## 固定判断与收益归因

固定阈值：clean≥＋1 pp，clean macro F1≥0 pp，seen V≥−0.5 pp，两组 final L CE≤0.5，完整 E20/4440 updates/六次 L 前向/3330 SupCon steps/合法数据和完整覆盖。六种 practical 全部报告，本次独立 clean 资格不以 practical 升降为门槛。结果：{"clean": false, "macroF1": true, "seenV": true, "healthy": true}。

FAIL 关闭这组固定 RX 分组条件；不改算子子集、分组、RX、seed、阈值或预算救援。

两组均从零，网络／参数量／默认头／原 SupCon／六次 L 前向／raw 监督权重／U 规则匹配。唯一训练差别是原 anti-shortcut 和 general-channel 条件在不同物理 RX 之间的相关性，含原整批 taper、多径核及触发条件；PA／DAC 保持原调用。不能将任何差值单独归因于某一个 gate 或某一种物理损伤。

| low urban 分层 | n | BATCH % | RXGROUP % | 差 pp | BATCH margin/30 | RXGROUP margin/30 |
|---|---:|---:|---:|---:|---:|---:|
| state:LOS | 384 | 41.6667 | 44.2708 | 2.6042 | -0.0286 | -0.0268 |
| state:BLOCKED | 714 | 19.3277 | 19.4678 | 0.1401 | -0.0603 | -0.0599 |
| state:PARTIAL | 162 | 36.4198 | 33.9506 | -2.4691 | -0.0320 | -0.0322 |
| lock:locked | 467 | 41.7559 | 44.1113 | 2.3555 | -0.0270 | -0.0253 |
| lock:degraded | 228 | 24.5614 | 21.0526 | -3.5088 | -0.0571 | -0.0581 |
| lock:unlocked | 565 | 18.7611 | 19.4690 | 0.7080 | -0.0595 | -0.0588 |
| quality:[20,30) | 396 | 42.4242 | 44.9495 | 2.5253 | -0.0275 | -0.0258 |
| quality:[5,10) | 157 | 22.2930 | 19.7452 | -2.5478 | -0.0636 | -0.0642 |
| quality:[0,5) | 385 | 18.9610 | 20.7792 | 1.8182 | -0.0582 | -0.0567 |
| quality:<0 | 180 | 18.3333 | 16.6667 | -1.6667 | -0.0622 | -0.0631 |
| quality:[10,20) | 142 | 33.8028 | 31.6901 | -2.1127 | -0.0334 | -0.0335 |

上述按固定同批 IQ／同信道状态分层，是表现定位，不是去除噪声／多径后的受控因果消融。本轮不训练 As，不能证明 practical 适应机制改善或与 As 存在正交互。added-noise SNR 不是总 SNR，quality 为既有白输入功率代理；保留原 boundary、数值 FIR 延迟和 RX 补偿。

## 固定 BATCH clean 正确集合与错误变化

固定集合 n=1029；所有场景复用这一集合，不按新模型重新挑选。

| view | BATCH 保持率 % | RXGROUP 保持率 % | 修复 | 损坏 | 净正确样本数 |
|---|---:|---:|---:|---:|---:|
| clean | 100.0000 | 98.9310 | 13 | 11 | 2 |
| practical_high | 36.6375 | 39.8445 | 49 | 8 | 41 |
| practical_mid | 37.4150 | 39.4558 | 42 | 13 | 29 |
| practical_low_suburban | 34.5967 | 36.1516 | 35 | 19 | 16 |
| practical_high_urban | 37.4150 | 39.2614 | 38 | 16 | 22 |
| practical_mid_urban | 30.1263 | 31.0982 | 29 | 20 | 9 |
| practical_low_urban | 28.4742 | 28.9602 | 37 | 30 | 7 |

## 完整覆盖、训练健康和成本

| row | updates | L NN | 总 NN packets | stage shadow 时间 s | 原训练墙钟 s | 到最终结果墙钟 s |
|---|---:|---:|---:|---:|---:|---:|
| BATCH | 4440 | 3409920 | 4686678 | 25.8575 | 1241.2048 | 1258.8211 |
| RXGROUP | 4440 | 3409920 | 4686678 | 26.2825 | 1246.5982 | 1264.2291 |

新训练总计 40 model-epochs／8880 updates；9373358 NN packets（包括 dispatcher 兼容 smoke2）。每组学生／EMA／optimizer 均从零，父训练成本0；旧 B00 仅 smoke 载入后丢弃，不继承。合成验证另计2322 NN／3 updates／8448 IQ call packets，离线评分及出图0 NN。NN 计数扣除了 EMA 复制时继承的2个评估计数，保留实际学生初始化 smoke；墙钟已由进程 start ticks 与产物 mtime 独立核对，近似精度1 s；E20 耗尽不能称为收敛。

两组各有4440 coupling calls、8880 stages、568320普通L视图、5040唯一L样本、完整72个TX×RX×day cell，完整4440训练步和20轮已检查；原U真值隐藏且未分组，默认头联合训练、无额外头。L/U 完整5994条序列相同：True（L=True，U=True）。各批 operator 实际触发数量可不同，不能把保持边际分布误写成实际计数完全相等。

| operator | BATCH 触发 L packets % | RXGROUP 触发 L packets % | RXGROUP 正样本对条件不同 % |
|---|---:|---:|---:|
| p_dc_offset | 6.3063 | 5.7278 | 10.6922 |
| p_bandedge_taper | 4.6396 | 4.7957 | 9.1110 |
| p_time_shift | 6.3964 | 6.8018 | 12.6804 |
| p_amp_scale | 8.9414 | 8.2366 | 14.9187 |
| p_phase_rot | 9.0991 | 8.6912 | 15.8239 |
| p_cfo | 6.7568 | 6.7905 | 12.5398 |
| p_phase_noise | 5.4955 | 5.7842 | 10.7859 |
| p_multipath | 2.8378 | 3.5755 | 6.8694 |
| p_awgn | 7.6351 | 7.3837 | 13.5663 |

## 图表与证据

图表均由完整已冻结源预测／完整日志生成。

- [source_scene_results PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_rxgroup_20261017/final/figures_v2/source_scene_results.png)；[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_rxgroup_20261017/final/figures_v2/source_scene_results.pdf)
- [complete_training_curves PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_rxgroup_20261017/final/figures_v2/complete_training_curves.png)；[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_rxgroup_20261017/final/figures_v2/complete_training_curves.pdf)
- [ordinary_coupling_exposure PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_rxgroup_20261017/final/figures_v2/ordinary_coupling_exposure.png)；[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_rxgroup_20261017/final/figures_v2/ordinary_coupling_exposure.pdf)
- [low_urban_strata PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_rxgroup_20261017/final/figures_v2/low_urban_strata.png)；[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_rxgroup_20261017/final/figures_v2/low_urban_strata.pdf)
- [fixed_clean_retention_and_errors PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_rxgroup_20261017/final/figures_v2/fixed_clean_retention_and_errors.png)；[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_rxgroup_20261017/final/figures_v2/fixed_clean_retention_and_errors.pdf)

独立结果：[analysis_summary.json](E:/type10-7/local_artifacts/cvs_FcrossRX_rxgroup_20261017/final/analysis_summary.json)；完整分层和曲线同目录。原始 IQ／checkpoint 留在不可变 N607 run root，未提交或拉取。执行命令／环境／模块文件路径／数据角色／停止规则见同 run 预登记。

## 最终问题的回答与未完成项

当前只有一个源 RX／一个模型 seed。正式目标域默认头提升、clean 代价、最差目标 RX、low urban>60% 和跨 seed 重现均未证明；尚无已验证新 Ag、B00/B10/B01/B11 因子矩阵或 Ag×As 正交互。本轮 source practical 的状态表现变化如上表，不能声称固定正式测试下某类退化已获缓解。

已见源 RX 的 V 最差项均为 RX3：95.4259%→95.5556%（＋0.1296 pp），不能代替正式目标域最差 RX。历史 B00 smoke 权重来自 scratch E20／4440 updates，只作兼容检查，0状态继承；历史训练成本与本轮新训练分开。


## 实际执行的验证与分析命令

以下本地命令 CWD 均为 E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014；模型合成检查 wrapper 自行设置已核实的原生模块路径。

- E:/type10-7/local_envs/ssr-gpu/python.exe -m experiments.cvs_f_rxgroup20.checks_runtime --native E:/type10-7/code/worktrees/cvs_iq_timing_20261016/experiments/adv3b02_xuc/code/DataAugmentation.py --schedule E:/type10-7/code/worktrees/cvs_iq_timing_20261016/experiments/adv3b02_xuc/code/cvsrffi/schedule.py --actual E:/type10-7/local_artifacts/cvs_FcrossRX_timegrid20_20261017/final/TIMEGRID-RX1-s2026092701/source/ordinary_augmentation_actual.json
- E:/type10-7/local_envs/ssr-gpu/python.exe E:/type10-7/local_artifacts/cvs_FcrossRX_rxgroup_20261017/run_model_checks.py
- F:/App/miniconda3/python.exe -m experiments.cvs_f_rxgroup20.checks_analysis
- F:/App/miniconda3/python.exe -m experiments.cvs_f_rxgroup20.harvest --output E:/type10-7/local_artifacts/cvs_FcrossRX_rxgroup_20261017/final
- F:/App/miniconda3/python.exe -m experiments.cvs_f_rxgroup20.analyze --input E:/type10-7/local_artifacts/cvs_FcrossRX_rxgroup_20261017/final
- F:/App/miniconda3/python.exe E:/type10-7/local_artifacts/cvs_FcrossRX_rxgroup_20261017/plot_local_v2.py

runtime 12＋模型10＋离线评分10项反例检查均 PASS；真实 checkpoint 兼容 smoke2 PASS；独立完整记录复算 exit0。图表初版的 U 接受量图注曾误写每步，已用全部 step 日志求和确认每轮总数并生成 figures_v2，初版保留、不作最终交付图。更正未改变任何训练、预测、评分或阈值。
