# F 公共相位与观察窗原点：完整源域分解

状态 **ANALYZED_SOURCE_ONLY / VERIFIED**。实际代码 `187cd47a17cd1a0413b118027d02fd26acdf2129`。两固定final20/默认头、同seed2026092701、未参与两父训练的原source RX1 L1260。raw预测与上一轮原clean两模型均逐包完全相同；实测输入复RMS范围0.999999906至1.000000094，排除了额外输入归一化造成的大幅掉分。

| 模型 | H相位来自场景 | raw accuracy | phase | delay24 | phase+delay | accuracy交互项pp |
|---|---|---:|---:|---:|---:|---:|
| A0 | practical_high | 81.1905% | 75.4762% | 48.6508% | 47.6190% | +4.6825 |
| A0 | practical_mid | 81.1905% | 74.6825% | 48.6508% | 46.3492% | +4.2063 |
| A0 | practical_low_suburban | 81.1905% | 70.9524% | 48.6508% | 45.7143% | +7.3016 |
| A0 | practical_high_urban | 81.1905% | 73.6508% | 48.6508% | 45.6349% | +4.5238 |
| A0 | practical_mid_urban | 81.1905% | 58.3333% | 48.6508% | 40.3175% | +14.5238 |
| A0 | practical_low_urban | 81.1905% | 47.6984% | 48.6508% | 36.9048% | +21.7460 |
| W30 | practical_high | 80.0000% | 75.0000% | 48.2540% | 46.5873% | +3.3333 |
| W30 | practical_mid | 80.0000% | 74.3651% | 48.2540% | 45.7937% | +3.1746 |
| W30 | practical_low_suburban | 80.0000% | 71.0317% | 48.2540% | 44.6825% | +5.3968 |
| W30 | practical_high_urban | 80.0000% | 72.2222% | 48.2540% | 45.0000% | +4.5238 |
| W30 | practical_mid_urban | 80.0000% | 58.2540% | 48.2540% | 39.2857% | +12.7778 |
| W30 | practical_low_urban | 80.0000% | 48.7302% | 48.2540% | 36.6667% | +19.6825 |

A0固定raw正确集为1023包：仅delay保留468包（45.7478%），low urban的phase保留579包（56.5982%），组合保留368包（35.9726%）。delay相对raw修复145/破坏555；low phase修复22/破坏444，组合修复97/破坏655。交互项是分类指标上的2×2差，不是互相独立的物理损伤份额，不能直接与上一轮去噪/flat_H增益相加。

原点因素有明确分类影响：A0 delay24准确率48.6508%，比raw81.1905%低32.5397pp，三day为50.0000/48.5714/47.3810%。六个TX各210包，准确率依次6.1905/0.4762/98.5714/86.6667/100.0000/0.0000%；强烈的TX间错误迁移保留，不能只报均值或宣称所有身份证据丢失。相位影响随原H状态分布增加，low urban相位单因素跌33.4921pp，组合跌44.2857pp。W30存在相似脆弱性，不因此重新晋级。

上述分解支持“当前冻结F对公共相位/短窗口原点敏感”的解释，预先的全部效应<1pp放弃条件未触发。它不证明随机时移训练有效，也不证明真实full practical已获得提升。delay包含截尾和人工前缀引起的窗口能量变化；其前24点是reflect合成过去，未假设存在未来/更长真实记录。phase只取原传播核最大tap的单位复相位，不代表全部振荡器相位，不反演原TX硬件。

pure delay对所有scene是同一波形/同一预测，重复六行只为按原full队列分层，不能当六次独立重复。原full状态/锁定/质量仅作为固定cohort标签，并非这三个无噪声反事实的接收状态。正式clean+六practical IQ/信道/SNR/边界始终未改。

原time/frequency/PA/reference及三SIG的真实前向均已记录，捕获加性项重建joint160/defaulthead logits通过。观察到的SIG/参考变化不用于替代身份成绩、拟合新头或宣布门控致因；具体均值、/30间隔、状态/TX/day分层accuracy和fixed6macroF1在完整私有汇总中。没有梯度求导或参数/buffer更新。

r02实际47884网络包（每模型23942含2CPsmoke），47880附加readout-only logits重建，0optimizer/EMA/headfit/epoch，1260独立物理包、一个modelseed。worker墙钟39.416530s；两父既有E20/4440各单列。r01技术失败另有4个CPsmoke包，故本问题累计47888网络包；r01确切墙钟未记录，保留UNKNOWN。该失败只因tuple/list序列化误判，sourcecache已读但源IQ前向0；原root保留、无科学分数。一次本地0IQ/CP/前向复现、定向修复和定向P0/P1复核已通过，三种真实配置改变仍拒绝。

r02单archive transfer/remotecompile、普通账户preflight、真实CPsmoke、运行中PID/CWD/argv/GPU/log增长及自然终态均已核实。唯一原P0/P1 PASS；数学与图检查含独立检查累计8个合成网络包/12原核核对/28原负例断言，修复另3项配置负例、0额外前向。其他健康任务未修改或停止。

本轮没有新训练曲线；两父完整训练曲线保存在原报告。逐scene/条件、默认头accuracy/F1/保持率、TX/day/state/lock/quality与实际分支统计已经完成独立后处理。尚无新正式target指标、worst target RX或多seed联合收益，不能宣称lowurban>60%或clean与星地双突破。下一项应检验由该源证据支持的有界原点抖动E20，而不是直接扩展冻结结果或叠对齐损失。

- [完整逐行指标](E:/type10-7/local_artifacts/cvs_f_phase_origin20_20261016_r02/final/analysis/metrics.csv)
- [完整机制与配对明细汇总](E:/type10-7/local_artifacts/cvs_f_phase_origin20_20261016_r02/final/analysis/summary.json)
- [phase_origin_factors.png](E:/type10-7/local_artifacts/cvs_f_phase_origin20_20261016_r02/final/phase_origin_factors.png)
- [phase_origin_source_strata.png](E:/type10-7/local_artifacts/cvs_f_phase_origin20_20261016_r02/final/phase_origin_source_strata.png)
- [phase_origin_actual_branches.png](E:/type10-7/local_artifacts/cvs_f_phase_origin20_20261016_r02/final/phase_origin_actual_branches.png)
