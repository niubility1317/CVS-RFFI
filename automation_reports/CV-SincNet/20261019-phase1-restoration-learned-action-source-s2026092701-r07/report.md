# 学习接收作用的自监督物理恢复：源域验证

状态：RUNNING / VERIFIED。归档单次SHA比较、远端compile/import入口、实际2条L_s及原F50/defaulthead smoke和checkpoint往返均通过；独立读回原PID/CWD/命令与日志增长，当前CPU构建fresh源教师缓存，GPU训练worker尚未启动。代码 `c8ccdfdbd03ab5ad8828b1eb9210b99a664ebe7b` 已提交/push并核实远端OID，11项聚焦检查通过，一次独立P0/P1无阻断问题。N607普通账户预检通过，GPU5空闲约24GB，原source/父权重/F50可用，新run/release不存在。

本轮从用户六类网络报告R3和R06源域算子差距出发，训练一个观测条件估计器。只读取当前received和冻结R02粗恢复，估计64tap复FIR、逐时主/镜像系数和单样本噪声；6步实内积CG软MAP产生一份IQ。FFT、PA等继续由原F实际计算。没有oracle/clean/场景/TX部署输入、IQ硬裁剪或单独重训头。

原6300L/56700U、fresh20epochs、15%clean/85%residual、4940updates，0TX标签/F训练信号；两组新白探针监督真实复合算子作用，另加0.1log噪声方差MSE，没有配对恢复或分类损失。这是具有clean源波形的模拟自监督。父E20冻结，新增E20，累计E40。该残差配方与R03mixed不同，不能冒称严格配方匹配。

最先用真实2条L_s验证父和原F50/defaulthead及checkpoint往返，PASS即继续。单候选固定lastE20；原sourceV1800×7预测全部固定后，独立CPUscorer才读取V标签。逐clean/三郊区/三城区/TX/RX/day报告accuracy、Macro-F1、repair/damage和波形指标。原F50来自CROSS_RX E50/11100，区别于历史cleanFE200和AsFE30。

六practical平均IQNMSE相对coarse改善至少5%、cleanNMSE≤0.02仅代表源波形可行性。身份诊断另行报告，不能用NMSE宣称身份忠实。此run不访问target、不依据目标调参、无选择性救跑；多seed、独立物理族及最终身份目标仍未完成。

完整矩阵、命令、环境、路径、GPU、来源、停止规则和预期产物见 [experiment.json](experiment.json)，设计追踪见 [R07追踪](../../../docs/PAMF_ACTION_RESTORE_R07_20261019.md)。执行时机器私有路径/原日志/权重/预测保留本地产物，不进入Git。
