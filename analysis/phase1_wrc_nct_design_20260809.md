# Phase1 WRC-NCT冻结设计

状态：`DESIGN_FROZEN / IMPLEMENTING`；目标模式：`GOAL_MODE=ACTIVE`。

1.候选名称：`WRC-NCT`（Worst-Receiver-Covered Normalized Class-Tail）。
2.候选只复用同fold v3 GI-EpiOR bundle中已经封存的class geometry和连续`NCT=d_(1)/(d_(2)+epsilon)`；不使用GI二元head，不重训backbone，不新增feature normalization或域对齐。
3.输入限于每fold五个source-known TX。先沿用GI的canonical physical ID 50/50 reference/query划分；每TX内部再按`SHA256(physical_id)`排序，把query前半作为calibration、后半作为evaluation，形成50/25/25的R/C/E物理互斥划分。
4.R已经用于上游bundle的prototype/radius；C只估计一个source-known拒识边界；E只评估known。proxy与outer-held全部零fit、零校准、零阈值选择。
5.每个source RX在C上至少需要50行，否则该fold fail-closed。
6.对RX`r`的`n_r`个升序NCT分数定义有限样本顺序统计量`k_r=min(n_r,ceil(0.98*(n_r+1)))`，`tau_r=s_(k_r)`；全fold唯一阈值为`tau=max_r tau_r`。
7.唯一决策是`accept iff NCT<=tau`；`p_local`保持冻结checkpoint的`tx_logits argmax`。registered reject/defer按known错误计数。
8.输出不可变readout JSON、阈值TorchScript runtime、split/coverage/parity receipt、逐行score CSV和同行metrics JSON；readout必须绑定上游GI bundle SHA256。
9.本轮矩阵固定为6个LOTO fold，每fold一条CPU命令；不运行新seed、不扫alpha、分位数、RX聚合或阈值。
10.clean门：6/6模型/数据闭环；每foldknown overall、min-class、min-RX、min-day相对同一E子集冻结C均不低于-2个百分点；source proxy在6/6 fold方向优于无拒识FAR=100%且AUROC均高于0.5；readout/runtime可导出且parity不高于`1e-5`。
11.outer-held只作跨TX方向诊断，不恢复Phase3式`FAR<=5%`强门，也不能把它写成真实unknown性能。
12.clean任一门失败即`REJECT`；通过才发布三种LEO视图验证最低类别和LEO弱信道floor。
13.这仍是Q98阈值族的一次最终反证。相对历史绝对density-Q98，唯一新假设是类相对NCT比例与worst-source-RX上包络能够同时保留known floor和产生proxy排序信号；失败后关闭该阈值族，不继续改分位数。
14.既有GI score的只读可行性计算只用于确认C/E计数与公式可执行，不作为WRC-NCT正式性能或阈值选择证据。
