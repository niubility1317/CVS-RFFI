# MILKR 适配器设备留出与 K1 重新校准

预登记R18。Tri-SIG-TIME SSL完整REPL+固定时序排列、DURA v1.0、神经默认头冻结。R17目标产物只作历史证据，不继承其权重；本版从相同合法source F0 L特征训练新线性encoder。命令/环境/CWD、来源、路径、owner、GPU、stoprule见experiment.json。

源五折按RX排序，held TX对固定(0,1)/(1,2)/(2,3)/(3,4)/(4,5)。每折从新CCA/metric训练中排除heldRX全部记录、heldTX在所有RX的全部记录，包含两aug epoch91/92。只在其它RX的四TX拟合；heldRX内old4+heldnew2每类52个物理记录，support前K≤20/query后32，eval固定epoch91，K1/5/20。F基座本身见过全部五sourceRX/六TX，本方法只叫adapter-TX/RX-heldout，不是真F-novel验证。TX0/5仅held一次、1..4两次，保持小源代理而非全部15组合。

表示候选rank16/32/64/96×metric0/0.5/0.9。每fold maxrank96的同一CCA SVD截断，类内cov显式允许四个非连续TX标签并inverse重映射，冻结T融合两路projection；不上传源cov/类均值/样本。每表示K1完整24核×5ridge重新校准，不沿用变换前R9超参；K5/20原R14 nested支持选择。共1440源候选，midK结果共享，约14400个小K1拟合及240个midK拟合。控制为同fold rank16/metric0/旧R9K1规则，不用旧源均值替代。

源门槛相对control：四路H≥+0.5pp、held-new macro≥+1pp、每个源K/S Q0H≥+0.25pp、K1 held-new Q0两S各≥+1pp、每K/S Q1−Q0≥0且两S均值>0。通过者按H/newgain/较大ridge/较小rank/较小metric/固定rule序tie；没有通过即source-only停止target0。只改变K1而高K数学上等于control的候选不能通过该全K源门槛。源已见F六类、old4/new2、三景增强与目标old6/new20六景不同，保留外推局限。

最终只在全source L6300物理/12600rawG配对观察上按选定rank/metric拟合普通PairedCCA encoder。K1 ridge明确采用mean-loss归一化：源6类参考lambda×当前已分配support注册类数/6；只读support类数，不读query数/角色/配额。源C6到目标C8..26未直接验证，明确披露；K>1仍原R14路径，不混称全部lambda缩放。无F/G训练、无U/V真值。

通过源门槛才firstRX1-1/P01/MIX01/seed2026092701，K1/5/10/20×N2/5/10/20四路80行，全补集query。S0/Q0只raw编码，S1/Q1只实际G编码，同一support状态共享query路径。80预测固定后独立truth-last；原版/R14/R17同case历史结果复用。按用户每格Q0较原版≥5pp且Q1>Q0，保留原均值旧新≥原版−1pp guard并逐格列四路原版差；只有全部16格通过才同参数六RX确认，mean不能替代。

唯一P0/P1独立审查PASS。源RX/TX两轮隔离、非连续类/rank32、K1手工缩放等价/K>1保持、只改善K1不能通过全K源门槛及旧噪声测试PASS。真实F/G no-query smoke为launcher第一步，通过即继续。源GPU4/targetGPU0一worker、actualfree检查，不改其他owner；技术失败保留产物，不盲重试。
