# MILKR 源跨RX对比有界残差编码

R41在新拟合的R40相关性加权CCA上增加源域神经残差编码器。新增NN使用合法L标签监督训练，不能称整个方法无神经更新；Tri-SIG-TIME SSL基座、DURA、原默认头及buffers保持冻结。NN输出每block64维表示，不另训六类分类头，原C4 head概率函数保持，最终MILKR判决会改变。

每orbit4/8各有raw/G两路径、八block MLP：2d→64 tanh→64 tanh，最后线性层零初始化。delta=.25*MLP/8，每向量L2≤.25；output=unit(baseCorrCCA64+delta)，归一化前范数至少.75。Source训练FP32，权重导出FP32只读数组；source冻结校准与target均用同一NumPy逐记录FP64算术，加载不训练、无跨query状态。源CCA仍557440FP64，新增NN684032FP32，总1241472数值参数/7195648未压缩字节；checkpoint只留模型参数，不携源IQ、例样本、cov或类库。

训练复用source Episodes的物理ID去重分组，每step取四TX和两个不同RX，sourceSUP K按1/5/20循环、另一RX每TX8条，分别36/52/112物理记录；相同索引提供raw/G两处理视图。SupCon正对仅同TX不同RX，同TX同RX从分母排除，异TX作负对，temperature=.1。总loss=SupCon+mean||e_raw−e_G||²+.1mean||e−base||²。每orbit固定600步AdamW lr.001/wd.0001，无早停/最佳步挑选，每orbit40000配对记录曝光。五fold验证模型加最终模型，共6×2×600=7200 NN optimizer步；最终部署模型为每orbit600步、两orbit合1200步，不把验证训练步数或原F训练步数混称为部署训练预算。

源L6300、ep91/92共12600配对。五RX/两TX留出同时排除两个CCA及两个NN的held RX/held TX，fit只接当前合法source L的y/RX/物理ID；component seed=source_proxy_seed+heldRX*10000+303+orbit*100003，最终无heldRX项。源F已见六TX，proxy old4/new2、源3景与目标old6/new20/6景、已曝光开发集边界保持。held-new head列softmax前裁除，NN不接head payload。源120K1规则只按Q0/QF宏H选择、ref6类数缩放；源只合法/数值门槛后固定验证，不重开旧失败gate。

target只加载同一冻结源encoder，每K/N/S仅本格SUP的R31成员gate/双RF核/残差KRR物理OOF，所有带宽/gate/KRR仅trainSUP。没有目标NN更新、query统计/角色/配额/真值。support严格纯S0raw/S1G，query raw/G-only/同记录融合完整；R38RF成员gate、R39类方向prior和R36密度均不引入。F每路径8次、G原记录一次，新增encoder推理开销实际记录。

首RX1-1/P01/MIX01/seed2026092701全部16KN，K1/5/10/20×N2/5/10/20，B24+C96=120预测、40fit/40fusion全部固定后独立truth-last。两S每KN raw较匹配原raw≥5pp、QF>raw及raw旧新均值guard≥原版−1pp才同冻结法扩其余六RX；最终七RX等权all16目标不变。G-only同输入配对和融合跨输入差分开。

sourceGPU4/targetGPU0实际容量核实并保护其他owner。真实F/G无query smoke为launcher首步PASS继续；唯一独立P0/P1、一次归档传输校验/remotecompile、实际PID/CWD/argv/GPU/log增长及独立评分保持。实际commit、命令、环境CWD、来源、数据句柄、矩阵和技术停止条件见experiment.json。

## 发布前核验

33项相关检查及4项非法协议输入检查通过，compile/diff检查通过。唯一独立P0/P1审查PASS，独立新增4项+worker alias共5项PASS：两epoch特征和y/RX/IDs同train索引过滤、两orbit CCA/rho/NN同步heldRX/TX；Episodes物理ID去重/4类跨两RX、SupCon正负与排除mask正确；零初/残差L2≤.25；FP32导出/重载及源冻结评估/target共用NumPy路径；参数/schema/head列映射正确；旧R30收益gate仅匹配旧method，新route合法数值后固定验证；worker实际加载新NN双桥且无target fit。真实F/G无query smoke在launcher首步PASS继续。
