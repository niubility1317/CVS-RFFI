# MILKR 当前 support 核密度归一化

R36 在 R34 的等权 C4/C8 核上检验 support hub 影响。固定 Tri-SIG-TIME SSL、DURA v1.0、原 C4 head、R31 成员 ridge 先验和 R26 同记录融合；不采用 R35 的类别校准，不回调旧 run 参数。

在当前训练 support S 上，A(x,S)=两路独立带宽后的 RBF 各半权，沿用所选 single/multi scales，只在密度 A 中去除线性分量以保证非负。d(x)=mean_j A(x,S_j)，support 自项包含在内；dbar=mean_i d(S_i)。新核为 dbar*K_parent(x,y)/sqrt(d(x)*d(y))。它是正对角合同变换，保留正半定；常数密度时严格还原父核。没有新增可调强度、截断底值或静默回退；非法或非正数值报错。核密度只降低 support 中相似度普遍偏高的记录权重，不用 query 真值、old/new 角色、配额或批次统计。

每个物理 OOF 的 tau、degree、dbar、成员 gate 和 KRR 只由 train support 拟合。held/query 逐记录对固定 train support 计算自身 d(x)，不写回状态；support 路径始终纯 S0 raw 或纯 S1 DURA，不增加物理 K。密度额外状态为 (physicalSUP+1)*8 bytes；F/G/head 参数和 buffers 冻结，最终 MILKR 判决有意改变。RF 最近距离及融合门控保持原定义。

双 source CCA64 重新拟合，源 L6300/ep91、92共12600配对，557440 FP64参数；五源RX/两TX留出同时保护两编码器，源held-new head列先移除再softmax。F已见源六TX，源3景/new2与目标6景/new20、已曝光开发基准边界如实保留。源120规则只选K1 Q0/QF宏H，ref6类别数缩放，K>1物理OOF。源合法/数值后直接固定角点证伪，无源H收益否决。

首RX1-1/P01/MIX01/seed2026092701，K1/20×N2/20，B12+C24共36行，Q0raw/QG仅DURA/QF同记录融合完整；全部预测固定后独立truth-last。两S每角点raw较匹配原raw≥5pp、QF>raw及raw旧新均值guard≥原版−1pp才冻结同法补12KN，再六RX。最终K1/5/10/20×N2/5/10/20的七RX等权逐格目标不变；G-only同输入配对和融合跨输入差分开。

source GPU4、target GPU0，启动前核实容量并保护其他owner。实际commit、命令、环境CWD、合法数据句柄、路径、矩阵和技术停止条件见experiment.json。真实F/G无query smoke为launcher第一步PASS继续；唯一P0/P1审查、单归档传输校验、remotecompile、PID/CWD/argv/GPU/log增长及独立评分保持。

## 发布前核验

29项相关检查及4项非法协议输入检查通过，本地compile/diff检查通过。唯一独立P0/P1审查PASS，独立新增4项加density worker alias共5项通过：非负A/mix0、正对角合同PSD、训练和held/query degree公式一致；每物理OOF只trainSUP的tau/degree/dbar/gate/KRR；源grid携带density rule，source/worker实际双CCA/fitter同路；R31 prior/R26 fusion/strictSUP保持。真实F/G无query smoke在launcher首步，PASS继续。
