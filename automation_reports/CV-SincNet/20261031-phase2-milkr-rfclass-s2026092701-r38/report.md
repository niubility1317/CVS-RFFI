# MILKR 当前support RF旧类先验方向校准

R39保留R38双C4/C8核、R31head成员gate、R38RF成员gate与R26融合，检验RF对“哪一个旧类”的证据能否改善旧类先验方向。A=.5RBF4+.5RBF8，使用当前trainSUP各自tau和所选single/multi尺度，只在likelihood A去掉线性成分以保持非负。a_c(x)=mean_{S_j属于旧c} A(x,S_j)，pcal=normalize(original_head_p*a)。旧类RF似然均匀时还原原p，无新增混合权重。head成员gate始终输入原p，不将校准p喂回旧gate。prior旧槽=.5*g_head(originalp)*(pcal−1/m)*R38_RFfactor，新槽0，因此prior零和及幅度界保持。

每物理OOF的old类均值/labels、两tau、两原gate和KRR只使用trainSUP，held/query逐条到固定trainSUP算affinity，无query统计/真值/角色/配额；不同K/N/S独立。源m4、目标m6依合法标签表对齐原head列；新SUP不进入旧类likelihood均值，但保持R38成员gate/KRR训练。RFclass状态为int64物理SUP labels及m个counts，额外8*(physicalSUP+m)bytes。不是神经头训练，Tri-SIG-TIME SSL、DURA、原head及buffers冻结，最终MILKR判决改变。非正likelihood normalizer明确报错，无静默替代。

source L6300/ep91、92共12600配对，fresh双CCA64合557440FP64；五源RX/两TX留出同时保护两个编码器，held-new head列在softmax前裁去。F已见源六TX、源3景/new2与目标6景/new20及已曝光开发集边界保留。源120K1规则只按源Q0/QF宏H选择，ref6类数缩放，K>1当前SUP物理OOF；源合法/数值后固定验证，不用target选择参数。

本新候选首RX1-1/P01/MIX01/seed2026092701覆盖全部K1/5/10/20×N2/5/10/20，B24+C96共120行、40fit/40fusion。它仍是单capsule小矩阵，补充中间K/N证据；旧角点未通过run不重开、不改gate。全部预测固定后独立truth-last。两S每格raw较匹配原raw≥5pp、QF>raw且raw旧新均值guard≥原版−1pp才同冻结方法扩其余六RX；首RX已覆盖all16，无剩余12KN阶段。最终七RX等权每KN要求保持，首RX全过也不代表最终达成。

support严格纯S0raw/S1G，Q0raw/QG仅DURA/QF同记录融合完整保留。G-only同输入配对与QF跨输入差分开。sourceGPU4/targetGPU0实际容量检查并保护其他owner。真实F/G无query smoke在launcher首步PASS继续；唯一P0/P1审查、单归档传输校验/remotecompile、实际PID/CWD/argv/GPU/log增长及独立评分保持。完整commit、命令、环境CWD、数据句柄、来源、逐行矩阵和技术停止条件见experiment.json。

## 发布前核验

36项相关检查及4项非法协议输入检查通过，compile/diff检查通过。唯一独立P0/P1审查PASS，独立新增4项+worker alias共5项PASS：source old4/target old6的旧类列序正确；RF likelihood去linear且context尺度一致；新pcal仅替换先验方向，原headgate仍接original p，R38 factor有界，prior新槽0/零和/幅度保持；每OOF仅TRAIN全部likelihood/tau/两gate/KRR，held/query逐条同cross规则；source grid两marker和实际双CCA/fitter同路。单capsule16KN对应40fit/40fusion/120row动态计数闭合，无density/center串入。真实F/G无query smoke为launcher首步PASS继续。
