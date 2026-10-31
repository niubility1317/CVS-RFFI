# MILKR 源域跨接收机回合式度量学习

R30 改为用合法源标签学习可迁移的身份距离，Tri-SIG-TIME SSL、DURA v1.0 和原神经分类头保持冻结。原C4moments/sourceCCA64不变；八个block各64轴学习一个正权重，共512参数，g=exp(log4·tanhθ)，范围[0.25,4]，初始为1。每个block按sqrt(g)加权再单位化，输出仍填充原维度。raw/G共享这些权重，保持PSD内积几何，不增加身份分类头。

备选是继续无标签配对拟合或改变视图数；此前两类机制未解决源H，本轮直接利用合法源TX标签约束跨RX判别。目标不是只降低同记录raw/G误差。全部预算在目标访问前固定：CPU/FP64、Adam学习率.01、600步；四类source episode，support K轮流1/5/20、每类8个held样本，support与held来自不同源RX，raw/G四种路径组合轮流出现。每个support只取唯一physical ID的一个ep91/92观察，不能把同物理两增强算两个shot。四类原型由归一化support特征均值再归一化得到；loss=八block平均cos/.1的CE + .01·mean(tanhθ²)。源held样本都是合法L，不是目标query。

source L6300、既有ep91/92共12600配对；五折同时排除held RX和两个held TX，CCA与可学习g均不接触这些行。每fold其余4TX/4RX训练，最终全source6TX/5RX训练只在源门槛通过后进行，训练同样保持四类episode。源验证old4/new2的head在softmax前剔除held新类两列，原F已见源6TX的限制保留。source3scenes/目标6scenes、source2new/target20new不外推为新盲测。

K1仍用120核/正则规则按Q0/QF宏H选择，K20当前support真实物理OOF。control复用R27已冻结source-only规则及60行源验证结果，不用其目标分数/适配状态。源主H(Q0,QF)平均差须≥+0.5个百分点，每个源K1/20、S0/S1的raw差须≥−0.25个百分点；否则source-only停止、target0，不生成最终encoder。通过后最终checkpoint只含CCA均值偏置/投影和512个g，不含源样本、类原型、协方差、标签/ID、optimizer或训练日志。逐fold记录实际step/物理数/源TX-RX/g范围和训练loss，训练loss不是科学收益门槛。

target仍strict S0raw/S1G，Q0raw/QG仅DURA/QF同记录融合。R27 currentSUP先验gate、残差KRR与物理OOF、R26融合规则不变。target只load冻结encoder，无metric训练或跨query统计。原received经G一次后C4，无新增观测。首RX1-1/P01/MIX01/seed2026092701 K1/20×N2/20，B12+C24共36行固定后独立truth-last。每角点两S raw较匹配原raw≥5pp、QF>raw、raw旧新均值guard≥原版−1pp才补12KN；all16后补其他6RX。最终七RX等权逐KN目标不变，QG匹配对照完整，QF对原QG明确路径改变。

sourceGPU4/targetGPU0启动前核实容量，source metric训练CPU1thread，其他owner不动。完整commit/命令/环境CWD/来源/data句柄/路径和技术停止规则见experiment.json。真实F/G noquery smoke为launcher首步，PASS直接继续；一次独立P0/P1审查、单包传输/remotecompile与实际PID/CWD/argv/GPU/log增长核实保持。

## 发布前核验

21项相关检查与4项非法协议输入检查通过，新增metric worker加载路由回归通过，本地compile/diff检查通过。唯一独立P0/P1审查PASS：episode类别顺序与CE标签一致、物理支持去重、support/held RX不同；两轮metadata和特征索引对齐；训练/部署sqrt(gains)一致，边界/加载/旧Torch转换一致；源失败仍停止target。实际F/G noquery smoke将在launcher首步运行，PASS直接继续。

本地准备目录纠正：最初与历史pooledmetric私有辅助目录重名，delivery guard在提交/发布前阻止继续。现改为独立 `local_artifacts/milkr_epmetric_20261031`，旧runtime由其原package恢复、旧spec由原实验记录恢复；科学结果/包/远端不可变run未改。旧未版本化辅助脚本内容未完全恢复，已明确标记停用该旧辅助目录。独立远端核实原run仍SCORED_COMPLETE，R30新run/release均不存在，尚未启动；算法/预算/来源和review不变。
