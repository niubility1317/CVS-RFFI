# MILKR 同记录时间相位轨道编码

R29 固定时间循环移位 [0,64,128,192] 与四相位组合，共16个RF视图。假设是时序位置的变化不应改变身份，而单C4只消除常相位影响。源域先证伪，位置/数量在运行前固定，不依据目标成绩选择。另一备选是神经残差适配器，训练及验证成本更高，本轮先验证明确的同记录不变性。

每条原received 256记录，DURA只在移位之前处理一次，然后各移位执行四相位F，归一化各RF block并聚合16视图均值/标准差，输出仍为2d；Hanning FFT对每个移位分别重算，再与该移位四相位共享。原默认head概率只用未移位的原四相位，R27的prior/置信度gate函数保持，新增移位head输出不用。F/G/head不训练，原记录没有增加新观测或标签。每record独立处理，无query批统计。

source L6300的ep91/92共12600raw/G配对，CCA64五折均排除held RX及两个held TX；旧F已见源六TX的限制保留，源old4/new2 head在softmax前剔除held新类列。120规则选择K1，K20物理OOF全train-only。control直接复用R27已冻结source-only的规则和60行source验证结果，不复用其target分数、support状态或源样本库。候选/对照源主H(Q0,QF)平均差至少+0.5个百分点、各K1/20和S0/S1 raw差至少−0.25个百分点，失败source-only停止、target0、无最终checkpoint；通过才全source拟合最终encoder。

support严格S0raw/S1DURA；query保留Q0raw/QG仅DURA/QF同记录融合，R26半权条件融合规则与R27当前SUP置信度先验保持。当前SUP物理OOF各折gate/带宽/系数仅trainSUP拟合。首RX1-1/P01/MIX01/seed2026092701，K1/20×N2/20，B12+C24共36行先全部预测固定再独立truth-last评分。两S每角点raw较匹配原版≥5个百分点、QF>raw、raw旧新均值guard通过，才冻结方法补12KN；全16通过后补六RX。最终七RX等权逐KN目标不缩小。QG同输入配对完整保留，QF对原QG明确为输入改变。

sourceGPU4/targetGPU0实际容量核实，其他owner不动。commit、命令、环境/CWD、来源/data句柄和技术停止条件见experiment.json。实际F/G无query smoke为launcher第一步，另加timeRF不变性；独立P0/P1审查、单包传输/remotecompile和实际PID/CWD/argv/GPU/log增长核实保持。权重仍普通CCA参数，无PCA或源统计库。

## 发布前核验

22项相关检查及4项非法协议输入检查通过（末次定向9项含time worker与原worker回归），本地compile/diff检查通过。唯一独立P0/P1审查PASS：16视图只同记录，G先且一次，各shift重算FFT，head保留未移位C4；source及worker的SUP/query均使用extract_time；source失败不生成最终encoder或启动target，三路36行评分接口闭合。实际checkpoint无query smoke将在launcher首步运行，PASS直接继续。

## 实际结果

**VERIFIED**：实际 checkpoint 无 query smoke PASS；release archive 传输比较与 remotecompile PASS；所属 PID/CWD/argv/GPU 与日志增长已读取，当前全部所属进程退出。执行 commit `cad271fc3ce781c5d5c457f567442c82387d8b81`，源流程 106.909 秒。

候选和 C4 对照均完成 120 个 K1 规则及 K20 物理 OOF 源验证（对照复用已冻结 R27 源结果）。候选 source 主宏H差 -0.6884 个百分点，源门槛 False。

| 源 K | support | raw 宏H配对差（百分点） |
|---:|---|---:|
| 1 | S0 | -0.3185 |
| 1 | S1 | -0.1688 |
| 20 | S0 | +0.1396 |
| 20 | S1 | -3.7205 |

按预登记科学门槛 source-only 停止，未拟合最终 encoder，target IQ/预测/评分均 0；没有把失败源参数送入目标再挑结果。F/G/head 更新 0。这个负结果不能替代最终每16KN宏H要求，goal ACTIVE。
