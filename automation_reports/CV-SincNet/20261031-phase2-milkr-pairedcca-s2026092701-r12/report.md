# MILKR 源配对 CCA 共同特征空间

预登记。完整Tri-SIG-TIME SSL基座与DURA v1.0冻结；方法、命令、环境/CWD、数据/模型来源、路径、owner、GPU和stop rule见experiment.json。

R14仅在合法source L6300物理记录×既有模拟ep91/92的12600配对观察学习参数。八个原特征块分别采用正则CCA：边缘协方差加0.01 trace/d，inverse-root白化，交叉协方差SVD，两路投到相同rank，再补零到原维度、逐块unit。raw和DURA各自有冻结线性编码器。严格S0/Q0只raw编码器；S1/Q1只真实G后使用restored编码器；不引入raw/G混合输入。R12当前K标签nested物理支持留出head不变。目标只加载冻结参数，不携带源样本/特征缓存。

源RX留出选择rank16/32/64/96。每次投影以及R13源控制的bridge训练均完全排除heldRX的两个模拟epoch。heldRX内每类选52个物理记录，support前K≤20，query固定位置20至51共32/class，无physical overlap。五RX×两episode×K1/5/10/20×四路。source旧/新各三类，F已见所有六类，不能称真新类泛化。

固定源门槛：四路平均H较sourceR13至少+0.5pp，两个S的Q0均不低于R13超过0.5pp，两个S的Q1−Q0均至少+0.5pp。满足者取源四路均H最大，平局小rank；没有rank通过即source-only科学停止，目标0行。通过才用全部sourceL拟合选定参数checkpoint，启动首RX1-1 P01/MIX01 seed2026092701的全部16KN/四路单arm80行。R13和原版同case各80行复用。全部预测固定后独立truth-last。

目标验收仍为两个S各自Q0较原版≥+5pp、Q1高于Q0，Q0旧/新macro≥原版−1pp，逐KN完整列出。通过仅允许固定方法其余6RX确认，不能以singleRX先导宣布goal完成。失败不扩展、原root不覆盖、不据query调参。源GPU4、目标GPU0一worker，实际余量检查，其他owner不动。

本候选唯一独立P0/P1审查PASS。配对旋转合成held数据对齐、逐query独立、保存重载逐位一致及错误path/rank拒绝测试PASS；原nested与协议非法输入拒绝检查PASS。真实checkpoint source-only smoke为launcher第一步，PASS即继续源留出筛选。

## 实际源端结果

**VERIFIED**：真实checkpoint smoke PASS，源6300物理记录/12600配对观察及五RX留出矩阵全部完成。投影/bridge训练排除heldRX，当前source support与held32/class物理互斥；无U/V真值，F/G更新0。执行commit `bbfc25f3313b3cdd53e628b741f4a6d71dda3bb0`。归档传输/compile、PID/CWD/argv/GPU及log增长均已核实。

| rank | 四路平均H | 较源R13 | S0/S1的Q0差 | S0/S1的Q1−Q0 | 源晋级 |
|---:|---:|---:|---|---|---|
| 16 | 81.173% | +1.483pp | +1.055, +3.405pp | +13.570, +14.448pp | True |
| 32 | 79.414% | -0.276pp | -0.501, +0.580pp | +12.988, +16.755pp | False |
| 64 | 78.144% | -1.546pp | -2.169, -1.797pp | +13.404, +19.346pp | False |
| 96 | 77.422% | -2.268pp | -2.644, -3.514pp | +13.173, +21.075pp | False |

## 目标四路先导

全部80预测及80行独立truth-last评分完成，匹配原版80行复用，所属进程全部退出。

| 路径 | 宏H | 原版宏H | 配对差 | Q1−Q0 |
|---|---:|---:|---:|---:|
| S0Q0 | 55.847% | 48.777% | +7.070pp | +0.592pp |
| S1Q0 | 51.854% | 13.091% | +38.763pp | +15.916pp |
| S0Q1 | 56.439% | 42.243% | +14.196pp | +0.592pp |
| S1Q1 | 67.770% | 62.069% | +5.702pp | +15.916pp |

完整16KN/old/new/F1/floor见result_summary.json；先导晋级：True。单RX不能声明goal完成；未通过则不扩展，不据target调参或选择性重跑。
