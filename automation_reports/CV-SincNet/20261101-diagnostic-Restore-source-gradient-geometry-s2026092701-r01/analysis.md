# 冻结恢复器源端目标梯度几何

**末轮冻结点未发现持续的编码器梯度反向或全局clip激活。** 此结果不支持据当前证据启用PCGrad/分组裁剪；不能外推为训练全过程没有冲突。完整性能目标仍未达到，默认DURA保持。

![梯度诊断](E:/type10-7/automation_reports/CV-SincNet/20261101-diagnostic-Restore-source-gradient-geometry-s2026092701-r01/gradient_geometry.png)

## 完整统计

| 冻结模型 | 参数组 | latent权重 | 有效cos批数 | 平均main/latent余弦 | 负余弦批数 | 平均总梯度范数 | 全局clip激活 | main·total<0 | physical·total<0 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MSAT_FeatureCond | encoder | 0.1 | 40/40 | 0.547769 | 0 | 0.068619 | 0/40 | 0/40 | 0/40 |
| MSAT_FeatureCond | encoder | 1.0 | 40/40 | 0.547769 | 0 | 0.288411 | 0/40 | 0/40 | 0/40 |
| MSAT_FeatureCond | latent_head | 0.1 | 40/40 | 0.353178 | 0 | 0.019205 | 0/40 | 0/40 | 0/40 |
| MSAT_FeatureCond | latent_head | 1.0 | 40/40 | 0.353178 | 0 | 0.118500 | 0/40 | 0/40 | 0/40 |
| MSAT_FeatureCond | FiLM | 0.1 | 0/40 | N/A（aux零梯度） | 0 | 0.043731 | 0/40 | 0/40 | 0/40 |
| MSAT_FeatureCond | FiLM | 1.0 | 0/40 | N/A（aux零梯度） | 0 | 0.043731 | 0/40 | 0/40 | 0/40 |
| MSAT_FeatureCond | decoder | 0.1 | 0/40 | N/A（aux零梯度） | 0 | 0.133122 | 0/40 | 0/40 | 0/40 |
| MSAT_FeatureCond | decoder | 1.0 | 0/40 | N/A（aux零梯度） | 0 | 0.133122 | 0/40 | 0/40 | 0/40 |
| MSAT_FeatureCond | all | 0.1 | 40/40 | 0.188601 | 0 | 0.158733 | 0/40 | 0/40 | 0/40 |
| MSAT_FeatureCond | all | 1.0 | 40/40 | 0.188601 | 0 | 0.345147 | 0/40 | 0/40 | 0/40 |
| MSAT_FeatureStrong | encoder | 0.1 | 40/40 | 0.599657 | 1 | 0.077329 | 0/40 | 0/40 | 0/40 |
| MSAT_FeatureStrong | encoder | 1.0 | 40/40 | 0.599657 | 1 | 0.278693 | 0/40 | 0/40 | 0/40 |
| MSAT_FeatureStrong | latent_head | 0.1 | 40/40 | 0.311160 | 0 | 0.024165 | 0/40 | 0/40 | 0/40 |
| MSAT_FeatureStrong | latent_head | 1.0 | 40/40 | 0.311160 | 0 | 0.109312 | 0/40 | 0/40 | 0/40 |
| MSAT_FeatureStrong | FiLM | 0.1 | 0/40 | N/A（aux零梯度） | 0 | 0.055207 | 0/40 | 0/40 | 0/40 |
| MSAT_FeatureStrong | FiLM | 1.0 | 0/40 | N/A（aux零梯度） | 0 | 0.055207 | 0/40 | 0/40 | 0/40 |
| MSAT_FeatureStrong | decoder | 0.1 | 0/40 | N/A（aux零梯度） | 0 | 0.133110 | 0/40 | 0/40 | 0/40 |
| MSAT_FeatureStrong | decoder | 1.0 | 0/40 | N/A（aux零梯度） | 0 | 0.133110 | 0/40 | 0/40 | 0/40 |
| MSAT_FeatureStrong | all | 0.1 | 40/40 | 0.205536 | 1 | 0.167749 | 0/40 | 0/40 | 0/40 |
| MSAT_FeatureStrong | all | 1.0 | 40/40 | 0.205536 | 1 | 0.335029 | 0/40 | 0/40 | 0/40 |

main定义为IQ+.05STFT+.1输出F0特征指导，latent_unit为未加权连续特征cos+源方差标准化MSE。每个冻结theta均计算latent权重.1和1两个反事实，但只执行一次前向及三次autograd.grad，不更新模型。FiLM/decoder的latent直接梯度为0，余弦未定义，不能把null当作0相似或冲突。

## 结论与边界

编码器平均main/latent余弦分别为.547769与.599657，负余弦批数分别0/40与1/40。全模型平均余弦.188601/.205536。80个源批次在两种权重下均未触发全局clip1，main·total及physical·total为负的批数均为0。当前末轮点的局部梯度大多同向，不能把上轮性能下降归因于“持续梯度打架”或“末轮裁剪压制重建”。

这些梯度来自final30/eval和固定512源包，不是实际63000随机排列的训练批次。没有重放AdamW动量、二阶矩或weight decay；点积只描述当前欧氏梯度的局部几何，不预测实际AdamW参数步。末轮没有clip并不证明早期从未clip；本实验不读取技术恢复快照作科学扫描。

因而暂不新增PCGrad、分组裁剪或依据这些点积改变优化器。源latent误差改善而IQ/输出F误差变差仍是已核实事实，其原因也可能涉及优化轨迹、表示和条件解码结构；本诊断没有分离这些因素，更不能解释未见RX的全部性能损失。

已有独立源路径诊断还显示：原FeatureCond投影前raw平均复功率75.83，而实际输出被单位功率投影。该尺度方向未被直接监督，不能把raw当作可部署输出。下一项可检验源假设是为投影前输出增加与原clean单位功率匹配的幅度约束，抑制任意尺度漂移；这只来自源数据约定和冻结路径证据，不是已证明的性能修复，也不据Main错误选择权重。

## 执行、复算与范围

两个G均为原LsUs从零single30来源，FeatureCond/c865及FeatureStrong/3e541 strict final30/7410；原F0/3ff8始终冻结。原R07 L_s128/U_s384固定512包×20draw、B256，每G40批，共80批/20480条G前向。输出F0特征20480条+clean参考512=20992，sourceLs2 smoke另G4/F0特征6；默认头调用0。实际耗时27.118665秒。

G只临时允许autograd导数，不执行backward、optimizer，不积累.grad，最终恢复requires_grad=False并逐state核保持；F0参数与buffers始终保持。0TX/V/OE/novel/target、0模型更新或统计拟合。两G、全部20draw、五参数组、三目标及两权重均保留。512包重复draw不等于20480条独立物理记录。

3聚焦测试/一次P0P1及真实Ls2三目标导数smoke通过。独立NumPy复算80批的原源索引、所有Gram关系、全局clip、点积/余弦及全部汇总PASS，最大统计差2.78e-17；分组Gram和与all的最大差1.86e-08属于FP32归约误差，latent对FiLM/decoder直接梯度严格为0。复算0新增NN、更新或TX读取。

VERIFIED_ARTIFACTS_COMPLETE：控制器初始PID/CWD/命令独立存活读回、真实smoke、全部80批日志、完整统计和worker/controller exit0均已核实。GPU4准入free24110MiB/compute0，提交环境指定对应UUID。worker在两次读取间27.12秒内自然完成，未捕获其中途GPU存活快照；这一项保留观测缺项，不伪称已核验，也不为补监控重跑。GPU分配由提交环境/执行代码和完成产物间接支持，本耗时不作隔离GPU速度比较。

相关性能：[权重1完整Main七景](../20261101-phase1-Restore-FeatureStrong-E30-s2026092701-r01/analysis.md)，0/6，六景均低于原权重.1，cross_both clean还少38正确，默认DURA保持。源机制依据：[冻结条件路径](../20261101-diagnostic-Restore-FeatureCond-source-paths-s2026092701-r01/analysis.md)。本诊断不构成新的恢复器版本、Main重测或理论恢复上限，完整目标保持ACTIVE。
