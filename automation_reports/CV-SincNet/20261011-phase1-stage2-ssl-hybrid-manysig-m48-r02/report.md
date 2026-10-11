# 半监督混合机制与更强权重调优探索

用户明确恢复半监督，纯CE仅为对照；关闭星地增强。源域角色 L_s6300、U_s56700、V27000不变。U的TX在数据集边界隐藏，方法只看到IQ、opaque ID及合法RX/day元数据。目标数据不用于训练、选模或方案选择。

48行：16方案×3seed（2026092701/02/03），每行匹配原始纯CE旧门控cosine E200，freshAdamW续训40epoch/8880更新；全部身份骨干、融合及分类参数可训练。逐行冻结后自动clean168000预测，独立truth-last评分；报告全部方案、每seed、RX/TX/day分层、配对差和实际成本。主方案S13，目标三seed均值84%。不读取当前H批次目标分数决定本设计。

半监督：教师初始化为同seed E200身份模型，eval/no_grad；每成功身份更新做一次cache-safe EMA（.999）。每步L128＋U256，U完整随机排列后才复用；每个U有实际遍历pass。teacherclean预测→RX/day置信门控quantile.86 clipped.92–.97→稳定性→studentstrong一致；S0沿用batch邻近window2/minconf.80，其余用同physicalID连续两次U遍历预测一致。pseudoCE权重.16，Uentropy.01，444step渐进。studentstrong只加std.015高斯，不生成星地信道。S11只将已通过全部门控的U伪标签加入跨RX对比，不重复计L对比。SAM两个closure共享teacher/history/strongIQ计划，EMA只在一次真实AdamW更新后执行。U真准确率N/A，绝不取隐藏标签计算。

| Arm | 机制 | encoder/head峰值LR |
|---|---|---|
|S0|旧邻近半监督CE|3e-5/1e-4|
|S1|遍历稳定半监督CE对照|3e-5/1e-4|
|S2|S1＋跨RX监督对比|3e-5/1e-4|
|S3|S1＋同TX跨RX特征混合|3e-5/1e-4|
|S4|S1＋manifoldmix|3e-5/1e-4|
|S5|S1＋对比＋跨RX混合|3e-5/1e-4|
|S6|S1＋SAM|3e-5/1e-4|
|S7|S6＋固定尾部平均|3e-5/1e-4|
|S8|S5＋coverage有限LT|3e-5/1e-4|
|S9|S8＋SAM＋固定尾部平均|3e-5/1e-4|
|S10|S5＋条件有限LT|3e-5/1e-4|
|S11|S5对比加入已接纳U|3e-5/1e-4|
|S12|S1加大权重更新|1e-4/3e-4|
|S13|S5加大权重更新（主）|1e-4/3e-4|
|S14|S9加大权重更新|1e-4/3e-4|
|S15|S5十倍峰值LR|3e-4/1e-3|

LR warmup444后cosine；平均变体E31–40固定2epoch循环并平均E32/34/36/38/40，不以V挑选。已有source-only E200辅助LT和关系采样网络复用，严格核对sourceIDs/seed/parent；训练中冻结，不新增在线辅助拟合。更高LR仅表示更强调优，不能预先认定更高精度；记录实际梯度、权重位移、损失、源V和U接纳/覆盖。

8GPU，每卡≤3总进程（包含其他任务），启动前至少12GB空闲；技术失败保留，不自动重试，不因低分停止健康任务。本记录在任何新目标成绩读取前冻结。当前为预登记，尚未发布，84%未验证。


技术恢复 20261011-phase1-stage2-ssl-hybrid-manysig-m48-r02：仅调整原生bootstrap/导入顺序；沿用原冻结48行与全部科学参数。旧发布未启动训练，原记录和远端目录保留。
