# PAMF-OW 完整贡献与参考条件残差

本轮把已完成的未知身份几何诊断落实为同结构两行完整方法试验。原 M 的源未知 R_U45.396825%、target A_K60.590278%/R_U64.583333%；open64 源跨RX/day近邻76.984127%仍保留结构，但RX组间比例16.286785%高于原fused160的2.887514%。这些是开发观察，不证明单个分支造成失败。

实际代码边界是：fused160由时频base_norm、带原增益PA、带原增益reference和已经限幅SIG相加；reference只在末端加性融合，未条件校正t2/PA/SIG。原open64保留了GN/PA-LN/SIG-LN，却没有完整继承时频、reference贡献及PA尺度。因此本轮先改变开放输入的边界，再以匹配控制检验参考条件信息。

## 选择与范围

已排除把独立fused→MLP、raw-fused VICReg、uniform-six OE或Gram保持换名作为主创新，这些路线已有负结果。直接跳到大容量target伪标签适应也不能解释当前输入边界，且此前适应损伤未知拒识。本轮采用原生可微完整贡献和小容量条件残差，不宣称任何RX逆或硬件恢复保证。

这是Agent依据现有证据作出的实施选择，延续用户“自行设计、分工并持续推进”的授权；不另建设计批准门槛。与项目最小流程冲突的额外设计审批、builder review、固定规划链均NONBLOCKING/REJECTED_EXTRA_GATE；仅完成一次下一真实run所需的直接P0/P1审查。矩阵是同结构学习机制比较，不是PAMF-CE-only网络结构面板。

## 同次可微贡献与条件模块

从原 `reader.features` 同次前向捕获：

1. `cls_head.base_norm` 输出160，原时频联合贡献。
2. `cls_head.pa_norm` 输出160乘原 `tanh(cls_head.gain)`，实际PA贡献。
3. `response_projection` 输出160乘原 `tanh(response_gain)`，实际参考贡献；同时捕获NoExplicitCFO后的29维response。
4. `sequence_pool` 实际输出160，已经包含三视图固定排列、投影/LN和有界增益。

不得重算一份不同的分支、detach或借历史NumPy缓存充当端到端优化。四贡献按原次序相加应与原raw fused160数值一致；分类残差的基底直接使用原前向返回值，零增益时原默认头logits逐值一致。NoExplicitCFO坐标27保持0，内部局部补偿保留；不新增canonical分支或回写IQ。

设四个贡献为c_j∈R160。条件网络h为`29→32/GELU→640`，最后linear权重和bias置零。每分支：

`delta_j = 0.1 * ||c_j||_2 / sqrt(160) * tanh(h_j(r))`，`c'_j=c_j+delta_j`。

因此每分支残差范数≤0.1||c_j||，原贡献为0时保留0。这个界不是跨RX准确率保证。C0给h的条件是全零29向量，仍允许学习常量残差；C1给本条真实masked reference29。两组都运行同一网络、参数均可训练、同一初始化，不用目标batch统计或Q拟合。

concat四个c'_j共640维，经`Linear640→64/GELU/Linear64→64/LayerNorm64/unit`为open64。共享关系评分器和原默认头身份残差沿用：`f_new=f_base+0.1||f_base||tanh(g160)*unit(H z_open)`，g初始0。原raw fused不整体单位化，仍由原cosine头内部处理。旧类身份仍取原默认六类头，关系仅判断注册集合成员。

参数设计计算：projector45312、H+g10560、relation4161、conditioner22080，共新增82113；识别器315106，原native容器900722加新增为982835，未用域参数仍667729。C0/C1参数和trainable数量相同。实际构造器和烟测须核对，未制作剥离容器部署权重。

## 固定训练矩阵

| 行 | 输入贡献/全部损失 | 条件信息 |
| --- | --- | --- |
| C0 | 完整四贡献，轮换五类成员任务 | 零reference输入，学习常量残差 |
| C1 唯一主 | 同C0 | 本条NoExplicitCFO reference29 |

两行从原Tri-SIG-TIME SSL E50/11100独立开始，不从目标成绩较好的控制或已训练M继续。newmodule seed2026092701，采样2026103017、槽排列2026103018；同L/OE批和2220 fresh CENTER_UNITS AdamW步骤/lr1e-4/原wd/clip5，累计13320，非自然epoch。

原完整L128六前向768行、CRIC.1、ordinary/raw KL、AsCE且AsKL0、LS.01；As64重放既定P1，不新造频道，encoder eval关闭dropout/BN统计，顶层train执行真实Lhook。另raw同P1 KD128行/T2权重1，加support96/query96/OE64行。相对集合五类任务与已完成M完全相同，两行不detach成员梯度。共享L+KD梯度冲突投影/成员范数cap，relation独立；conditioner属于受保护共享参数。每行训练F2557440实物行，所有重复曝光单列。

所有原生Sinc/骨干/原默认头保持联合训练。原U/PL/EMA在本增量关闭，原phase1 SSL祖先保持；这不是从头重训或把当前完整PAMF改名纯监督。旧phase1/phase2默认和原权重不变。

## 数据、评分与边界

源L6300/OE6134/V27000/novel3150，沿原G10MIX01 cachedP1和原F0/DURA ancestry；源5RX/3day，真旧6标签只L/V合法，OE仅outside-six与RX/day、真实TX/K隐藏。sourceL支持/查询跨RX且跨day、物理ID互斥；原F0祖先已见所有源旧类/RX，留一类是相对注册集合的训练代理。

模型冻结后L形成六原型，V仅按既定42组known风险校准六个predicted-class阈值；随后两行都完成novel3150和RX24-1 futureQ864预测，再独立CPU truth-last。source未知90不作额外目标入口；源novel和开发Q已暴露，不宣称新盲测，不用其成绩换主/选checkpoint/调参/重采样。target U384和正式RX24-18保持不读。所有postPhase1新IQ仍星地received/P1，P0不表示clean。

主报逐真实TX宏A_K/R_U/H，伴C_K、原同P1F0closed差和候选自身closed差，逐TX/RX/day/六scene。targets v3 CK≥90%、RU≥90%、A_K≥原F0closed−2pp；H不能抵单项。Q practical_high未知计数0/4/2缺类，保持N/A。仅DA0_REG0，AN/HOW/UDA/匿名发现注册仍N/A，完整目标ACTIVE。

默认N607、最多两行并发，每卡最多新增2个own训练/预测进程，每次实际free≥8192MB，允许foreign共卡而不停止/修改/重启。技术失败保留、健康行继续，无自动重试或按科学分数提前停/调；UNKNOWN先核对原PID/root。固定新run、代码commit、最小预登记、一次preflight/单archive比较/compile、真实checkpoint首步smoke及独立进程/log证据，预测齐后truth-last；具体命令与路径在run登记。
