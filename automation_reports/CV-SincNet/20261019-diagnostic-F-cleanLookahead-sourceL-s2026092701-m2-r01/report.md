# 纯 F：clean 一步更新后的 As 响应诊断

状态：DEVELOPMENT_NOT_RELEASED。已实现并验证可丢弃的一步更新内核；真实源 worker、统一登记、限定独立 P0/P1 和远端发布尚未完成。没有新真实 IQ、checkpoint、源模型前向或正式测试访问，不能报告源诊断结果，也没有后台新任务。

总体目标仍为固定 F 的 low urban >60%、同预算 NoAs clean 可重复正收益及逐场景 suburban 接近自身 clean。本诊断只检验下一假设，不替代识别或多种子证据，不因一个梯度或虚拟步骤通过就自动延长 E50。

## 已排除的近邻及现有证据

本轮只基于源证据推进。先前 nontrue teacher、signed input Jacobian、场景风险和 capability-label CE 配方的原关闭决定保持；不换系数、教师、seed 或名称救回。另一聊天的 U practical 路由、HQ 标签视图及噪声 KL 不纳入本轮。旧 PCGrad 是同一点的梯度投影，splitAdam 是同一点的独立矩状态，SAM 是参数上升扰动；它们不等同于下述普通 clean 下降一步后的有限非线性响应。旧“所有最终模型持续受 As 梯度支配”解释已有源反证，本诊断不恢复该假设。

一个仅使用既有 B1 FINAL20 V360、两个固定 draw logits 的离线数学检查已完成：wrong 样本中至少两个非真类超过真类的数量为 mid0 51/85、low0 88/129、mid1 56/94、low1 94/139。但把 multiclass CE 与 conditional-smoothed pairwise logistic 各占一半后，wrong 样本梯度相对 CE 的平均转角仅为 2.5641°、2.5151°、2.3646°、2.3457°，主要变化是逐包梯度尺度（整体中位数约0.60至0.66）。因此暂不为这条近邻损失发布 E20，不把“数学不完全相等”当作新的有效监督。该检查0新NN/IQ/更新/目标访问，不拟合权重或阈值，不使用教师输出；代码 `analysis/cvs_F_pair_direction_cached_20261019.py`，聚合输出保留私有目录。

## 唯一拟检验问题与范围

问题：同一固定模型上，先沿 raw clean CE 做一次有界且可丢弃的下降更新，是否改变 full As 的梯度，并使一阶 lookahead 合成更新比直接 joint CE 在另一批合法 L 包上有更好的实际误差响应。当前没有证据证明答案为是。

这借鉴 [MLDG 原始研究](https://aaai.org/papers/11596-learning-to-generalize-meta-learning-for-domain-generalization/) 中在虚拟训练更新后检查另一条件损失的原则。本项目拟用 raw clean/full practical、fresh AdamW/CENTER_UNITS 和一阶传递；不是原论文的精确二阶 MLDG，不声称论文成绩可迁移至 CVS。

两份固定旧纯 F checkpoint 为 `20261019-phase1-F-nontrue-CSIL-e20-s2026092701-m4-r01` 的 B0、B1 FINAL20，各自实际完整 L6300/U56700 hidden TX/V27000、RX1/3/4/6/8、day1/2/3、split392005、scratch/final_only/no target 来源已核实。真实模型加载继续复用现有 own_model 严格来源检查。原训练、权重和测试产物只读，未追加训练，不继承 CSIL。

拟固定每个 TX×RX×day 的既有 L 缓存前两包作 step 集、接着两包作 response 集，各180包，共360个不同物理L；按day/RX/TX固定交错，每batch30覆盖全部6TX×5RX。两组包都曾被历史模型训练，response只表示本次算子的物理包互斥，不冒称历史未见验证集或新的L_dev。V/U/target不参与梯度、响应或选择；复用原 source handle 和缓存，不重新做数据验证。缓存中其他角色/旧特征即使随容器反序列化也不消费数值。

raw clean 不经过 ordinary 随机增强，不能冒充 ordinary 更新；As 从同一 raw 一次完整引擎生成，mid/low urban 各一份预先命名的独立诊断随机流，真实传播、噪声、接收质量、锁定/补偿/AGC依赖和noEQ不变。既有正式IQ/SNR/困难样本不改。两模型及所有反事实共用同一实际源观测，硬件session按RX/day，不按TX。

拟比较原参数处的 raw/As loss、raw-only一步、直接0.5raw＋0.5As一步，以及0.5raw原梯度＋0.5As在raw虚拟点的梯度合成一步。每个步骤从同一原参数重新复制，Adam moments为空；lr0.0002、WD0.0001、betas0.9/0.999、eps1e-8、原全局physical clip5及CENTER_UNITS均固定。它是明确重置优化器的有限反事实，不重建原E20 Adam轨迹，也不把源一步结果当作E20训练收益。

原默认cosine6×160/scale30、实际joint160、900722参数及F结构保持。每个副本的Sinc、骨干与默认头共同更新，不另训头；副本最终丢弃，原模型state/.grad/EMA/RNG保持。预计2模型×6批×3副本=36次可丢弃更新，另各一个真实CP无query smoke的副本，共38次；新持久模型更新0。预计实际NN6128（含8 synthetic real-CP smoke），原模型梯度与lookahead梯度的样本计算另列；真实计数以worker完成后核对。当前worker未实现，因此上述矩阵和成本在正式登记前仍需与实际入口一致固定。

## 当前实际实现与验证

`experiments/cvs_clean_lookahead/virtual.py` 使用原Sinc物理单位optimizer函数，在完整非别名副本上执行恰好一次fresh AdamW/clip5，返回实际参数位移；不在原模型写grad或参数。`checks.py` 已通过独立第一步闭式公式（最大参数绝对误差5.82e-11）、原state/grad/global RNG保留、非别名、4输入负测，以及实际F/defaulthead前向、真实梯度和改变logits路径。当前成功版本8个合成NN、2次丢弃副本更新、0真实IQ/CP/target；早期测试以通用forward错误索引字典而失败，改为实际forward_identity_only接口后通过，未发生远端执行。

开发时可运行：`ssr-gpu/python -X utf8 -B -m experiments.cvs_clean_lookahead.checks`。真正来源核对、一次真实CP无query smoke、唯一独立P0/P1、统一experiment/events/row登记、不可覆盖release/run、普通N607 preflight/单archive传输比对/compile及PID/CWD/argv/GPU/log增长均待worker完成后执行；不能把本地合成PASS写成发布或源性能成功。当前输出保护目录为 `local_artifacts/F_cleanLookahead_source_20261019`，不入Git。

本轮明确只做源诊断。无新训练完成事件，不启动正式目标预测或评分；后续真实E20机制需要该问题的实际源证据和单独冻结配方。总体目标保持ACTIVE。
