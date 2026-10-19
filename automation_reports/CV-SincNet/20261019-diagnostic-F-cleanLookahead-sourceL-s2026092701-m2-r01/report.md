# 纯 F：clean 一步更新后的 As 响应诊断

状态：LOCAL_VERIFIED，未发布。真实源 worker、完整批路径及统一两行登记已完成；限定独立 P0/P1 和远端发布尚未完成。没有新真实 IQ、checkpoint、源模型前向或正式测试访问，不能报告源诊断结果，也没有后台新任务。

总体目标仍为固定 F 的 low urban >60%、同预算 NoAs clean 可重复正收益及逐场景 suburban 接近自身 clean。本诊断只检验下一假设，不替代识别或多种子证据，不因一个梯度或虚拟步骤通过就自动延长 E50。

## 已排除的近邻及现有证据

本轮只基于源证据推进。先前 nontrue teacher、signed input Jacobian、场景风险和 capability-label CE 配方的原关闭决定保持；不换系数、教师、seed 或名称救回。另一聊天的 U practical 路由、HQ 标签视图及噪声 KL 不纳入本轮。旧 PCGrad 是同一点的梯度投影，splitAdam 是同一点的独立矩状态，SAM 是参数上升扰动；它们不等同于下述普通 clean 下降一步后的有限非线性响应。旧“所有最终模型持续受 As 梯度支配”解释已有源反证，本诊断不恢复该假设。

一个仅使用既有 B1 FINAL20 V360、两个固定 draw logits 的离线数学检查已完成：wrong 样本中至少两个非真类超过真类的数量为 mid0 51/85、low0 88/129、mid1 56/94、low1 94/139。但把 multiclass CE 与 conditional-smoothed pairwise logistic 各占一半后，wrong 样本梯度相对 CE 的平均转角仅为 2.5641°、2.5151°、2.3646°、2.3457°，主要变化是逐包梯度尺度（整体中位数约0.60至0.66）。因此暂不为这条近邻损失发布 E20，不把“数学不完全相等”当作新的有效监督。该检查0新NN/IQ/更新/目标访问，不拟合权重或阈值，不使用教师输出；代码 `analysis/cvs_F_pair_direction_cached_20261019.py`，聚合输出保留私有目录。

## 唯一拟检验问题与范围

问题：同一固定模型上，先沿 raw clean CE 做一次有界且可丢弃的下降更新，是否改变 full As 的梯度，并使一阶 lookahead 合成更新比直接 joint CE 在另一批合法 L 包上有更好的实际误差响应。当前没有证据证明答案为是。

这借鉴 [MLDG 原始研究](https://aaai.org/papers/11596-learning-to-generalize-meta-learning-for-domain-generalization/) 中在虚拟训练更新后检查另一条件损失的原则。本项目拟用 raw clean/full practical、fresh AdamW/CENTER_UNITS 和一阶传递；不是原论文的精确二阶 MLDG，不声称论文成绩可迁移至 CVS。

两份固定旧纯 F checkpoint 为 `20261019-phase1-F-nontrue-CSIL-e20-s2026092701-m4-r01` 的 B0、B1 FINAL20，各自实际完整 L6300/U56700 hidden TX/V27000、RX1/3/4/6/8、day1/2/3、split392005、scratch/final_only/no target 来源已核实。真实模型加载继续复用现有 own_model 严格来源检查。原训练、权重和测试产物只读，未追加训练，不继承 CSIL。

固定每个 TX×RX×day 的既有 L 缓存前两包作 step 集、接着两包作 response 集，各180包，共360个不同物理L；按day/RX/TX固定交错，每batch30覆盖全部6TX×5RX。两组包都曾被历史模型训练，response只表示本次算子的物理包互斥，不冒称历史未见验证集或新的L_dev。V/U/target不参与梯度、响应或选择；复用原 source handle 和缓存，不重新做数据验证。缓存中其他角色/旧特征即使随容器反序列化也不消费数值。

raw clean 不经过 ordinary 随机增强，不能冒充 ordinary 更新；As 从同一 raw 一次完整引擎生成，mid/low urban 各一份预先命名的独立诊断随机流，真实传播、噪声、接收质量、锁定/补偿/AGC依赖和noEQ不变。既有正式IQ/SNR/困难样本不改。两模型及所有反事实共用同一实际源观测，硬件session按RX/day，不按TX。

比较原参数处的 raw/As loss、raw-only一步、直接0.5raw＋0.5As一步，以及0.5raw原梯度＋0.5As在raw虚拟点的梯度合成一步。每个步骤从同一原参数重新复制，Adam moments为空；lr0.0002、WD0.0001、betas0.9/0.999、eps1e-8、原全局physical clip5及CENTER_UNITS均固定。它是明确重置优化器的有限反事实，不重建原E20 Adam轨迹，也不把源一步结果当作E20训练收益。

原默认cosine6×160/scale30、实际joint160、900722参数及F结构保持。每个副本的Sinc、骨干与默认头共同更新，不另训头；副本最终丢弃，原模型state/.grad/EMA/RNG保持。预计2模型×6批×3副本=36次可丢弃更新，另各一个真实CP无query smoke的完整三副本批路径，共42次；新持久模型更新0。固定实际NN6188（含68 synthetic real-CP smoke），原模型梯度与lookahead梯度的样本计算另列；真实计数以worker完成后核对。已由worker实际调用路径和固定配置一致定义，network backward样本计算1820包；原始模型更新0，副本不保存成新checkpoint。

## 当前实际实现与验证

`experiments/cvs_clean_lookahead/virtual.py` 使用原Sinc物理单位optimizer函数，在完整非别名副本上执行恰好一次fresh AdamW/clip5，返回实际参数位移；不在原模型写grad或参数。`checks.py` 已通过独立第一步闭式公式（最大参数绝对误差5.82e-11）、原state/grad/global RNG保留、非别名、4输入负测，以及实际F/defaulthead前向、真实梯度和改变logits路径。当前成功版本8个合成NN、2次丢弃副本更新、0真实IQ/CP/target；早期测试以通用forward错误索引字典而失败，改为实际forward_identity_only接口后通过，未发生远端执行。

开发时可运行：`ssr-gpu/python -X utf8 -B -m experiments.cvs_clean_lookahead.checks`。来源实际payload核对、一次真实CP无query smoke、唯一独立P0/P1、不可覆盖release/run、普通N607 preflight/单archive传输比对/compile及PID/CWD/argv/GPU/log增长按最小流程执行；不能把本地合成PASS写成发布或源性能成功。当前输出保护目录为 `local_artifacts/F_cleanLookahead_source_20261019`，不入Git。

本轮明确只做源诊断。无新训练完成事件，不启动正式目标预测或评分；后续真实E20机制需要该问题的实际源证据和单独冻结配方。总体目标保持ACTIVE。


共享worker批路径本地检查通过：34合成NN/10 backward包/3丢弃更新，六项协议负测、180＋180选择及每批6TX×5RX、原state/grad/RNG保持、源决定PASS/FAIL反例均已验证。单步公式有效版本8合成NN/2丢弃更新另列；开发早期调用接口错误及其修正测试保留，按已观察到的调用本地累计48合成NN/7丢弃更新，0真实IQ/CP/目标。

固定源反证规则：分别对B0和B1，lookahead−joint的响应mid/low均值CE须低于−1e-6，raw CE差不大于+1e-6，且至少4/6批urban CE改善。数值1e-6只区分浮点近零，不是有意义识别提升门槛。PASS仅支持另行冻结E20假设，不自动训练或延长；FAIL关闭本次fresh-Adam raw-lookahead理由，不扫步长、inner状态或seed。所有源单步变化均不是历史未见RX、正式识别增量或多种子证据。

实际入口 `CVS-RFFI/python -u experiments/cvs_clean_lookahead/runner.py`，N607普通账户既有FP32/TF32off/CPU2；CWD `PROJECT/releases/cvs_F_cleanLookahead_source_20261019_r01`；不可覆盖输出 `PROJECT/runs/20261019-diagnostic-F-cleanLookahead-sourceL-s2026092701-m2-r01`，每arm的responses/complete独占子目录。唯一owner `codex/root/F-clean-lookahead`；每GPU含其他owner最多2进程、空闲至少12000MB，CUDA init在共享capacity锁内。只实际来源/权限/非有限/接口/预算/输出错误停止自身并保留，不自动重试或干预健康旧任务。精确row/seed/config及预期产物由experiment.json维护。


唯一独立P0/P1审查PASS，无会直接使本次源诊断错误、越权、覆盖或不可运行的缺陷。审查核对完整9文件/登记/配置与成本，新增NN0/未修改文件/未操作远端。实际code/config commit `0740b3e010e4c0cad39642767ee3690d96fb40ff`，发布前已完成本地聚焦验证；远端尚未启动，不以本地PASS冒充真实源证据。
