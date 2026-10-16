# F＋跨 RX：clean 与 As 梯度关系的冻结源诊断

上一轮 LA16 已按原规则关闭：源 RX1 clean +2.555556 个百分点，但六 practical 全降、low −1.370370 个百分点。新的循环移位旁路结果同样未晋级；另一个 owner 正在运行 additive As 的完整监督对照，本任务不重复启动。

当前唯一问题是：在现有 F＋跨 RX 同 TX 基线中，完整 As 的监督梯度是否比同 IQ 重复 clean 控制更经常反向于原监督，并在范数上支配原监督。实际模型静态构建核对为 900722 参数、GroupNorm/LayerNorm，0 BatchNorm 和运行统计，已排除 BN 状态污染这一实现解释。

使用同一合法 RX1-held 契约的已冻结 C0/C1 final20 默认头，各自 scratch、无祖先或目标选模；不继承为新训练模型。只用 fit RX3/4/6/8 的 L5040（72 个 TX×RX×day 单元各70包），原源缓存读取权限和 VALIDATED_ONCE 结论复用。旧缓存同时反序列化源 V 及旧父特征，但不消费其数值或做网络前向；不读取 U 标签、target，heldRX1 IQ 也不参与。

全部5040按独立、标签无关 permutation 分成40批128/最后48，保留全部样本。两条命名观测流，每批只生成一次普通 IQ、一次完整 practical(raw)，两模型复用；As 独立选择原三个训练场景的 nominal 先验，无质量拒绝或采样过滤。RX 硬件保持 RX/day/eq session 稳定，session 不由 TX 定义。实际普通配置与原 C0、末期 E13 截断 configure 保持，PA/shift不旁路，As 不叠加 ordinary 的噪声或多径。实际 ch.Config 与 C1逐项对比，25MHz/residual/post_sync/EQoff G=1/reference30/reflect，正式测试完全不动。

每模型每批做2次原ordinary主前向、2次同ordinary控制和2次As前向；同IQ RDrop(alpha1/LS.01)、原浅中层 SupCon .1只使用主对，两次secondary使用相同命名dropout流。p=.5*dRD_primary+.1*dSupCon，s=.5*dRD_secondary；按原F约定把Sinc q梯度除25000换回Hz坐标，记录whole/defaulthead/SIG_TIME/Sinc/timepath/remaining的平方范数和内积，独立复算cosine、范数比、反向比例和p+s方向。

这只是欧氏一阶几何，不是实际 AdamW 步长、held 泛化、纯RX损伤或身份提升。保存实际默认头每包预测和margin/30作描述，不增加或拟合分类头。不调整任何参数、optimizer、EMA、归一化或持久buffer；所有state_dict末状态必须逐tensor相等，.grad必须仍为None。

固定证据规则：C0/C1 × 两条流各40批都要求whole的As/primary范数比中位数>1，且As与primary负内积批比例至少高于匹配clean控制0.20。全部通过只支持另行登记训练组织实验，不自动训练或晋级；否则关闭梯度支配解释，不按层/RX/流/阈值救援。任何结果都不称Ag已验证或low已改善。梯度或健康检查永远不替代后续真实识别。

新增训练0模型轮/0更新/0头拟合/0EMA；载入2个既有E20 CP，120964网络包=2模型×2流×5040×6前向+两CP各2包zero smoke；60480默认头描述性源决策。2个父模型已有40模型轮/8880更新成本单列，不把历史训练隐藏。源候选仅诊断，不进入正式target。

本轮最小流程：局部协议负例和真实网络合成梯度检查、一次限定P0/P1、代码提交、一个run和release、单归档比较/远端compile、worker先做两真实CP no-query smoke并立即继续、唯一owner和独立PID/CWD/argv/GPU/log增长核验。只允许本run的协议、来源、非有限、状态变化、输出碰撞或无法闭合预测等技术失败停止，保留partial、无自动重试、不干预健康外部任务。公共小型汇总入Git，IQ/CP/逐包轨迹/原日志留私有产物。

精确config、命令、环境/CWD、路径、GPU、stop和预期产物以同run experiment.json为准。REJECTED_EXTRA_GATE：brainstorming 的独立设计审批超出当前项目最小流程白名单；本任务已有研究授权，不增加审批等待。
