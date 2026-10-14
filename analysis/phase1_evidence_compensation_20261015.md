# 下一独立机制：有停止条件的身份主路径证据补偿

状态：**辅助目标已实现，全源冻结机会普查已 ANALYZED / VERIFIED**。尚无本机制实际训练或泛化收益；84% 默认头 clean 三种子目标仍未完成。

新的源实验排除了两个固定变体。跨 RX PA 完整交换源 gate 0/8；直接修正 AdamW 的源 RX 风险位移也为 0/8。后一项末态修正保留约 92% 位移，改变真实浅／中层特征，却在独立 L 上比 RAW 损失更少的平均 CE 改善。问题不能靠放大原方向或仅使一个风险子集的更新可行来解决。已有集合候选版本错误排除真实类别，重复观测预测版本又未通过源身份条件，也不重启原配置。

下一机制重新关注**监督内容**。保留原主 CE、默认头联合训练、原高可信 U 监督。额外任务只问：当前完整证据足以给出可信身份时，时频主路径是否能在物理增量适度减弱后补足身份判断？这个任务不增加低可信 U 的硬类别，也不要求学生在受挑战的证据上先预测一致。

真实部署特征的四项为

\[
z=a+b+c+s,
\quad b=\tanh(g_{PA})p,
\quad c=\tanh(g_{ref})r,
\quad s=0.2\tanh(g_{SIG})s_{raw}.
\]

其中 `a` 是原 head.base_norm 输出，`p/r/s_raw` 必须从实际同次前向直接取得，不能从 joint 相减猜分支。主分类器仍是既有 scale30／eps1e-4 cosine 头。新任务不添加头或参数。

定义两个受控特征视图，先固定使用报告中的 β=.5、π=.9，不依据任何 target 结果扫描：

\[
z^{-PA}=a+s+(1-\beta)\operatorname{sg}(b)+\operatorname{sg}(c),
\qquad
z^{-ref}=a+s+\operatorname{sg}(b)+(1-\beta)\operatorname{sg}(c).
\]

这里每个辅助视图的 PA／reference 端点都停止梯度。辅助 W 和全部融合 gain 也停止梯度，仅时频主路径、SIG_TIME 及原 base normalization 取得辅助梯度；辅助 `s` 用原 SIG gain 的 detached 值计算。实现保留原 `((a+b)+c)+s` 加法顺序，SELF 控制须精确还原完整 joint，不能以交换加法顺序带来的 FP32 差异作为新信号。共享参数若还能经 `a/s` 收到梯度，必须如实记录，不能声称某个模块整体被冻结。原主 CE 仍训练全部既有特征和头，实际部署 z 仍含全部四项。

身份目标 `y*` 在 L 上来自真实标签，在 U 上仅复用原已接受的可信教师目标；不放宽原 U 的身份接受链、不用 U 真值匹配类别。受挑战视图是否已预测正确不是新增入口条件。用当前完整学生表示计算 SELF 停止损失作匹配控制，再比较两个挑战视图：

\[
\ell_{stop}(v,y^*)=[\log\pi-\log p_{\operatorname{sg}(W)}(y^*\mid v)]_+,
\qquad
\mathcal L_{comp}=\frac{0.1}{2B}\sum_i q_i
\{\ell_{stop}(z_i^{-PA},y_i^*)+\ell_{stop}(z_i^{-ref},y_i^*)\}.
\]

分母固定为完整当前 batch 的 B，不能随活跃数或接受数变化。`q` 在 L 恒为1，在 U 是原接受 mask。停止条件防止该辅助项在已有证据足够时无限锐化概率；原主 CE 的锐化行为并不因此消失。两个视图均保留一半相应增量，既不全删物理路径，也不称真实信道／硬件反事实。

候选集合监督原来试图给低可信样本有限类别约束，本方案先在可信身份内检查“还有哪部分证据没有学会”，两者不能混称成功版本。PA donor 重组改动跨包证据，本方案仍使用同包增量，并通过端点 stop-gradient 阻止辅助损失仅加强剩余 PA/ref 分量。关键待检验点是：a/s 的实际学习能否改善其他源 RX 的身份特征，而不只是降低受挑战视图 CE。

学习受挑战特征已有 RSC 基础；用可信弱视图目标监督另一种视图已有 FixMatch 基础。因此“挑战＋教师”概念不宣称原创。这里拟研究的具体特点是 CVS 真实四项结构、保持物理侧路部署贡献、原身份主路径定向辅助梯度和有停止条件的可信 U 学习，仍需文献核验及实证。[RSC](https://arxiv.org/abs/2007.02454)，[FixMatch](https://arxiv.org/abs/2001.07685)。

下一轮最多两个动作：先做冻结模型的监督机会普查，再验证真正的主路径梯度／AdamW 位移。

1. 普查保留原身份监督，记录完整／PA减弱／ref减弱的目标概率、停止项活跃率、独立物理样本规模及 RX/day 分布。源 V 只作无拟合 proxy 计数，不用于梯度；不仅报告所有接受样本的 teacher 精度，还必须报告**辅助活跃子集**的源代理精度，不能把总体高精度套到困难子集。真实 U 标签保持隐藏，采样仅依合法物理 ID、RX/day 或固定随机顺序，不读取 U 的 TX 元数据。若只剩很小规模、活跃子集错误率高或几乎全部已停止，不提高 π、扩大 β 或增加门控制造机会。
2. 仅在有足够合法机会时，做源 L 与隐藏 U 的真实图、定向参数梯度、同状态 AdamW 及独立 source-L 有限步诊断。比较 ZERO／SELF_STOP／EVIDENCE_STOP，相同原主损失、有效曝光和 readout 计算预算；默认 W、三 gain、PA/ref 端点的辅助梯度范围与主 CE 的原联合梯度分别验证。V 和 target 不作元训练数据。

真实 source 诊断发布前另行预登记矩阵、状态来源、规则、预算和停止条件；现在不提供不存在的执行状态。来源优先考察各 seed 固定 BASE_CE E20 父权重及由该学生复制的初始教师，不按目标最高分选权重；如果普查成立，优先准备明确重置优化器的 E20+20 续训，累计40轮，匹配纯CE／原SSL／SELF_STOP／EVIDENCE_STOP从同父独立分叉。新增20轮阶段、U启用与cos日程在发布前固定，不冒充scratch20成本，不从历史E50串接。是否实际发布仍取决于科学否证，逐row原默认头clean truth-last保持。SIG_TIME、固定CENTER25K、COMPACT_LINEAR、LONG_SILU历史正收益独立保留，不能相加或按目标结果挑选父权重。


2026-10-15 固定普查 run `20261015-phase1-SIGTIME-evidence-compensation-sourceVU-s2026092701-r01`，执行`ee303b067ebd614986bc5cd59e2d634935979394`，source gate NOT_PASSED。完整V27000/U56700、167404forwards、原q不改，活跃机会及困难子集源proxy精度逐RX/day统计；U真值始终隐藏，0训练/头拟合/EMA/target，真实CPsmoke及独立终态核实。判定`CLOSE_FIXED_TRUSTED_U_COMPENSATION_ENTRY`，不根据结果扫描beta/pi/门控/sampler。详见[完整普查报告](../automation_reports/CV-SincNet/20261015-phase1-SIGTIME-evidence-compensation-sourceVU-s2026092701-r01/report.md)。

普查明确关闭固定原可信U入口：V活跃4、U活跃3，SELF均为0，不能靠参数扫描恢复。下一独立假设见[跨RX类间特征差方向学习](phase1_crossrx_identity_difference_20261015.md)，由合法L真实身份产生监督，尚未实现/发布/训练。
