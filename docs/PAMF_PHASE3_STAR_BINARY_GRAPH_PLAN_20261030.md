# Phase3 冻结身份的二元图适应与匿名注册

本轮检验：保持原 PAMF 身份表征不变，只在无标签 U 池上适应“已知／未知”分数，是否能取得旧类正确接收收益并保护未知拒识。它是新的独立开发候选，不调整已完成 reciprocal 运行的模型、阈值或判定。

Run 为 `20261030-phase3-star-binary-graph-s2026092701-r01`，唯一主为 `DA1_REG1`，完整四态为 `DA0_REG0/DA1_REG0/DA0_REG1/DA1_REG1`。配置见 `configs/pamf_phase3_star_binary_graph_20261030.json`；真实命令、代码提交、资源、输出和状态在同名 `experiment.json/report.md`。

## 输入和监督

原 F0 完整三视图 Tri-SIG-TIME SSL E50/11100、232993 参数及默认六类身份头冻结；G 为 DURA v1.0 R69、266951 参数、FP32/TF32off/reuse=true/graph=true/native=false。原 source split392005 的 L6300/U_s56700/V27000 和 OE6134 成员监督仍是祖先。本轮仅复用 `20261029-phase3-star-neighborhood-s2026092701-r01/source`（code `cb58217de28eb2cb5536d2392aef268b79006877`）的 P0/P1 聚合 geometry、四个3233参数的 paired 读出快照和六个源域阈值，实际选 paired1000。不更新任何神经参数、geometry、normalizer 或阈值，不回读 source IQ／逐样本 source feature。

目标输入为原 G10 MIX01：RX24-1 的 U384（2021-03-08）用于无标签适应和发现，互斥 Q864（2021-03-15）只在冻结后逐条预测。Source RX1/3/4/6/8 与目标 RX 互斥；不是未见日期测试。U 为旧六类288和两到达设备96，Q 为旧576、两到达192及 later96；这些数量和名单只用于 builder/scorer 说明，真实 TX/K/配额不传入算法。原开发 RX/设备/Q 已暴露，正式 RX24-18 未访问，不称新盲测。

沿用现有 VALIDATED_ONCE 句柄：U capsule `p3dev-star-g10-6e1292b62ab09286ebf1a0df`、split `83c5cc1fe2f2a1512dbc1287`；Q capsule `p3dev-star-g10-71a5a0dec23ac6ccc3b479c2`、split `05de65cd4818f3e5e33ced5c`。不重抽记录、信道或场景以改善覆盖。P0 是 received，P1 是同条 received 经冻结 G；均为星地。Receiver2027/scene2026103001/channel2026104001、600km/25MHz/2.45GHz/residual-post_sync-noEQ、六分量面积概率混合保持原定义。

## 固定算法

原 paired 条件 margin 记 m，原 P1 单位 fused160 为 x。DA0 使用 tanh(m)，接收符号和旧身份与原 paired 基线精确一致。正种子为 m≥0 且原 MSP≥.95，负种子为 m<0 且 MSP≤.5。这里只生成已知／未知二元伪标签，不以六类身份伪标签训练特征。

在全 U 的 FP64 cosine 上构造 mutual10NN 图，排 self，同分按 opaque ID。`d2=max(0,2−2*clip(cos,-1,1))`；σ 是第 k 个其他 U 邻居距离，floor1e-6。互近邻边权 `W_ij=exp(-d2/(σ_iσ_j))`，diag0，`L=diag(W.sum(1))−W`。正负种子以 D_s 和符号 s 表示，固定求解一次：

\[
(I+L+10D_s)f=\tanh(m)+10s.
\]

这相当于源分数先验、图平滑和固定强度二元种子约束的严格凸二次目标。若任一类种子为空，显式 SKIPPED，DA1 对 U/Q 精确退回 DA0，不改阈值或重启。一般情况下报告完整种子数、图边数、求解残差、目标能量及 f 范围；无神经优化器、更新轮次、EMA 或早停。

每条未来 q 独立寻找固定 U 的 k10 近邻，σ_q 仅用该 q 到 U 的第 k 距离，再算相同 w。输出 `f_q=(tanh(m_q)+Σw*f_U)/(1+Σw)`，接收阈值恒为0。没有 Q 间关系、Q 批统计或 Q 再拟合。原默认六类头 argmax 始终不变，不能把 gate 收益写成身份特征改善。

本轮允许并明确保存合法**无标签 U memory**：384×160 FP32 unit、opaque IDs、384维 FP64 σ/f 和 schema/skipped；不是无记忆部署。Source 逐样本 bank 仍为0。图权重／L 只用于 U 求解，不要求部署；实际数组／文件字节和逐 q 距离成本分别报告。

图平滑的出发点可见 [Zhu、Ghahramani 和 Lafferty 的原论文](https://mlg.eng.cam.ac.uk/pub/pdf/ZhuGhaLaf03a.pdf)。本实现加入源 margin 先验、自动二元伪种子和独立未来行插值，不声称原论文或现有 SF-OSDA 方法的复现。

## 注册与四态评分

DA0/DA1 分别在原、当前 margin 均负的 U 上，用原 frozen unit160 mutual10NN 候选诱导图取分量；自主 K，可为0。沿用 min16、与正 trusted 邻接点比例≤.05、交替两半中心距≤LOO95%半径、中心／半径floor1e-6。注册不额外拟合神经模型。

REG 先保留 old margin≥0 的原旧 label，仅 old 拒识后允许匿名原型球；否则拒识。同 DA 注册旧正确接收保持是算法性质，仍须实际预测验证；不能代替未来新类或 later 拒识证据。所有图／注册状态在 Q 首次访问前冻结。

全部四态预测先完整固定，独立 CPU scorer 再核公有完成、读取 U truth，冻结 DA0/DA1 两个独立 Hungarian 匿名映射；两 map 全完成才读 Q truth。映射不回算法，未知真 K 不参与发现。漏发现的新类仍计 A_N 分母；未匹配匿名是接收错误，不是拒识。

主报 true-TX 宏 A_K/R_U/H，附 C_K、原 F0 同输入闭集损伤、A_N/later R_U/H_OW、同 DA REG 旧损失、逐 TX/RX/day/六场景与固定 cohort 配对。REG0 的 R_U 为全部三未知，REG1 为计划 later 集合，不能直接比较两个 H/R_U 当同分母增益。缺必需类别的场景保留 N/A、coverage=false，不重抽样。

v3 最终门槛保持：DA1_REG0 的 overall及六主景 C_K≥90%、R_U≥90%、A_K≥原同输入 F0 closed−2个百分点，并有正旧 A_K 适应收益；DA1_REG1 的 A_N≥80%、应注册旧六＋全部到达类 C_K≥90%、later拒识≥90%、原 F0 保护2个百分点和同DA REG旧损失≤2个百分点及覆盖。DA0绝对失败只是对照，不加 source90 或 DA0先通过的入口门槛。主失败不换状态或控制，不自动扩 seed／正式集；完整目标 ACTIVE。

## 执行和停止

N607 单个本任务 GPU 提取／预测进程，卡0至7取实际free≥4096MB的最高者；每卡最多新增2个own，本轮仅1，不停止／修改其他用户。独立 CPU scorer 随后自动执行。一个不可覆盖 root、一个owner；代码/config单Git提交、一次P0/P1、定向测试、一次preflight、一次archive SHA比较／remote compile。真实 F0/G 合成 G10 smoke 在 U 前首步 PASS 后立即继续。

预期 real F0 full2496/G1248，合成 smoke4/2、默认头feature2另计；source四快照每次均实际计算，真实MLP4992行、合成8行，F/G+四快照512876参数，无新adapter。源祖先成本保留。图求解／Q插值／注册／U memory 非神经成本逐项报告，不预报 B1 延迟或加速。

仅直接协议／来源／非有限／输出冲突／无法合法预测等技术错误停止并保留；低分、缺种子、K0 不触发重选、重启或目标标签访问。外部状态 UNKNOWN 先核原 root/PID，不重复提交。
