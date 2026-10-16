# 旧普通增强失败后：先检验额外 PA 的源身份破坏

状态：DESIGN_NOT_RUN。五个源 RX 配对训练已完整分析，整套旧 ordinary 作为可靠 Ag 的候选按原规则结束；不启动该配方的 full-source 联合或扩 seed。下面是不同的问题：定位已接收 IQ 上再次施加 PA 的直接影响，不将冻头诊断冒充训练收益。

## 当前证据

当前 SIG_TIME/RAW_HZ/NoExplicitCFO/joint160/defaultcos＋RDrop、LS.01、scratch E20=13L+7U 的五个源 RX 留出中，N1−N0 accuracy 依次为 RX1 +6.4074、RX3 +4.4259、RX4 +2.8333、RX6 −0.9259、RX8 −2.5926 pp。均值 +2.0296 pp、macroF1 +2.3152 pp，但只有 3/5 折为正且最差折低于 −1 pp，源规则 FAIL。RX8 的 TX2/TX3 分别下降约 18.4/13.3 pp，不能以平均净收益称身份保持。

这不是技术故障：四对新行的全部 L/U 实际样本序列一致，新 160 epoch/35520 更新和完整日志已解析，所有模型及默认头正常联合训练，无 U TX、As 或 target。训练代码为 2cbaee17215772bc50ec63d77b771675894991ec。完整结果在 [五折报告](../automation_reports/CV-SincNet/20261016-phase1-ordinary-augmentation-sourceRX-confirm-e20-s2026092701-m8-r01/report.md)。

实际代码有一个可以精确干预的边界：native train_ssdg 调用 augmentor 时显式传入 no_pa=(not args.aug_enable_pa_normal)，实际 args 为 true，因此原 ordinary 确实允许额外 PA。RFFIAugmentor 正常视图先做 anti-shortcut/general channel，再在最后对抽中的样本执行 simulate_pa；DAC 概率为 0、class_signature 关闭。只修改 no_pa，复用同一 RNG 起点，PA 之前的所有输入、抽样和操作均可完全相同。不能仅凭先前输入峰值 204.893 就认定 PA 是原因。

## 唯一优先问题与最小实验

优先问题：**在已接收 IQ 上再施加当前 PA，是否在其余观测扰动固定时，直接破坏冻结默认头原本能够辨认的源身份？** 用户已明确要求 PA 等非线性不能未经验证就视为标签保持。先检验此项风险，不默认它解释了全部跨 RX 训练损失。

只读使用上述五折各自的 N0 final20 默认头，全部冻结，不使用 N1 或历史 E50/E200、不拟合任何分类器。每个模型仅诊断其留出 RX 的原 L_s 1260 个 IQ，五折合计原合法 L6300，6TX×3day×70/折。模型训练使用相应四 RX 的 L5040/U45360，来源、原角色/物理 ID 和 held RX 在载入前核对。V 不复测，U 不使用，target 不构造。

固定三条件：RAW、原 ordinary FULL（no_pa=false）、其余过程相同的 NO_PA（no_pa=true）。使用原实际 E13 概率/范围，不寻找更有利强度；每个物理样本 16 个命名 draw，诊断流与已完成训练、验证、目标测试分离，随机流不由 TX 定义。FULL/NO_PA 每次从完全相同的随机状态开始；由于 PA 位于末端，必须实际验证其前级 IQ 一致、未受 PA 作用的包输出一致。原 RMS 归一化位置与完整 CVS 前向不改，不裁剪峰值、不新增 PA、不反演未知 TX 硬件。

首要结果是默认头准确率、macroF1、RAW 正确固定集合上的保持率、修复/新增错误；记录所有 RX/TX/day 及 PA 实际作用子集，类别间隔只作机制支持。每个输入记录前后功率、最大分量、峰均比和非有限值，区别原始录音极值、前级变化和 PA 增量。只保留统计/合法源决策及必要日志，不把数据/权重放入 Git。

预期只读网络前向为 RAW6300＋FULL100800＋NO_PA100800＝207900，另计真实 checkpoint no-query smoke；0 optimizer 更新、0 新模型。最后不足一个 batch 的计数按实际保存，不通过丢包使计数变整。

## 解释边界与放弃条件

首要对比 NO_PA−FULL。事前以保持率均值改善至少 0.5 pp、至少 4/5 RX 为正、最差 RX 不低于 −0.5 pp 作为“存在跨 RX 一致的直接 PA 损伤”的探索判据；同时完整报告准确率/F1，不能只挑条件正确子集。对实际 PA 作用样本的差额只解释算子作用，不代替整个分布上的效果。

若不满足该判据，不以更强 PA、改变概率、换 seed 或把次要 RX/类别选成新主要结果救援；关闭“额外 PA 直接损伤是一致主因”的假设。即使满足，也只能说明当前冻结头的条件输入风险，不能证明 PA 导致 N1 训练后的 RX6/8 下降，不能证明 NO_PA 是有效 Ag。训练因果还需要另立同预算 scratch E20 对照；不自动启动。

AWGN、多径、移位、DC、taper 等还可能有单独影响和交互；本首轮不把它们全堆成候选。FULL/NO_PA 的同随机对照先给出一个能够解释的结果，避免根据源分数扫描开关组合。

正式联合目标保持：clean 净改善、原 low urban 超过 60%、suburban 接近自身 clean；测试 IQ/信道和默认头固定。当前尚无该诊断结果、可靠 Ag、新 full-source 混合结果或跨 seed 结论。
