# 跨 RX 脆弱方向的特征学习

候选假设：一个源 RX 上的监督梯度，可能揭示其他 RX 上也敏感的特征方向；训练应抑制这种跨样本、跨 RX 的损失增量。它不依赖 U 真标签、不增加伪标签接受量，也不要求末期 AdamW 一小步 CE 增量达到贡献门槛。当前尚无该候选的训练收益或创新性证明。

## 已有证据与保留路线

- SIG_TIME 是已验证有潜力的实际时序特征路线；固定 CENTER_25K 的历史 3 种子各有正差。保留各自证据，不相加增益、不据目标分数挑父权重。
- 固定 BASE/final50 的 RAW_HZ primary Sinc 更新受 FP32 分辨率及 AdamW eps 影响。CENTER_25K 源假设步骤可改变 12 至 13 个滤波核，但跨样本 CE credit 仍小于 1e−6。该检查不支持直接扩训练，也不能否定所有早中期机制。
- 旧 contribution selector 的 functional-call 参数缓存存在版本键碰撞，归因限制已补记；原模型、预测、评分和未启动行保留。

## 明确机制

以原默认分类头 W 和部署特征参数 θ_F 定义真实 L 的 CE。θ_F 只包括 primary identity Sinc、time_fuse、t1、t2 和 SIG_TIME sequence_pool（排除 gain_raw）。其他参数在辅助分支中取 detached 当前值，原主训练仍联合更新默认头及骨干。Sinc 固定原构造 origin 与 25,000 Hz 中心坐标，不扫描单位。

对当前 L 梯度批次的每个源 RX r，计算 g_r=∇θ_F CE_r，再取停止梯度的 δ_r=0.05g_r/max(||g_r||₂,1e−12)。内步骤不执行 optimizer，也不扰动 W/gain。使用与当前批次完整物理 ID/base_index 分离的原 L bank，评价另外 4 个源 RX。令 E_r(θ_F) 为这些样本的平均 CE，辅助项为：

`L_cross = (0.1/5) Σ_r relu(E_r(θ_F + stop(δ_r)) − E_r(θ_F))`。

clean 与 perturbed 两项都对 θ_F 保留梯度；不是把 clean 当 detached 常数后重复加普通 CE。δ 停止梯度，所以没有二阶元梯度。零增量停止施压，所有 5 个 RX 维持固定分母。每个 functional state 克隆 buffer，临时 eval、保留 RNG，逐次清空并最终恢复参数派生缓存，live 参数、buffer、gradient 和 moment 保持。

固定对照 ZERO、CLEAN_CE、GLOBAL_GAP、CROSS_GAP。GLOBAL_GAP 从完整 current L 的平均 CE 产生一条共同 δ，但采用相同 5 个 other-RX masks；CLEAN_CE 使用相同 bank/masks 的普通 CE；ZERO 的辅助值为 0。各控制均执行相同 clean 和 5 次 perturb forward，披露实际 source exposure/forward/backward 成本。新增参数、头拟合、真实 inner optimizer、V/U truth 和目标访问均为 0。

held bank 仍是模型此前训练过的源 L，只在当前梯度步骤排除，不称未训练 holdout 或未见 RX。它检验跨样本可迁移的局部脆弱性，不等于真正目标域泛化。

## 与文献的关系

[SAM](https://arxiv.org/abs/2010.01412) 已提出在参数邻域内优化最坏损失；不能把权重扰动本身称作新方法。[SAGM](https://arxiv.org/abs/2303.10353) 已研究 empirical/perturbed risk 与其 gap 的梯度关系。[DISAM](https://arxiv.org/abs/2405.18861) 使用域损失方差调整 perturbation；[DGSAM](https://arxiv.org/abs/2503.23430) 研究逐域 sharpness；2026 年 [Dual Adaptive SAM](https://proceedings.mlr.press/v306/he26m.html) 进一步处理域差异与多方向 flatness。

这里待检验的区别是：只干预实际部署特征，固定当前默认读出；每个 RX 生成脆弱方向，却在排除该 RX 和当前物理样本的其他 RX 上学习损失 gap。必须用 GLOBAL_GAP 和 CLEAN_CE 验证其是否提供额外信号。文献核查不证明独创，也不证明这种定向信号有性能收益。

## 下一项有界检查

先在固定 BASE 首种子/final50 上做 source-only 原型；只是合法既有权重的诊断，不续训该 E50 checkpoint。4 个固定 L90 cohorts、原 L180 bank，固定 ρ=.05、系数=.1、4 个控制。检查跨 RX 正 gap、auxiliary 梯度进入浅中层与实际 kernel、W/gain 不接辅助梯度、CROSS 与 GLOBAL 的梯度差及 live 状态恢复。通过后才预登记实际训练。

后续新训练遵循当前 E20 预算（scratch20 或合规 E20+20），另保留同预算纯 CE 对照；不默认复制旧 E50 日程。研究变体按明确安排使用隐藏真值的 U；默认头保持联合训练，最终固定 clean 预测后独立 truth-last。旧 E50 成本及结果保留，不冒充 E20 同成本。当前默认头 clean 三种子 84% 目标未达成。


2026-10-15：跨RX feature sharpness源原型 `20261015-phase1-SIGTIME-crossRX-feature-sharpness-sourceL-manysig-s2026092701-r01` 4/4通过固定信号可用性条件。currentRX造detachedρ=.05特征方向，在otherRX／排除currentID的seenL上优化relu(pertCE-cleanCE)，两项均向17个部署参数传梯度，固定preW/defaulthead。CROSS最大RXgap6.325e−4至4.005e−3，梯度范数.001616至.007193，与GLOBAL相对差.441至3.532；actualAdamW aux增量改变2至3个Sinc核。540L IQ／14762forwards／8tempsteps，0realtrain/headfit/U/V/target。没有证明DG收益或原创性。保留SIG_TIME／固定CENTER既有正路线，已经准备新的E20匹配矩阵接口；BASE_CE／BASE_SSL／CLEAN_CE／GLOBAL_GAP／CROSS_GAP计划先首seed再固定source规则扩展，实际阶段与预算须预登记，不默认E50续训。84%默认头clean三种子仍active／未完成。
